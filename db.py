"""
db.py – Snowflake connection and query execution.

Responsibilities:
  - Manage a single reusable Snowflake connection per process.
  - Execute SQL safely, returning rows as a list of dicts.
  - Expose a schema inspector that joins INFORMATION_SCHEMA.COLUMNS
    with the dataset's own metadata table to produce rich descriptions
    (e.g. "Estimate: Sex By Age — Male, 35 to 39 years").

Schema notes (from live inspection):
  - 71 tables, all in PUBLIC schema, all named <year>_CBG_<code>
  - Two years of ACS data: 2019 and 2020
  - Column naming convention:
      <field_id>e<n>  → Estimate (use these for calculations)
      <field_id>m<n>  → Margin of Error (skip unless uncertainty needed)
  - CENSUS_BLOCK_GROUP (TEXT) is the join key across all tables
  - 2019_METADATA_CBG_FIELD_DESCRIPTIONS maps column codes → human descriptions
  - 2019_CBG_GEOMETRY has STATE, COUNTY, STATE_FIPS, COUNTY_FIPS columns
  - 2019_METADATA_CBG_GEOGRAPHIC_DATA has LATITUDE, LONGITUDE per CBG
"""

import logging
import os
from typing import Any, List, Optional

import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection singleton
# ---------------------------------------------------------------------------

_connection: Optional[snowflake.connector.SnowflakeConnection] = None


def get_connection() -> snowflake.connector.SnowflakeConnection:
    """Return (or create) the shared Snowflake connection."""
    global _connection
    if _connection is None or _connection.is_closed():
        logger.info("Opening Snowflake connection …")
        _connection = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            role=os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
            session_parameters={"QUERY_TAG": "text-to-sql-census-agent"},
        )
        logger.info("Snowflake connection established.")
    return _connection


def close_connection() -> None:
    """Explicitly close the connection (call on shutdown)."""
    global _connection
    if _connection and not _connection.is_closed():
        _connection.close()
        _connection = None
        logger.info("Snowflake connection closed.")


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

MAX_ROWS = 500  # Safety cap — avoids dumping huge result sets into the LLM

# DDL/DML verbs that must never appear as the leading statement keyword.
# This covers SQL injection and prompt-injection payloads even when the
# attacker tries to smuggle a write behind a comment or semicolon.
_BLOCKED_KEYWORDS = frozenset({
    "insert", "update", "delete", "merge", "upsert",
    "create", "alter", "drop", "truncate", "replace",
    "grant", "revoke", "deny",
    "copy", "put", "get",                  # Snowflake bulk-load / unload
    "call", "execute", "exec",             # stored procedures
    "begin", "commit", "rollback",         # transaction control
    "use",                                 # role / database switching
})


class QueryError(Exception):
    """Raised when Snowflake returns an error for a generated SQL statement."""


class UnsafeQueryError(QueryError):
    """Raised when SQL fails the safety check before reaching Snowflake."""


def _validate_sql(sql: str) -> None:
    """
    Reject any SQL that is not a read-only SELECT (or WITH…SELECT) statement.

    Strategy — two complementary checks:
    1. Leading-keyword check: the first real keyword must be SELECT or WITH.
    2. Blocked-keyword scan: no statement in the batch may start with a
       write/admin verb (catches semicolon-separated payloads).

    Both checks operate on normalised, comment-stripped text so that tricks
    like `/* DROP */ SELECT` or `--\nDROP` are caught.

    Raises UnsafeQueryError for anything that fails.
    """
    import re

    # Strip single-line (--) and block (/* */) comments, then collapse whitespace.
    stripped = re.sub(r"--[^\n]*", " ", sql)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL)
    stripped = stripped.strip()

    if not stripped:
        raise UnsafeQueryError("Empty query rejected.")

    # Extract the first keyword of the whole statement.
    first_keyword = re.match(r"[a-zA-Z_]+", stripped)
    if not first_keyword or first_keyword.group(0).lower() not in ("select", "with"):
        raise UnsafeQueryError(
            f"Only SELECT queries are permitted. Got leading keyword: "
            f"'{first_keyword.group(0) if first_keyword else stripped[:20]}'"
        )

    # Scan every semicolon-separated segment for blocked leading keywords.
    # This catches `SELECT 1; DROP TABLE foo` payloads.
    for segment in re.split(r";", stripped):
        seg = segment.strip()
        if not seg:
            continue
        kw_match = re.match(r"[a-zA-Z_]+", seg)
        if kw_match and kw_match.group(0).lower() in _BLOCKED_KEYWORDS:
            raise UnsafeQueryError(
                f"Statement contains forbidden keyword '{kw_match.group(0)}'. "
                "Only read-only SELECT queries are allowed."
            )


def run_query(sql: str, max_rows: int = MAX_ROWS) -> List[dict]:
    """
    Execute *sql* against Snowflake and return at most *max_rows* rows.

    Validates that the query is read-only before sending it to Snowflake.

    Returns:
        A list of dicts, one per row (column-name → value).

    Raises:
        UnsafeQueryError: if the SQL fails the read-only safety check.
        QueryError: on any Snowflake-level or network error.
    """
    _validate_sql(sql)
    conn = get_connection()
    logger.debug("Executing SQL:\n%s", sql)
    try:
        cur = conn.cursor(snowflake.connector.DictCursor)
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        logger.debug("Query returned %d row(s).", len(rows))
        return list(rows)
    except snowflake.connector.errors.ProgrammingError as exc:
        raise QueryError(f"Snowflake SQL error: {exc}") from exc
    except Exception as exc:
        raise QueryError(f"Unexpected database error: {exc}") from exc


# ---------------------------------------------------------------------------
# Schema inspection — enriched with metadata table descriptions
# ---------------------------------------------------------------------------

def fetch_schema_metadata() -> List[dict]:
    """
    Return a flat list of schema entries for every ESTIMATE column
    (suffix 'e') in the dataset, enriched with human-readable descriptions
    from the metadata table.

    Each entry is:
        {
          "table":    "2019_CBG_B01",
          "column":   "B01001e1",
          "data_type":"NUMBER",
          "comment":  "Estimate: Sex By Age — Total population, Total",
        }

    Joins:
      INFORMATION_SCHEMA.COLUMNS
      → 2019_METADATA_CBG_FIELD_DESCRIPTIONS ON TABLE_ID = COLUMN_NAME

    We also include the geometry/geography columns and a few special columns
    (CENSUS_BLOCK_GROUP, STATE, COUNTY) so the LLM can always join tables.
    """

    # Step 1: fetch all estimate columns (ending in 'e' + digits) + key join/geo cols
    columns_sql = """
        SELECT
            c.TABLE_NAME  AS "table",
            c.COLUMN_NAME AS "column",
            c.DATA_TYPE   AS "data_type"
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_SCHEMA = 'PUBLIC'
          AND (
              -- Estimate columns (skip margin-of-error 'm' columns)
              REGEXP_LIKE(c.COLUMN_NAME, '^[A-Z][0-9]+e[0-9]+$', 'i')
              -- Always include the join key and geographic identifiers
              OR c.COLUMN_NAME IN (
                  'CENSUS_BLOCK_GROUP', 'STATE', 'COUNTY',
                  'STATE_FIPS', 'COUNTY_FIPS', 'TRACT_CODE',
                  'LATITUDE', 'LONGITUDE', 'AMOUNT_LAND', 'AMOUNT_WATER'
              )
          )
        ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
    """

    # Step 2: fetch field descriptions from the metadata table
    descriptions_sql = """
        SELECT
            TABLE_ID,
            TABLE_TITLE,
            TABLE_TOPICS,
            FIELD_LEVEL_1,
            FIELD_LEVEL_2,
            FIELD_LEVEL_3,
            FIELD_LEVEL_4,
            FIELD_LEVEL_5,
            FIELD_LEVEL_6
        FROM "2019_METADATA_CBG_FIELD_DESCRIPTIONS"
    """

    try:
        col_rows = run_query(columns_sql, max_rows=15_000)
        desc_rows = run_query(descriptions_sql, max_rows=15_000)
    except QueryError as exc:
        logger.warning("Could not fetch schema metadata: %s", exc)
        return []

    # Build description lookup: column_name_lower → readable string
    desc_lookup: dict = {}
    for d in desc_rows:
        fid = (d.get("TABLE_ID") or "").strip()
        if not fid:
            continue
        parts = [
            d.get("TABLE_TITLE") or "",
            d.get("FIELD_LEVEL_5") or d.get("FIELD_LEVEL_4") or "",
            d.get("FIELD_LEVEL_6") or "",
        ]
        readable = " — ".join(p for p in parts if p)
        key = f"Estimate: {readable}" if readable else (d.get("TABLE_TITLE") or fid)
        desc_lookup[fid.lower()] = key

    # Merge
    enriched = []
    for row in col_rows:
        col = row.get("column", "")
        comment = desc_lookup.get(col.lower(), "")
        enriched.append({
            "table": row.get("table", ""),
            "column": col,
            "data_type": row.get("data_type", ""),
            "comment": comment,
        })

    logger.info("Fetched enriched metadata for %d column(s).", len(enriched))
    return enriched
