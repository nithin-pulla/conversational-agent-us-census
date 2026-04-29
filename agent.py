"""
agent.py – Multi-step ReAct agent for US Census Text-to-SQL.

Pipeline per user message:
  1. Guardrail   → fast-fail off-topic queries.
  2. ReAct Loop  → LLM reasons step-by-step, calling SQL tools iteratively:
                     - Lookup FIPS codes, field descriptions, schema info
                     - Run the final data query
                   Up to MAX_STEPS iterations, each result fed back to LLM.
  3. Synthesis   → LLM writes the final conversational answer.

Key improvements over single-shot:
  - Fully qualified table names (DB.SCHEMA.TABLE) for both PUBLIC and INFORMATION_SCHEMA
  - LLM can chain lookups: county FIPS → field name → INFORMATION_SCHEMA check → data query
  - "Chain of thought" for ambiguous geography / unknown field names
"""

import json
import logging
import os
import re
import time
from typing import Any, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from db import QueryError, UnsafeQueryError, run_query
from retrieval import get_retriever

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

_LLM_MODEL = "openai/gpt-oss-120b:free"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MAX_STEPS = 5   # max SQL tool calls per question


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=_OPENROUTER_BASE,
    )


def _chat(messages: list, temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Chat completion with 429 retry (3x exponential backoff)."""
    client = _get_client()
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=_LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate" in err_str.lower():
                wait = 2 ** (attempt + 1)
                logger.warning("Rate-limited (attempt %d/3), retrying in %ds …", attempt + 1, wait)
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc or RuntimeError("LLM call failed after retries.")


# ---------------------------------------------------------------------------
# Step 1 – Guardrail
# ---------------------------------------------------------------------------

_GUARDRAIL_SYSTEM = """
You are a strict input validator for a US Census data assistant.
Your ONLY job is to decide if a user message is related to:
  - US population, demographics, or census statistics
  - Geographic regions (states, counties, cities, zip codes)
  - Age, sex, race, ethnicity, income, education, housing, or employment data

Reply with EXACTLY one word: ALLOW or DENY.
""".strip()


class OffTopicError(Exception):
    """Raised when the guardrail rejects a query."""


def check_guardrail(user_message: str) -> None:
    verdict = _chat(
        messages=[
            {"role": "system", "content": _GUARDRAIL_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    verdict = verdict.strip().upper().split()[0] if verdict.strip() else "DENY"
    logger.debug("Guardrail verdict: %s", verdict)
    if verdict != "ALLOW":
        raise OffTopicError(
            "I'm a US Census data assistant and can only answer questions about "
            "US population statistics, demographics, housing, income, education, "
            "and related topics. Please rephrase your question accordingly."
        )


# ---------------------------------------------------------------------------
# Step 2 – Multi-step ReAct reasoning loop
# ---------------------------------------------------------------------------

def _db_name() -> str:
    return os.environ.get("SNOWFLAKE_DATABASE", "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET")


def _make_react_system() -> str:
    db = _db_name()
    return f"""
You are a data analyst agent with access to a Snowflake SQL tool.
Answer US Census questions by querying the database step-by-step.

DATABASE: {db}
Two schemas available:
  - {db}.PUBLIC          → 71 census data tables + metadata tables
  - {db}.INFORMATION_SCHEMA → schema metadata (COLUMNS, TABLES, etc.)

════════════════════════════════════════════
KEY LOOKUP TABLES (query these first for geography/field lookup):
  {db}.PUBLIC."2020_METADATA_CBG_FIPS_CODES"
      Columns: STATE, STATE_FIPS, COUNTY_FIPS, COUNTY, CLASS_CODE
      Use: find FIPS codes by county/state name
      Example: WHERE LOWER("COUNTY") LIKE '%san francisco%'

  {db}.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS"
      Columns: TABLE_ID, TABLE_NUMBER, TABLE_TITLE, TABLE_TOPICS,
               FIELD_LEVEL_1 … FIELD_LEVEL_10
      Use: find which column code maps to a concept (income, population, race…)
      Example: WHERE LOWER("TABLE_TITLE") LIKE '%income%'

  {db}.INFORMATION_SCHEMA.COLUMNS
      Use: verify a column exists in a table before querying it
      Example: WHERE TABLE_NAME = '2020_CBG_B19' AND COLUMN_NAME = 'B19013e1'

DATA TABLES:
  Pattern: {db}.PUBLIC."<year>_CBG_<code>"
  Examples: "2020_CBG_B01" (population), "2020_CBG_B19" (income/earnings),
            "2020_CBG_B25" (housing), "2020_CBG_B15" (education)
  Default year: 2020 unless asked for 2019.
  Columns: "CENSUS_BLOCK_GROUP" (TEXT, join key) + estimate fields "<ID>e<n>"

GEOGRAPHY FILTERING:
  CENSUS_BLOCK_GROUP starts with STATE_FIPS (2 digits) + COUNTY_FIPS (3 digits)
  San Francisco CA = '06075', Los Angeles CA = '06037', New York NY = '36061'
  Filter: WHERE "CENSUS_BLOCK_GROUP" LIKE '06075%'
  For state-level: LIKE '06%' (CA), LIKE '48%' (TX), LIKE '36%' (NY)

STATISTICAL ACCURACY — CRITICAL FOR INCOME QUERIES:
  B19013e1 is a PRE-COMPUTED MEDIAN at the CBG level. Aggregating medians (via AVG or
  MEDIAN) across CBGs does NOT give the true county/state median — it can be off by $10,000+.

  THE CORRECT METHOD for county/state MEDIAN HOUSEHOLD INCOME:
  Use B19001e2–B19001e17 (income bracket COUNTS) to reconstruct the distribution,
  then interpolate. This matches the Census Bureau's official figure within ~$50.

  Bracket definitions (B19001 in 2020_CBG_B19):
    e2=<$10k, e3=$10-15k, e4=$15-20k, e5=$20-25k, e6=$25-30k,
    e7=$30-35k, e8=$35-40k, e9=$40-45k, e10=$45-50k, e11=$50-60k,
    e12=$60-75k, e13=$75-100k, e14=$100-125k, e15=$125-150k,
    e16=$150-200k, e17=$200k+

  CORRECT QUERY PATTERN (interpolated median via bracket counts):
  WITH brackets AS (
    SELECT
      SUM("B19001e2")  AS b2,  SUM("B19001e3")  AS b3,
      SUM("B19001e4")  AS b4,  SUM("B19001e5")  AS b5,
      SUM("B19001e6")  AS b6,  SUM("B19001e7")  AS b7,
      SUM("B19001e8")  AS b8,  SUM("B19001e9")  AS b9,
      SUM("B19001e10") AS b10, SUM("B19001e11") AS b11,
      SUM("B19001e12") AS b12, SUM("B19001e13") AS b13,
      SUM("B19001e14") AS b14, SUM("B19001e15") AS b15,
      SUM("B19001e16") AS b16, SUM("B19001e17") AS b17,
      SUM("B19001e1")  AS total
    FROM DB.PUBLIC."2020_CBG_B19"
    WHERE "CENSUS_BLOCK_GROUP" LIKE '06075%' AND "B19001e1" > 0
  ),
  cumulative AS (
    SELECT total, total/2.0 AS median_rank,
      b2,  b2+b3 AS c3,  b2+b3+b4 AS c4,  b2+b3+b4+b5 AS c5,
      b2+b3+b4+b5+b6 AS c6, b2+b3+b4+b5+b6+b7 AS c7,
      b2+b3+b4+b5+b6+b7+b8 AS c8, b2+b3+b4+b5+b6+b7+b8+b9 AS c9,
      b2+b3+b4+b5+b6+b7+b8+b9+b10 AS c10,
      b2+b3+b4+b5+b6+b7+b8+b9+b10+b11 AS c11,
      b2+b3+b4+b5+b6+b7+b8+b9+b10+b11+b12 AS c12,
      b2+b3+b4+b5+b6+b7+b8+b9+b10+b11+b12+b13 AS c13,
      b2+b3+b4+b5+b6+b7+b8+b9+b10+b11+b12+b13+b14 AS c14,
      b14, b15, b16, b17
    FROM brackets
  )
  SELECT ROUND(
    CASE
      WHEN median_rank <= b2   THEN 0      + (median_rank/NULLIF(b2,0))*10000
      WHEN median_rank <= c3   THEN 10000  + ((median_rank-b2)/NULLIF(b3,0))*5000
      WHEN median_rank <= c4   THEN 15000  + ((median_rank-c3)/NULLIF(b4,0))*5000
      WHEN median_rank <= c5   THEN 20000  + ((median_rank-c4)/NULLIF(b5,0))*5000
      WHEN median_rank <= c6   THEN 25000  + ((median_rank-c5)/NULLIF(b6,0))*5000
      WHEN median_rank <= c7   THEN 30000  + ((median_rank-c6)/NULLIF(b7,0))*5000
      WHEN median_rank <= c8   THEN 35000  + ((median_rank-c7)/NULLIF(b8,0))*5000
      WHEN median_rank <= c9   THEN 40000  + ((median_rank-c8)/NULLIF(b9,0))*5000
      WHEN median_rank <= c10  THEN 45000  + ((median_rank-c9)/NULLIF(b10,0))*5000
      WHEN median_rank <= c11  THEN 50000  + ((median_rank-c10)/NULLIF(b11,0))*10000
      WHEN median_rank <= c12  THEN 60000  + ((median_rank-c11)/NULLIF(b12,0))*15000
      WHEN median_rank <= c13  THEN 75000  + ((median_rank-c12)/NULLIF(b13,0))*25000
      WHEN median_rank <= c14  THEN 100000 + ((median_rank-c13)/NULLIF(b14,0))*25000
      WHEN median_rank <= c14+b15 THEN 125000 + ((median_rank-c14)/NULLIF(b15,0))*25000
      WHEN median_rank <= c14+b15+b16 THEN 150000 + ((median_rank-(c14+b15))/NULLIF(b16,0))*50000
      ELSE 200000
    END, 0) AS interpolated_median_income
  FROM cumulative;

  NEVER use AVG("B19013e1") or MEDIAN("B19013e1") for county/state-level income — they
  ignore household distribution within CBGs and produce results ~$10,000 off.
  Only use B19013e1 for single-CBG lookups.

CRITICAL QUOTING RULES:
  - ALWAYS double-quote table names: "2020_CBG_B19" (they start with digits)
  - ALWAYS double-quote column names: "B19013e1", "CENSUS_BLOCK_GROUP"
  - Aliases are fine without quotes: AS median_income

════════════════════════════════════════════
OUTPUT FORMAT — follow exactly:

To run a SQL query, output:
THOUGHT: <your reasoning>
SQL:
<your fully-qualified sql>
END_SQL

After seeing results, continue with more THOUGHT/SQL blocks.

When ready to give the final answer, output:
FINAL_ANSWER:
<your natural language answer with numbers formatted with commas>
END_ANSWER

Rules:
  - Max {MAX_STEPS} SQL steps before FINAL_ANSWER
  - Use fully qualified names in every query
  - If empty results, try a different approach (different year, fuzzy LIKE, etc.)
  - Never guess — if data is unavailable, say so clearly
""".strip()


_SQL_PATTERN = re.compile(r"SQL:\s*\n(.*?)END_SQL", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"FINAL_ANSWER:\s*\n(.*?)END_ANSWER", re.DOTALL)


def _extract_sql(text: str) -> Optional[str]:
    m = _SQL_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _extract_answer(text: str) -> Optional[str]:
    m = _ANSWER_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _rows_to_text(rows: List[dict]) -> str:
    """Compact rows representation for LLM context (cap at 20 rows)."""
    trimmed = rows[:20]
    return json.dumps(trimmed, indent=2, default=str)


def run_react_loop(
    user_question: str,
    chat_history: List[dict],
    schema_context: str,
) -> Tuple[str, Optional[str]]:
    """
    Run the multi-step ReAct reasoning loop.
    Returns (final_answer, last_sql_used).
    """
    system_prompt = _make_react_system()
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        *chat_history[-6:],
        {
            "role": "user",
            "content": (
                f"Schema context (BM25-retrieved relevant columns):\n{schema_context}\n\n"
                f"Question: {user_question}"
            ),
        },
    ]

    last_sql: Optional[str] = None

    for step in range(MAX_STEPS):
        response_text = _chat(messages, temperature=0.0, max_tokens=2048)
        logger.debug("ReAct step %d:\n%s", step + 1, response_text)

        messages.append({"role": "assistant", "content": response_text})

        # Check if LLM gave a final answer
        answer = _extract_answer(response_text)
        if answer:
            return answer, last_sql

        # Check if LLM wants to run SQL
        sql = _extract_sql(response_text)
        if sql:
            last_sql = sql
            logger.info("Step %d SQL:\n%s", step + 1, sql)
            try:
                rows = run_query(sql, max_rows=50)
                result_text = _rows_to_text(rows)
                row_count = len(rows)
                feedback = f"SQL executed successfully. Returned {row_count} row(s):\n{result_text}"
            except UnsafeQueryError as uqe:
                feedback = (
                    f"SAFETY VIOLATION — query blocked before execution: {uqe}\n"
                    "You must only generate read-only SELECT statements. "
                    "Rewrite the query as a SELECT and try again."
                )
            except QueryError as qe:
                feedback = f"SQL ERROR: {qe}\nPlease fix the query and try again."

            messages.append({"role": "user", "content": f"Tool result:\n{feedback}"})
        else:
            # LLM produced neither SQL nor FINAL_ANSWER — try to extract plain text answer
            logger.warning("Step %d: no SQL or FINAL_ANSWER found in response.", step + 1)
            if response_text:
                return response_text, last_sql
            break

    # Exhausted steps — ask for a direct answer
    messages.append({
        "role": "user",
        "content": (
            "You have used the maximum number of SQL steps. "
            "Based on all information gathered so far, provide your FINAL_ANSWER now.\n"
            "FINAL_ANSWER:\n<your answer>\nEND_ANSWER"
        ),
    })
    final = _chat(messages, temperature=0.1, max_tokens=1024)
    answer = _extract_answer(final) or final
    return answer, last_sql


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(
    user_message: str,
    chat_history: List[dict],
) -> Tuple[str, Optional[str]]:
    """
    Full agent pipeline. Returns (answer_text, sql_used_or_None).
    Never raises — all exceptions are caught and converted to user-friendly messages.
    """
    # Step 1 – Guardrail
    try:
        check_guardrail(user_message)
    except OffTopicError as exc:
        return str(exc), None
    except Exception as exc:
        logger.error("Guardrail call failed: %s", exc)
        return "I'm having trouble validating your request right now. Please try again.", None

    # Step 2 – Hybrid schema retrieval (BM25) for context
    try:
        retriever = get_retriever()
        schema_ctx = retriever.schema_context(user_message, top_k=25)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        schema_ctx = "(schema context unavailable)"

    # Step 3 – Multi-step ReAct reasoning
    try:
        answer, sql_used = run_react_loop(user_message, chat_history, schema_ctx)
        return answer, sql_used
    except Exception as exc:
        logger.exception("Unexpected error in ReAct loop: %s", exc)
        return (
            "An unexpected error occurred while processing your question. "
            "Please try again or rephrase your request.",
            None,
        )
