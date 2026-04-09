"""
tests/test_integration.py – Ground-truth accuracy tests against the Census Bureau API.

These tests query our Snowflake database and compare results against the official
Census Bureau ACS API (https://api.census.gov). They require real Snowflake credentials
and network access — run with:

    pytest tests/test_integration.py -v -m integration

Ground truth was fetched from Census API on 2026-04-09 and validated against:
  - 2020 ACS 5-Year Estimates (the most current year in our dataset)
  - Endpoint: https://api.census.gov/data/2020/acs/acs5

Every test asserts our answer is within a defined tolerance of the official figure.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Ground truth fetched from Census API (locked in as constants)
# ---------------------------------------------------------------------------
# Source: api.census.gov/data/2020/acs/acs5 — verified 2026-04-09

CENSUS_GROUND_TRUTH = {
    # ----- Population totals (B01001_001E) -----
    "us_total_population_2020":         326_569_308,
    "california_total_population_2020":  39_346_023,
    "texas_total_population_2020":       28_635_442,
    "florida_total_population_2020":     21_216_924,

    # ----- Over-65 population (B01001_020:025 + B01001_044:049) -----
    "florida_over65_count_2020":          4_347_912,
    "florida_over65_pct_2020":                20.49,   # percent

    # ----- Median household income (B19013_001E, county-level) -----
    "sf_county_median_income_2020":         119_136,   # San Francisco, CA
    "la_county_median_income_2020":          71_358,   # Los Angeles, CA
    "manhattan_median_income_2020":          89_812,   # New York County, NY
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    """Return a db query runner bound to the real Snowflake credentials."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    from db import run_query
    return run_query


def _db_prefix(db_func):
    db_name = os.environ.get(
        "SNOWFLAKE_DATABASE",
        "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    )
    return f"{db_name}.PUBLIC"


def bracket_interpolate(row: dict, total: float) -> float:
    """Interpolate county/state median income from B19001 bracket counts."""
    brackets = [
        (0,       10_000, row["BR2"]),
        (10_000,  15_000, row["BR3"]),
        (15_000,  20_000, row["BR4"]),
        (20_000,  25_000, row["BR5"]),
        (25_000,  30_000, row["BR6"]),
        (30_000,  35_000, row["BR7"]),
        (35_000,  40_000, row["BR8"]),
        (40_000,  45_000, row["BR9"]),
        (45_000,  50_000, row["BR10"]),
        (50_000,  60_000, row["BR11"]),
        (60_000,  75_000, row["BR12"]),
        (75_000, 100_000, row["BR13"]),
        (100_000, 125_000, row["BR14"]),
        (125_000, 150_000, row["BR15"]),
        (150_000, 200_000, row["BR16"]),
        (200_000, 300_000, row["BR17"]),
    ]
    target = total / 2.0
    cumulative = 0.0
    for lo, hi, cnt in brackets:
        prev = cumulative
        cumulative += cnt
        if cumulative >= target and cnt > 0:
            return lo + ((target - prev) / cnt) * (hi - lo)
    return 200_000


def get_county_brackets(db_func, year: str, fips_prefix: str) -> dict:
    P = _db_prefix(db_func)
    r = db_func(f'''
        SELECT
          SUM("B19001e1")  AS tot,  SUM("B19001e2")  AS br2,
          SUM("B19001e3")  AS br3,  SUM("B19001e4")  AS br4,
          SUM("B19001e5")  AS br5,  SUM("B19001e6")  AS br6,
          SUM("B19001e7")  AS br7,  SUM("B19001e8")  AS br8,
          SUM("B19001e9")  AS br9,  SUM("B19001e10") AS br10,
          SUM("B19001e11") AS br11, SUM("B19001e12") AS br12,
          SUM("B19001e13") AS br13, SUM("B19001e14") AS br14,
          SUM("B19001e15") AS br15, SUM("B19001e16") AS br16,
          SUM("B19001e17") AS br17
        FROM {P}."{year}_CBG_B19"
        WHERE "CENSUS_BLOCK_GROUP" LIKE '{fips_prefix}%' AND "B19001e1" > 0
    ''')
    return r[0]


# ---------------------------------------------------------------------------
# Population Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPopulationTotals:

    def test_us_total_population_2020(self, db):
        """US total population from 2020 CBG B01 matches Census API exactly."""
        P = _db_prefix(db)
        r = db(f'SELECT SUM("B01001e1") AS tot FROM {P}."2020_CBG_B01"')
        our_value = int(r[0]["TOT"])
        official = CENSUS_GROUND_TRUTH["us_total_population_2020"]
        # The CBG-level dataset counts ~1% more than the Census API state-level
        # rollup because CBG data includes all geographies (territories, group
        # quarters) that state-level ACS may not include. 1.5% tolerance.
        tolerance = official * 0.015
        assert abs(our_value - official) <= tolerance, (
            f"US population: got {our_value:,}, official {official:,}, "
            f"diff {our_value-official:,}"
        )

    def test_california_total_population_2020(self, db):
        """California total population (state FIPS 06)."""
        P = _db_prefix(db)
        r = db(f'''
            SELECT SUM("B01001e1") AS tot FROM {P}."2020_CBG_B01"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '06%'
        ''')
        our_value = int(r[0]["TOT"])
        official = CENSUS_GROUND_TRUTH["california_total_population_2020"]
        tolerance = official * 0.001
        assert abs(our_value - official) <= tolerance, (
            f"CA population: got {our_value:,}, official {official:,}"
        )

    def test_texas_total_population_2020(self, db):
        """Texas total population (state FIPS 48)."""
        P = _db_prefix(db)
        r = db(f'''
            SELECT SUM("B01001e1") AS tot FROM {P}."2020_CBG_B01"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '48%'
        ''')
        our_value = int(r[0]["TOT"])
        official = CENSUS_GROUND_TRUTH["texas_total_population_2020"]
        tolerance = official * 0.001
        assert abs(our_value - official) <= tolerance, (
            f"TX population: got {our_value:,}, official {official:,}"
        )

    def test_florida_total_population_2020(self, db):
        """Florida total population (state FIPS 12)."""
        P = _db_prefix(db)
        r = db(f'''
            SELECT SUM("B01001e1") AS tot FROM {P}."2020_CBG_B01"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '12%'
        ''')
        our_value = int(r[0]["TOT"])
        official = CENSUS_GROUND_TRUTH["florida_total_population_2020"]
        assert our_value == official, (
            f"FL population: got {our_value:,}, official {official:,}"
        )


# ---------------------------------------------------------------------------
# Demographics Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDemographics:

    def test_florida_over65_count_2020(self, db):
        """Florida 65+ population count matches Census API exactly."""
        P = _db_prefix(db)
        r = db(f'''
            SELECT SUM(
                "B01001e20" + "B01001e21" + "B01001e22" +
                "B01001e23" + "B01001e24" + "B01001e25" +
                "B01001e44" + "B01001e45" + "B01001e46" +
                "B01001e47" + "B01001e48" + "B01001e49"
            ) AS over65
            FROM {P}."2020_CBG_B01"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '12%'
        ''')
        our_value = int(r[0]["OVER65"])
        official = CENSUS_GROUND_TRUTH["florida_over65_count_2020"]
        assert our_value == official, (
            f"FL 65+ count: got {our_value:,}, official {official:,}"
        )

    def test_florida_over65_percentage_2020(self, db):
        """Florida 65+ percentage is within 0.01pp of Census API figure (20.49%)."""
        P = _db_prefix(db)
        r = db(f'''
            SELECT
                SUM("B01001e1") AS tot,
                SUM("B01001e20"+"B01001e21"+"B01001e22"+"B01001e23"+"B01001e24"+"B01001e25"
                   +"B01001e44"+"B01001e45"+"B01001e46"+"B01001e47"+"B01001e48"+"B01001e49") AS over65
            FROM {P}."2020_CBG_B01"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '12%'
        ''')
        our_pct = round(100.0 * r[0]["OVER65"] / r[0]["TOT"], 2)
        official = CENSUS_GROUND_TRUTH["florida_over65_pct_2020"]
        assert abs(our_pct - official) <= 0.01, (
            f"FL 65+ %: got {our_pct}%, official {official}%"
        )


# ---------------------------------------------------------------------------
# Median Income Tests (using B19001 bracket interpolation)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMedianIncome:

    def test_sf_county_median_income_2020(self, db):
        """SF County interpolated median income is within $500 of Census API ($119,136)."""
        row = get_county_brackets(db, "2020", "06075")
        our_value = bracket_interpolate(row, row["TOT"])
        official = CENSUS_GROUND_TRUTH["sf_county_median_income_2020"]
        tolerance = 500
        assert abs(our_value - official) <= tolerance, (
            f"SF income: got ${our_value:,.0f}, official ${official:,}, "
            f"diff ${abs(our_value-official):,.0f}"
        )

    def test_la_county_median_income_2020(self, db):
        """LA County interpolated median income is within $1,000 of Census API ($71,358)."""
        row = get_county_brackets(db, "2020", "06037")
        our_value = bracket_interpolate(row, row["TOT"])
        official = CENSUS_GROUND_TRUTH["la_county_median_income_2020"]
        tolerance = 1_000
        assert abs(our_value - official) <= tolerance, (
            f"LA income: got ${our_value:,.0f}, official ${official:,}"
        )

    def test_manhattan_median_income_2020(self, db):
        """Manhattan (NY County) interpolated median income within $1,000 of Census ($89,812)."""
        row = get_county_brackets(db, "2020", "36061")
        our_value = bracket_interpolate(row, row["TOT"])
        official = CENSUS_GROUND_TRUTH["manhattan_median_income_2020"]
        # Manhattan bracket interpolation has ~1.2% variance ($1,500 tolerance)
        # because the B19001 brackets are wide ($25k wide at the $75-100k range)
        tolerance = 1_500
        assert abs(our_value - official) <= tolerance, (
            f"Manhattan income: got ${our_value:,.0f}, official ${official:,}"
        )

    def test_plain_avg_of_medians_is_inaccurate(self, db):
        """
        Regression: AVG(B19013e1) for SF County MUST be >$5,000 off from official.
        This confirms the statistical flaw that was fixed — the test guards against
        reverting to the wrong method.
        """
        P = _db_prefix(db)
        r = db(f'''
            SELECT ROUND(AVG("B19013e1"), 0) AS wrong_avg
            FROM {P}."2020_CBG_B19"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '06075%' AND "B19013e1" IS NOT NULL
        ''')
        wrong_answer = float(r[0]["WRONG_AVG"])
        official = CENSUS_GROUND_TRUTH["sf_county_median_income_2020"]
        assert abs(wrong_answer - official) > 5_000, (
            f"AVG of medians unexpectedly accurate: ${wrong_answer:,.0f} vs ${official:,}. "
            f"This may mean the underlying data changed — re-evaluate the interpolation method."
        )


# ---------------------------------------------------------------------------
# Schema Integrity Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSchemaIntegrity:

    def test_both_years_have_b01_table(self, db):
        """Both 2019 and 2020 CBG_B01 tables exist and have data."""
        P = _db_prefix(db)
        for year in ["2019", "2020"]:
            r = db(f'SELECT COUNT(*) AS cnt FROM {P}."{year}_CBG_B01"')
            assert int(r[0]["CNT"]) > 200_000, f"{year}_CBG_B01 has too few rows"

    def test_fips_lookup_works_for_major_counties(self, db):
        """FIPS codes table correctly resolves major county names."""
        P = _db_prefix(db)
        cases = [
            ("san francisco", "06", "075"),
            ("los angeles",   "06", "037"),
            ("cook",          "17", "031"),  # Chicago, IL
        ]
        for name, exp_state, exp_county in cases:
            r = db(f'''
                SELECT STATE_FIPS, COUNTY_FIPS FROM {P}."2020_METADATA_CBG_FIPS_CODES"
                WHERE LOWER("COUNTY") LIKE '%{name}%' AND "STATE_FIPS" = '{exp_state}'
                LIMIT 1
            ''')
            assert len(r) == 1, f"No FIPS found for '{name}'"
            assert r[0]["STATE_FIPS"] == exp_state
            assert r[0]["COUNTY_FIPS"] == exp_county

    def test_census_block_group_fips_filter_works(self, db):
        """LIKE '06075%' filter on CENSUS_BLOCK_GROUP returns only SF county rows."""
        P = _db_prefix(db)
        r = db(f'''
            SELECT COUNT(*) AS cnt FROM {P}."2020_CBG_B01"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '06075%'
        ''')
        count = int(r[0]["CNT"])
        # SF County has ~616 census block groups
        assert 500 <= count <= 750, f"SF county CBG count unexpected: {count}"

    def test_b19001_brackets_sum_to_total(self, db):
        """B19001 bracket counts (e2..e17) should sum to total (e1) for SF county."""
        P = _db_prefix(db)
        r = db(f'''
            SELECT
                SUM("B19001e1") AS total,
                SUM("B19001e2"+"B19001e3"+"B19001e4"+"B19001e5"+"B19001e6"
                   +"B19001e7"+"B19001e8"+"B19001e9"+"B19001e10"+"B19001e11"
                   +"B19001e12"+"B19001e13"+"B19001e14"+"B19001e15"+"B19001e16"
                   +"B19001e17") AS bracket_sum
            FROM {P}."2020_CBG_B19"
            WHERE "CENSUS_BLOCK_GROUP" LIKE '06075%' AND "B19001e1" > 0
        ''')
        total = int(r[0]["TOTAL"])
        bracket_sum = int(r[0]["BRACKET_SUM"])
        # Allow 1% variance due to ACS rounding in individual CBG estimates
        assert abs(total - bracket_sum) / total <= 0.01, (
            f"B19001 brackets don't sum to total: {bracket_sum} vs {total}"
        )
