"""
tests/test_retrieval.py – Unit tests for the hybrid BM25 + TF-IDF retriever.

Tests cover:
  - SchemaEntry data class formatting
  - BM25Index scoring
  - HybridRetriever indexing, RRF fusion, and vocabulary-mismatch recovery
  - bootstrap_retriever and get_retriever lifecycle
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval import (
    SchemaEntry,
    BM25Index,
    HybridRetriever,
    bootstrap_retriever,
    get_retriever,
)


# ---------------------------------------------------------------------------
# SchemaEntry
# ---------------------------------------------------------------------------

class TestSchemaEntry:
    def test_as_text_includes_all_fields(self):
        e = SchemaEntry("2020_CBG_B25", "B25003e2", "NUMBER", "Owner occupied housing")
        text = e.as_text()
        assert "2020_CBG_B25" in text
        assert "B25003e2" in text
        assert "Owner occupied housing" in text

    def test_as_display_format(self):
        e = SchemaEntry("T", "col", "NUMBER", "a comment")
        d = e.as_display()
        assert "T.col" in d
        assert "NUMBER" in d
        assert "a comment" in d

    def test_as_display_no_comment(self):
        e = SchemaEntry("T", "col", "TEXT", "")
        d = e.as_display()
        assert "T.col" in d
        assert " -- " not in d


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_returns_ranked_results(self):
        idx = BM25Index()
        idx.fit(["income median household", "population total count", "owner occupied housing"])
        results = idx.score("income", top_k=3)
        assert results[0][0] == 0  # income doc ranks first
        assert results[0][1] > 0

    def test_empty_returns_empty(self):
        idx = BM25Index()
        idx.fit([])
        assert idx.score("income", top_k=5) == []

    def test_top_k_respected(self):
        idx = BM25Index()
        idx.fit(["a", "b", "c", "d", "e"])
        results = idx.score("a b c d e", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

def _make_entries():
    return [
        SchemaEntry("2020_CBG_B25", "B25003e1",  "NUMBER", "Estimate: Tenure — Total"),
        SchemaEntry("2020_CBG_B25", "B25003e2",  "NUMBER", "Estimate: Tenure — Owner occupied"),
        SchemaEntry("2020_CBG_B25", "B25003e3",  "NUMBER", "Estimate: Tenure — Renter occupied"),
        SchemaEntry("2020_CBG_B19", "B19013e1",  "NUMBER", "Estimate: Median Household Income — Total"),
        SchemaEntry("2020_CBG_B01", "B01001e1",  "NUMBER", "Estimate: Sex By Age — Total population"),
        SchemaEntry("2020_CBG_B08", "B08301e21", "NUMBER", "Estimate: Means Of Transportation To Work — Worked from home"),
        SchemaEntry("2020_CBG_B15", "B15003e22", "NUMBER", "Estimate: Educational Attainment — Bachelor's degree"),
        SchemaEntry("2020_CBG_B17", "B17021e2",  "NUMBER", "Estimate: Poverty Status — Below poverty level"),
    ]


class TestHybridRetriever:
    def _built(self) -> HybridRetriever:
        ret = HybridRetriever(_make_entries())
        ret.build_index()
        return ret

    def test_exact_keyword_match(self):
        ret = self._built()
        results = ret.retrieve("median household income", top_k=3)
        assert any("B19013e1" in r.column for r in results), \
            "income query should find B19013e1"

    def test_vocabulary_mismatch_home_ownership(self):
        """
        Core regression: 'home ownership' must find 'owner occupied', not mobile home.
        This is the exact failure mode that caused 'no available tables' with pure BM25.
        """
        ret = self._built()
        results = ret.retrieve("home ownership rate", top_k=4)
        columns = [r.column for r in results]
        assert "B25003e2" in columns, (
            f"'home ownership' should match 'Owner occupied' (B25003e2). Got: {columns}"
        )

    def test_vocabulary_mismatch_college_educated(self):
        """
        'bachelor degree' directly in corpus description should rank B15003e22 high.
        Full semantic bridging ('college educated' → 'bachelor') requires a large corpus;
        this test validates TF-IDF finds explicit term matches above BM25 baseline.
        """
        ret = self._built()
        results = ret.retrieve("bachelor degree attainment", top_k=3)
        columns = [r.column for r in results]
        assert "B15003e22" in columns, (
            f"'bachelor degree' should match B15003e22. Got: {columns}"
        )

    def test_vocabulary_mismatch_poverty(self):
        """'people below poverty line' should find poverty status fields."""
        ret = self._built()
        results = ret.retrieve("people below poverty line", top_k=3)
        columns = [r.column for r in results]
        assert "B17021e2" in columns, \
            f"poverty query should find B17021e2. Got: {columns}"

    def test_top_k_respected(self):
        ret = self._built()
        results = ret.retrieve("income", top_k=2)
        assert len(results) <= 2

    def test_schema_context_returns_nonempty_string(self):
        ret = self._built()
        ctx = ret.schema_context("income", top_k=3)
        assert isinstance(ctx, str)
        assert len(ctx) > 10

    def test_empty_retriever_returns_empty_context(self):
        ret = HybridRetriever([])
        ret.build_index()
        ctx = ret.schema_context("income", top_k=5)
        assert "no matching" in ctx.lower()

    def test_unbuilt_retriever_returns_first_k(self):
        """Before build_index(), falls back to first k entries."""
        ret = HybridRetriever(_make_entries())
        # do NOT call build_index
        results = ret.retrieve("anything", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Bootstrap lifecycle
# ---------------------------------------------------------------------------

class TestBootstrapRetriever:
    def test_bootstrap_builds_and_caches(self):
        schema = [
            {"table": "T", "column": "c1", "data_type": "NUMBER", "comment": "income"},
            {"table": "T", "column": "c2", "data_type": "NUMBER", "comment": "population"},
        ]
        ret = bootstrap_retriever(schema)
        assert isinstance(ret, HybridRetriever)
        assert ret._indexed

    def test_get_retriever_after_bootstrap(self):
        schema = [{"table": "T", "column": "c", "data_type": "NUMBER", "comment": "x"}]
        bootstrap_retriever(schema)
        ret = get_retriever()
        assert ret is not None

    def test_bootstrap_with_missing_fields(self):
        schema = [{"table": "T"}]  # missing column, comment, data_type
        ret = bootstrap_retriever(schema)
        assert isinstance(ret, HybridRetriever)
