"""
tests/test_agent.py – Unit tests for the ReAct agent pipeline.

All Snowflake and OpenRouter calls are mocked — no network required.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent as agent_module
from agent import (
    check_guardrail,
    OffTopicError,
    _extract_sql,
    _extract_answer,
    _rows_to_text,
    _db_name,
    answer_question,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chat_mock(return_value: str):
    """Return a patch context that makes _chat return a fixed string."""
    return patch("agent._chat", return_value=return_value)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

class TestGuardrail:
    def test_allow_census_question(self):
        with make_chat_mock("ALLOW"):
            check_guardrail("What is the population of California?")

    def test_deny_offtopic(self):
        with make_chat_mock("DENY"):
            with pytest.raises(OffTopicError, match="US Census data assistant"):
                check_guardrail("What is the capital of France?")

    def test_guardrail_case_insensitive(self):
        with make_chat_mock("allow"):
            # should not raise
            check_guardrail("How many people live in Texas?")

    def test_guardrail_empty_response_denies(self):
        with make_chat_mock(""):
            with pytest.raises(OffTopicError):
                check_guardrail("anything")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestExtractors:
    def test_extract_sql_basic(self):
        text = "THOUGHT: need data\nSQL:\nSELECT 1\nEND_SQL"
        assert _extract_sql(text) == "SELECT 1"

    def test_extract_sql_multiline(self):
        text = "SQL:\nSELECT a,\nb\nFROM t\nEND_SQL"
        assert _extract_sql(text) == "SELECT a,\nb\nFROM t"

    def test_extract_sql_returns_none_when_absent(self):
        assert _extract_sql("No SQL here") is None

    def test_extract_answer_basic(self):
        text = "FINAL_ANSWER:\nThe answer is 42\nEND_ANSWER"
        assert _extract_answer(text) == "The answer is 42"

    def test_extract_answer_none_when_absent(self):
        assert _extract_answer("nothing") is None

    def test_extract_answer_strips_whitespace(self):
        text = "FINAL_ANSWER:\n  Hello  \nEND_ANSWER"
        assert _extract_answer(text) == "Hello"


# ---------------------------------------------------------------------------
# _rows_to_text
# ---------------------------------------------------------------------------

class TestRowsToText:
    def test_returns_json_string(self):
        rows = [{"col": "val"}]
        result = _rows_to_text(rows)
        assert '"col"' in result
        assert '"val"' in result

    def test_caps_at_20_rows(self):
        import json
        rows = [{"i": i} for i in range(30)]
        result = json.loads(_rows_to_text(rows))
        assert len(result) == 20

    def test_handles_empty(self):
        assert _rows_to_text([]) == "[]"


# ---------------------------------------------------------------------------
# _db_name
# ---------------------------------------------------------------------------

class TestDbName:
    def test_reads_from_env(self):
        with patch.dict(os.environ, {"SNOWFLAKE_DATABASE": "MY_DB"}):
            assert _db_name() == "MY_DB"

    def test_fallback_value(self):
        env = {k: v for k, v in os.environ.items() if k != "SNOWFLAKE_DATABASE"}
        with patch.dict(os.environ, env, clear=True):
            result = _db_name()
            assert len(result) > 0


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

class TestReactLoop:
    def _make_retriever(self):
        from retrieval import HybridRetriever, SchemaEntry
        ret = HybridRetriever(entries=[SchemaEntry("2020_CBG_B19", "B19013e1", "NUMBER", "Median income")])
        ret.build_index()
        return ret

    def test_single_sql_then_answer(self):
        """LLM runs one SQL query then gives a FINAL_ANSWER."""
        responses = iter([
            "THOUGHT: query\nSQL:\nSELECT 1\nEND_SQL",      # step 1
            "FINAL_ANSWER:\n$100,000\nEND_ANSWER",           # step 2
        ])
        with patch("agent._chat", side_effect=lambda *a, **k: next(responses)):
            with patch("agent.run_query", return_value=[{"result": 100000}]):
                with patch("agent.get_retriever", return_value=self._make_retriever()):
                    answer, sql = agent_module.run_react_loop("income?", [], "ctx")
        assert "$100,000" in answer
        assert sql == "SELECT 1"

    def test_direct_answer_no_sql(self):
        """LLM immediately gives FINAL_ANSWER without running SQL."""
        with patch("agent._chat", return_value="FINAL_ANSWER:\nNo data\nEND_ANSWER"):
            answer, sql = agent_module.run_react_loop("population?", [], "ctx")
        assert "No data" in answer
        assert sql is None

    def test_sql_error_feeds_back(self):
        """A SQL error should be fed back to the LLM, not crash."""
        from db import QueryError
        responses = iter([
            "SQL:\nBAD SQL\nEND_SQL",
            "FINAL_ANSWER:\nCould not get data\nEND_ANSWER",
        ])
        with patch("agent._chat", side_effect=lambda *a, **k: next(responses)):
            with patch("agent.run_query", side_effect=QueryError("syntax error")):
                answer, sql = agent_module.run_react_loop("q?", [], "ctx")
        assert "Could not get data" in answer

    def test_max_steps_triggers_final_request(self):
        """After MAX_STEPS, a final FINAL_ANSWER request is sent."""
        sql_resp = "SQL:\nSELECT 1\nEND_SQL"
        calls = {"n": 0}
        def side_effect(*a, **k):
            calls["n"] += 1
            if calls["n"] <= agent_module.MAX_STEPS:
                return sql_resp
            return "FINAL_ANSWER:\nTimeout answer\nEND_ANSWER"

        with patch("agent._chat", side_effect=side_effect):
            with patch("agent.run_query", return_value=[{"x": 1}]):
                answer, _ = agent_module.run_react_loop("q?", [], "ctx")
        assert answer  # got something back


# ---------------------------------------------------------------------------
# answer_question (full pipeline)
# ---------------------------------------------------------------------------

class TestAnswerQuestion:
    def _make_retriever(self):
        from retrieval import HybridRetriever, SchemaEntry
        ret = HybridRetriever(entries=[SchemaEntry("T", "col", "NUMBER", "a column")])
        ret.build_index()
        return ret

    def test_offtopic_returns_friendly_message(self):
        with patch("agent._chat", return_value="DENY"):
            ans, sql = answer_question("Who won the World Cup?", [])
        assert "Census" in ans
        assert sql is None

    def test_happy_path(self):
        responses = iter([
            "ALLOW",
            "SQL:\nSELECT 1\nEND_SQL",
            "FINAL_ANSWER:\nThe population is 1M\nEND_ANSWER",
        ])
        with patch("agent._chat", side_effect=lambda *a, **k: next(responses)):
            with patch("agent.run_query", return_value=[{"pop": 1_000_000}]):
                with patch("agent.get_retriever", return_value=self._make_retriever()):
                    ans, sql = answer_question("California population?", [])
        assert "population is 1M" in ans
        assert sql is not None

    def test_guardrail_error_returns_message(self):
        with patch("agent._chat", side_effect=Exception("network error")):
            ans, sql = answer_question("population?", [])
        assert "trouble" in ans.lower() or "error" in ans.lower()
        assert sql is None

    def test_react_exception_returns_friendly_message(self):
        with patch("agent._chat", side_effect=["ALLOW", Exception("boom")]):
            with patch("agent.get_retriever", return_value=self._make_retriever()):
                ans, sql = answer_question("income?", [])
        assert "unexpected error" in ans.lower() or "error" in ans.lower()
