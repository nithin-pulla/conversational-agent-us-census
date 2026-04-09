"""
tests/test_db.py – Unit tests for the database module.
Snowflake is fully mocked — no real connection needed.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "dummy")
    monkeypatch.setenv("SNOWFLAKE_USER", "dummy")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "dummy")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "dummy")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "dummy")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "dummy")


@pytest.fixture(autouse=True)
def reset_connection():
    """Ensure module singleton is cleared between tests."""
    import db
    db._connection = None
    yield
    db._connection = None


class TestRunQuery:
    def test_happy_path_returns_rows(self, mocker):
        import db
        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchmany.return_value = [{"col": 1}, {"col": 2}]
        mock_conn.is_closed.return_value = False
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("db.snowflake.connector.connect", return_value=mock_conn)

        rows = db.run_query("SELECT 1")
        assert rows == [{"col": 1}, {"col": 2}]

    def test_programming_error_raises_query_error(self, mocker):
        import db
        import snowflake.connector.errors as sfe

        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.execute.side_effect = sfe.ProgrammingError("bad sql")
        mock_conn.is_closed.return_value = False
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("db.snowflake.connector.connect", return_value=mock_conn)

        with pytest.raises(db.QueryError, match="Snowflake SQL error"):
            db.run_query("SELECT BAD")

    def test_unexpected_error_raises_query_error(self, mocker):
        import db

        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.execute.side_effect = RuntimeError("network gone")
        mock_conn.is_closed.return_value = False
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("db.snowflake.connector.connect", return_value=mock_conn)

        with pytest.raises(db.QueryError, match="Unexpected database error"):
            db.run_query("SELECT 1")

    def test_max_rows_cap(self, mocker):
        import db

        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchmany.return_value = [{"n": i} for i in range(5)]
        mock_conn.is_closed.return_value = False
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("db.snowflake.connector.connect", return_value=mock_conn)

        rows = db.run_query("SELECT 1", max_rows=5)
        mock_cursor.fetchmany.assert_called_once_with(5)
        assert len(rows) == 5


class TestFetchSchemaMetadata:
    def test_returns_list_of_dicts(self, mocker):
        import db
        sample = [{"table": "T", "column": "c", "data_type": "NUMBER", "comment": ""}]
        mocker.patch("db.run_query", return_value=sample)
        result = db.fetch_schema_metadata()
        assert result == sample

    def test_returns_empty_list_on_error(self, mocker):
        import db
        mocker.patch("db.run_query", side_effect=db.QueryError("error"))
        result = db.fetch_schema_metadata()
        assert result == []
