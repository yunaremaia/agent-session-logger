"""Tests for the FTS indexing write path."""

import sqlite3

import pytest

from asl.indexer import Indexer
from asl.store import Store


def _fail_insert_after_transaction_opens(monkeypatch):
    """Make the *second* INSERT raise, after the write transaction is open.

    The first good INSERT opens the transaction (Python's sqlite3 begins one
    implicitly); the second has an unbindable type and raises while that
    transaction is still open. That is the only window in which the raw
    connect()/commit()/close() pattern leaked a lock.
    """
    real = Store.get_messages

    def bad_messages(self, session_id):
        rows = real(self, session_id)
        return [dict(rows[0]), dict(rows[0], content={"unbindable": True})]

    monkeypatch.setattr(Store, "get_messages", bad_messages)


def _make_session(tmp_path, session_id):
    store = Store(str(tmp_path))
    store.init_db()
    store.create_session(session_id, "agent")
    store.append_message(session_id, "user", "hello world")
    return store


def test_failed_index_does_not_leak_the_write_lock(tmp_path, monkeypatch):
    """A failed index must not leave a connection holding the database lock.

    The raw connect()/commit()/close() pattern closed the connection only on
    success, so a raised INSERT leaked an open connection still inside its
    write transaction, and every later write failed with `database is locked`.
    """
    store = _make_session(tmp_path, "s1")
    _fail_insert_after_transaction_opens(monkeypatch)

    with pytest.raises(sqlite3.Error):
        Indexer(str(tmp_path)).index_session("s1")

    monkeypatch.undo()
    # The follow-up write is what a leaked lock breaks. It must succeed.
    store.append_message("s1", "user", "written after the failure")


def test_failed_index_can_be_retried(tmp_path, monkeypatch):
    """Re-indexing after a failed attempt must still succeed."""
    _make_session(tmp_path, "s2")
    _fail_insert_after_transaction_opens(monkeypatch)

    with pytest.raises(sqlite3.Error):
        Indexer(str(tmp_path)).index_session("s2")

    monkeypatch.undo()
    Indexer(str(tmp_path)).index_session("s2")