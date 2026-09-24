"""SQLite schema of the Learning Module experience store.

Phase 0 owns `experiences`. Phase 1 adds `pending_states`, the transient handoff
between the two hook invocations: the pre-hook and post-hook are separate
processes, so the state captured before a model call must be durable until the
step's action is known. A row is deleted as soon as its outcome is recorded.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Columns of `experiences`, in declaration order.
EXPERIENCE_COLUMNS = (
    'id',
    'state_error_type',
    'state_tool_context',
    'state_file_context',
    'state_summary',
    'action_category',
    'action_detail',
    'reward',
    'success',
    'success_count',
    'fail_count',
    'total_count',
    'average_reward',
    'value',
    'rewards',
    'confidence',
    'last_used_at',
    'created_at',
    'reward_source',
    'flagged_for_review',
    'flagged_reason',
)

#: Column triple identifying one experience row.
EXPERIENCE_KEY = ('state_error_type', 'state_tool_context', 'action_category')

#: Column pair the retriever looks experiences up by.
EXPERIENCE_LOOKUP_INDEX = ('state_error_type', 'action_category')

#: Columns of `pending_states`, in declaration order.
PENDING_STATE_COLUMNS = (
    'session_id',
    'turn',
    'step',
    'state_error_type',
    'state_tool_context',
    'state_file_context',
    'state_summary',
    'created_at',
)

_QUOTED_INDEX_COLUMNS = ', '.join(f'"{column}"' for column in EXPERIENCE_LOOKUP_INDEX)
_QUOTED_KEY_COLUMNS = ', '.join(f'"{column}"' for column in EXPERIENCE_KEY)

# `created_at` uses a column default rather than DEFAULT CURRENT_TIMESTAMP so
# the value stays a plain UTC ISO-8601 string, matching every other timestamp
# this table stores.
_EXPERIENCES_DDL = """
CREATE TABLE IF NOT EXISTS experiences (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  state_error_type   TEXT    NOT NULL DEFAULT 'none',
  state_tool_context TEXT    NOT NULL DEFAULT '',
  state_file_context TEXT    NOT NULL DEFAULT '',
  state_summary      TEXT    NOT NULL DEFAULT '',
  action_category    TEXT    NOT NULL DEFAULT 'none',
  action_detail      TEXT    NOT NULL DEFAULT '',
  reward             REAL    NOT NULL DEFAULT 0.0,
  success            INTEGER NOT NULL DEFAULT 0,
  success_count      INTEGER NOT NULL DEFAULT 0,
  fail_count         INTEGER NOT NULL DEFAULT 0,
  total_count        INTEGER NOT NULL DEFAULT 0,
  average_reward     REAL    NOT NULL DEFAULT 0.0,
  value              REAL,
  rewards            TEXT,
  confidence         REAL    NOT NULL DEFAULT 0.0,
  last_used_at       TEXT,
  created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  reward_source      TEXT    NOT NULL DEFAULT 'none',
  flagged_for_review INTEGER NOT NULL DEFAULT 0,
  CHECK (success IN (0, 1)),
  CHECK (reward_source IN ('none', 'rule', 'human')),
  CHECK (flagged_for_review IN (0, 1))
)
"""

_INDEX_DDL = (
    f'CREATE INDEX IF NOT EXISTS idx_experiences_state_action'
    f' ON experiences ({_QUOTED_INDEX_COLUMNS})'
)

# The unique index is what makes `record_outcome` an upsert: the same triple
# must update one row instead of accumulating duplicates.
_UNIQUE_KEY_DDL = (
    f'CREATE UNIQUE INDEX IF NOT EXISTS idx_experiences_key'
    f' ON experiences ({_QUOTED_KEY_COLUMNS})'
)

_PENDING_STATES_DDL = """
CREATE TABLE IF NOT EXISTS pending_states (
  session_id         TEXT    NOT NULL,
  turn               INTEGER NOT NULL,
  step               INTEGER NOT NULL,
  state_error_type   TEXT    NOT NULL,
  state_tool_context TEXT    NOT NULL,
  state_file_context TEXT    NOT NULL,
  state_summary      TEXT    NOT NULL DEFAULT '',
  created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (session_id, turn, step)
)
"""

_SCHEMA_STATEMENTS = (
    _EXPERIENCES_DDL,
    _INDEX_DDL,
    _UNIQUE_KEY_DDL,
    _PENDING_STATES_DDL,
)

#: Tables this module owns, for diagnostics and tests.
TABLES = ('experiences', 'pending_states')

#: Columns added after the table's first release, with their declarations. All
#: are nullable additions, so an older store keeps every row and column it had.
_ADDED_COLUMNS = (
    # The loss-averse EMA value: what retrieval decays and ranks on.
    ('value', 'REAL'),
    # The observed reward sequence, so `value` stays reconstructible from the
    # row itself instead of being an unexplainable number.
    ('rewards', 'TEXT'),
    # Phase 4's reason, beside the existing 0/1 `flagged_for_review` marker: the
    # marker keeps the meaning it has always had (this row needs a look), and the
    # reason records which validation rule raised it.
    ('flagged_reason', 'TEXT'),
)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the experience store, creating its parent directory when missing.

    @param path - SQLite file path.
    @returns an open connection with row access by column name.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add columns this phase introduced to a store created by an earlier one.

    `CREATE TABLE IF NOT EXISTS` leaves an existing table untouched, so a store
    carried over from an earlier phase needs the new columns added explicitly.
    Each addition is a bare nullable column, which leaves existing rows and
    existing columns exactly as they were.

    @param conn - open connection.
    """
    existing = {row[1] for row in conn.execute('PRAGMA table_info(experiences)')}
    for column, declaration in _ADDED_COLUMNS:
        if column not in existing:
            conn.execute(f'ALTER TABLE experiences ADD COLUMN {column} {declaration}')


def ensure_schema(path: Path | str) -> Path:
    """Create every table and index if absent, and add later-phase columns.

    @param path - SQLite file path.
    @returns the resolved path.
    """
    resolved = Path(path)
    # `with sqlite3.connect(...)` commits but does not close, and a leaked
    # handle keeps the file locked on Windows, so the connection is closed here.
    conn = connect(resolved)
    try:
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        _add_missing_columns(conn)
        conn.commit()
    finally:
        conn.close()
    return resolved


def schema_statements() -> tuple[str, ...]:
    """Return the DDL this module applies, for inspection and tests.

    @returns every schema statement in application order.
    """
    return _SCHEMA_STATEMENTS
