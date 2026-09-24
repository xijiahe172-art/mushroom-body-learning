"""Structured experience retrieval.

No embeddings and no vector store: candidates come from an exact key match,
with a bounded relaxation to predefined neighbouring failure kinds when the
exact match is too thin. The retriever either returns trustworthy experience or
nothing at all — it never pads a result set with weak evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from learning_module import schema, value

#: Neighbouring failure kinds, used only when the exact kind yields fewer than
#: {@link MIN_CANDIDATES} rows. Benign kinds (`none`, `warning`) are absent by
#: design: a run that merely warned is not a failure to relax towards, and a
#: clean state must never be padded with failures.
ADJACENT_ERROR_KINDS: Mapping[str, tuple[str, ...]] = {
    'test_failure': ('test_collection_error', 'import_error'),
    'test_collection_error': ('test_failure', 'import_error'),
    'build_failure': ('syntax_error', 'dependency_error'),
    'syntax_error': ('build_failure',),
    'import_error': ('dependency_error', 'test_collection_error'),
    'dependency_error': ('import_error', 'file_not_found'),
    'file_not_found': ('dependency_error',),
    'command_failed': ('tool_error', 'timeout'),
    'tool_error': ('command_failed', 'tool_not_found'),
    'tool_not_found': ('tool_error',),
    'timeout': ('command_failed',),
    'permission_denied': ('tool_error',),
    'aborted': ('timeout',),
}

#: The exact match must yield at least this many rows before relaxation stops.
MIN_CANDIDATES = 3


@dataclass(frozen=True)
class RetrievalResult:
    """What the retriever decided, including why."""

    #: Keys that were looked up, in query order.
    error_types_searched: tuple[str, ...]
    #: Rows the exact kind matched.
    exact_matches: int
    #: Candidates dropped because their decayed value was too weak.
    skipped_stale: int
    #: Candidates dropped for low confidence.
    skipped_low_confidence: int
    #: Ranked candidates, best first, at most {@link value.TOP_K}.
    candidates: tuple[value.Candidate, ...] = ()
    #: Whether these candidates may be injected.
    injectable: bool = False
    #: Why the result is empty or withheld.
    reason: str = ''

    @property
    def entries(self) -> tuple[value.Candidate, ...]:
        """The candidates that would be shown, or nothing when withheld.

        @returns the injectable candidates.
        """
        return self.candidates if self.injectable else ()


def candidates_for(
    database: object,
    error_type: str,
    *,
    now: datetime | None = None,
) -> RetrievalResult:
    """Retrieve experience for one state.

    @param database - SQLite file path.
    @param error_type - the state's exact failure kind.
    @param now - reference time for decay; defaults to the current UTC time.
    @returns the ranked result and the reason behind it.
    """
    # A store that does not exist yet is an empty store, not an error: the
    # retriever answers "nothing to offer" so a first run behaves like a run
    # with no matching history.
    schema.ensure_schema(database)
    conn = schema.connect(database)
    try:
        rows, searched = _collect(conn, error_type)
    finally:
        conn.close()

    built = [(value.build_candidate(row, now), _tier(row, error_type)) for row in rows]
    fresh = [entry for entry in built if not entry[0].stale]
    skipped_stale = len(built) - len(fresh)
    ranked = sorted(fresh, key=lambda entry: abs(entry[0].adjusted_score), reverse=True)
    confident = [entry for entry in ranked if not entry[0].below_confidence]
    skipped_low = len(ranked) - len(confident)
    # Relaxation only fills what the exact kind could not: an exact match is
    # always more relevant than a neighbouring kind's experience, so a
    # neighbour never displaces one.
    top = tuple(candidate for candidate, _ in confident if _ == 0)[:value.TOP_K]
    if len(top) < value.TOP_K:
        filler = tuple(candidate for candidate, tier in confident if tier == 1)
        top = top + filler[:value.TOP_K - len(top)]

    if not top:
        reason = 'no candidate passed the value and confidence floors'
        return RetrievalResult(searched, len(built), skipped_stale, skipped_low, (), False, reason)
    best = max(candidate.confidence for candidate in top)
    if best < value.INJECTION_CONFIDENCE_FLOOR:
        reason = f'best confidence {best:.3f} is below the injection floor'
        return RetrievalResult(searched, len(built), skipped_stale, skipped_low, top, False, reason)
    return RetrievalResult(searched, len(built), skipped_stale, skipped_low, top, True, '')


def _tier(row: object, error_type: str) -> int:
    """Rank a row's relevance tier: an exact kind match outranks a neighbour.

    @param row - a stored experience row.
    @param error_type - the state's exact failure kind.
    @returns `0` for an exact match, `1` for a relaxed neighbour.
    """
    return 0 if str(row['state_error_type']) == error_type else 1  # type: ignore[index]


def _collect(conn: object, error_type: str) -> tuple[list[object], tuple[str, ...]]:
    """Collect candidate rows, relaxing to neighbouring kinds only when thin.

    @param conn - open connection.
    @param error_type - the state's exact failure kind.
    @returns the rows and the kinds that were queried.
    """
    searched = [error_type]
    rows = _query(conn, [error_type])
    if len(rows) >= MIN_CANDIDATES:
        return rows, tuple(searched)
    for neighbour in ADJACENT_ERROR_KINDS.get(error_type, ()):
        if len(rows) >= MIN_CANDIDATES:
            break
        searched.append(neighbour)
        rows.extend(_query(conn, [neighbour]))
    return rows, tuple(searched)


def _query(conn: object, error_types: Sequence[str]) -> list[object]:
    """Read every row whose failure kind is one of `error_types`.

    @param conn - open connection.
    @param error_types - kinds to match exactly.
    @returns the matching rows.
    """
    placeholders = ', '.join('?' for _ in error_types)
    cursor = conn.execute(  # type: ignore[attr-defined]
        f'SELECT * FROM experiences WHERE state_error_type IN ({placeholders})',
        tuple(error_types),
    )
    return list(cursor.fetchall())
