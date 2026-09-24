"""Experience persistence: the transient state handoff and the outcome upsert.

`record_outcome` implements the Phase 1 experience identity: rows are keyed by
`(state_error_type, state_tool_context, action_category)`, so the same concrete
practice in the same situation updates one row's counters instead of inserting
a duplicate. `action_detail` is overwritten as the most recent debug reference
and never participates in the key or the aggregates.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from learning_module import modes, schema, validation, value


@dataclass(frozen=True)
class StateSnapshot:
    """Structural state captured before a model call."""

    #: Failure kind observed before the call.
    error_type: str
    #: Tools the step's request offers.
    tool_context: str
    #: Files the step can touch, as `kind:basename` entries.
    file_context: str
    #: Short display-only description; never used for retrieval or injection.
    summary: str


@dataclass(frozen=True)
class RecordedOutcome:
    """What one `record_outcome` call did to the store."""

    #: `True` when an existing row was updated instead of inserted.
    updated: bool
    #: Reward the call recorded.
    reward: float
    #: Which evaluator layer produced the reward.
    reward_source: str
    #: Counters after the call.
    success_count: int
    fail_count: int
    total_count: int
    #: Rolling mean of recorded rewards after the call.
    average_reward: float
    #: Loss-averse EMA value after the call.
    value: float = 0.0
    #: Observations recorded in the row's reward sequence after the call.
    rewards: int = 0
    #: Flag Phase 4 validation stored on this observation, or `''`.
    flagged_for_review: str = ''
    #: Whether validation discounted the reward before recording it.
    discounted: bool = False


def save_pending_state(
    database: Path | str,
    session_id: str,
    turn: int,
    step: int,
    state: StateSnapshot,
) -> None:
    """Persist the pre-hook state until the step's action is known.

    A step that ends the session (the model answers without calling a tool)
    never reaches the post-action hook, so its row would linger forever. The
    write therefore keeps only this step's row per session: the handoff is
    between two consecutive hook calls, and an older unclaimed row can never be
    claimed again.

    @param database - SQLite file path.
    @param session_id - session identity.
    @param turn - turn number of the step.
    @param step - step number within the turn.
    @param state - the state captured before the model call.
    """
    conn = schema.connect(database)
    try:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            'DELETE FROM pending_states WHERE session_id = ? AND NOT (turn = ? AND step = ?)',
            (session_id, turn, step),
        )
        conn.execute(
            'INSERT INTO pending_states'
            ' (session_id, turn, step, state_error_type, state_tool_context, state_file_context, state_summary)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?)'
            ' ON CONFLICT (session_id, turn, step) DO UPDATE SET'
            '   state_error_type = excluded.state_error_type,'
            '   state_tool_context = excluded.state_tool_context,'
            '   state_file_context = excluded.state_file_context,'
            '   state_summary = excluded.state_summary',
            (session_id, turn, step, state.error_type, state.tool_context, state.file_context, state.summary),
        )
        conn.commit()
    finally:
        conn.close()


def take_pending_state(
    database: Path | str,
    session_id: str,
    turn: int,
    step: int,
) -> StateSnapshot | None:
    """Read and delete the pre-hook state for one step.

    @param database - SQLite file path.
    @param session_id - session identity.
    @param turn - turn number of the step.
    @param step - step number within the turn.
    @returns the captured state, or `None` when the pre-hook never ran.
    """
    conn = schema.connect(database)
    try:
        row = conn.execute(
            'SELECT state_error_type, state_tool_context, state_file_context, state_summary'
            ' FROM pending_states WHERE session_id = ? AND turn = ? AND step = ?',
            (session_id, turn, step),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            'DELETE FROM pending_states WHERE session_id = ? AND turn = ? AND step = ?',
            (session_id, turn, step),
        )
        conn.commit()
    finally:
        conn.close()
    return StateSnapshot(
        error_type=str(row['state_error_type']),
        tool_context=str(row['state_tool_context']),
        file_context=str(row['state_file_context']),
        summary=str(row['state_summary']),
    )


def record_outcome(
    database: Path | str,
    state: StateSnapshot,
    action_category: str,
    action_detail: str,
    reward: float,
    reward_source: str,
    *,
    base_reward: float | None = None,
    churn: int = 0,
    touch_test_file: bool = False,
    final_reward: float | None = None,
) -> RecordedOutcome:
    """Insert or update the experience row for one settled action.

    The row carries both central tendencies: `average_reward` stays the plain
    arithmetic mean for diagnosis, while `value` is the loss-averse EMA that
    retrieval decays and ranks on. The reward sequence is stored alongside them
    so the value stays reconstructible from the row.

    Phase 4 validation runs here for a caller that has not run it already. A
    caller that holds a settled verdict — the hook does, because it had to
    report the reward to the harness before recording it — passes the already
    discounted number as `final_reward`, and the row stores exactly that. The
    `flagged_for_review` marker and the `flagged_reason` beside it are written
    either way, and the stored reward, the mean, the EMA, and the reward
    sequence all carry the same number so the row cannot be read two ways. The
    flag is a reason string, so a later clean observation never clears an
    earlier flag on the same row.

    If `final_reward` is given, the caller decides the discount and passes its
    bases so the flag stays explainable: this module validates a reward by
    default, and a caller that already validated one keeps that decision.

    @param database - SQLite file path.
    @param state - structural state captured before the action.
    @param action_category - the action's category; part of the identity.
    @param action_detail - most recent call description; overwritten, not aggregated.
    @param reward - reward before the discount; with `final_reward`, the settled reward.
    @param reward_source - evaluator layer that produced the reward.
    @param base_reward - the evaluator's base tier; defaults to `reward`, which
        means "no penalty, and the reward is the base tier".
    @param churn - changed lines the step's file-mutating calls produced.
    @param touch_test_file - whether the step wrote or edited a test file.
    @param final_reward - reward a caller already validated and discounted.
    @returns the counters after the update.
    """
    if final_reward is None:
        final = validation.finalize(
            base_reward=reward if base_reward is None else base_reward,
            reward=reward,
            churn=churn,
            touch_test_file=touch_test_file,
        )
        assessment = final.assessment
        reward = final.reward
    else:
        assessment = validation.assess(
            base_reward=reward if base_reward is None else base_reward,
            reward=reward,
            churn=churn,
            touch_test_file=touch_test_file,
        )
        reward = final_reward
    flagged = 1 if assessment.flag else 0
    success = 1 if reward > 0 else 0
    conn = schema.connect(database)
    try:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute(
            f'SELECT id, success_count, fail_count, total_count, average_reward, value, rewards'
            f' FROM experiences WHERE state_error_type = ? AND state_tool_context = ? AND action_category = ?',
            (state.error_type, state.tool_context, action_category),
        ).fetchone()
        if row is None:
            success_count = success
            fail_count = 1 - success
            total_count = 1
            average_reward = float(reward)
            new_value = value.update_value(value.initial_value(), float(reward), has_history=False)
            rewards = [float(reward)]
            conn.execute(
                'INSERT INTO experiences'
                ' (state_error_type, state_tool_context, state_file_context, state_summary,'
                '  action_category, action_detail, reward, success, success_count, fail_count,'
                '  total_count, average_reward, value, rewards, last_used_at, reward_source,'
                '  flagged_for_review, flagged_reason)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
                "  strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?)",
                (
                    state.error_type, state.tool_context, state.file_context, state.summary,
                    action_category, action_detail, float(reward), success,
                    success_count, fail_count, total_count, average_reward,
                    new_value, json.dumps(rewards), reward_source, flagged, assessment.flag,
                ),
            )
            updated = False
        else:
            success_count = int(row['success_count']) + success
            fail_count = int(row['fail_count']) + (1 - success)
            total_count = int(row['total_count']) + 1
            # The stored mean is the running average of recorded rewards.
            average_reward = (
                (float(row['average_reward']) * int(row['total_count']) + float(reward)) / total_count
            )
            new_value, rewards = _next_value(row, float(reward))
            conn.execute(
                'UPDATE experiences SET'
                ' success_count = ?, fail_count = ?, total_count = ?, average_reward = ?,'
                ' value = ?, rewards = ?,'
                ' reward = ?, success = ?, action_detail = ?, state_file_context = ?,'
                ' state_summary = ?, reward_source = ?,'
                ' flagged_for_review = CASE WHEN ? = 1 THEN 1 ELSE flagged_for_review END,'
                # The first reason wins: a row already flagged for tampering must
                # not have that reason rewritten by a later, weaker rule, or the
                # reviewer would read the wrong evidence.
                " flagged_reason = CASE WHEN flagged_reason IS NULL OR flagged_reason = ''"
                ' THEN ? ELSE flagged_reason END,'
                " last_used_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
                ' WHERE id = ?',
                (
                    success_count, fail_count, total_count, average_reward,
                    new_value, json.dumps(rewards),
                    float(reward), success, action_detail, state.file_context,
                    state.summary, reward_source, flagged, assessment.flag,
                    int(row['id']),
                ),
            )
            updated = True
        conn.commit()
    finally:
        conn.close()
    return RecordedOutcome(
        updated=updated,
        reward=float(reward),
        reward_source=reward_source,
        success_count=success_count,
        fail_count=fail_count,
        total_count=total_count,
        average_reward=average_reward,
        value=new_value,
        rewards=len(rewards),
        flagged_for_review=assessment.flag,
        discounted=assessment.discounted and final_reward is None,
    )


def _next_value(row: object, reward: float) -> tuple[float, list[float]]:
    """Advance one row's loss-averse value and append the observation.

    A row from a store that predates the value column has no recorded sequence.
    Its observation history cannot be reconstructed, and inventing one would
    fake loss-aversion weighting that never happened, so the row is treated as
    held out of learning: only the reward sequence starts now.

    @param row - the stored row before this observation.
    @param reward - the observed reward.
    @returns the new value and the reward sequence after appending.
    """
    raw = row['rewards']  # type: ignore[index]
    rewards: list[float] = []
    if isinstance(raw, str) and raw.strip() != '':
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = []
        if isinstance(parsed, list):
            rewards = [float(entry) for entry in parsed if isinstance(entry, (int, float))]
    elif raw is None:
        # Pre-value rows: no sequence exists, so no loss aversion is applied.
        return float(row['value']) if row['value'] is not None else float(reward), [reward]  # type: ignore[index]
    rewards.append(float(reward))
    old_value = float(row['value']) if row['value'] is not None else value.initial_value()  # type: ignore[index]
    new_value = value.update_value(old_value, float(reward), has_history=len(rewards) > 1)
    return new_value, rewards


def fetch_experience(database: Path | str, key: tuple[str, str, str]) -> sqlite3.Row | None:
    """Read one experience row by its identity triple.

    @param database - SQLite file path.
    @param key - `(state_error_type, state_tool_context, action_category)`.
    @returns the row, or `None` when the triple has no experience yet.
    """
    conn = schema.connect(database)
    try:
        return conn.execute(
            'SELECT * FROM experiences'
            ' WHERE state_error_type = ? AND state_tool_context = ? AND action_category = ?',
            key,
        ).fetchone()
    finally:
        conn.close()


#: Why each flag exists, in the reviewer's words.
FLAG_REASONS: Mapping[str, str] = {
    validation.FLAG_TEST_TAMPERING: 'the tests were edited and then passed',
    validation.FLAG_TEST_FILE_EDIT: 'a test file was edited',
    validation.FLAG_SMALL_CHANGE_HIGH_REWARD: 'a high reward for fewer than 3 changed lines',
}


def flagged_experiences(
    database: Path | str | None = None,
    *,
    flag: str | None = None,
    limit: int = 100,
) -> list[Mapping[str, Any]]:
    """List the rows a human should spot-check.

    The query is the read side of Phase 4: a flag is only useful if a person can
    find it. Only flagged rows are returned — an unflagged row needs no review —
    and each entry carries the structural evidence a reviewer needs (the identity
    triple, the reward as recorded, the call description, and the reason), never
    the free-text state summary.

    @param database - store path; defaults to the configured one.
    @param flag - restrict to one flag name; defaults to every flagged row.
    @param limit - maximum rows to return.
    @returns the flagged rows, most recently observed first.
    """
    path = resolve_database(database)
    conn = schema.connect(path)
    try:
        if not _has_experiences(conn):
            return []
        query = (
            'SELECT id, state_error_type, state_tool_context, action_category, action_detail,'
            ' reward, average_reward, value, rewards, total_count, reward_source,'
            ' flagged_for_review, flagged_reason, last_used_at'
            ' FROM experiences WHERE flagged_for_review = 1'
        )
        parameters: list[Any] = []
        if flag is not None:
            query += ' AND flagged_reason = ?'
            parameters.append(flag)
        query += ' ORDER BY last_used_at DESC, id DESC LIMIT ?'
        parameters.append(max(0, int(limit)))
        rows = [dict(row) for row in conn.execute(query, tuple(parameters))]
    finally:
        conn.close()
    for row in rows:
        reason = str(row.get('flagged_reason') or '')
        row['reason'] = FLAG_REASONS.get(reason, 'flagged for manual review')
    return rows


def _has_experiences(conn: sqlite3.Connection) -> bool:
    """Whether this store already carries the `experiences` table.

    A store created by an earlier phase, or a path that has never been written,
    is not an error: the reviewer gets an empty list rather than a traceback.

    @param conn - open connection.
    @returns `True` when the table exists.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'experiences'"
    ).fetchone()
    return row is not None


def resolve_database(database: Path | str | None = None) -> Path | str:
    """Resolve the store path, honouring an explicit override.

    @param database - explicit path, or `None` for the configured default.
    @returns the store path to use.
    """
    return modes.database_path() if database is None else database


@dataclass(frozen=True)
class BackfillReport:
    """What a value backfill did, and what it could not do."""

    #: Rows whose value was recomputed from their recorded reward sequence.
    recomputed_from_sequence: int
    #: Rows with no recorded sequence, set to their arithmetic mean.
    from_arithmetic_mean: int
    #: Rows that already carried a value.
    already_present: int
    #: Rows skipped because they recorded no observation at all.
    skipped_empty: int
    #: Mean-vs-mean difference over the rows that were substituted.
    mean_spread: float

    @property
    def total(self) -> int:
        """Rows the report covers.

        @returns the sum of every bucket.
        """
        return (self.recomputed_from_sequence + self.from_arithmetic_mean
                + self.already_present + self.skipped_empty)


def backfill_values(database: Path | str) -> BackfillReport:
    """Populate the `value` column of rows written before it existed.

    A row that recorded its reward sequence is recomputed through the real EMA,
    so the stored value is exactly what the engine would have produced. A row
    without a sequence cannot be recomputed — loss-aversion weighting depends on
    the order and size of the individual rewards, and inventing a plausible
    sequence would fabricate results that were never observed. Those rows are
    seeded with their arithmetic mean, which is the only recorded central
    tendency, and the report counts them separately so the substitution stays
    visible. From the next observation on, every row follows the real EMA.

    @param database - SQLite file path.
    @returns the report of what was backfilled and how.
    """
    schema.ensure_schema(database)
    conn = schema.connect(database)
    recomputed = 0
    substituted = 0
    present = 0
    skipped = 0
    differences: list[float] = []
    try:
        conn.execute('BEGIN IMMEDIATE')
        rows = list(conn.execute(
            'SELECT id, value, rewards, average_reward, total_count FROM experiences'
        ))
        for row in rows:
            if int(row['total_count']) <= 0:
                skipped += 1
                continue
            raw = row['rewards']
            sequence: list[float] = []
            if isinstance(raw, str) and raw.strip() != '':
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = []
                if isinstance(parsed, list):
                    sequence = [float(entry) for entry in parsed if isinstance(entry, (int, float))]
            if sequence:
                rebuilt = value.replay_value(sequence)
                conn.execute('UPDATE experiences SET value = ? WHERE id = ?', (rebuilt, int(row['id'])))
                if row['value'] is not None:
                    present += 1
                else:
                    recomputed += 1
                continue
            if row['value'] is not None:
                present += 1
                continue
            mean = float(row['average_reward'])
            conn.execute('UPDATE experiences SET value = ?, rewards = ? WHERE id = ?',
                         (mean, json.dumps([mean]), int(row['id'])))
            differences.append(abs(mean - value.replay_value([mean])))
            substituted += 1
        conn.commit()
    finally:
        conn.close()
    spread = sum(differences) / len(differences) if differences else 0.0
    return BackfillReport(
        recomputed_from_sequence=recomputed,
        from_arithmetic_mean=substituted,
        already_present=present,
        skipped_empty=skipped,
        mean_spread=spread,
    )
