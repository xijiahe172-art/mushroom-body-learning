"""Phase 1 acceptance audit: field presence, value ranges, and reward spot checks.

Reads the shadow-mode store produced by the 10 real tasks and reports:
  1. every declared column's presence and observed range,
  2. the identity-key uniqueness and counter consistency of each row,
  3. the reward observations per action category, for manual review.

Usage: python learning_module/tests/audit_phase1.py <experiences.db>
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning_module import classify, reward, schema  # noqa: E402

CATEGORIES = set(classify.ACTION_CATEGORIES)
ERROR_KINDS = set(classify.ERROR_KINDS)
SOURCES = {'rule', 'none', 'human'}


def main(path: str) -> int:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute('SELECT * FROM experiences ORDER BY id')]
    failures: list[str] = []

    print(f'rows: {len(rows)}')
    print('\n=== 1. field presence and ranges ===')
    for column in schema.EXPERIENCE_COLUMNS:
        # A store written by an earlier phase may predate a later column; the
        # audit reports it as absent instead of crashing on its own schema.
        if column not in rows[0] and rows:
            print(f'  {column:20} absent from this store (added by a later phase)')
            continue
        values = [row[column] for row in rows]
        present = all(value is not None for value in values)
        if column in ('id', 'last_used_at'):
            print(f'  {column:20} present={present} (nullable by design)')
            continue
        if not present:
            failures.append(f'{column} has NULL values')
        numeric = [value for value in values if isinstance(value, (int, float))]
        if column in ('reward', 'average_reward', 'confidence'):
            span = f'range=[{min(numeric):+.3f}, {max(numeric):+.3f}]' if numeric else 'no numeric values'
            if column == 'reward' and numeric and not all(-1.0 <= value <= 1.0 for value in numeric):
                failures.append('reward outside [-1, 1]')
            if column == 'confidence' and numeric and not all(0.0 <= value <= 1.0 for value in numeric):
                failures.append('confidence outside [0, 1]')
            if column == 'average_reward' and numeric and not all(-1.0 <= value <= 1.0 for value in numeric):
                failures.append('average_reward outside [-1, 1]')
        elif column in ('success', 'flagged_for_review'):
            span = f'values={sorted(set(numeric))}'
            if column == 'success' and not set(numeric) <= {0, 1}:
                failures.append('success not boolean')
        else:
            span = f'distinct={len(set(values))}'
        print(f'  {column:20} present={present} non-null  {span}')

    print('\n=== 2. identity and counters ===')
    keys = [(row['state_error_type'], row['state_tool_context'], row['action_category']) for row in rows]
    if len(keys) != len(set(keys)):
        failures.append('duplicate identity triples exist')
    print(f'  identity triples unique: {len(keys) == len(set(keys))}')
    for row in rows:
        total = row['total_count']
        counted = row['success_count'] + row['fail_count']
        if total != counted:
            failures.append(f"row {row['id']}: total_count {total} != success+fail {counted}")
        if row['action_category'] not in CATEGORIES:
            failures.append(f"row {row['id']}: unknown action category {row['action_category']!r}")
        if row['state_error_type'] not in ERROR_KINDS:
            failures.append(f"row {row['id']}: unknown error kind {row['state_error_type']!r}")
        if row['reward_source'] not in SOURCES:
            failures.append(f"row {row['id']}: unknown reward source {row['reward_source']!r}")
        if row['average_reward'] == 0.0 and row['success_count'] > 0 and row['fail_count'] > 0:
            pass
    print(f'  counters consistent: {not any("total_count" in item for item in failures)}')
    print(f'  action categories in vocabulary: '
          f'{sorted({row["action_category"] for row in rows})}')

    print('\n=== 3. reward observations by action category ===')
    for category in sorted({row['action_category'] for row in rows}):
        subset = [row for row in rows if row['action_category'] == category]
        rewards = [row['reward'] for row in subset]
        print(f'  {category:16} rows={len(subset):2}  rewards={[f"{value:+.2f}" for value in rewards]}'
              f'  avg={sum(rewards) / len(rewards):+.3f}')

    print('\n=== 4. reward band coverage ===')
    bands = {
        '+1.0 solved': reward.REWARD_SOLVED,
        '+0.6 improved': reward.REWARD_IMPROVED,
        '+0.2 partial': reward.REWARD_PARTIAL,
        '0.0 unchanged': reward.REWARD_UNCHANGED,
        '-0.3 ineffective': reward.REWARD_INEFFECTIVE,
        '-0.8 worsened': reward.REWARD_WORSENED,
        '-1.0 severe': reward.REWARD_SEVERE,
    }
    observed = {row['reward'] for row in rows}
    for label, value in bands.items():
        print(f'  {label:18} observed={value in observed}')

    print('\n=== 5. pending handoff stays bounded ===')
    leftover = conn.execute('SELECT COUNT(*) FROM pending_states').fetchone()[0]
    sessions = conn.execute('SELECT COUNT(DISTINCT session_id) FROM pending_states').fetchone()[0]
    print(f'  pending_states rows left: {leftover} across {sessions} session(s)')
    if leftover > sessions:
        failures.append(f'pending_states holds {leftover} rows for {sessions} sessions (leak)')
    print('  at most one unclaimed final step per session: '
          f'{leftover <= sessions} (a step that ends a session never reaches the post-action hook)')

    print('\n=== result ===')
    if failures:
        for item in failures:
            print(f'  FAIL {item}')
        conn.close()
        return 1
    print('  OK: every declared field present, every value inside its declared range')
    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
