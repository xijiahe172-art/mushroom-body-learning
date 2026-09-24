"""Phase 1 acceptance: pick five experience rows for manual reward review.

Prints the full decision context of each chosen row so the reward can be judged
against the session log it came from — no judgement is made here.

Usage: python learning_module/tests/spotcheck_phase1.py <experiences.db> [count]
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(path: str, count: int = 5) -> int:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute(
        'SELECT * FROM experiences ORDER BY reward DESC, id ASC'
    )]
    conn.close()
    if not rows:
        print('no rows recorded')
        return 1

    # Spread the sample across the reward range: the extremes and the middle.
    ordered = sorted(rows, key=lambda row: row['reward'])
    picks: list[dict] = []
    if len(ordered) <= count:
        picks = ordered
    else:
        indexes = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
        picks = [ordered[index] for index in dict.fromkeys(indexes)]

    for position, row in enumerate(picks, start=1):
        print(f'=== spot check {position}/{len(picks)} (id {row["id"]}) ===')
        print(f'  state_error_type   : {row["state_error_type"]}')
        print(f'  state_tool_context : {row["state_tool_context"]}')
        print(f'  state_file_context : {row["state_file_context"]}')
        print(f'  state_summary      : {row["state_summary"]}')
        print(f'  action_category    : {row["action_category"]}')
        print(f'  action_detail      : {row["action_detail"]}')
        print(f'  reward             : {row["reward"]:+.2f}  (source={row["reward_source"]})')
        print(f'  counters           : success={row["success_count"]} fail={row["fail_count"]}'
              f' total={row["total_count"]} average={row["average_reward"]:+.3f}')
        observations = row['total_count']
        if observations > 1:
            print(f'  NOTE               : this row aggregates {observations} observations;'
                  f' `reward` is the most recent one and `average_reward` their mean.'
                  f' The session in `action_detail` is the most recent contributor.')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5))
