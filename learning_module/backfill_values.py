"""Backfill the loss-averse value column for rows written before it existed.

Transparency is the point of this script: a row that recorded its reward
sequence is recomputed through the real EMA, while a row without one cannot be
recomputed at all (loss aversion depends on the order and size of individual
rewards) and is therefore seeded with its arithmetic mean. The report prints
both counts so a substituted row is never mistaken for a recomputed one.

Usage: python learning_module/backfill_values.py <db> [--apply]
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_module import schema, value  # noqa: E402


def survey(database: str) -> None:
    """Print what the backfill would do, changes nothing but the schema.

    Adding the new columns is part of reading them: a store written by an
    earlier phase has neither, and `ensure_schema` upgrades in place without
    touching existing rows or columns.

    @param database - SQLite file path.
    """
    schema.ensure_schema(database)
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute(
            'SELECT id, value, rewards, average_reward, total_count FROM experiences ORDER BY id'
        )]
    finally:
        conn.close()
    with_sequence = 0
    without = 0
    already = 0
    empty = 0
    for row in rows:
        if int(row['total_count']) <= 0:
            empty += 1
        elif row['rewards'] not in (None, '') and row['rewards'] != '[]':
            with_sequence += 1
        elif row['value'] is not None:
            already += 1
        else:
            without += 1
    print(f'rows: {len(rows)}')
    print(f'  recomputable from a recorded reward sequence : {with_sequence}')
    print(f'  already carrying a value                     : {already}')
    print(f'  no sequence and no value (mean substitution)  : {without}')
    print(f'  no observations at all (skipped)              : {empty}')
    if without:
        print('\n  Substituted rows keep their arithmetic mean, which is the only recorded')
        print('  central tendency. Loss-aversion weighting is NOT retroactively applied:')
        print('  the individual rewards are gone, and inventing a sequence would fabricate')
        print('  results that were never observed. From the next observation each such row')
        print('  follows the real EMA (' f'alpha={value.ALPHA}, lambda={value.LAMBDA}).')


def main(argv: list[str]) -> int:
    """Run the survey, or apply the backfill with `--apply`.

    @param argv - arguments after the script name.
    @returns the process exit status.
    """
    if not argv:
        print(__doc__)
        return 2
    apply_change = '--apply' in argv
    database = next(argument for argument in argv if not argument.startswith('--'))
    survey(database)
    if not apply_change:
        print('\n(dry run — pass --apply to write)')
        return 0
    from learning_module import store
    report = store.backfill_values(database)
    print('\nbackfill applied:')
    print(f'  recomputed from a recorded sequence : {report.recomputed_from_sequence}')
    print(f'  seeded from the arithmetic mean     : {report.from_arithmetic_mean}')
    print(f'  already carrying a value            : {report.already_present}')
    print(f'  skipped (no observations)           : {report.skipped_empty}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
