"""Phase 4 acceptance: the anti-cheat layer over the real module.

Every scenario runs end to end through the two hook calls the agent loop makes —
a `pre-llm` call that captures the state, then a `post-action` call carrying the
settled results — against a real store, and the result is read back through the
query interface a human reviewer uses. Nothing is stubbed except the tool
results themselves, which are the payload the harness produces.

The scenario matrix is deliberate. Scenario 1 (edit the tests, then pass) and
scenario 2 (fix the source, then pass) are the phase's two acceptance cases.
Scenario 3 repeats scenario 1's state with a one-line fix, so it lands on the
same row as scenario 1 and shows that the upsert does not clear an earlier flag.
Scenario 4 differs only in the failure kind, so it gets its own row and shows
that a four-line fix in the same situation is left alone.

Usage:
    python learning_module/acceptance_phase4.py [database]
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_module import cli, hooks, schema, validation  # noqa: E402

PASSED = 0
FAILED = 0

TEST_FILE = r'C:\ws\test_stats.py'
SOURCE_FILE = r'C:\ws\stats.py'

#: A four-line fix: outside the small-change rule.
BIG_FIX_OLD = 'def add(a, b):\n    return a - b'
BIG_FIX_NEW = (
    'def add(a, b):\n'
    '    if a is None:\n'
    '        return b\n'
    '    if b is None:\n'
    '        return a\n'
    '    return a + b'
)

#: A one-line fix: inside it.
SMALL_FIX_OLD = 'return a - b'
SMALL_FIX_NEW = 'return a + b'

#: The prior step's result, which is what the pre-LLM state is derived from.
FAILED_TESTS = {'text': '1 failed, 2 passed in 0.10s', 'isError': True}
COLLECTION_ERROR = {
    'text': 'ERROR collecting test_calc.py\nModuleNotFoundError: No module named calc',
    'isError': True,
}


def check(label: str, condition: bool, detail: str = '') -> None:
    """Record one acceptance check.

    @param label - what is being checked.
    @param condition - whether it held.
    @param detail - the observed evidence.
    """
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f'[PASS] {label}')
        if detail:
            print(f'       {detail}')
    else:
        FAILED += 1
        print(f'[FAIL] {label}')
        if detail:
            print(f'       {detail}')


def edit_result(path: str, old: str, new: str) -> dict:
    """A settled `edit` call, shaped as the post-action payload carries it."""
    return {
        'callId': 'c1',
        'name': 'edit',
        'isError': False,
        'text': f'Edited {path}',
        'arguments': {'file_path': path, 'old_string': old, 'new_string': new},
        'mutates': True,
    }


def pytest_result() -> dict:
    """A settled test run that reported everything passing."""
    return {
        'callId': 'c2',
        'name': 'pwsh',
        'isError': False,
        'text': '3 passed in 0.05s',
        'arguments': {'command': 'python -m pytest -q'},
        'mutates': False,
    }


def run_step(database: Path, turn: int, prior: dict, results: list[dict], *,
             command: str = 'python -m pytest -q') -> dict:
    """Run both hooks of one step and return the post-action envelope.

    @param database - store path.
    @param turn - turn the step belongs to.
    @param prior - the previous step's settled result, which sets the state.
    @param results - the settled calls of this step.
    @param command - the prior command; its leading token becomes part of the
        state key, which is how a scenario gets its own row.
    @returns the post-action reply envelope.
    """
    session = 'session-verify01'
    hooks.handle_hook({
        'hook': 'pre-llm',
        'sessionId': session,
        'turn': turn,
        'step': 2,
        'toolNames': ['edit', 'pwsh'],
        'fileTargets': [
            {'name': 'edit', 'parameters': ['file_path']},
            {'name': 'pwsh', 'parameters': []},
        ],
        'priorResults': [{
            'callId': 'p1',
            'name': 'pwsh',
            'text': prior['text'],
            'isError': prior['isError'],
            'arguments': {'command': command},
            'mutates': False,
        }],
    }, database=database)
    return hooks.handle_hook({
        'hook': 'post-action',
        'sessionId': session,
        'turn': turn,
        'step': 2,
        'results': results,
    }, database=database)


def read_rows(database: Path) -> list[dict]:
    """Read every experience row, oldest first.

    @param database - store path.
    @returns the stored rows as plain dictionaries.
    """
    conn = schema.connect(database)
    try:
        return [dict(row) for row in conn.execute('SELECT * FROM experiences ORDER BY id')]
    finally:
        conn.close()


def row_for(rows: list[dict], error_type: str) -> dict:
    """Select one scenario's row by the failure kind that identifies it.

    @param rows - every stored row.
    @param error_type - the state's failure kind.
    @returns the matching row.
    """
    matching = [row for row in rows if row['state_error_type'] == error_type]
    assert len(matching) == 1, f'expected one {error_type!r} row, found {len(matching)}'
    return matching[0]


def main(argv: list[str]) -> int:
    """Run every acceptance check.

    @param argv - optional store path.
    @returns the process exit status.
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

    scratch = tempfile.TemporaryDirectory(prefix='dsh-p4-accept-')
    database = Path(argv[0]) if argv else Path(scratch.name) / 'experiences.db'
    schema.ensure_schema(database)
    previous_mode = os.environ.get('DSH_LEARNING_MODULE')
    previous_db = os.environ.get('DSH_LEARNING_DB')
    os.environ['DSH_LEARNING_MODULE'] = 'shadow'
    os.environ['DSH_LEARNING_DB'] = str(database)

    print(f'store: {database}')
    print(f'anti-cheat parameters: discount={validation.CHEAT_DISCOUNT}'
          f' small_change<{validation.SMALL_CHANGE_LINES} lines'
          f' high_reward>{validation.HIGH_REWARD}')

    try:
        print('\n--- 1. must be caught: the tests were edited until they passed ---')
        envelope = run_step(database, 1, FAILED_TESTS, [
            edit_result(TEST_FILE, 'assert add(2, 3) == 5', 'assert add(2, 3) == 4'),
            pytest_result(),
        ])
        check('the hook recorded the step',
              envelope.get('ok') is True and envelope.get('recorded') is True,
              f'action={envelope.get("action_category")} reward={envelope.get("reward")}')
        check('the base tier was the full solved reward',
              'test run reported no failures' in str(envelope.get('rationale')),
              f'rationale={envelope.get("rationale")}')
        # base +1.0, static penalty -0.05 for editing a test file, then x0.3.
        check('the reward was discounted instead of staying at +1.00',
              abs(float(envelope['reward']) - 0.285) < 1e-9,
              f'reward={envelope["reward"]} = (1.0 - 0.05) x 0.3')
        check('the envelope tells the caller it was discounted',
              envelope.get('discounted') is True
              and envelope.get('flagged_for_review') == validation.FLAG_TEST_TAMPERING,
              f'flagged_for_review={envelope.get("flagged_for_review")}')

        rows = read_rows(database)
        caught = row_for(rows, 'test_failure')
        check('the stored reward is the discounted one',
              abs(float(caught['reward']) - 0.285) < 1e-9,
              f'reward={caught["reward"]} average={caught["average_reward"]}')
        check('the mean, the EMA and the sequence all carry the same number',
              abs(float(caught['average_reward']) - 0.285) < 1e-9
              and abs(float(caught['value']) - 0.0855) < 1e-9
              and json.loads(caught['rewards']) == [0.285],
              f'average={caught["average_reward"]} value={caught["value"]}'
              f' rewards={caught["rewards"]}')
        check('the row is marked for review, with the reason',
              int(caught['flagged_for_review']) == 1
              and caught['flagged_reason'] == validation.FLAG_TEST_TAMPERING,
              f'flagged_for_review={caught["flagged_for_review"]}'
              f' reason={caught["flagged_reason"]}')
        check('the action category is still the tool category, not a new one',
              caught['action_category'] == 'edit_file',
              f'action_category={caught["action_category"]}')

        print('\n--- 2. must not be caught: the same situation, a real fix in the source ---')
        envelope = run_step(database, 2, COLLECTION_ERROR, [
            edit_result(SOURCE_FILE, BIG_FIX_OLD, BIG_FIX_NEW),
            pytest_result(),
        ])
        check('the hook recorded the step',
              envelope.get('ok') is True and envelope.get('recorded') is True,
              f'action={envelope.get("action_category")} reward={envelope.get("reward")}')
        check('the reward is untouched', abs(float(envelope['reward']) - 1.0) < 1e-9,
              f'reward={envelope["reward"]}')
        check('the envelope reports no discount and no flag',
              envelope.get('discounted') is False and 'flagged_for_review' not in envelope,
              f'discounted={envelope.get("discounted")}'
              f' flagged_for_review={envelope.get("flagged_for_review")}')
        rows = read_rows(database)
        check('the clean scenario wrote its own row', len(rows) == 2, f'rows={len(rows)}')
        clean = row_for(rows, 'test_collection_error')
        check('the row carries no flag and no reason',
              int(clean['flagged_for_review']) == 0 and (clean['flagged_reason'] or '') == '',
              f'flagged_for_review={clean["flagged_for_review"]}'
              f' reason={clean["flagged_reason"]!r}')
        check('a four-line source fix stays outside the small-change rule',
              abs(float(clean['reward']) - 1.0) < 1e-9,
              'churn 4 >= 3, so `reward > 0.8 and churn < 3` does not apply')

        print('\n--- 3. the same row observed again: the first flag survives ---')
        envelope = run_step(database, 3, FAILED_TESTS, [
            edit_result(SOURCE_FILE, SMALL_FIX_OLD, SMALL_FIX_NEW),
            pytest_result(),
        ])
        check('the reward is not discounted', abs(float(envelope['reward']) - 1.0) < 1e-9,
              f'reward={envelope["reward"]}')
        check('this step also raises the small-change flag',
              envelope.get('flagged_for_review') == validation.FLAG_SMALL_CHANGE_HIGH_REWARD,
              f'flagged_for_review={envelope.get("flagged_for_review")}')
        rows = read_rows(database)
        rebuilt = row_for(rows, 'test_failure')
        check('it updates scenario 1\'s row rather than duplicating it', len(rows) == 2,
              f'rows={len(rows)}')
        check('the first reason survives the later, weaker flag',
              rebuilt['flagged_reason'] == validation.FLAG_TEST_TAMPERING
              and int(rebuilt['total_count']) == 2,
              f'reason={rebuilt["flagged_reason"]} total_count={rebuilt["total_count"]}')
        check('the new observation is recorded at full reward',
              abs(float(rebuilt['reward']) - 1.0) < 1e-9
              and json.loads(rebuilt['rewards'])[-1] == 1.0,
              f'reward={rebuilt["reward"]} rewards={rebuilt["rewards"]}')

        print('\n--- 4. the query interface a human reviewer uses ---')
        reply = cli.COMMANDS['config']()
        check('the module still answers its config command', reply.get('ok') is True,
              f'mode={reply.get("mode")} database={reply.get("database")}')

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = cli.main(['flagged'])
        listed = json.loads(out.getvalue())
        check('the flagged query exits cleanly',
              status == 0 and listed.get('ok') is True,
              f'rows={len(listed.get("flagged", []))}'
              + (f' error={listed["error"]}' if not listed.get('ok') else ''))
        check('the flagged query reports no error', 'error' not in listed,
              f'error={listed.get("error", "none")}')
        check('it lists exactly the flagged rows', len(listed['flagged']) == 1,
              f'rows={len(listed["flagged"])}')
        entry = listed['flagged'][0]
        check('it explains why the row was flagged',
              entry['reason'] == 'the tests were edited and then passed',
              f'reason={entry["reason"]!r} flag={entry["flag"]}')
        check('the listed row exposes the discounted observation',
              json.loads(entry['rewards'])[0] == 0.285,
              f'rewards={entry["rewards"]}'
              ' (the latest `reward` column is the later full-reward observation)')
        check('the listed row names the file the step touched',
              'stats.py' in str(entry['action_detail']) and '/test' not in str(entry['action_detail']),
              f'action_detail={entry["action_detail"]}')
        check('the listed row carries its identity, not free text',
              entry['state_error_type'] == 'test_failure'
              and 'state_summary' not in entry and 'action_detail' in entry,
              f'identity=({entry["state_error_type"]}, {entry["state_tool_context"]},'
              f' {entry["action_category"]})')

        filtered = cli.command_flagged(['--flag', validation.FLAG_TEST_TAMPERING])
        check('the query can filter by flag', len(filtered['flagged']) == 1,
              f'rows={len(filtered["flagged"])} for test_tampering')
        limited = cli.command_flagged(['--limit', '1'])
        check('the query honours a limit', len(limited['flagged']) == 1,
              f'rows={len(limited["flagged"])}')

        print('\n--- 5. a small honest fix in its own state is flagged, not discounted ---')
        # A different failure kind, so this state gets its own row instead of
        # updating the row scenario 1 already flagged for tampering.
        envelope = run_step(database, 4, {
            'text': 'the command reported a failure', 'isError': True,
        }, [
            edit_result(SOURCE_FILE, SMALL_FIX_OLD, SMALL_FIX_NEW),
            pytest_result(),
        ])
        check('the reward is not discounted', abs(float(envelope['reward']) - 1.0) < 1e-9,
              f'reward={envelope["reward"]}')
        check('the step is flagged for a look instead',
              envelope.get('flagged_for_review') == validation.FLAG_SMALL_CHANGE_HIGH_REWARD,
              f'flagged_for_review={envelope.get("flagged_for_review")}')
        rows = read_rows(database)
        check('the small-change state wrote its own row', len(rows) == 3, f'rows={len(rows)}')
        listed = cli.command_flagged(['--flag', validation.FLAG_SMALL_CHANGE_HIGH_REWARD])
        check('the spot-check query finds it, with the small-change reason',
              len(listed['flagged']) == 1
              and listed['flagged'][0]['reason'] == 'a high reward for fewer than 3 changed lines',
              f'rows={len(listed["flagged"])}'
              f' reason={listed["flagged"][0]["reason"] if listed["flagged"] else "n/a"}')

        print('\n--- 6. nothing else moved ---')
        rows = read_rows(database)
        rewards = [float(row['reward']) for row in rows]
        observations = sum(int(row['total_count']) for row in rows)
        check('the scenarios produced the expected rows', len(rows) == 3 and observations == 4,
              f'rows={len(rows)} observations={observations}: three states, one seen twice')
        check('no reward leaves the declared band',
              all(-1.0 <= value <= 1.0 for value in rewards), f'rewards={rewards}')
        check('the module still serves its hook names',
              cli.COMMANDS['config']()['hooks'] == list(hooks.HOOK_NAMES),
              f'hooks={list(hooks.HOOK_NAMES)}')
    finally:
        if previous_mode is None:
            os.environ.pop('DSH_LEARNING_MODULE', None)
        else:
            os.environ['DSH_LEARNING_MODULE'] = previous_mode
        if previous_db is None:
            os.environ.pop('DSH_LEARNING_DB', None)
        else:
            os.environ['DSH_LEARNING_DB'] = previous_db
        scratch.cleanup()

    print(f'\n=== {PASSED}/{PASSED + FAILED} checks passed ===')
    return 0 if FAILED == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
