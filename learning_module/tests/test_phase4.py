"""Phase 4 acceptance tests: reward validation and the anti-cheat layer.

Three things are checked here, in the order the phase defines them: the
deterministic inputs the validation reads (changed lines, a test file, a passing
test run), the decision itself (discount and flag), and the store and query
behaviour a human reviewer depends on.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from learning_module import (  # noqa: E402
    classify,
    cli,
    reward,
    schema,
    state as state_encoder,
    store,
    validation,
)

TEST_FILE = r'C:\ws\test_stats.py'
SOURCE_FILE = r'C:\ws\stats.py'


def _result(name: str, arguments: dict, text: str, *, mutates: bool = True) -> dict:
    """One settled call shaped like the post-action payload carries it."""
    return {'name': name, 'arguments': arguments, 'text': text, 'isError': False, 'mutates': mutates}


def _edit(path: str, old: str, new: str, *, tool: str = 'edit') -> dict:
    """A file-editing call whose argument names follow the given tool's schema."""
    if tool == 'str_replace_editor':
        arguments = {'command': 'str_replace', 'path': path, 'old_str': old, 'new_str': new}
    else:
        arguments = {'file_path': path, 'old_string': old, 'new_string': new}
    return _result(tool, arguments, 'ok')


#: A real remediation: four lines change, so its churn sits outside the
#: small-change rule instead of tripping it.
REAL_FIX_OLD = 'def add(a, b):\n    return a - b'
REAL_FIX_NEW = (
    'def add(a, b):\n'
    '    if a is None:\n'
    '        return b\n'
    '    if b is None:\n'
    '        return a\n'
    '    return a + b'
)


def _source_fix() -> dict:
    """A step that fixes a source file by a change larger than the small-change rule."""
    return _edit(SOURCE_FILE, REAL_FIX_OLD, REAL_FIX_NEW)


def _two_line_fix() -> dict:
    """A step whose source change sits just inside the small-change rule."""
    return _edit(SOURCE_FILE, REAL_FIX_OLD, REAL_FIX_NEW[:REAL_FIX_NEW.index('\n    if b')])


def _run_tests() -> dict:
    """A `pwsh` call that ran pytest and reported everything passing."""
    return _result('pwsh', {'command': 'python -m pytest -q'}, '3 passed in 0.05s', mutates=False)


class DiffLineTests(unittest.TestCase):
    """The changed-line count is derived from the model's own arguments."""

    def test_an_in_place_edit_reports_its_line_delta(self) -> None:
        churn = validation.diff_lines('edit', {
            'file_path': SOURCE_FILE, 'old_string': 'a\nb', 'new_string': 'a\nc\nd',
        })
        self.assertEqual(churn, 1)

    def test_a_whole_file_write_reports_the_lines_it_wrote(self) -> None:
        self.assertEqual(validation.diff_lines('write', {'file_path': SOURCE_FILE, 'content': 'a\nb\nc'}), 3)

    def test_a_trailing_newline_does_not_add_a_line(self) -> None:
        self.assertEqual(validation.diff_lines('write', {'file_path': SOURCE_FILE, 'content': 'a\nb\n'}), 2)

    def test_an_empty_write_has_no_churn(self) -> None:
        self.assertEqual(validation.diff_lines('write', {'file_path': SOURCE_FILE, 'content': ''}), 0)

    def test_a_call_carrying_no_text_has_no_churn(self) -> None:
        self.assertEqual(validation.diff_lines('edit', {'file_path': SOURCE_FILE}), 0)

    def test_a_removal_without_a_replacement_still_counts(self) -> None:
        self.assertEqual(validation.diff_lines('edit', {
            'file_path': SOURCE_FILE, 'old_string': 'a\nb', 'new_string': None,
        }), 2)

    def test_the_alternative_editor_reports_the_same_churn(self) -> None:
        self.assertEqual(validation.diff_lines('str_replace_editor', {
            'command': 'str_replace', 'path': SOURCE_FILE, 'old_str': 'a', 'new_str': 'a\nb\nc',
        }), 2)

    def test_the_alternative_editor_counts_a_created_file(self) -> None:
        self.assertEqual(validation.diff_lines('str_replace_editor', {
            'command': 'create', 'path': SOURCE_FILE, 'file_text': 'a\nb',
        }), 2)

    def test_the_alternative_editor_counts_an_inserted_block(self) -> None:
        self.assertEqual(validation.diff_lines('str_replace_editor', {
            'command': 'insert', 'path': SOURCE_FILE, 'new_str': 'a\nb\nc',
        }), 3)

    def test_a_read_only_editor_command_changes_nothing(self) -> None:
        self.assertEqual(validation.diff_lines('str_replace_editor', {
            'command': 'view', 'path': SOURCE_FILE,
        }), 0)


class AssessmentTests(unittest.TestCase):
    """The two rules, including the asymmetries the specification requires."""

    def test_a_positive_reward_for_editing_the_tests_is_discounted(self) -> None:
        verdict = validation.assess(base_reward=1.0, reward=1.0, churn=1, touch_test_file=True)
        self.assertAlmostEqual(verdict.discount, 0.3, places=10)
        self.assertEqual(verdict.flag, validation.FLAG_TEST_TAMPERING)
        self.assertTrue(verdict.discounted)

    def test_a_negative_reward_for_editing_the_tests_is_left_alone(self) -> None:
        verdict = validation.assess(base_reward=-0.8, reward=-0.8, churn=1, touch_test_file=True)
        self.assertAlmostEqual(verdict.discount, 1.0, places=10)
        self.assertEqual(verdict.flag, validation.FLAG_TEST_FILE_EDIT)

    def test_a_negative_reward_is_never_discounted(self) -> None:
        verdict = validation.assess(base_reward=-1.0, reward=-1.0, churn=0, touch_test_file=False)
        self.assertAlmostEqual(verdict.discount, 1.0, places=10)
        self.assertEqual(verdict.flag, validation.FLAG_NONE)

    def test_editing_the_tests_without_a_pass_is_left_alone(self) -> None:
        verdict = validation.assess(base_reward=0.2, reward=0.2, churn=1, touch_test_file=True)
        self.assertAlmostEqual(verdict.discount, 1.0, places=10)
        self.assertEqual(verdict.flag, validation.FLAG_TEST_FILE_EDIT)

    def test_a_normal_fix_is_not_flagged(self) -> None:
        verdict = validation.assess(base_reward=1.0, reward=1.0, churn=5, touch_test_file=False)
        self.assertAlmostEqual(verdict.discount, 1.0, places=10)
        self.assertEqual(verdict.flag, validation.FLAG_NONE)

    def test_a_high_reward_for_a_two_line_change_is_flagged_not_discounted(self) -> None:
        verdict = validation.assess(base_reward=1.0, reward=1.0, churn=2, touch_test_file=False)
        self.assertAlmostEqual(verdict.discount, 1.0, places=10)
        self.assertEqual(verdict.flag, validation.FLAG_SMALL_CHANGE_HIGH_REWARD)

    def test_the_boundary_churn_is_not_a_small_change(self) -> None:
        verdict = validation.assess(base_reward=1.0, reward=1.0, churn=3, touch_test_file=False)
        self.assertEqual(verdict.flag, validation.FLAG_NONE)

    def test_a_modest_reward_is_never_flagged_at_all(self) -> None:
        verdict = validation.assess(base_reward=0.6, reward=0.6, churn=1, touch_test_file=False)
        self.assertEqual(verdict.flag, validation.FLAG_NONE)

    def test_a_clean_observation_carries_no_assessment(self) -> None:
        verdict = validation.assess(base_reward=0.0, reward=0.0, churn=3, touch_test_file=False)
        self.assertFalse(verdict.discounted)
        self.assertEqual(verdict.flag, validation.FLAG_NONE)


class RewardLayerTests(unittest.TestCase):
    """The discount multiplies the settled reward, penalties included."""

    def _tampering(self, results: list[dict]) -> reward.RewardVerdict:
        return reward.evaluate(
            'test_failure', 'none', results,
            modified_files=[TEST_FILE], observed_files=('stats.py',),
            churn=1, touch_test_file=True,
        )

    def test_the_penalty_is_discounted_together_with_the_positive_part(self) -> None:
        # Base +1.0, the static-penalty cap -0.1, then x0.3: the penalty is inside
        # the discount, not subtracted from the discounted reward afterwards.
        verdict = self._tampering([
            _edit(TEST_FILE, 'assert add(2, 3) == 5', 'assert add(2, 3) == 5\n'),
            _edit(r'C:\ws\unrelated.py', 'a', 'a\nb'),
            _run_tests(),
        ])
        self.assertAlmostEqual(verdict.base, 1.0, places=10)
        self.assertAlmostEqual(verdict.penalty, -0.1, places=10)
        self.assertAlmostEqual(verdict.reward, 0.27, places=10)
        self.assertTrue(verdict.discounted)
        self.assertEqual(verdict.flag, validation.FLAG_TEST_TAMPERING)

    def test_a_discounted_verdict_explains_itself(self) -> None:
        verdict = self._tampering([
            _edit(TEST_FILE, 'assert add(2, 3) == 5', 'assert add(2, 3) == 5\n'),
            _run_tests(),
        ])
        self.assertIn(validation.FLAG_TEST_TAMPERING, verdict.rationale)
        self.assertIn('discounted', verdict.rationale)

    def test_a_normal_fix_is_not_discounted_by_the_evaluator(self) -> None:
        verdict = reward.evaluate(
            'test_failure', 'none',
            [_source_fix(), _run_tests()],
            modified_files=[SOURCE_FILE], observed_files=('stats.py',),
            churn=4, touch_test_file=False,
        )
        self.assertAlmostEqual(verdict.reward, 1.0, places=10)
        self.assertFalse(verdict.discounted)
        self.assertEqual(verdict.flag, validation.FLAG_NONE)

    def test_a_two_line_fix_is_flagged_but_keeps_its_full_reward(self) -> None:
        # The specification's spot-check rule, at its boundary: flagged for a
        # look, not discounted, and not blocked.
        verdict = reward.evaluate(
            'test_failure', 'none',
            [_two_line_fix(), _run_tests()],
            modified_files=[SOURCE_FILE], observed_files=('stats.py',),
            churn=2, touch_test_file=False,
        )
        self.assertAlmostEqual(verdict.reward, 1.0, places=10)
        self.assertFalse(verdict.discounted)
        self.assertEqual(verdict.flag, validation.FLAG_SMALL_CHANGE_HIGH_REWARD)


class OutcomeDerivationTests(unittest.TestCase):
    """The post-action state must expose the two signals validation reads."""

    def test_a_source_fix_reports_its_churn(self) -> None:
        post = state_encoder.complete_state({
            'results': [_source_fix(), _run_tests()],
        })
        self.assertFalse(post.touch_test_file)
        self.assertEqual(post.churn, 4)

    def test_editing_a_test_file_is_detected(self) -> None:
        post = state_encoder.complete_state({
            'results': [_edit(TEST_FILE, 'assert add(2, 3) == 5', 'assert add(2, 3) == -1'), _run_tests()],
        })
        self.assertTrue(post.touch_test_file)

    def test_a_test_directory_is_detected_too(self) -> None:
        post = state_encoder.complete_state({
            'results': [_edit(r'C:\ws\tests\helpers.py', 'a', 'a\nb')],
        })
        self.assertTrue(post.touch_test_file)

    def test_a_read_only_step_has_no_churn_and_no_test_file(self) -> None:
        post = state_encoder.complete_state({
            'results': [_result('read', {'file_path': TEST_FILE}, 'contents', mutates=False)],
        })
        self.assertFalse(post.touch_test_file)
        self.assertEqual(post.churn, 0)

    def test_churn_accumulates_over_a_step_with_several_edits(self) -> None:
        post = state_encoder.complete_state({
            'results': [
                _edit(SOURCE_FILE, 'a', 'a\nb'),
                _edit(SOURCE_FILE, 'x', 'x\ny'),
            ],
        })
        self.assertEqual(post.churn, 2)


class RecordingTests(unittest.TestCase):
    """What actually lands in the store."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-p4-store-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        schema.ensure_schema(self.db)

    def _assert_flag(self, key: tuple[str, str, str], marker: int, reason: str) -> None:
        row = store.fetch_experience(self.db, key)
        assert row is not None
        self.assertEqual(int(row['flagged_for_review']), marker)
        self.assertEqual(row['flagged_reason'] or '', reason)

    def test_a_cheating_observation_is_recorded_discounted_and_flagged(self) -> None:
        recorded = store.record_outcome(
            self.db,
            _state('test_failure'),
            'edit_file',
            'session=abcd1234 edit test_stats.py',
            1.0,
            reward.SOURCE_RULE,
            base_reward=1.0,
            churn=1,
            touch_test_file=True,
        )
        self.assertAlmostEqual(recorded.reward, 0.3, places=10)
        self.assertTrue(recorded.discounted)
        self.assertEqual(recorded.flagged_for_review, validation.FLAG_TEST_TAMPERING)
        key = ('test_failure', 'pwsh:pytest', 'edit_file')
        row = store.fetch_experience(self.db, key)
        assert row is not None
        self.assertAlmostEqual(float(row['reward']), 0.3, places=10)
        self.assertAlmostEqual(float(row['average_reward']), 0.3, places=10)
        self.assertAlmostEqual(float(row['value']), 0.09, places=10)
        self.assertEqual(json.loads(row['rewards']), [0.3])
        self._assert_flag(key, 1, validation.FLAG_TEST_TAMPERING)

    def test_a_normal_fix_is_recorded_untouched(self) -> None:
        recorded = store.record_outcome(
            self.db,
            _state('test_failure'),
            'edit_file',
            'session=abcd1234 edit stats.py',
            1.0,
            reward.SOURCE_RULE,
            base_reward=1.0,
            churn=4,
            touch_test_file=False,
        )
        self.assertAlmostEqual(recorded.reward, 1.0, places=10)
        self.assertFalse(recorded.discounted)
        self.assertEqual(recorded.flagged_for_review, validation.FLAG_NONE)
        self._assert_flag(('test_failure', 'pwsh:pytest', 'edit_file'), 0, '')

    def test_a_small_change_with_a_high_reward_is_flagged_but_keeps_its_reward(self) -> None:
        recorded = store.record_outcome(
            self.db,
            _state('test_failure'),
            'edit_file',
            'session=abcd1234 edit stats.py',
            1.0,
            reward.SOURCE_RULE,
            base_reward=1.0,
            churn=2,
            touch_test_file=False,
        )
        self.assertAlmostEqual(recorded.reward, 1.0, places=10)
        self.assertEqual(recorded.flagged_for_review, validation.FLAG_SMALL_CHANGE_HIGH_REWARD)
        self._assert_flag(
            ('test_failure', 'pwsh:pytest', 'edit_file'), 1, validation.FLAG_SMALL_CHANGE_HIGH_REWARD,
        )

    def test_a_negative_observation_is_never_discounted(self) -> None:
        recorded = store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'edit test_stats.py', -0.8, reward.SOURCE_RULE,
            base_reward=-0.8, churn=1, touch_test_file=True,
        )
        self.assertAlmostEqual(recorded.reward, -0.8, places=10)
        self.assertEqual(recorded.flagged_for_review, validation.FLAG_TEST_FILE_EDIT)

    def test_a_later_clean_observation_does_not_clear_the_flag(self) -> None:
        key = ('test_failure', 'pwsh:pytest', 'edit_file')
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'edit test_stats.py', 1.0, reward.SOURCE_RULE,
            base_reward=1.0, churn=1, touch_test_file=True,
        )
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'edit stats.py', 0.6, reward.SOURCE_RULE,
            base_reward=0.6, churn=4, touch_test_file=False,
        )
        self._assert_flag(key, 1, validation.FLAG_TEST_TAMPERING)

    def test_a_later_flag_does_not_rewrite_the_first_reason(self) -> None:
        # The row was flagged for tampering; a later small-change observation
        # raises a weaker flag and must not replace the reason a reviewer reads.
        key = ('test_failure', 'pwsh:pytest', 'edit_file')
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'edit test_stats.py', 1.0, reward.SOURCE_RULE,
            base_reward=1.0, churn=1, touch_test_file=True,
        )
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'edit stats.py', 1.0, reward.SOURCE_RULE,
            base_reward=1.0, churn=1, touch_test_file=False,
        )
        self._assert_flag(key, 1, validation.FLAG_TEST_TAMPERING)

    def test_a_call_without_the_signals_still_records(self) -> None:
        recorded = store.record_outcome(
            self.db, _state('none'), 'read_file', 'read stats.py', 0.6, reward.SOURCE_RULE,
        )
        self.assertAlmostEqual(recorded.reward, 0.6, places=10)
        self.assertEqual(recorded.flagged_for_review, validation.FLAG_NONE)

    def test_the_recorded_category_stays_the_tool_category(self) -> None:
        # The test-file signal is an internal flag, not a new action category.
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'edit test_stats.py', 1.0, reward.SOURCE_RULE,
            base_reward=1.0, churn=1, touch_test_file=True,
        )
        row = store.fetch_experience(self.db, ('test_failure', 'pwsh:pytest', 'edit_file'))
        assert row is not None
        self.assertEqual(row['action_category'], 'edit_file')

    def test_the_stored_reward_matches_what_the_evaluator_settled(self) -> None:
        # A verdict is settled once: the recorder stores the number the caller
        # already validated, instead of discounting it a second time (0.27 must
        # not become 0.081).
        recorded = store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'edit test_stats.py', 0.27, reward.SOURCE_RULE,
            base_reward=1.0, final_reward=0.27, churn=1, touch_test_file=True,
        )
        self.assertAlmostEqual(recorded.reward, 0.27, places=10)
        self.assertFalse(recorded.discounted)
        self.assertEqual(recorded.flagged_for_review, validation.FLAG_TEST_TAMPERING)
        row = store.fetch_experience(self.db, ('test_failure', 'pwsh:pytest', 'edit_file'))
        assert row is not None
        self.assertAlmostEqual(float(row['reward']), 0.27, places=10)
        self.assertEqual(json.loads(row['rewards']), [0.27])


class QueryTests(unittest.TestCase):
    """The manual spot-check interface."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-p4-query-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        schema.ensure_schema(self.db)
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'session=aaaa1111 edit test_stats.py',
            1.0, reward.SOURCE_RULE, base_reward=1.0, churn=1, touch_test_file=True,
        )
        store.record_outcome(
            self.db, _state('none'), 'write_file', 'session=bbbb2222 write stats.py',
            1.0, reward.SOURCE_RULE, base_reward=1.0, churn=1, touch_test_file=False,
        )
        store.record_outcome(
            self.db, _state('none'), 'read_file', 'session=cccc3333 read stats.py',
            0.6, reward.SOURCE_RULE, base_reward=0.6, churn=0, touch_test_file=False,
        )

    def test_the_query_returns_only_flagged_rows(self) -> None:
        rows = store.flagged_experiences(self.db)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['flagged_reason'] for row in rows}, {
            validation.FLAG_TEST_TAMPERING, validation.FLAG_SMALL_CHANGE_HIGH_REWARD,
        })
        self.assertTrue(all(int(row['flagged_for_review']) == 1 for row in rows))

    def test_the_query_explains_why_each_row_was_flagged(self) -> None:
        rows = store.flagged_experiences(self.db)
        reasons = {row['flagged_reason']: row['reason'] for row in rows}
        self.assertEqual(reasons[validation.FLAG_TEST_TAMPERING], 'the tests were edited and then passed')
        self.assertEqual(
            reasons[validation.FLAG_SMALL_CHANGE_HIGH_REWARD],
            'a high reward for fewer than 3 changed lines',
        )

    def test_the_query_can_filter_by_reason(self) -> None:
        rows = store.flagged_experiences(self.db, flag=validation.FLAG_TEST_TAMPERING)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['action_detail'], 'session=aaaa1111 edit test_stats.py')

    def test_the_query_reports_an_empty_store_as_empty(self) -> None:
        empty = Path(self._tmp.name) / 'empty.db'
        self.assertEqual(store.flagged_experiences(empty), [])

    def test_the_query_honours_its_limit(self) -> None:
        self.assertEqual(len(store.flagged_experiences(self.db, limit=1)), 1)

    def test_the_query_reveals_the_discounted_reward_and_its_source(self) -> None:
        rows = store.flagged_experiences(self.db, flag=validation.FLAG_TEST_TAMPERING)
        self.assertAlmostEqual(float(rows[0]['reward']), 0.3, places=10)
        self.assertEqual(rows[0]['reward_source'], reward.SOURCE_RULE)
        self.assertIn('edit test_stats.py', rows[0]['action_detail'])

    def test_a_store_with_only_unflagged_rows_returns_nothing(self) -> None:
        clean = Path(self._tmp.name) / 'clean.db'
        schema.ensure_schema(clean)
        store.record_outcome(
            clean, _state('none'), 'read_file', 'read stats.py', 0.6, reward.SOURCE_RULE,
        )
        self.assertEqual(store.flagged_experiences(clean), [])

    def test_a_store_created_before_phase4_is_still_queryable(self) -> None:
        # An older store has the marker column but neither `flagged_reason` nor
        # any reason value: the upgrade adds the column and the query stays empty
        # instead of failing on the missing one.
        legacy = Path(self._tmp.name) / 'legacy.db'
        conn = sqlite3.connect(legacy)
        try:
            conn.execute(
                'CREATE TABLE experiences ('
                ' id INTEGER PRIMARY KEY AUTOINCREMENT, state_error_type TEXT, state_tool_context TEXT,'
                ' action_category TEXT, action_detail TEXT, reward REAL, average_reward REAL,'
                ' total_count INTEGER, reward_source TEXT, flagged_for_review INTEGER DEFAULT 0,'
                ' last_used_at TEXT)'
            )
            conn.execute(
                "INSERT INTO experiences (state_error_type, state_tool_context, action_category,"
                " action_detail, reward, average_reward, total_count, reward_source,"
                " flagged_for_review, last_used_at)"
                " VALUES ('test_failure', 'pwsh', 'edit_file', 'old row', 0.6, 0.6, 1, 'rule', 0, '2026-01-01T00:00:00Z')"
            )
            conn.commit()
        finally:
            conn.close()
        schema.ensure_schema(legacy)
        self.assertEqual(store.flagged_experiences(legacy), [])


class HookTests(unittest.TestCase):
    """The signal has to survive the two-process hook handoff."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-p4-hook-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        self._previous = os.environ.get('DSH_LEARNING_MODULE')
        os.environ['DSH_LEARNING_MODULE'] = 'shadow'
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._previous is None:
            os.environ.pop('DSH_LEARNING_MODULE', None)
        else:
            os.environ['DSH_LEARNING_MODULE'] = self._previous

    def _payload(self, results: list[dict]) -> dict:
        return {'hook': 'post-action', 'sessionId': 'session-aaaa1111', 'turn': 1, 'step': 2, 'results': results}

    def _begin(self, step: int = 2) -> None:
        from learning_module import hooks
        hooks.handle_hook({
            'hook': 'pre-llm', 'sessionId': 'session-aaaa1111', 'turn': 1, 'step': step,
            'toolNames': ['edit', 'pwsh'],
            'fileTargets': [{'name': 'edit', 'parameters': ['file_path']}],
            'priorResults': [{
                'name': 'pwsh', 'isError': True, 'text': '1 failed, 2 passed',
                'arguments': {'command': 'python -m pytest -q'},
            }],
        }, database=self.db)

    def test_the_post_action_hook_discounts_and_flags_test_editing(self) -> None:
        from learning_module import hooks
        self._begin()
        envelope = hooks.handle_hook(self._payload([
            _edit(TEST_FILE, 'assert add(2, 3) == 5', 'assert add(2, 3) == 5\n'),
            _run_tests(),
        ]), database=self.db)
        self.assertTrue(envelope['ok'])
        # (base +1.0, static penalty -0.05 for editing the test file) x0.3.
        self.assertAlmostEqual(envelope['reward'], 0.285, places=10)
        self.assertTrue(envelope['discounted'])
        self.assertEqual(envelope['flagged_for_review'], validation.FLAG_TEST_TAMPERING)
        row = store.fetch_experience(self.db, ('test_failure', 'pwsh:pytest', 'edit_file'))
        assert row is not None
        self.assertAlmostEqual(float(row['reward']), 0.285, places=10)
        self.assertEqual(row['action_category'], 'edit_file')
        self.assertEqual(row['flagged_reason'], validation.FLAG_TEST_TAMPERING)

    def test_the_post_action_hook_leaves_a_normal_fix_unflagged(self) -> None:
        from learning_module import hooks
        self._begin()
        envelope = hooks.handle_hook(self._payload([
            _source_fix(),
            _run_tests(),
        ]), database=self.db)
        self.assertTrue(envelope['ok'])
        self.assertAlmostEqual(envelope['reward'], 1.0, places=10)
        row = store.fetch_experience(self.db, ('test_failure', 'pwsh:pytest', 'edit_file'))
        assert row is not None
        self.assertEqual(int(row['flagged_for_review']), 0)
        self.assertEqual(row['flagged_reason'] or '', validation.FLAG_NONE)

    def test_the_envelope_reports_that_the_reward_was_discounted(self) -> None:
        from learning_module import hooks
        self._begin()
        envelope = hooks.handle_hook(self._payload([
            _edit(TEST_FILE, 'assert add(2, 3) == 5', 'assert add(2, 3) == 5\n'),
            _run_tests(),
        ]), database=self.db)
        self.assertIn('discounted', envelope['rationale'])
        self.assertEqual(envelope['flagged_for_review'], validation.FLAG_TEST_TAMPERING)


class CliQueryTests(unittest.TestCase):
    """The command a human runs for the spot check."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-p4-cli-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        schema.ensure_schema(self.db)
        self._previous = os.environ.get('DSH_LEARNING_DB')
        os.environ['DSH_LEARNING_DB'] = str(self.db)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._previous is None:
            os.environ.pop('DSH_LEARNING_DB', None)
        else:
            os.environ['DSH_LEARNING_DB'] = self._previous

    def _run(self, argv: list[str]) -> dict:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = cli.main(argv)
        self.assertEqual(status, 0)
        return json.loads(out.getvalue())

    def test_an_empty_store_reports_no_flagged_rows(self) -> None:
        reply = self._run(['flagged'])
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['flagged'], [])

    def test_the_command_lists_a_flagged_row_with_its_reason(self) -> None:
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'session=aaaa1111 edit test_stats.py',
            1.0, reward.SOURCE_RULE, base_reward=1.0, churn=1, touch_test_file=True,
        )
        reply = self._run(['flagged'])
        self.assertEqual(len(reply['flagged']), 1)
        entry = reply['flagged'][0]
        self.assertEqual(entry['flag'], validation.FLAG_TEST_TAMPERING)
        self.assertAlmostEqual(entry['reward'], 0.3, places=10)
        self.assertEqual(entry['action_category'], 'edit_file')

    def test_the_command_accepts_a_reason_filter_and_a_limit(self) -> None:
        store.record_outcome(
            self.db, _state('test_failure'), 'edit_file', 'session=aaaa1111 edit test_stats.py',
            1.0, reward.SOURCE_RULE, base_reward=1.0, churn=1, touch_test_file=True,
        )
        reply = self._run(['flagged', '--flag', validation.FLAG_SMALL_CHANGE_HIGH_REWARD, '--limit', '5'])
        self.assertEqual(reply['flagged'], [])


def _state(error_type: str, tool_context: str = 'pwsh:pytest') -> store.StateSnapshot:
    """One pre-action state shaped like the pre-LLM hook leaves it."""
    return store.StateSnapshot(
        error_type=error_type, tool_context=tool_context, file_context='-', summary='seeded',
    )


class TestFileClassificationTests(unittest.TestCase):
    """The test-file signal reuses the classifier that already exists."""

    def test_a_test_module_is_a_test_file(self) -> None:
        self.assertEqual(classify.file_kind(TEST_FILE), 'test')

    def test_ordinary_source_is_not_a_test_file(self) -> None:
        self.assertEqual(classify.file_kind(SOURCE_FILE), 'source')


if __name__ == '__main__':
    unittest.main()
