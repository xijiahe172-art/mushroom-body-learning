"""Phase 1 acceptance tests: classification, state, store, and reward.

Every expectation here is a deterministic rule from the phase specification:
the action vocabulary, the experience identity, the three reward layers, and
the guarantee that shadow mode never reaches a model request.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from learning_module import classify, reward, schema, state, store  # noqa: E402
from learning_module.hooks import get_experience_context, handle_hook  # noqa: E402

PRE_LLM = 'pre-llm'
POST_ACTION = 'post-action'


class ScratchCase(unittest.TestCase):
    """Gives every test its own store file and database override."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-learning-p1-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        schema.ensure_schema(self.db)

    def rows(self) -> list[dict]:
        import sqlite3
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute('SELECT * FROM experiences ORDER BY id')]
        finally:
            conn.close()


class ActionVocabularyTests(unittest.TestCase):
    def test_tool_names_map_to_concrete_actions(self) -> None:
        cases = {
            'read': 'read_file',
            'write': 'write_file',
            'edit': 'edit_file',
            'str_replace_editor': 'edit_file',
            'grep': 'search_files',
            'glob': 'search_files',
            'read_image': 'inspect_image',
            'web_fetch': 'fetch_url',
            'web_search': 'web_search',
            'todo_write': 'track_task',
        }
        for tool, expected in cases.items():
            with self.subTest(tool=tool):
                self.assertEqual(classify.action_category(tool, {}), expected)

    def test_unknown_tool_falls_back_to_other(self) -> None:
        self.assertEqual(classify.action_category('teleport', {}), 'other')

    def test_shell_commands_split_into_distinct_actions(self) -> None:
        cases = {
            'pnpm test': 'run_test',
            'python -m pytest tests/': 'run_test',
            'npx vitest run foo.spec.ts': 'run_test',
            'npm run build:lib:host': 'run_build',
            'npx tsc -b tsconfig.json': 'run_build',
            'pnpm install': 'install_deps',
            'node scripts/probe.mjs': 'run_shell',
            'python app.py --serve': 'run_shell',
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(classify.action_category('pwsh', {'command': command}), expected)

    def test_every_category_is_a_documented_value(self) -> None:
        self.assertIn(classify.action_category('read', {}), classify.ACTION_CATEGORIES)
        self.assertIn(classify.action_category('pwsh', {'command': 'ls'}), classify.ACTION_CATEGORIES)

    def test_action_detail_records_the_call_and_stays_bounded(self) -> None:
        detail = classify.action_detail('edit', {'file_path': 'a/b.py', 'old_string': 'x' * 500})
        self.assertTrue(detail.startswith('edit a/b.py'))
        self.assertLessEqual(len(detail), 200)

    def test_file_kinds(self) -> None:
        cases = {
            'src/app.py': 'source',
            'tests/test_app.py': 'test',
            'spec/app.spec.ts': 'test',
            'config/settings.yaml': 'config',
            'README.md': 'doc',
            'data/blob.bin': 'other',
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(classify.file_kind(path), expected)

    def test_file_targets_reads_declared_arguments(self) -> None:
        self.assertEqual(
            classify.file_targets({'name': 'edit', 'arguments': {'file_path': ' src/a.py '}}),
            ('src/a.py',),
        )
        self.assertEqual(classify.file_targets({'name': 'read'}), ())

    def test_contexts_are_sorted_and_empty_safe(self) -> None:
        self.assertEqual(classify.tool_context(['write', 'read', 'read']), 'read,write')
        self.assertEqual(classify.tool_context([]), classify.NO_CONTEXT)
        self.assertEqual(classify.file_context(['b/tests/t.py', 'a/src/s.py']), 'source:s.py,test:t.py')
        self.assertEqual(classify.file_context([]), '-')


class ErrorKindTests(unittest.TestCase):
    def test_clean_step_has_no_failure(self) -> None:
        self.assertEqual(classify.step_error_kind([{'name': 'read', 'isError': False}]), 'none')

    def test_failing_test_run_is_a_failure_even_when_the_tool_succeeded(self) -> None:
        result = {'name': 'pwsh', 'isError': False, 'arguments': {'command': 'pnpm test'}, 'text': '2 failed | 16 passed'}
        self.assertEqual(classify.step_error_kind([result]), 'test_failure')

    def test_successful_result_without_failure_text_is_clean(self) -> None:
        result = {'name': 'pwsh', 'isError': False, 'arguments': {'command': 'pnpm test'}, 'text': '18 passed'}
        self.assertEqual(classify.step_error_kind([result]), 'none')

    def test_tool_usage_separates_what_a_shell_call_ran(self) -> None:
        cases = [
            ({'name': 'pwsh', 'arguments': {'command': 'python -m pytest tests/'}}, 'pwsh:pytest'),
            ({'name': 'pwsh', 'arguments': {'command': 'npx tsc -b'}}, 'pwsh:tsc'),
            ({'name': 'read', 'arguments': {'file_path': 'a.py'}}, 'read'),
            ({'name': ''}, classify.NO_CONTEXT),
        ]
        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify.tool_usage(result), expected)

    def test_failure_codes_map_to_kinds(self) -> None:
        cases = [
            ({'isError': True, 'error': {'code': 'TOOL_NOT_FOUND'}}, 'tool_not_found'),
            ({'isError': True, 'error': {'code': 'LLM_TIMEOUT'}}, 'timeout'),
            ({'isError': True, 'error': {'code': 'FS_PERMISSION_DENIED'}}, 'permission_denied'),
            ({'isError': True, 'error': {'code': 'TOOL_ABORTED_BEFORE_DISPATCH'}}, 'aborted'),
            ({'isError': True, 'error': {'code': 'TEST_FAILED'}, 'text': '3 failed | 2 passed'}, 'test_failure'),
            ({'isError': True, 'error': {'code': 'BUILD_FAILED'}, 'text': 'error TS2339: Property missing'}, 'build_failure'),
            ({'isError': True, 'error': {'code': 'EXIT_2'}, 'text': 'command failed'}, 'command_failed'),
            ({'isError': True, 'error': {'code': 'SOMETHING_ELSE'}, 'text': 'went sideways'}, 'tool_error'),
        ]
        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify.step_error_kind([result]), expected)

    def test_text_only_failures_still_classify(self) -> None:
        cases = [
            ({'isError': True, 'text': 'spawnSync python ENOENT'}, 'tool_error'),
            ({'isError': True, 'text': 'Error: unknown tool "bash"'}, 'tool_not_found'),
            ({'isError': True, 'text': '3 failed | 2 passed'}, 'test_failure'),
            ({'isError': True, 'text': 'error TS2339: Property missing'}, 'build_failure'),
        ]
        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify.step_error_kind([result]), expected)

    def test_failure_forms_are_distinguished(self) -> None:
        cases = [
            (
                {'isError': True, 'text': 'ERROR collecting test_geometry.py\nImportError: cannot import name X\n1 error'},
                'test_collection_error',
            ),
            (
                {'isError': True, 'text': 'ModuleNotFoundError: No module named "requests"\n1 failed'},
                'import_error',
            ),
            (
                {'isError': True, 'text': 'json.decoder.JSONDecodeError: Expecting \',\' delimiter: line 3'},
                'syntax_error',
            ),
            (
                {'isError': True, 'text': "python: command not found"},
                'dependency_error',
            ),
            (
                {'isError': True, 'text': "FileNotFoundError: [Errno 2] No such file or directory: 'data/rows.csv'"},
                'file_not_found',
            ),
            (
                {'isError': False, 'text': '2 passed, 1 warning\nPytestCacheWarning: could not create cache'},
                'warning',
            ),
        ]
        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify.step_error_kind([result]), expected)

    def test_collection_error_is_not_absorbed_by_the_runner_verdict(self) -> None:
        # A collection failure also reads "1 error", which the generic verdict
        # would swallow; the more specific form must win.
        result = {'isError': True, 'text': 'ERROR collecting tests/test_x.py\n1 error in 0.05s'}
        self.assertEqual(classify.step_error_kind([result]), 'test_collection_error')
        self.assertIn('test_collection_error', classify.ERROR_KINDS)

    def test_warning_is_a_benign_state_for_reward(self) -> None:
        self.assertTrue(reward.is_benign('none'))
        self.assertTrue(reward.is_benign('warning'))
        self.assertFalse(reward.is_benign('test_failure'))
        verdict = reward.evaluate('warning', 'warning', [{'name': 'pwsh', 'isError': False}])
        self.assertEqual(verdict.reward, reward.REWARD_UNCHANGED)
        cleared = reward.evaluate('test_failure', 'warning', [{'name': 'pwsh', 'isError': False}])
        self.assertEqual(cleared.reward, reward.REWARD_IMPROVED)

    def test_test_outcome_reads_the_runner_summary(self) -> None:
        passing = [{'name': 'pwsh', 'arguments': {'command': 'pnpm test'}, 'isError': False, 'text': '18 passed (18)'}]
        failing = [{'name': 'pwsh', 'arguments': {'command': 'pnpm test'}, 'isError': True, 'text': '2 failed | 16 passed'}]
        self.assertEqual(classify.test_outcome(passing), (True, True))
        self.assertEqual(classify.test_outcome(failing), (True, False))
        self.assertEqual(classify.test_outcome([{'name': 'read', 'arguments': {}, 'isError': False}]), (False, False))


class StateEncoderTests(unittest.TestCase):
    def test_prior_step_defines_the_observed_state(self) -> None:
        pre = state.begin_state({
            'toolNames': ['read', 'edit', 'pwsh'],
            'fileTargets': [{'name': 'edit', 'parameters': ['file_path']}],
            'priorResults': [
                {'name': 'pwsh', 'isError': True, 'error': {'code': 'TIMEOUT'}},
                {'name': 'read', 'isError': False, 'arguments': {'file_path': 'src\\app.py'}},
            ],
        })
        self.assertEqual(pre.error_type, 'timeout')
        # Only what the previous step actually used, never the whole request.
        self.assertEqual(pre.tool_context, 'pwsh,read')
        self.assertEqual(pre.file_context, 'source:app.py')

    def test_first_step_has_no_observed_context(self) -> None:
        pre = state.begin_state({'toolNames': ['read', 'edit']})
        self.assertEqual(pre.error_type, 'none')
        self.assertEqual(pre.tool_context, classify.NO_CONTEXT)
        self.assertEqual(pre.file_context, '-')

    def test_state_is_stable_for_the_same_observation(self) -> None:
        payload = {
            'priorResults': [{'name': 'run' if False else 'pwsh', 'isError': False,
                              'arguments': {'command': 'python app.py'}}],
        }
        first = state.begin_state(payload)
        second = state.begin_state(payload)
        self.assertEqual(first.tool_context, second.tool_context)
        self.assertEqual(first.file_context, second.file_context)
        self.assertEqual(first.summary, second.summary)

    def test_pre_state_carries_no_action_fields(self) -> None:
        pre = state.begin_state({'toolNames': ['read']})
        self.assertFalse(hasattr(pre, 'action_category'))
        self.assertFalse(hasattr(pre, 'action_detail'))

    def test_summary_is_structural_and_bounded(self) -> None:
        pre = state.begin_state({'toolNames': ['read'], 'priorResults': [{'name': 'read', 'isError': False}]})
        self.assertIn('error=none', pre.summary)
        self.assertIn('used=read', pre.summary)
        self.assertIn('offered=1', pre.summary)
        state.assert_summary_is_structural(pre.summary)

    def test_post_state_derives_action_and_modified_files(self) -> None:
        post = state.complete_state({
            'results': [
                {'name': 'edit', 'isError': False, 'mutates': True, 'arguments': {'file_path': 'src/app.py'}},
                {'name': 'pwsh', 'isError': False, 'mutates': False, 'arguments': {'command': 'pnpm test'}, 'text': '5 passed'},
            ],
        })
        self.assertEqual(post.action_category, 'edit_file')
        self.assertEqual(post.modified_files, ('src/app.py',))
        self.assertEqual(post.error_type, 'none')

    def test_post_state_reports_the_failure_it_left(self) -> None:
        post = state.complete_state({'results': [{'name': 'pwsh', 'isError': True, 'text': 'exit code 1'}]})
        self.assertEqual(post.action_category, 'run_shell')
        self.assertEqual(post.error_type, 'command_failed')

    def test_post_state_without_results_is_neutral(self) -> None:
        post = state.complete_state({})
        self.assertEqual(post.action_category, 'other')
        self.assertEqual(post.error_type, 'none')


class StoreTests(ScratchCase):
    def snapshot(self, error_type: str = 'none', tool_context: str = 'read', file_context: str = '-') -> store.StateSnapshot:
        return store.StateSnapshot(
            error_type=error_type, tool_context=tool_context, file_context=file_context, summary='s',
        )

    def test_pending_state_round_trips_and_is_deleted(self) -> None:
        store.save_pending_state(self.db, 's1', 1, 2, self.snapshot('timeout'))
        taken = store.take_pending_state(self.db, 's1', 1, 2)
        self.assertIsNotNone(taken)
        self.assertEqual(taken.error_type, 'timeout')  # type: ignore[union-attr]
        self.assertIsNone(store.take_pending_state(self.db, 's1', 1, 2))

    def test_pending_state_upserts_per_step(self) -> None:
        store.save_pending_state(self.db, 's1', 1, 1, self.snapshot('none'))
        store.save_pending_state(self.db, 's1', 1, 1, self.snapshot('timeout'))
        taken = store.take_pending_state(self.db, 's1', 1, 1)
        self.assertEqual(taken.error_type, 'timeout')  # type: ignore[union-attr]

    def test_unclaimed_pending_state_does_not_accumulate(self) -> None:
        # A final step's pre-hook never gets a post-hook, so the next pre-hook
        # must drop it rather than let the handoff table grow forever.
        for step in range(1, 6):
            store.save_pending_state(self.db, 's1', 1, step, self.snapshot('none'))
        store.save_pending_state(self.db, 's2', 1, 1, self.snapshot('none'))
        import sqlite3
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute('SELECT session_id, step FROM pending_states ORDER BY session_id').fetchall()
        finally:
            conn.close()
        self.assertEqual(rows, [('s1', 5), ('s2', 1)])

    def test_first_outcome_inserts_one_row(self) -> None:
        outcome = store.record_outcome(self.db, self.snapshot(), 'read_file', 'read a.py', 1.0, reward.SOURCE_RULE)
        self.assertFalse(outcome.updated)
        self.assertEqual((outcome.success_count, outcome.fail_count, outcome.total_count), (1, 0, 1))
        self.assertEqual(outcome.average_reward, 1.0)
        self.assertEqual(len(self.rows()), 1)

    def test_same_identity_updates_instead_of_inserting(self) -> None:
        store.record_outcome(self.db, self.snapshot(), 'read_file', 'first', 1.0, reward.SOURCE_RULE)
        outcome = store.record_outcome(self.db, self.snapshot(), 'read_file', 'second', -1.0, reward.SOURCE_RULE)
        self.assertTrue(outcome.updated)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual((outcome.success_count, outcome.fail_count, outcome.total_count), (1, 1, 2))
        self.assertEqual(outcome.average_reward, 0.0)
        self.assertEqual(self.rows()[0]['action_detail'], 'second')

    def test_identity_ignores_action_detail(self) -> None:
        store.record_outcome(self.db, self.snapshot(), 'run_test', 'a', 1.0, reward.SOURCE_RULE)
        store.record_outcome(self.db, self.snapshot(), 'run_test', 'completely different', 1.0, reward.SOURCE_RULE)
        self.assertEqual(len(self.rows()), 1)

    def test_different_identity_inserts_a_second_row(self) -> None:
        store.record_outcome(self.db, self.snapshot(), 'run_test', 'a', 1.0, reward.SOURCE_RULE)
        store.record_outcome(self.db, self.snapshot(error_type='timeout'), 'run_test', 'a', 1.0, reward.SOURCE_RULE)
        store.record_outcome(self.db, self.snapshot(), 'read_file', 'a', 1.0, reward.SOURCE_RULE)
        self.assertEqual(len(self.rows()), 3)

    def test_success_flag_follows_the_reward_sign(self) -> None:
        store.record_outcome(self.db, self.snapshot(), 'run_test', 'a', 0.0, reward.SOURCE_RULE)
        self.assertEqual(self.rows()[0]['success'], 0)
        self.assertEqual(self.rows()[0]['fail_count'], 1)

    def test_zero_reward_is_the_first_failure(self) -> None:
        outcome = store.record_outcome(self.db, self.snapshot(), 'run_test', 'a', 0.0, reward.SOURCE_RULE)
        self.assertEqual((outcome.success_count, outcome.fail_count, outcome.total_count), (0, 1, 1))

    def test_fetch_experience_reads_the_identity(self) -> None:
        store.record_outcome(self.db, self.snapshot(), 'read_file', 'a', 1.0, reward.SOURCE_RULE)
        row = store.fetch_experience(self.db, ('none', 'read', 'read_file'))
        self.assertIsNotNone(row)
        self.assertIsNone(store.fetch_experience(self.db, ('none', 'read', 'run_test')))


class RewardTests(unittest.TestCase):
    def test_passing_tests_reach_the_top_tier(self) -> None:
        verdict = reward.evaluate('none', 'none', [
            {'name': 'pwsh', 'arguments': {'command': 'pnpm test'}, 'isError': False, 'text': '10 passed'},
        ])
        self.assertEqual(verdict.reward, reward.REWARD_SOLVED)
        self.assertEqual(verdict.source, reward.SOURCE_RULE)

    def test_build_failure_is_a_severe_failure(self) -> None:
        verdict = reward.evaluate('none', 'build_failure', [
            {'name': 'pwsh', 'arguments': {'command': 'npx tsc'}, 'isError': True, 'text': 'error TS2339: missing'},
        ])
        self.assertEqual(verdict.reward, reward.REWARD_SEVERE)

    def test_failing_tests_without_a_failure_signal_are_partial(self) -> None:
        verdict = reward.evaluate('none', 'none', [
            {'name': 'pwsh', 'arguments': {'command': 'pnpm test'}, 'isError': True, 'text': '2 failed'},
        ])
        self.assertEqual(verdict.reward, reward.REWARD_PARTIAL)

    def test_clearing_a_failure_through_action_is_an_improvement(self) -> None:
        verdict = reward.evaluate('timeout', 'none', [{'name': 'read', 'isError': False}])
        self.assertEqual(verdict.reward, reward.REWARD_IMPROVED)

    def test_action_without_a_prior_failure_is_unchanged(self) -> None:
        verdict = reward.evaluate('none', 'none', [{'name': 'read', 'isError': False}])
        self.assertEqual(verdict.reward, reward.REWARD_UNCHANGED)
        self.assertEqual(verdict.penalty, 0.0)

    def test_new_failure_is_a_worsening(self) -> None:
        verdict = reward.evaluate('none', 'command_failed', [{'name': 'pwsh', 'isError': True}])
        self.assertEqual(verdict.reward, reward.REWARD_WORSENED)

    def test_unchanged_failure_is_ineffective(self) -> None:
        verdict = reward.evaluate('timeout', 'timeout', [{'name': 'read', 'isError': True}])
        self.assertEqual(verdict.reward, reward.REWARD_INEFFECTIVE)

    def test_changed_failure_is_a_worsening(self) -> None:
        verdict = reward.evaluate('timeout', 'build_failure', [{'name': 'read', 'isError': True}])
        self.assertEqual(verdict.reward, reward.REWARD_WORSENED)

    def test_static_penalties_never_exceed_the_cap(self) -> None:
        results = [{'name': f'read{i}', 'isError': False} for i in range(12)]
        verdict = reward.evaluate(
            'none', 'none', results,
            modified_files=['unrelated.py', 'tests/test_x.py'],
            observed_files=['src/app.py'],
        )
        self.assertGreaterEqual(verdict.penalty, reward.PENALTY_CAP)
        self.assertEqual(verdict.penalty, reward.PENALTY_CAP)
        self.assertIn('unrelated', verdict.rationale)

    def test_penalty_cannot_override_a_correctness_verdict(self) -> None:
        tests = [{
            'name': 'pwsh', 'arguments': {'command': 'pnpm test'}, 'isError': False, 'text': '9 passed',
        }]
        extra = [{'name': f'read{index}', 'isError': False} for index in range(12)]
        verdict = reward.evaluate(
            'none', 'none', tests + extra,
            modified_files=['/elsewhere/other.py'],
            observed_files=['src/app.py'],
        )
        self.assertEqual(verdict.reward, reward.REWARD_SOLVED + reward.PENALTY_CAP)
        self.assertGreater(verdict.reward, 0.5)

    def test_learning_modes_never_return_context_except_active(self) -> None:
        # Phase 2 turned this into a real retriever; the mode gate still keeps
        # `off` and `shadow` from ever handing text to a caller.
        import os
        import tempfile
        from pathlib import Path
        previous = os.environ.get('DSH_LEARNING_MODULE')
        self.addCleanup(_restore_mode, previous)
        with tempfile.TemporaryDirectory(prefix='dsh-p1-mode-') as workdir:
            empty = Path(workdir) / 'empty.db'
            for mode in ('off', 'shadow'):
                os.environ['DSH_LEARNING_MODULE'] = mode
                self.assertIsNone(get_experience_context({}, empty))
            os.environ['DSH_LEARNING_MODULE'] = 'active'
            # An empty store has nothing to offer, so active mode still yields None.
            self.assertIsNone(get_experience_context({}, empty))


def _restore_mode(previous: str | None) -> None:
    """Restore the module switch after a mode-sensitive test.

    @param previous - the value the switch held before the test.
    """
    import os
    if previous is None:
        os.environ.pop('DSH_LEARNING_MODULE', None)
    else:
        os.environ['DSH_LEARNING_MODULE'] = previous


class HookFlowTests(ScratchCase):
    def test_pre_hook_records_pending_state_only(self) -> None:
        envelope = handle_hook(
            {
                'hook': PRE_LLM, 'sessionId': 's1', 'turn': 1, 'step': 1,
                'toolNames': ['edit'], 'fileTargets': [{'name': 'edit', 'parameters': ['file_path']}],
            },
            mode='shadow', database=self.db,
        )
        self.assertTrue(envelope['ok'])
        self.assertEqual(self.rows(), [])
        pending = store.take_pending_state(self.db, 's1', 1, 1)
        self.assertIsNotNone(pending)

    def test_post_hook_writes_an_experience_row(self) -> None:
        handle_hook(
            {'hook': PRE_LLM, 'sessionId': 's1', 'turn': 1, 'step': 1, 'toolNames': ['pwsh']},
            mode='shadow', database=self.db,
        )
        envelope = handle_hook(
            {
                'hook': POST_ACTION, 'sessionId': 's1', 'turn': 1, 'step': 1,
                'results': [{
                    'name': 'pwsh', 'isError': False, 'mutates': False,
                    'arguments': {'command': 'pnpm test'}, 'text': '8 passed',
                }],
            },
            mode='shadow', database=self.db,
        )
        self.assertTrue(envelope['ok'])
        self.assertEqual(envelope['action_category'], 'run_test')
        self.assertEqual(envelope['reward'], reward.REWARD_SOLVED)
        row = self.rows()[0]
        self.assertEqual(row['action_category'], 'run_test')
        self.assertEqual(row['total_count'], 1)
        self.assertEqual(row['reward_source'], 'rule')
        self.assertEqual(row['state_error_type'], 'none')

    def test_repeated_identical_step_aggregates_into_one_row(self) -> None:
        for _ in range(3):
            handle_hook(
                {'hook': PRE_LLM, 'sessionId': 's1', 'turn': 1, 'step': 1, 'toolNames': ['read']},
                mode='shadow', database=self.db,
            )
            handle_hook(
                {
                    'hook': POST_ACTION, 'sessionId': 's1', 'turn': 1, 'step': 1,
                    'results': [{'name': 'read', 'isError': False, 'arguments': {'file_path': 'a.py'}}],
                },
                mode='shadow', database=self.db,
            )
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]['total_count'], 3)

    def test_action_detail_carries_the_session_origin_for_spot_checks(self) -> None:
        session = 'session-6f341416-37f7-4b55-9147-d36de3344be9'
        handle_hook(
            {'hook': PRE_LLM, 'sessionId': session, 'turn': 1, 'step': 1, 'toolNames': ['read']},
            mode='shadow', database=self.db,
        )
        handle_hook(
            {
                'hook': POST_ACTION, 'sessionId': session, 'turn': 1, 'step': 1,
                'results': [{'name': 'read', 'isError': False, 'arguments': {'file_path': 'a.py'}}],
            },
            mode='shadow', database=self.db,
        )
        detail = self.rows()[0]['action_detail']
        self.assertTrue(detail.startswith('session=6f341416 '), detail)
        self.assertIn('read a.py', detail)

    def test_session_origin_does_not_change_the_identity(self) -> None:
        for session in ('session-aaaaaaaa-0000', 'session-bbbbbbbb-1111'):
            handle_hook(
                {'hook': PRE_LLM, 'sessionId': session, 'turn': 1, 'step': 1, 'toolNames': ['read']},
                mode='shadow', database=self.db,
            )
            handle_hook(
                {
                    'hook': POST_ACTION, 'sessionId': session, 'turn': 1, 'step': 1,
                    'results': [{'name': 'read', 'isError': False, 'arguments': {'file_path': 'a.py'}}],
                },
                mode='shadow', database=self.db,
            )
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]['total_count'], 2)

    def test_post_hook_without_a_pending_state_still_records(self) -> None:
        envelope = handle_hook(
            {
                'hook': POST_ACTION, 'sessionId': 's1', 'turn': 4, 'step': 2,
                'results': [{'name': 'grep', 'isError': False}],
            },
            mode='shadow', database=self.db,
        )
        self.assertTrue(envelope['ok'])
        self.assertEqual(self.rows()[0]['state_error_type'], 'none')

    def test_off_mode_writes_nothing(self) -> None:
        envelope = handle_hook(
            {'hook': PRE_LLM, 'sessionId': 's1', 'turn': 1, 'step': 1, 'toolNames': ['read']},
            mode='off', database=self.db,
        )
        self.assertFalse(envelope.get('recorded', False))
        self.assertEqual(self.rows(), [])

    def test_active_mode_records_but_injects_nothing(self) -> None:
        import os
        previous = os.environ.get('DSH_LEARNING_MODULE')
        self.addCleanup(_restore_mode, previous)
        handle_hook(
            {'hook': PRE_LLM, 'sessionId': 's1', 'turn': 1, 'step': 1, 'toolNames': ['read']},
            mode='active', database=self.db,
        )
        handle_hook(
            {
                'hook': POST_ACTION, 'sessionId': 's1', 'turn': 1, 'step': 1,
                'results': [{'name': 'read', 'isError': False}],
            },
            mode='active', database=self.db,
        )
        self.assertEqual(len(self.rows()), 1)
        # Recording happens in active mode; retrieval needs enough evidence, and
        # a single observation is below the confidence floor, so still None.
        os.environ['DSH_LEARNING_MODULE'] = 'active'
        self.assertIsNone(get_experience_context('none', self.db))

    def test_broken_store_fails_silently(self) -> None:
        envelope = handle_hook(
            {'hook': PRE_LLM, 'sessionId': 's1', 'turn': 1, 'step': 1},
            mode='shadow', database='bad\x00path.db',
        )
        self.assertFalse(envelope['ok'])
        self.assertIn('error', envelope)


class SchemaTests(ScratchCase):
    def test_every_table_is_created(self) -> None:
        import sqlite3
        conn = sqlite3.connect(self.db)
        try:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        self.assertEqual(set(schema.TABLES) - names, set())

    def test_experience_identity_index_is_unique(self) -> None:
        ddl = _index_ddl()
        self.assertIn('CREATE UNIQUE INDEX', ddl)
        for column in schema.EXPERIENCE_KEY:
            self.assertIn(column, ddl)

    def test_ensuring_the_schema_twice_is_idempotent(self) -> None:
        schema.ensure_schema(self.db)
        schema.ensure_schema(self.db)


def _index_ddl() -> str:
    """Return the unique-key DDL this module applies."""
    return next(statement for statement in schema.schema_statements() if 'idx_experiences_key' in statement)


if __name__ == '__main__':
    with contextlib.suppress(Exception):
        os.environ.pop('DSH_LEARNING_MODULE', None)
    unittest.main()
