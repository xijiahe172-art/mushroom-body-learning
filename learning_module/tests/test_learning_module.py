"""Phase 0 acceptance tests for the Learning Module infrastructure.

Covers: the `experiences` table and its index, the `DSH_LEARNING_MODULE`
switch (default off), the hook CLI, and the fail-silent rule (a broken hook
must never raise out of the module).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from learning_module import hooks, schema  # noqa: E402
from learning_module.hooks import handle_hook  # noqa: E402
from learning_module.modes import (  # noqa: E402
    LEARNING_MODULE_ENV,
    current_mode,
)

CLI = REPO_ROOT / 'learning_module' / 'cli.py'

REQUIRED_COLUMNS = (
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
    'confidence',
    'last_used_at',
    'created_at',
    'reward_source',
    'flagged_for_review',
)


class EnvSandbox(unittest.TestCase):
    """Restores DSH_* environment state around every test."""

    def setUp(self) -> None:
        self._saved = {k: v for k, v in os.environ.items() if k.startswith('DSH_')}
        for key in list(os.environ):
            if key.startswith('DSH_'):
                del os.environ[key]
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-learning-test-')
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self) -> None:
        for key in list(os.environ):
            if key.startswith('DSH_'):
                del os.environ[key]
        os.environ.update(self._saved)

    @property
    def tmp(self) -> Path:
        return Path(self._tmp.name)

    def db_path(self) -> Path:
        return self.tmp / 'learning.db'

    def connection(self) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        """Open the scratch store read-only for assertions, closing it afterwards.

        `with sqlite3.connect(...)` commits but never closes; a leaked handle
        keeps the file locked, which fails the scratch-directory cleanup on Windows.
        """
        return contextlib.closing(sqlite3.connect(self.db_path()))


class ModeSwitchTests(EnvSandbox):
    def test_defaults_to_off_when_unset(self) -> None:
        self.assertEqual(current_mode(), 'off')

    def test_accepts_the_three_declared_modes(self) -> None:
        for mode in ('off', 'shadow', 'active'):
            os.environ[LEARNING_MODULE_ENV] = mode
            self.assertEqual(current_mode(), mode)

    def test_unrecognized_value_falls_back_to_off_without_raising(self) -> None:
        for raw in ('OFF', 'yes', ' ', 'shadow '):
            os.environ[LEARNING_MODULE_ENV] = raw
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(current_mode(), 'off')
            self.assertIn('error', stderr.getvalue())


class SchemaTests(EnvSandbox):
    def test_ensure_schema_creates_every_declared_column(self) -> None:
        schema.ensure_schema(self.db_path())
        with self.connection() as conn:
            columns = {row[1] for row in conn.execute('PRAGMA table_info(experiences)')}
        self.assertEqual(set(REQUIRED_COLUMNS) - columns, set())

    def test_ensure_schema_is_idempotent(self) -> None:
        schema.ensure_schema(self.db_path())
        schema.ensure_schema(self.db_path())
        with self.connection() as conn:
            columns = {row[1] for row in conn.execute('PRAGMA table_info(experiences)')}
        self.assertEqual(set(REQUIRED_COLUMNS) - columns, set())

    def test_indexes_experience_lookup_by_error_type_and_action_category(self) -> None:
        schema.ensure_schema(self.db_path())
        with self.connection() as conn:
            declarations = {
                row[1]: tuple(column[2] for column in conn.execute(f'PRAGMA index_info({row[1]!r})'))
                for row in conn.execute('PRAGMA index_list(experiences)')
                if row[3] == 'c'
            }
        self.assertIn(('state_error_type', 'action_category'), declarations.values())

    def test_ensure_schema_creates_missing_parent_directories(self) -> None:
        nested = self.tmp / 'nested' / 'deeper' / 'learning.db'
        schema.ensure_schema(nested)
        self.assertTrue(nested.is_file())

    def test_defaults_match_the_declared_experience_row(self) -> None:
        schema.ensure_schema(self.db_path())
        with self.connection() as conn:
            conn.execute('INSERT INTO experiences (state_error_type, state_summary) VALUES (?, ?)', ('none', 'x'))
            conn.commit()
            row = conn.execute(
                'SELECT action_category, success, success_count, fail_count, total_count,'
                ' average_reward, confidence, reward_source, flagged_for_review, created_at'
                ' FROM experiences'
            ).fetchone()
        self.assertEqual(row[0], 'none')
        self.assertEqual(row[1:9], (0, 0, 0, 0, 0.0, 0.0, 'none', 0))
        self.assertTrue(row[9])


class HookCliTests(EnvSandbox):
    def run_cli(self, args: list[str], payload: object | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            input=None if payload is None else json.dumps(payload),
            capture_output=True, text=True, env=dict(os.environ), check=False,
        )

    def test_hook_reports_the_hook_it_served(self) -> None:
        result = self.run_cli(['hook'], {'hook': 'pre-llm', 'sessionId': 's1', 'turn': 1, 'step': 1})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['hook'], 'pre-llm')

    def test_hook_logs_that_it_was_called(self) -> None:
        # Phase 1: an off-mode hook is completely silent, so the module's own
        # diagnostics are asserted in a learning mode instead.
        os.environ[LEARNING_MODULE_ENV] = 'shadow'
        os.environ['DSH_LEARNING_DB'] = str(self.db_path())
        result = self.run_cli(['hook'], {'hook': 'post-action', 'sessionId': 's1', 'turn': 1, 'step': 1})
        self.assertIn('hook called', result.stderr)

    def test_hook_stays_silent_when_off(self) -> None:
        result = self.run_cli(['hook'], {'hook': 'post-action', 'sessionId': 's1', 'turn': 1, 'step': 1})
        self.assertEqual(result.stderr, '')

    def test_hook_stays_off_by_default(self) -> None:
        os.environ.pop(LEARNING_MODULE_ENV, None)
        self.assertEqual(json.loads(self.run_cli(['hook'], {'hook': 'pre-llm'}).stdout)['mode'], 'off')

    def test_hook_serves_shadow_mode_payloads(self) -> None:
        os.environ[LEARNING_MODULE_ENV] = 'shadow'
        os.environ['DSH_LEARNING_DB'] = str(self.db_path())
        payload = {
            'hook': 'post-action', 'sessionId': 's1', 'turn': 2, 'step': 3,
            'results': [{'name': 'read', 'isError': False, 'arguments': {'file_path': 'a.py'}}],
        }
        result = self.run_cli(['hook'], payload)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body['mode'], 'shadow')
        self.assertIs(body['ok'], True)

    def test_malformed_stdin_fails_silently(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI), 'hook'],
            input='{not json', capture_output=True, text=True, env=dict(os.environ), check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(json.loads(result.stdout)['ok'], False)

    def test_unknown_hook_name_fails_silently(self) -> None:
        result = self.run_cli(['hook'], {'hook': 'not-a-hook'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(json.loads(result.stdout)['ok'], False)

    def test_init_db_creates_the_experiences_table(self) -> None:
        os.environ['DSH_LEARNING_DB'] = str(self.db_path())
        self.assertEqual(self.run_cli(['init-db']).returncode, 0)
        with self.connection() as conn:
            self.assertIsNotNone(conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='experiences'"
            ).fetchone())

    def test_config_reports_the_active_mode_and_database_path(self) -> None:
        os.environ[LEARNING_MODULE_ENV] = 'active'
        os.environ['DSH_LEARNING_DB'] = str(self.db_path())
        body = json.loads(self.run_cli(['config']).stdout)
        self.assertEqual(body['mode'], 'active')
        self.assertEqual(body['database'], str(self.db_path()))


class FailSilentTests(EnvSandbox):
    def test_internal_exception_is_swallowed_and_reported(self) -> None:
        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError('retriever exploded')

        original = hooks.serve
        hooks.serve = explode
        self.addCleanup(lambda: setattr(hooks, 'serve', original))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = handle_hook(
                {'hook': 'pre-llm', 'sessionId': 's1', 'turn': 1, 'step': 1},
                mode='active',
                database=str(self.db_path()),
            )
        self.assertIs(result['ok'], False)
        self.assertIn('retriever exploded', stderr.getvalue())

    def test_invalid_database_path_does_not_raise(self) -> None:
        # A NUL byte is never a legal path on any platform, so the store fails
        # at connect time exactly like a corrupt file would.
        with contextlib.redirect_stderr(io.StringIO()):
            result = handle_hook(
                {'hook': 'pre-llm', 'sessionId': 's1', 'turn': 1, 'step': 1},
                mode='active',
                database='bad\x00path.db',
            )
        self.assertIs(result['ok'], False)

    def test_unexpected_state_shape_does_not_raise(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            result = handle_hook({'hook': 'pre-llm', 'turn': 'not-a-number'}, mode='shadow', database=str(self.db_path()))
        self.assertIsInstance(result, dict)

    def test_error_is_reported_on_stderr_not_stdout(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            handle_hook(
                {'hook': 'pre-llm', 'sessionId': 's1', 'turn': 1, 'step': 1},
                mode='active',
                database='bad\x00path.db',
            )
        self.assertIn('error', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
