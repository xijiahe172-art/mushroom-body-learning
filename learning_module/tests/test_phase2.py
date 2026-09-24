"""Phase 2 acceptance tests: value computation and structured retrieval.

Every numeric expectation here is taken from the phase specification, including
the decay check at exactly one half-life and the two value-update cases that
pin the loss-aversion branch.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from learning_module import context, retrieve, schema, store, value  # noqa: E402

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)


class ValueEngineTests(unittest.TestCase):
    def test_first_observation_uses_the_base_rate(self) -> None:
        self.assertAlmostEqual(value.update_value(0.0, 1.0, has_history=False), 0.30, places=10)

    def test_loss_branch_matches_the_specified_case(self) -> None:
        # From old_value=0, a negative reward is a loss and moves at alpha*lambda.
        self.assertAlmostEqual(value.update_value(0.0, -1.0, has_history=True), -0.75, places=10)

    def test_both_specified_cases_from_zero(self) -> None:
        self.assertAlmostEqual(value.replay_value([1.0]), 0.30, places=10)
        self.assertAlmostEqual(value.replay_value([0.0, -1.0]), -0.75, places=10)

    def test_loss_is_measured_against_expectation_not_sign(self) -> None:
        # A positive reward below the stored value is still a loss.
        old = 0.8
        expected = old + (value.ALPHA * value.LAMBDA) * (0.4 - old)
        self.assertAlmostEqual(value.update_value(old, 0.4, has_history=True), expected, places=10)
        # A negative reward above the stored value is a gain.
        old_negative = -0.9
        expected_gain = old_negative + value.ALPHA * (-0.5 - old_negative)
        self.assertAlmostEqual(value.update_value(old_negative, -0.5, has_history=True), expected_gain, places=10)

    def test_a_first_observation_never_takes_the_loss_branch(self) -> None:
        self.assertAlmostEqual(value.update_value(0.0, -1.0, has_history=False), -0.30, places=10)

    def test_parameters_are_the_specified_constants(self) -> None:
        self.assertEqual(value.ALPHA, 0.3)
        self.assertEqual(value.LAMBDA, 2.5)
        self.assertEqual(value.HALF_LIFE_DAYS, 30.0)
        self.assertEqual(value.ENDOWMENT_WEIGHT, 0.2)

    def test_decay_at_one_half_life_is_exactly_one_half(self) -> None:
        # This is what makes "half-life = 30 days" a真实 statement rather than a name.
        self.assertAlmostEqual(value.decay_factor(30.0), 0.5, places=12)

    def test_decay_at_documented_ages(self) -> None:
        self.assertAlmostEqual(value.decay_factor(0.0), 1.0, places=12)
        self.assertAlmostEqual(value.decay_factor(60.0), 0.25, places=12)
        self.assertAlmostEqual(value.decay_factor(90.0), 0.125, places=12)

    def test_effective_value_multiplies_the_mean_by_the_decay(self) -> None:
        self.assertAlmostEqual(value.effective_value(0.8, 30.0), 0.4, places=12)
        self.assertAlmostEqual(value.effective_value(-1.0, 60.0), -0.25, places=12)

    def test_age_uses_the_last_use_timestamp(self) -> None:
        stamp = (NOW - timedelta(days=30)).isoformat().replace('+00:00', 'Z')
        self.assertAlmostEqual(value.age_in_days(stamp, NOW), 30.0, places=6)

    def test_age_of_a_never_used_row_is_zero(self) -> None:
        self.assertEqual(value.age_in_days(None, NOW), 0.0)
        self.assertEqual(value.age_in_days('', NOW), 0.0)

    def test_age_never_goes_negative(self) -> None:
        future = (NOW + timedelta(days=5)).isoformat().replace('+00:00', 'Z')
        self.assertEqual(value.age_in_days(future, NOW), 0.0)


class ConfidenceTests(unittest.TestCase):
    def test_six_successes_beat_three_of_each(self) -> None:
        consistent = value.confidence(6, 0)
        wavering = value.confidence(3, 3)
        self.assertGreater(consistent, wavering)
        # The specified schedule: min(0.9, 0.15*6) * max(stability, 0.2)
        self.assertAlmostEqual(consistent, 0.9, places=10)
        self.assertAlmostEqual(wavering, 0.18, places=10)
        self.assertGreater(consistent, wavering * 4)

    def test_wavering_keeps_a_fifth_of_its_base(self) -> None:
        self.assertAlmostEqual(value.confidence(5, 5), min(0.9, 0.15 * 6) * 0.2, places=10)

    def test_confidence_grows_with_the_observation_count(self) -> None:
        self.assertLess(value.confidence(1, 0), value.confidence(3, 0))
        self.assertLess(value.confidence(3, 0), value.confidence(6, 0))

    def test_confidence_saturates_at_the_cap(self) -> None:
        self.assertAlmostEqual(value.confidence(40, 0), 0.9, places=10)

    def test_a_row_without_observations_has_no_confidence(self) -> None:
        self.assertEqual(value.confidence(0, 0), 0.0)

    def test_all_failures_are_as_confident_as_all_successes(self) -> None:
        self.assertAlmostEqual(value.confidence(0, 6), value.confidence(6, 0), places=10)


class EndowmentTests(unittest.TestCase):
    def test_a_verified_experience_outranks_a_thin_higher_one(self) -> None:
        verified = value.Candidate(
            row_id=1, error_type='test_failure', tool_context='pwsh', action_category='run_test',
            action_detail='', success_count=8, fail_count=0, total_count=8, average_reward=0.5,
            stored_value=0.5, age_days=0.0, decay=1.0, effective=0.5, confidence=0.8,
            adjusted_score=0.5 * (1 + value.ENDOWMENT_WEIGHT * 0.8),
        )
        thin = value.Candidate(
            row_id=2, error_type='test_failure', tool_context='pwsh', action_category='run_test',
            action_detail='', success_count=2, fail_count=0, total_count=2, average_reward=0.55,
            stored_value=0.55, age_days=0.0, decay=1.0, effective=0.55, confidence=0.2,
            adjusted_score=0.55 * (1 + value.ENDOWMENT_WEIGHT * 0.2),
        )
        ranked = sorted([thin, verified], key=lambda candidate: abs(candidate.adjusted_score), reverse=True)
        self.assertEqual(ranked[0].row_id, 1)
        self.assertGreater(verified.adjusted_score, thin.adjusted_score)
        # The weighting only orders them: the raw values still favour the thin row.
        self.assertLess(verified.effective, thin.effective)


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-p2-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        schema.ensure_schema(self.db)

    def insert(
        self,
        *,
        error_type: str,
        action_category: str,
        tool_context: str = 'pwsh',
        success_count: int = 6,
        fail_count: int = 0,
        average_reward: float = 0.8,
        age_days: float = 0.0,
        detail: str = '',
    ) -> int:
        stamp = (NOW - timedelta(days=age_days)).isoformat().replace('+00:00', 'Z')
        conn = sqlite3.connect(self.db)
        try:
            cursor = conn.execute(
                'INSERT INTO experiences (state_error_type, state_tool_context, state_file_context,'
                ' state_summary, action_category, action_detail, reward, success, success_count,'
                ' fail_count, total_count, average_reward, confidence, last_used_at, created_at,'
                ' reward_source, flagged_for_review)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (error_type, tool_context, '-', 's', action_category, detail, average_reward,
                 1 if average_reward > 0 else 0, success_count, fail_count,
                 success_count + fail_count, average_reward, value.confidence(success_count, fail_count),
                 stamp, stamp, 'rule', 0),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)
        finally:
            conn.close()

    def test_exact_match_is_found(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test')
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(result.injectable)
        # The exact kind is always searched first; relaxation only adds kinds.
        self.assertEqual(result.error_types_searched[0], 'test_failure')

    def test_a_single_exact_match_still_relaxes(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test')
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertGreater(len(result.error_types_searched), 1)

    def test_relaxation_kicks_in_only_below_the_minimum(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test')
        self.insert(error_type='test_collection_error', action_category='run_test')
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertIn('test_collection_error', result.error_types_searched)
        self.assertEqual(len(result.candidates), 2)

    def test_relaxation_stops_once_the_minimum_is_met(self) -> None:
        # Three distinct identities of the exact kind: enough to stop there.
        for index in range(3):
            self.insert(error_type='test_failure', action_category=f'action_{index}')
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(result.error_types_searched, ('test_failure',))
        self.assertEqual(result.exact_matches, 3)

    def test_nothing_is_returned_when_no_kind_matches(self) -> None:
        self.assertIsNone(retrieve.candidates_for(self.db, 'test_failure', now=NOW).candidates or None)
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(result.candidates, ())
        self.assertFalse(result.injectable)
        self.assertIsNone(context.render(result.entries).text or None)

    def test_low_confidence_rows_are_skipped(self) -> None:
        # One observation: base 0.15 * stability 1.0 = 0.15, under the floor.
        self.insert(error_type='test_failure', action_category='run_test', success_count=1, fail_count=0)
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.skipped_low_confidence, 1)

    def test_stale_rows_are_skipped_and_counted(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test', average_reward=0.08, age_days=90)
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.skipped_stale, 1)

    def test_a_row_between_the_two_floors_is_ranked_but_withheld(self) -> None:
        # 5 good and 3 bad: base min(0.9, 0.15*6) = 0.9, stability 2*|5/8-0.5| = 0.25,
        # so confidence is 0.225 —?it clears the per-row floor and fails the
        # injection floor, which is exactly the case the second gate exists for.
        expected_confidence = value.confidence(5, 3)
        self.assertGreaterEqual(expected_confidence, value.MIN_CONFIDENCE)
        self.assertLess(expected_confidence, value.INJECTION_CONFIDENCE_FLOOR)
        self.insert(error_type='test_failure', action_category='run_test',
                    success_count=5, fail_count=3, average_reward=0.30)
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(len(result.candidates), 1)
        self.assertAlmostEqual(result.candidates[0].confidence, expected_confidence, places=10)
        self.assertFalse(result.injectable)
        self.assertIn('below the injection floor', result.reason)
        self.assertIsNone(context.render(result.entries).text or None)

    def test_the_injection_floor_only_gates_the_gap_between_the_floors(self) -> None:
        # Reachable confidences are discrete; the second gate matters only for
        # the values that clear the first one and sit under the second.
        gap = [
            (success, fail)
            for success in range(0, 13)
            for fail in range(0, 13)
            if success + fail > 0
            and value.MIN_CONFIDENCE <= value.confidence(success, fail) < value.INJECTION_CONFIDENCE_FLOOR
        ]
        self.assertTrue(gap, 'the schedule must expose at least one gated value')
        for success, fail in gap:
            computed = value.confidence(success, fail)
            self.assertGreaterEqual(computed, value.MIN_CONFIDENCE)
            self.assertLess(computed, value.INJECTION_CONFIDENCE_FLOOR)

    def test_exactly_six_consistent_observations_reach_the_top_confidence(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test',
                    success_count=6, fail_count=0, average_reward=0.9)
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertTrue(result.injectable)
        self.assertAlmostEqual(result.candidates[0].confidence, 0.9, places=10)

    def test_strong_evidence_is_injectable_and_ordered(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test',
                    success_count=6, fail_count=0, average_reward=0.5)
        self.insert(error_type='test_failure', action_category='edit_file',
                    success_count=6, fail_count=0, average_reward=0.9)
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertTrue(result.injectable)
        self.assertEqual([c.action_category for c in result.candidates], ['edit_file', 'run_test'])

    def test_at_most_three_candidates_are_returned(self) -> None:
        for index in range(5):
            self.insert(error_type='test_failure', action_category=f'action_{index}')
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(len(result.candidates), value.TOP_K)

    def test_decay_lowers_a_stale_row_below_a_fresh_one(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test',
                    success_count=6, fail_count=0, average_reward=0.9, age_days=120)
        self.insert(error_type='test_failure', action_category='edit_file',
                    success_count=6, fail_count=0, average_reward=0.3, age_days=0)
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(result.candidates[0].action_category, 'edit_file')

    def test_the_injected_text_carries_no_internal_score(self) -> None:
        self.insert(error_type='test_failure', action_category='edit_file',
                    success_count=6, fail_count=0, average_reward=0.9, detail='edit a.py')
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        text = context.render(result.entries).text
        self.assertNotEqual(text, '')
        # The endowment-weighted score stays internal; the text carries the
        # evidence the model can reason about instead.
        self.assertNotIn('1.08', text)
        self.assertIn('+0.90', text)
        self.assertIn('6 good / 0 bad of 6', text)
        self.assertIn('Successful approaches', text)

    def test_a_benign_state_is_not_relaxed_into_failures(self) -> None:
        self.insert(error_type='test_failure', action_category='run_test')
        result = retrieve.candidates_for(self.db, 'none', now=NOW)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.error_types_searched, ('none',))

    def test_retrieval_against_a_missing_store_creates_it_empty(self) -> None:
        fresh = Path(self._tmp.name) / 'nested' / 'fresh.db'
        result = retrieve.candidates_for(fresh, 'test_failure', now=NOW)
        self.assertEqual(result.candidates, ())
        self.assertTrue(fresh.is_file())


class InjectionGateTests(unittest.TestCase):
    """The gate the agent loop depends on must never raise."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-p2-gate-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        self._previous = __import__('os').environ.get('DSH_LEARNING_MODULE')
        self.addCleanup(_restore_mode, self._previous)

    def test_off_and_shadow_never_touch_the_store(self) -> None:
        from learning_module.hooks import get_experience_context
        payload = {'priorResults': [{'name': 'pwsh', 'isError': True}]}
        for mode in ('off', 'shadow', ''):
            __import__('os').environ['DSH_LEARNING_MODULE'] = mode
            self.assertIsNone(get_experience_context(payload, self.db, now=NOW))
        self.assertFalse(self.db.exists())

    def test_active_returns_none_for_an_empty_store(self) -> None:
        from learning_module.hooks import get_experience_context
        __import__('os').environ['DSH_LEARNING_MODULE'] = 'active'
        payload = {'priorResults': [{'name': 'pwsh', 'isError': True}]}
        self.assertIsNone(get_experience_context(payload, self.db, now=NOW))

    def test_a_broken_store_yields_none_instead_of_raising(self) -> None:
        import contextlib
        import io
        from learning_module.hooks import get_experience_context
        __import__('os').environ['DSH_LEARNING_MODULE'] = 'active'
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertIsNone(get_experience_context({'priorResults': []}, 'bad\x00path.db', now=NOW))
        self.assertIn('retrieval failed', stderr.getvalue())

    def test_active_returns_labelled_text_once_evidence_is_strong(self) -> None:
        import sqlite3
        from learning_module.hooks import get_experience_context
        schema.ensure_schema(self.db)
        conn = sqlite3.connect(self.db)
        try:
            conn.execute(
                'INSERT INTO experiences (state_error_type, state_tool_context, state_file_context,'
                ' state_summary, action_category, action_detail, reward, success, success_count,'
                ' fail_count, total_count, average_reward, value, rewards, confidence, last_used_at,'
                ' created_at, reward_source, flagged_for_review)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                ('test_failure', 'pwsh', '-', 's', 'edit_file', 'session=abcd1234 edit a.py',
                 0.6, 1, 6, 0, 6, 0.6, 0.6, '[0.6, 0.6, 0.6, 0.6, 0.6, 0.6]', 0.9,
                 NOW.isoformat().replace('+00:00', 'Z'), NOW.isoformat().replace('+00:00', 'Z'),
                 'rule', 0),
            )
            conn.commit()
        finally:
            conn.close()
        __import__('os').environ['DSH_LEARNING_MODULE'] = 'active'
        payload = {'priorResults': [{'name': 'pwsh', 'isError': True, 'text': '1 failed'}]}
        text = get_experience_context(payload, self.db, now=NOW)
        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn('edit_file', text)
        self.assertIn('+0.60', text)
        self.assertIn('Reference Only', text)


class PersistedValueTests(unittest.TestCase):
    """The EMA value must land in the row, not only in a pure function."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix='dsh-p2-value-')
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / 'experiences.db'
        schema.ensure_schema(self.db)
        self.state = store.StateSnapshot(
            error_type='test_failure', tool_context='pwsh', file_context='-', summary='s',
        )

    def record(self, reward: float) -> store.RecordedOutcome:
        return store.record_outcome(self.db, self.state, 'run_test', 'detail', reward, 'rule')

    def row(self) -> dict:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return dict(conn.execute('SELECT * FROM experiences').fetchone())
        finally:
            conn.close()

    def test_the_first_observation_persists_alpha_times_the_reward(self) -> None:
        outcome = self.record(1.0)
        self.assertAlmostEqual(outcome.value, 0.30, places=10)
        self.assertAlmostEqual(self.row()['value'], 0.30, places=10)

    def test_the_specified_loss_case_is_persisted(self) -> None:
        self.record(0.0)
        outcome = self.record(-1.0)
        # old_value 0.0 -> reward -1.0 is a loss -> 0 + 0.75*(-1.0) = -0.75
        self.assertAlmostEqual(outcome.value, -0.75, places=10)
        self.assertAlmostEqual(self.row()['value'], -0.75, places=10)

    def test_average_reward_stays_the_arithmetic_mean(self) -> None:
        self.record(1.0)
        self.record(-1.0)
        row = self.row()
        # Arithmetic mean of [+1, -1] is 0. The value is not: the second
        # observation fell below the expectation the first one set, so the loss
        # branch moved it by 0.75 of the gap: 0.3 + 0.75*(-1.3) = -0.675.
        self.assertAlmostEqual(row['average_reward'], 0.0, places=10)
        self.assertAlmostEqual(row['value'], -0.675, places=10)
        self.assertNotAlmostEqual(row['value'], row['average_reward'], places=6)

    def test_the_loss_branch_uses_the_gap_not_the_sign(self) -> None:
        # Two gains in a row: the second is above expectation, so it uses the
        # base rate even though the first one already moved the value.
        self.record(1.0)
        outcome = self.record(1.0)
        self.assertAlmostEqual(outcome.value, 0.3 + value.ALPHA * (1.0 - 0.3), places=10)
        self.assertAlmostEqual(outcome.value, 0.51, places=10)

    def test_the_reward_sequence_is_recorded_for_reconstruction(self) -> None:
        for reward in (1.0, 0.2, -0.8):
            self.record(reward)
        row = self.row()
        self.assertEqual(json.loads(row['rewards']), [1.0, 0.2, -0.8])
        self.assertAlmostEqual(row['value'], value.replay_value([1.0, 0.2, -0.8]), places=10)

    def test_retrieval_decays_the_stored_value_not_the_mean(self) -> None:
        # Two observations, both gains: confidence 0.30 clears both floors, and
        # the value stays below the mean, so decay acting on either is visible.
        self.record(1.0)
        self.record(0.2)
        row = self.row()
        self.assertAlmostEqual(row['average_reward'], 0.6, places=10)
        self.assertAlmostEqual(row['value'], 0.3 + 0.75 * (0.2 - 0.3), places=10)
        self.assertLess(row['value'], row['average_reward'])
        stamp = (NOW - timedelta(days=30)).isoformat().replace('+00:00', 'Z')
        conn = sqlite3.connect(self.db)
        try:
            conn.execute('UPDATE experiences SET last_used_at = ?, created_at = ?', (stamp, stamp))
            conn.commit()
        finally:
            conn.close()
        result = retrieve.candidates_for(self.db, 'test_failure', now=NOW)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertAlmostEqual(candidate.stored_value, row['value'], places=10)
        self.assertAlmostEqual(candidate.decay, 0.5, places=12)
        self.assertAlmostEqual(candidate.effective, row['value'] * 0.5, places=10)
        # The mean is still reported for diagnosis, but decay did not act on it.
        self.assertAlmostEqual(candidate.average_reward, row['average_reward'], places=10)
        self.assertNotAlmostEqual(candidate.effective, row['average_reward'] * 0.5, places=6)

    def test_backfill_replays_a_recorded_sequence(self) -> None:
        for reward in (1.0, -0.5, 0.25):
            self.record(reward)
        expected = value.replay_value([1.0, -0.5, 0.25])
        conn = sqlite3.connect(self.db)
        try:
            conn.execute('UPDATE experiences SET value = NULL')
            conn.commit()
        finally:
            conn.close()
        report = store.backfill_values(self.db)
        self.assertEqual(report.recomputed_from_sequence, 1)
        self.assertEqual(report.from_arithmetic_mean, 0)
        self.assertAlmostEqual(self.row()['value'], expected, places=10)

    def test_backfill_substitutes_the_mean_when_no_sequence_exists(self) -> None:
        self.record(1.0)
        conn = sqlite3.connect(self.db)
        try:
            # Simulate a row from a store that predates the value column.
            conn.execute('UPDATE experiences SET value = NULL, rewards = NULL')
            conn.commit()
        finally:
            conn.close()
        report = store.backfill_values(self.db)
        self.assertEqual(report.from_arithmetic_mean, 1)
        self.assertEqual(report.recomputed_from_sequence, 0)
        self.assertAlmostEqual(self.row()['value'], self.row()['average_reward'], places=10)

    def test_backfill_is_idempotent(self) -> None:
        self.record(1.0)
        for reward in (0.5, -0.25):
            self.record(reward)
        first = store.backfill_values(self.db)
        second = store.backfill_values(self.db)
        self.assertEqual(second.recomputed_from_sequence, 0)
        self.assertEqual(second.from_arithmetic_mean, 0)
        self.assertGreaterEqual(first.recomputed_from_sequence + first.already_present, 1)


def _restore_mode(previous: str | None) -> None:
    """Restore the module switch after a mode-sensitive test.

    @param previous - the value the switch held before the test.
    """
    import os
    if previous is None:
        os.environ.pop('DSH_LEARNING_MODULE', None)
    else:
        os.environ['DSH_LEARNING_MODULE'] = previous

    def test_a_neighbour_never_displaces_an_exact_match(self) -> None:
        # A two-row exact match (below the relaxation minimum) plus a much
        # stronger neighbour: relevance order still puts the exact rows first.
        self.insert(error_type='file_not_found', action_category='read_file',
                    success_count=6, fail_count=0, average_reward=0.4)
        self.insert(error_type='file_not_found', action_category='search_files',
                    success_count=6, fail_count=0, average_reward=0.3)
        self.insert(error_type='dependency_error', action_category='install_deps',
                    success_count=6, fail_count=0, average_reward=0.9)
        result = retrieve.candidates_for(self.db, 'file_not_found', now=NOW)
        self.assertGreater(len(result.error_types_searched), 1)
        self.assertEqual([c.action_category for c in result.candidates],
                         ['read_file', 'search_files', 'install_deps'])

    def test_a_neighbour_fills_a_gap_the_exact_kind_cannot(self) -> None:
        self.insert(error_type='file_not_found', action_category='read_file',
                    success_count=6, fail_count=0, average_reward=0.4)
        self.insert(error_type='dependency_error', action_category='install_deps',
                    success_count=6, fail_count=0, average_reward=0.9)
        result = retrieve.candidates_for(self.db, 'file_not_found', now=NOW)
        self.assertEqual([c.action_category for c in result.candidates],
                         ['read_file', 'install_deps'])


if __name__ == '__main__':
    unittest.main()
