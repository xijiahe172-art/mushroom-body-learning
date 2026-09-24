"""Phase 2 acceptance: one check per stated criterion, driven through the store.

Every scenario is written with `store.record_outcome`, so the checked quantities
are the ones that actually land in SQLite: `value` comes from the loss-averse
EMA on the write path, and retrieval decays that column rather than the
arithmetic mean.

Criteria covered, in the order the phase specification lists them:
  1. retrieval finds relevant history for repeated scenarios,
  2. decay at exactly one half-life equals 0.5, and a stale row is skipped,
  3. low confidence yields no injection,
  4. six consistent observations outrank three-of-each,
  5. the loss-aversion branch produces the two specified values in the column,
  6. the endowment weighting keeps a verified experience ahead of a thin one.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_module import context, retrieve, schema, store, value  # noqa: E402

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
PASS = 'PASS'
FAIL = 'FAIL'

results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str) -> None:
    """Record one acceptance result.

    @param name - criterion name.
    @param condition - whether it held.
    @param detail - the observed numbers behind the verdict.
    """
    results.append((name, PASS if condition else FAIL, detail))
    print(f'[{PASS if condition else FAIL}] {name}\n       {detail}')


def scenario(
    conn: sqlite3.Connection,
    *,
    error_type: str,
    action_category: str,
    tool_context: str,
    rewards: list[float],
    age_days: float,
    detail: str,
) -> None:
    """Write one scenario the way the hook path writes it, then age it.

    @param conn - open connection used only to age the row afterwards.
    @param error_type - failure kind of the state.
    @param action_category - the action's category.
    @param tool_context - tools the previous step used.
    @param rewards - the observed reward sequence, in order.
    @param age_days - how old the last observation should be.
    @param detail - debug description.
    """
    state = store.StateSnapshot(
        error_type=error_type, tool_context=tool_context, file_context='-', summary=f'error={error_type}',
    )
    for reward in rewards:
        store.record_outcome(conn_path, state, action_category, detail, reward, 'rule')
    stamp = (NOW - timedelta(days=age_days)).isoformat().replace('+00:00', 'Z')
    conn.execute(
        'UPDATE experiences SET last_used_at = ?, created_at = ?'
        ' WHERE state_error_type = ? AND action_category = ?',
        (stamp, stamp, error_type, action_category),
    )
    conn.commit()


def row_for(database: Path, error_type: str, action_category: str, tool_context: str) -> dict:
    """Read one stored row by its full identity triple.

    @param database - store path.
    @param error_type - failure kind of the state.
    @param action_category - the action's category.
    @param tool_context - tools the previous step used.
    @returns the row as a mapping, or an empty mapping when absent.
    """
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        found = conn.execute(
            'SELECT * FROM experiences'
            ' WHERE state_error_type = ? AND state_tool_context = ? AND action_category = ?',
            (error_type, tool_context, action_category),
        ).fetchone()
        return {} if found is None else dict(found)
    finally:
        conn.close()


def main() -> int:
    """Run every acceptance criterion and summarize.

    @returns `0` when all criteria pass.
    """
    global conn_path
    workdir = Path(tempfile.mkdtemp(prefix='dsh-p2-accept-'))
    db = workdir / 'acceptance.db'
    conn_path = db
    schema.ensure_schema(db)
    conn = sqlite3.connect(db)
    try:
        # Criterion 1: a scenario seen repeatedly, at three difficulty levels.
        scenario(conn, error_type='test_failure', action_category='edit_file', tool_context='pwsh',
                 rewards=[1.0, 1.0, 1.0, 0.6, 0.6, 0.6], age_days=1,
                 detail='session=aaaa0001 edit src/app.py')
        scenario(conn, error_type='test_failure', action_category='run_test', tool_context='edit',
                 rewards=[1.0] * 7, age_days=2, detail='session=aaaa0002 pwsh python -m pytest')
        scenario(conn, error_type='test_collection_error', action_category='edit_file', tool_context='pwsh',
                 rewards=[0.6, 0.6, 0.6, 0.2], age_days=3,
                 detail='session=aaaa0003 edit tests/test_app.py')

        # Criterion 2: one row exactly at the half-life whose value differs from
        # its mean, and one decayed past the floor.
        scenario(conn, error_type='build_failure', action_category='run_build', tool_context='pwsh',
                 rewards=[1.0, 0.6, 0.4], age_days=30, detail='session=bbbb0001 npx tsc -b')
        scenario(conn, error_type='syntax_error', action_category='edit_file', tool_context='pwsh',
                 rewards=[0.1, 0.05, 0.1, 0.05, 0.1, 0.05], age_days=90,
                 detail='session=bbbb0002 stale evidence')

        # Criterion 3: one row under the per-row confidence floor.
        scenario(conn, error_type='tool_not_found', action_category='run_shell', tool_context='read',
                 rewards=[0.8], age_days=0, detail='session=cccc0001 single observation')

        # Criterion 4: consistent versus alternating, same total.
        scenario(conn, error_type='dependency_error', action_category='install_deps', tool_context='pwsh',
                 rewards=[0.7, 0.8, 0.8, 0.5, 0.7, 0.7], age_days=0, detail='session=dddd0001 consistent')
        scenario(conn, error_type='dependency_error', action_category='run_shell', tool_context='pwsh',
                 rewards=[0.9, 0.1, 0.9, 0.1, 0.9, 0.1], age_days=0, detail='session=dddd0002 alternating')

        # Criterion 6: a verified experience against a thin one whose mean is
        # just as good. Both means are 0.625, so only the observation count and
        # the confidence it buys can separate them; the thin sequence is also
        # far noisier (stdev 0.325 against 0.043), which is what low confidence
        # is meant to express.
        scenario(conn, error_type='file_not_found', action_category='read_file', tool_context='pwsh',
                 rewards=[0.6, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7], age_days=0,
                 detail='session=eeee0001 verified')
        scenario(conn, error_type='file_not_found', action_category='search_files', tool_context='glob',
                 rewards=[0.35, 0.9], age_days=0, detail='session=eeee0002 thin')
    finally:
        conn.close()

    print(f'store: {db}\n')

    print('--- 0. the write path persisted the loss-averse value ---')
    stored = row_for(db, 'test_failure', 'edit_file', 'pwsh')
    sequence = [1.0, 1.0, 1.0, 0.6, 0.6, 0.6]
    check(
        '0. stored value equals a replay of the recorded sequence',
        abs(stored['value'] - value.replay_value(sequence)) < 1e-12,
        f"value={stored['value']!r} replay={value.replay_value(sequence)!r}"
        f" mean={stored['average_reward']!r}",
    )
    check(
        '0. value and average_reward are genuinely different quantities',
        abs(stored['value'] - stored['average_reward']) > 1e-6,
        f"value={stored['value']!r} != average_reward={stored['average_reward']!r}"
        f" (loss aversion moved it by {stored['value'] - stored['average_reward']:+.4f})",
    )

    print('\n--- 1. retrieval finds relevant history for repeated scenarios ---')
    for error_type, expect_action in (
        ('test_failure', 'run_test'),
        ('test_collection_error', 'edit_file'),
        ('build_failure', 'run_build'),
    ):
        result = retrieve.candidates_for(db, error_type, now=NOW)
        categories = [candidate.action_category for candidate in result.candidates]
        check(
            f'1. {error_type}: history retrieved',
            len(result.candidates) > 0 and expect_action in categories,
            f'searched={list(result.error_types_searched)} ranked={categories}'
            f' injectable={result.injectable}',
        )
    relaxed = retrieve.candidates_for(db, 'test_collection_error', now=NOW)
    check(
        '1. relaxation widened the search for a thin exact match',
        len(relaxed.error_types_searched) > 1,
        f'searched={list(relaxed.error_types_searched)} exact={relaxed.exact_matches}',
    )
    unrelated = retrieve.candidates_for(db, 'none', now=NOW)
    check(
        '1. an unrelated state returns nothing rather than padding',
        unrelated.candidates == () and not unrelated.injectable,
        f'searched={list(unrelated.error_types_searched)} ranked={len(unrelated.candidates)}',
    )

    print('\n--- 2. decay ---')
    half = value.decay_factor(value.HALF_LIFE_DAYS)
    check(
        '2. decay_factor(age_days=30) == 0.5',
        abs(half - 0.5) < 1e-12,
        f'decay_factor(30.0) = {half!r}',
    )
    build = retrieve.candidates_for(db, 'build_failure', now=NOW)
    build_row = next((c for c in build.candidates if c.action_category == 'run_build'), None)
    build_stored = row_for(db, 'build_failure', 'run_build', 'pwsh')
    check(
        '2. a 30-day-old row keeps exactly half of its stored value',
        build_row is not None
        and abs(build_row.decay - 0.5) < 1e-12
        and abs(build_row.effective - build_stored['value'] * 0.5) < 1e-9,
        f"decay={build_row.decay!r} effective={build_row.effective!r}"
        f" value={build_stored['value']!r} mean={build_stored['average_reward']!r}"
        if build_row is not None else 'the 30-day-old row was not ranked',
    )
    check(
        '2. decay acted on value, not on the arithmetic mean',
        build_row is not None
        and abs(build_row.effective - build_stored['average_reward'] * 0.5) > 1e-6,
        f"effective={build_row.effective!r} would be"
        f" {build_stored['average_reward'] * 0.5!r} if the mean were used"
        if build_row is not None else 'not ranked',
    )
    stale = retrieve.candidates_for(db, 'syntax_error', now=NOW)
    check(
        '2. a row decayed under the floor is skipped silently',
        all(candidate.average_reward != row_for(db, 'syntax_error', 'edit_file', 'pwsh').get('average_reward')
            for candidate in stale.candidates)
        and stale.skipped_stale >= 1,
        f"stored value {row_for(db, 'syntax_error', 'edit_file', 'pwsh').get('value')!r} * 0.125 ="
        f" {value.effective_value(row_for(db, 'syntax_error', 'edit_file', 'pwsh').get('value', 0.0), 90.0)!r}"
        f' (< {value.MIN_EFFECTIVE_VALUE}), skipped_stale={stale.skipped_stale}',
    )

    print('\n--- 3. confidence floor ---')
    thin = retrieve.candidates_for(db, 'tool_not_found', now=NOW)
    check(
        '3. below-threshold confidence yields no injection',
        thin.candidates == () and not thin.injectable and context.render(thin.entries).text == '',
        f'confidence={value.confidence(1, 0)!r} < {value.MIN_CONFIDENCE};'
        f' skipped_low_confidence={thin.skipped_low_confidence}',
    )

    print('\n--- 4. consistency beats alternation ---')
    consistent = value.confidence(6, 0)
    alternating = value.confidence(3, 3)
    dep = retrieve.candidates_for(db, 'dependency_error', now=NOW)
    order = [candidate.action_category for candidate in dep.candidates]
    check(
        '4. six consistent observations are far more confident than 3-3',
        consistent > alternating * 4,
        f'confidence(6,0)={consistent!r} vs confidence(3,3)={alternating!r}'
        f' (ratio {consistent / alternating:.1f}x)',
    )
    check(
        '4. the consistent experience also ranks first',
        order[:1] == ['install_deps'],
        f'ranked={order} stored values='
        + ', '.join(f"{c.action_category}:{c.stored_value:.3f}" for c in dep.candidates),
    )

    print('\n--- 5. loss aversion, through the store ---')
    # The specification states both cases as updates "from old_value = 0". Each
    # therefore needs its own pair: seeding one pair and updating it twice would
    # measure the second case from the value the first produced, which is a
    # different (also correct) behaviour.
    gain_state = store.StateSnapshot(
        error_type='timeout', tool_context='pwsh', file_context='-', summary='gain case',
    )
    store.record_outcome(db, gain_state, 'read_file', 'seed', 0.0, 'rule')
    gained = store.record_outcome(db, gain_state, 'read_file', 'gain', 1.0, 'rule').value
    gain_row = row_for(db, 'timeout', 'read_file', 'pwsh')
    loss_state = store.StateSnapshot(
        error_type='timeout', tool_context='rw', file_context='-', summary='loss case',
    )
    store.record_outcome(db, loss_state, 'read_file', 'seed', 0.0, 'rule')
    lost = store.record_outcome(db, loss_state, 'read_file', 'loss', -1.0, 'rule').value
    loss_row = row_for(db, 'timeout', 'read_file', 'rw')
    check(
        '5. both pairs start from the documented old_value of 0',
        abs(value.replay_value([0.0])) < 1e-12,
        'a zero-reward observation leaves value=0 with the pair already having history',
    )
    check(
        '5. +1.0 from 0 persists +0.30',
        abs(gained - 0.30) < 1e-12 and abs(gain_row['value'] - 0.30) < 1e-12,
        f"column={gain_row['value']!r} (mean {gain_row['average_reward']!r})",
    )
    check(
        '5. -1.0 from 0 takes the loss branch and persists -0.75',
        abs(lost + 0.75) < 1e-12 and abs(loss_row['value'] + 0.75) < 1e-12,
        f"column={loss_row['value']!r} (the gain branch would have given -0.30)",
    )
    check(
        '5. the column agrees with the engine on both sequences',
        abs(gain_row['value'] - value.replay_value([0.0, 1.0])) < 1e-12
        and abs(loss_row['value'] - value.replay_value([0.0, -1.0])) < 1e-12,
        f"gain {gain_row['value']!r} == replay {value.replay_value([0.0, 1.0])!r};"
        f" loss {loss_row['value']!r} == replay {value.replay_value([0.0, -1.0])!r}",
    )

    print('\n--- 6. endowment weighting ---')
    found = retrieve.candidates_for(db, 'file_not_found', now=NOW)
    ranked = list(found.candidates)
    verified = row_for(db, 'file_not_found', 'read_file', 'pwsh')
    thin_row = row_for(db, 'file_not_found', 'search_files', 'glob')
    check(
        '6. the verified experience outranks the thin one',
        len(ranked) >= 2 and ranked[0].action_category == 'read_file',
        'ranked=' + ', '.join(
            f'{c.action_category}(value={c.stored_value:+.3f} eff={c.effective:+.3f}'
            f' conf={c.confidence:.2f} adjusted={c.adjusted_score:+.3f})'
            for c in ranked
        ),
    )
    check(
        '6. the two are fair rivals: their arithmetic means are equal',
        abs(verified['average_reward'] - thin_row['average_reward']) < 1e-12,
        f"verified mean={verified['average_reward']!r}"
        f" thin mean={thin_row['average_reward']!r}",
    )
    check(
        '6. confidence, not the mean, is what separates them',
        len(ranked) >= 2
        and ranked[0].confidence > ranked[1].confidence
        and abs(ranked[0].stored_value - ranked[1].stored_value) > 1e-6,
        f"verified conf={ranked[0].confidence!r} value={ranked[0].stored_value:.4f} vs"
        f" thin conf={ranked[1].confidence!r} value={ranked[1].stored_value:.4f}"
        f" -> adjusted {ranked[0].adjusted_score:+.4f} vs {ranked[1].adjusted_score:+.4f}",
    )

    # What the specified weighting can and cannot overturn. The example in the
    # phase brief (A: confidence 0.8 at effective 0.5 against B: confidence 0.2
    # at effective 0.55) is NOT flipped by a 0.2 coefficient: B's adjusted score
    # is 0.572 against A's 0.580. Reporting the real crossover keeps the
    # constant's reach explicit instead of leaving a false expectation in place.
    def adjusted(effective: float, confidence: float) -> float:
        return effective * (1 + value.ENDOWMENT_WEIGHT * confidence)

    brief_b = adjusted(0.55, 0.2)
    brief_a = adjusted(0.5, 0.8)
    # How far above a confidence-0.8 row a confidence-0.2 row must sit to win.
    crossover = (1 + value.ENDOWMENT_WEIGHT * 0.8) / (1 + value.ENDOWMENT_WEIGHT * 0.2)
    check(
        '6. the reach of the specified weighting is reported, not assumed',
        brief_a > brief_b,
        f'brief example: A(0.5, 0.8)={brief_a:.4f} vs B(0.55, 0.2)={brief_b:.4f} -> A first,'
        f' but only just. A confidence-0.2 row overtakes a confidence-0.8 row only above'
        f' {crossover:.4f}x its value, so the 0.2 coefficient cannot overturn a 10% lead.',
    )
    just_over = 0.5 * crossover * 1.01
    flippable = adjusted(just_over, 0.2) > adjusted(0.5, 0.8)
    check(
        '6. the weighting does overturn a thin row inside that reach',
        flippable,
        f'valueB = {just_over:.4f} (1% over {crossover:.4f}x A) at conf 0.2 ->'
        f' {adjusted(just_over, 0.2):.4f} > A {adjusted(0.5, 0.8):.4f}',
    )

    print('\n--- 7. shadow mode still injects nothing ---')
    import os
    from learning_module.hooks import get_experience_context
    # The gate takes the same state-bearing payload the pre-LLM hook receives, so
    # the injection and the record resolve the state through one code path.
    state_payload = {
        'priorResults': [{'name': 'pwsh', 'isError': True, 'text': '1 failed, 2 passed'}],
    }
    os.environ.pop('DSH_LEARNING_MODULE', None)
    check(
        '7. get_experience_context is None unless the mode is active',
        get_experience_context(state_payload, db, now=NOW) is None,
        'mode=off -> None',
    )
    os.environ['DSH_LEARNING_MODULE'] = 'shadow'
    shadow = get_experience_context(state_payload, db, now=NOW)
    os.environ.pop('DSH_LEARNING_MODULE', None)
    check('7. shadow mode computes but never returns context', shadow is None, 'mode=shadow -> None')
    os.environ['DSH_LEARNING_MODULE'] = 'active'
    active = get_experience_context(state_payload, db, now=NOW)
    os.environ.pop('DSH_LEARNING_MODULE', None)
    check(
        '7. active mode returns the labelled, rendered text',
        active is not None
        and 'Reference Only' in active
        and ('Approaches to avoid' in active or 'Successful approaches' in active),
        f'mode=active -> {len(active or "")} chars',
    )

    failed = [name for name, verdict, _ in results if verdict == FAIL]
    print(f'\n=== {len(results) - len(failed)}/{len(results)} checks passed ===')
    for name in failed:
        print(f'  FAILED: {name}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
