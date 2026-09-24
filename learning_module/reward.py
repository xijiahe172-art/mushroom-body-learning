"""Reward evaluation for one settled action.

Three layers, taken in priority order: a deterministic check of what actually
ran, then a before/after comparison of the structural error state, then static
penalties. Static penalties are capped at -0.1 so they can never outweigh a
correctness verdict. No layer asks a model to judge its own work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from learning_module import classify, validation

#: Reward tiers. These are the specification's reference bands.
REWARD_SOLVED = 1.0
REWARD_IMPROVED = 0.6
REWARD_PARTIAL = 0.2
REWARD_UNCHANGED = 0.0
REWARD_INEFFECTIVE = -0.3
REWARD_WORSENED = -0.8
REWARD_SEVERE = -1.0

#: Evaluator layers, stored as `reward_source`.
SOURCE_RULE = 'rule'
SOURCE_NONE = 'none'

#: Ceiling on the static-penalty layer's combined contribution.
PENALTY_CAP = -0.1

#: Static-penalty weights, summed and then capped at {@link PENALTY_CAP}.
PENALTY_UNRELATED_FILE = -0.05
PENALTY_TEST_FILE_TOUCHED = -0.05
PENALTY_EXCESSIVE_CALLS = -0.05

#: A step issuing more calls than this is penalized as excessive.
EXCESSIVE_CALL_THRESHOLD = 8

#: Error kinds that report no failure at all: a state resting on one of these
#: is a clean state, so a run that only emitted a warning still counts as
#: succeeding.
BENIGN_ERROR_KINDS = frozenset({'none', 'warning'})


def is_benign(error_type: str) -> bool:
    """Whether a structural failure kind means "no failure".

    @param error_type - one of {@link classify.ERROR_KINDS}.
    @returns `True` for the kinds that describe a clean state.
    """
    return error_type in BENIGN_ERROR_KINDS


@dataclass(frozen=True)
class RewardVerdict:
    """One evaluated reward plus why it was chosen."""

    #: Reward value in `[-1.0, 1.0]`.
    reward: float
    #: Layer that produced the base value: `rule` or `none`.
    source: str
    #: Base tier before static penalties.
    base: float
    #: Combined static penalty, already capped.
    penalty: float
    #: Human-readable explanation for logs and the manual spot check.
    rationale: str
    #: Multiplier Phase 4 validation applied; `1.0` when nothing was discounted.
    discount: float = 1.0
    #: Flag Phase 4 validation raised, or `''`.
    flag: str = ''

    @property
    def discounted(self) -> bool:
        """Whether validation cut this reward.

        @returns `True` when the recorded reward is below its evaluated value.
        """
        return self.discount < 1.0


def evaluate(
    before_error_type: str,
    after_error_type: str,
    results: Sequence[Mapping[str, object]],
    *,
    modified_files: Iterable[str] = (),
    observed_files: Iterable[str] = (),
    churn: int = 0,
    touch_test_file: bool = False,
) -> RewardVerdict:
    """Evaluate one settled action.

    @param before_error_type - structural failure kind observed before the action.
    @param after_error_type - structural failure kind after the action.
    @param results - settled calls of the step.
    @param modified_files - files the step wrote or edited.
    @param observed_files - files the previous step's calls made observable.
    @param churn - changed lines the step's file-mutating calls produced.
    @param touch_test_file - whether the step wrote or edited a test file.
    @returns the verdict, including which layer decided it.
    """
    base, rationale = _deterministic(before_error_type, after_error_type, results)
    if base is None:
        base, rationale = _state_comparison(before_error_type, after_error_type)
    penalty, penalty_notes = _static_penalties(results, modified_files, observed_files)
    reward = _clamp(base + penalty)
    if penalty_notes:
        rationale = f'{rationale}; {", ".join(penalty_notes)}'
    # Validation runs last and multiplies the settled reward, so the number a
    # reader sees in the row is the number the penalty and the discount left.
    final = validation.finalize(
        base_reward=base, reward=reward, churn=churn, touch_test_file=touch_test_file,
    )
    if final.discounted:
        rationale = (f'{rationale}; validation: {final.flag}'
                     f' (reward discounted x{final.assessment.discount:g})')
    return RewardVerdict(
        reward=final.reward,
        source=SOURCE_RULE,
        base=base,
        penalty=penalty,
        rationale=rationale,
        discount=final.assessment.discount,
        flag=final.flag,
    )


def _deterministic(
    before_error_type: str,
    after_error_type: str,
    results: Sequence[Mapping[str, object]],
) -> tuple[float | None, str]:
    """Highest-priority layer: what the tools actually reported."""
    if any(classify.looks_like_build_failure(result) for result in results):
        return REWARD_SEVERE, 'deterministic: build/typecheck failure'
    ran_tests, tests_passed = classify.test_outcome(results)
    if ran_tests:
        if tests_passed:
            return REWARD_SOLVED, 'deterministic: test run reported no failures'
        if is_benign(after_error_type):
            return REWARD_PARTIAL, 'deterministic: tests still failing, no new failure signal'
        return REWARD_WORSENED, 'deterministic: tests failing with a failure signal'
    if is_benign(after_error_type) and any(not result.get('isError') for result in results):
        if not is_benign(before_error_type):
            return REWARD_IMPROVED, 'deterministic: actions succeeded and the prior failure cleared'
        return REWARD_UNCHANGED, 'deterministic: actions succeeded without a failure to clear'
    return None, ''


def _state_comparison(before_error_type: str, after_error_type: str) -> tuple[float, str]:
    """Second layer: the structural error state before versus after."""
    if is_benign(before_error_type) and is_benign(after_error_type):
        return REWARD_UNCHANGED, 'state: no failure before or after'
    if is_benign(before_error_type) and not is_benign(after_error_type):
        return REWARD_WORSENED, f'state: new failure ({after_error_type})'
    if not is_benign(before_error_type) and is_benign(after_error_type):
        return REWARD_IMPROVED, f'state: prior failure ({before_error_type}) cleared'
    if before_error_type == after_error_type:
        return REWARD_INEFFECTIVE, f'state: failure ({after_error_type}) unchanged'
    return REWARD_WORSENED, f'state: failure changed {before_error_type} -> {after_error_type}'


def _static_penalties(
    results: Sequence[Mapping[str, object]],
    modified_files: Iterable[str],
    observed_files: Iterable[str],
) -> tuple[float, list[str]]:
    """Third layer: small penalties, capped so correctness always dominates."""
    notes: list[str] = []
    penalty = 0.0

    # Compare by file name: the same file reaches a step as an absolute path in
    # one place and a relative one in another.
    observed = {_name(path) for path in observed_files if path}
    modified = {path.replace('\\', '/') for path in modified_files if path}
    if observed:
        unrelated = sorted(path for path in modified if _name(path) not in observed)
        if unrelated:
            penalty += PENALTY_UNRELATED_FILE
            notes.append(f'penalty: modified unrelated file(s) {", ".join(unrelated[:3])}')

    if any(classify.file_kind(path) == 'test' for path in modified):
        penalty += PENALTY_TEST_FILE_TOUCHED
        notes.append('penalty: modified a test file')

    if len(results) > EXCESSIVE_CALL_THRESHOLD:
        penalty += PENALTY_EXCESSIVE_CALLS
        notes.append(f'penalty: {len(results)} calls in one step')

    capped = max(PENALTY_CAP, penalty) if penalty < 0 else 0.0
    return capped, notes


def _name(path: str) -> str:
    """Reduce a path to its file name for cross-step comparison.

    @param path - a relative or absolute path.
    @returns the final path component.
    """
    return path.replace('\\', '/').rsplit('/', 1)[-1]


def _clamp(value: float) -> float:
    """Keep a reward inside the declared band.

    @param value - raw reward.
    @returns the value clamped to `[-1.0, 1.0]`.
    """
    return max(-1.0, min(1.0, value))
