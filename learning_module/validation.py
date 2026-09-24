"""Reward validation: the anti-cheat layer of Phase 4.

Two signals are checked here, both deterministic and both derived from what the
step actually did:

* **Test tampering.** A step that edits the tests and is then rewarded for the
  tests passing has not demonstrated that the code works — it has demonstrated
  that the tests agree with it. A positive reward for that shape is discounted
  so it cannot accumulate the same credibility as a real fix, and the row is
  flagged for a human to look at.
* **Too-easy wins.** A large reward for a change of one or two lines is worth a
  look even when it is honest, so it is flagged without blocking.

Two deliberate asymmetries:

* A negative reward is never discounted. Making a failure look less bad would
  hide exactly the evidence the module exists to keep.
* A flag never blocks execution. `flagged_for_review` marks a row for the manual
  spot check (see `cli.py flagged`); it removes no experience from retrieval and
  changes no reward unless the tampering rule applies.

The discount applies to the final reward, after the static penalties, so a
discounted observation is stored as the number a reader sees: `+1.0` becomes
`+0.3`, and the recorded reward sequence replays to the same value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: Multiplier applied to a positive reward earned by editing the tests.
CHEAT_DISCOUNT = 0.3

#: Line churn below this counts as a small change; the specification's threshold.
SMALL_CHANGE_LINES = 3

#: A reward above this is "high" for the spot-check rule.
HIGH_REWARD = 0.8

#: Flags, stored in `flagged_reason` beside the `flagged_for_review` marker.
FLAG_NONE = ''
FLAG_TEST_TAMPERING = 'test_tampering'
FLAG_TEST_FILE_EDIT = 'test_file_edit'
FLAG_SMALL_CHANGE_HIGH_REWARD = 'small_change_high_reward'

#: Argument names carrying written text or an in-place edit, per tool.
_CONTENT_KEYS = ('content', 'file_text')
_OLD_KEYS = ('old_string', 'old_str')
_NEW_KEYS = ('new_string', 'new_str')


@dataclass(frozen=True)
class Assessment:
    """The validation verdict for one observation."""

    #: Multiplier the recorder must apply to the reward; `1.0` leaves it alone.
    discount: float = 1.0
    #: Flag to store in `flagged_reason`; {@link FLAG_NONE} for a clean one.
    flag: str = FLAG_NONE

    @property
    def discounted(self) -> bool:
        """Whether the reward is discounted.

        @returns `True` when {@link discount} is below `1.0`.
        """
        return self.discount < 1.0


@dataclass(frozen=True)
class FinalReward:
    """The reward a verdict settled on, plus the validation behind it."""

    #: Reward after the discount.
    reward: float
    #: The assessment that produced it.
    assessment: Assessment

    @property
    def discounted(self) -> bool:
        """Whether this reward was cut.

        @returns `True` when a discount was applied.
        """
        return self.assessment.discounted

    @property
    def flag(self) -> str:
        """The flag this observation carries.

        @returns one of the `FLAG_*` names, or {@link FLAG_NONE}.
        """
        return self.assessment.flag


def finalize(*, base_reward: float, reward: float, churn: int, touch_test_file: bool) -> FinalReward:
    """Apply the validation rules to a reward exactly once.

    Every caller that records a reward goes through here, and none of them
    re-assesses the result: applying the discount twice would turn a `0.3` into
    a `0.09` and hide the tampering it was meant to price.

    @param base_reward - the verdict's base tier, before penalties and discounts.
    @param reward - the settled reward, before the discount.
    @param churn - changed lines of the step's mutating calls.
    @param touch_test_file - whether the step wrote or edited a test file.
    @returns the final reward and the assessment behind it.
    """
    assessment = assess(
        base_reward=base_reward, reward=reward, churn=churn, touch_test_file=touch_test_file,
    )
    return FinalReward(reward=reward * assessment.discount, assessment=assessment)


def _text(arguments: Mapping[str, Any], names: tuple[str, ...]) -> str | None:
    """Read the first present, string-valued argument among `names`.

    @param arguments - one call's arguments.
    @param names - candidate argument names, in priority order.
    @returns the value, or `None` when the call carries none of them.
    """
    for name in names:
        value = arguments.get(name)
        if isinstance(value, str):
            return value
    return None


def _lines(text: str) -> int:
    """Count the lines one written text carries.

    An empty text writes no line at all, while a trailing newline does not open
    an extra line: `'a\\n'` is one line, not two.

    @param text - a text argument of a file-mutating call.
    @returns the line count.
    """
    if text == '':
        return 0
    return len(text.rstrip('\n').split('\n'))


def diff_lines(name: str, arguments: Mapping[str, Any]) -> int:
    """Measure the line churn one file-mutating call produced.

    This is the model's own edit, read back from the arguments it sent, not a
    filesystem diff: the module never re-reads the workspace, and the argument
    is what the harness already carries. An in-place edit costs the number of
    lines it replaced or added; a whole-file write costs the lines it wrote.

    @param name - the tool the model called.
    @param arguments - that call's arguments.
    @returns the changed-line count, `0` when the call carries no text at all.
    """
    if name == 'str_replace_editor':
        command = _text(arguments, ('command',))
        if command == 'create':
            return _lines(_text(arguments, _CONTENT_KEYS) or '')
        if command == 'str_replace':
            return _changed(_text(arguments, _OLD_KEYS), _text(arguments, _NEW_KEYS))
        if command == 'insert':
            return _lines(_text(arguments, _NEW_KEYS) or '')
        return 0
    old = _text(arguments, _OLD_KEYS)
    if old is not None:
        return _changed(old, _text(arguments, _NEW_KEYS))
    return _lines(_text(arguments, _CONTENT_KEYS) or '')


def _changed(old: str | None, new: str | None) -> int:
    """Measure the churn of a replacement, counting removals as work too.

    @param old - the replaced text, or `None` when the call replaces nothing.
    @param new - the replacement text, or `None` when nothing was added.
    @returns the absolute line difference.
    """
    return abs(_lines(old or '') - _lines(new or ''))


def is_test_pass(base_reward: float) -> bool:
    """Whether the evaluator's deterministic layer reported a passing test run.

    `reward_source` names the layer (`rule`), not the observation, so the
    condition the specification calls "reward_source is test_pass" is the
    deterministic layer's top verdict: the only way that layer reports `+1.0` is
    a test run that reported no failures.

    @param base_reward - the verdict's base tier, before penalties and discounts.
    @returns `True` for the solved-by-test tier.
    """
    return base_reward >= 1.0


def assess(*, base_reward: float, reward: float, churn: int, touch_test_file: bool) -> Assessment:
    """Validate one observation before it is recorded.

    @param base_reward - the verdict's base tier, before penalties and discounts.
    @param reward - the reward about to be recorded.
    @param churn - changed lines of the step's mutating calls.
    @param touch_test_file - whether the step wrote or edited a test file.
    @returns the discount to apply and the flag to store.
    """
    if touch_test_file and is_test_pass(base_reward) and reward > 0:
        # Flagged before the discount so the row records why the reward was cut,
        # not the small-change shape the cut reward would otherwise resemble.
        return Assessment(discount=CHEAT_DISCOUNT, flag=FLAG_TEST_TAMPERING)
    if touch_test_file:
        # An honest small edit to a test is not tampering, but a reviewer still
        # benefits from seeing that this step touched the tests.
        return Assessment(flag=FLAG_TEST_FILE_EDIT)
    if reward > HIGH_REWARD and churn < SMALL_CHANGE_LINES:
        return Assessment(flag=FLAG_SMALL_CHANGE_HIGH_REWARD)
    return Assessment()
