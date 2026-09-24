"""Value computation: EMA update, time decay, and confidence.

Every parameter here is fixed by the phase specification and must not be tuned
without an explicit change request. The functions are pure so each formula can
be verified against its stated numeric case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

#: Base learning rate for the value update.
ALPHA = 0.3

#: Loss-aversion coefficient: a loss moves the value 2.5x as hard as an equal gain.
LAMBDA = 2.5

#: Half-life of an experience's value, in days.
HALF_LIFE_DAYS = 30.0

#: Decay base. `ln(2)` is what makes `HALF_LIFE_DAYS` a true half-life; a plain
#: `exp(-age/30)` would only carry that name by coincidence.
DECAY_LN2 = math.log(2)

#: An experience whose decayed value is weaker than this is skipped silently.
MIN_EFFECTIVE_VALUE = 0.05

#: Confidence schedule: `base = min(0.9, 0.15 * min(total_count, 6))`.
CONFIDENCE_CAP = 0.9
CONFIDENCE_PER_OBSERVATION = 0.15
CONFIDENCE_COUNT_CAP = 6

#: A wavering experience keeps at least this share of its base confidence.
CONFIDENCE_STABILITY_FLOOR = 0.2

#: Candidates below this confidence are not injected.
MIN_CONFIDENCE = 0.2

#: Confidence below this in the best-ranked candidate suppresses the injection.
INJECTION_CONFIDENCE_FLOOR = 0.3

#: How many candidates the retriever returns.
TOP_K = 3

#: Endowment weighting applied to the internal ranking score only.
ENDOWMENT_WEIGHT = 0.2


def initial_value() -> float:
    """Return the value a `(state, action)` pair starts from.

    @returns the documented starting value, `0`.
    """
    return 0.0


def update_value(old_value: float, reward: float, *, has_history: bool) -> float:
    """Apply one asymmetric, loss-averse EMA update.

    A first observation has no expectation to fall short of, so it always uses
    the base rate; from the second observation on, a reward below the stored
    value is treated as a loss and moves the value `LAMBDA` times as hard.

    @param old_value - the stored value before this observation.
    @param reward - the observed reward.
    @param has_history - whether a stored value already existed for this pair.
    @returns the updated value.
    """
    loss = has_history and reward < old_value
    effective_alpha = ALPHA * LAMBDA if loss else ALPHA
    return old_value + effective_alpha * (reward - old_value)


def decay_factor(age_days: float) -> float:
    """Return the half-life decay factor for an age in days.

    @param age_days - age of the last observation.
    @returns `exp(-ln(2) * age_days / HALF_LIFE_DAYS)`.
    """
    return math.exp(-DECAY_LN2 * age_days / HALF_LIFE_DAYS)


def effective_value(value_now: float, age_days: float) -> float:
    """Combine the stored value with its time decay.

    @param value_now - the stored loss-averse value.
    @param age_days - age of the last observation.
    @returns `value_now * decay_factor(age_days)`.
    """
    return value_now * decay_factor(age_days)


def confidence(success_count: int, fail_count: int) -> float:
    """Compute the confidence of one experience.

    Confidence grows with the observation count and rewards consistency: an
    experience that always succeeded or always failed is more trustworthy than
    one that alternates, and a wavering experience keeps a fifth of its base.

    @param success_count - observations with a positive reward.
    @param fail_count - observations with a non-positive reward.
    @returns the confidence in `[0, CONFIDENCE_CAP]`.
    """
    total = success_count + fail_count
    if total <= 0:
        return 0.0
    success_rate = success_count / total
    stability = 2 * abs(success_rate - 0.5)
    base = min(CONFIDENCE_CAP, CONFIDENCE_PER_OBSERVATION * min(total, CONFIDENCE_COUNT_CAP))
    return base * max(stability, CONFIDENCE_STABILITY_FLOOR)


def replay_value(rewards: Sequence[float]) -> float:
    """Apply the value update to a sequence of observations.

    The first observation of a pair has no stored expectation, so the loss
    branch cannot trigger for it; every later observation is judged against the
    value the earlier ones produced.

    @param rewards - observed rewards in order.
    @returns the final value.
    """
    current = initial_value()
    for index, reward in enumerate(rewards):
        current = update_value(current, reward, has_history=index > 0)
    return current


def age_in_days(last_used_at: str | None, now: datetime | None = None) -> float:
    """Measure the age of an experience's last observation.

    @param last_used_at - UTC ISO-8601 timestamp, or `None` for a never-used row.
    @param now - reference time; defaults to the current UTC time.
    @returns the age in days, never negative.
    """
    if last_used_at is None or last_used_at == '':
        return 0.0
    reference = now if now is not None else datetime.now(timezone.utc)
    parsed = parse_timestamp(last_used_at)
    return max(0.0, (reference - parsed).total_seconds() / 86_400)


def parse_timestamp(value: str) -> datetime:
    """Parse a stored timestamp into an aware UTC datetime.

    @param value - an ISO-8601 timestamp, with or without a `Z` suffix.
    @returns the parsed datetime in UTC.
    """
    text = value.strip().replace('Z', '+00:00')
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Candidate:
    """One experience considered by the retriever."""

    #: Stored row identity.
    row_id: int
    #: State key components.
    error_type: str
    tool_context: str
    action_category: str
    #: Most recent call description, for the explained output only.
    action_detail: str
    #: Stored aggregates.
    success_count: int
    fail_count: int
    total_count: int
    #: Arithmetic mean of every reward; diagnostic only, never ranked on.
    average_reward: float
    #: Loss-averse EMA value; the quantity retrieval decays and ranks on.
    stored_value: float
    #: Derived quantities.
    age_days: float
    decay: float
    effective: float
    confidence: float
    adjusted_score: float

    @property
    def stale(self) -> bool:
        """Whether the decayed value is too weak to consider."""
        return abs(self.effective) < MIN_EFFECTIVE_VALUE

    @property
    def below_confidence(self) -> bool:
        """Whether the candidate falls under the confidence floor."""
        return self.confidence < MIN_CONFIDENCE


def build_candidate(row: object, now: datetime | None = None) -> Candidate:
    """Derive every computed quantity for one stored experience row.

    @param row - a row exposing the `experiences` columns.
    @param now - reference time for the decay; defaults to the current UTC time.
    @returns the candidate with its derived values.
    """
    read = row.__getitem__  # type: ignore[attr-defined]
    success_count = int(read('success_count'))
    fail_count = int(read('fail_count'))
    average_reward = float(read('average_reward'))
    # A row written before the value column existed carries no EMA value; its
    # arithmetic mean is the only recorded central tendency, and the backfill
    # script records that substitution explicitly.
    raw_value = read('value')
    stored_value = average_reward if raw_value is None else float(raw_value)
    age = age_in_days(read('last_used_at') or read('created_at'), now)
    decay = decay_factor(age)
    effective = stored_value * decay
    confidence_value = confidence(success_count, fail_count)
    return Candidate(
        row_id=int(read('id')),
        error_type=str(read('state_error_type')),
        tool_context=str(read('state_tool_context')),
        action_category=str(read('action_category')),
        action_detail=str(read('action_detail')),
        success_count=success_count,
        fail_count=fail_count,
        total_count=int(read('total_count')),
        average_reward=average_reward,
        stored_value=stored_value,
        age_days=age,
        decay=decay,
        effective=effective,
        confidence=confidence_value,
        adjusted_score=effective * (1 + ENDOWMENT_WEIGHT * confidence_value),
    )
