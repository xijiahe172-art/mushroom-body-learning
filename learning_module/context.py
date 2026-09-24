"""Rendering of retrieved experience as model-facing reference text.

Only structured fields reach this text: the action category, the observed mean
reward, the observation counts, and the confidence. Free-text fields
(`state_summary`, `action_detail`, session identity, paths, source snippets) are
deliberately absent — they can carry code or text that reads like an
instruction, and the prompt is the one place that must never receive them.

The text labels each entry's polarity explicitly. A strongly negative
experience rendered as a bare number invites the opposite reading ("this scored
-0.8" can be mistaken for a recommended action), so the negative group carries
its own heading and warning marker.

Token budget: {@link PRE_CAP_CHARS} is a conservative character pre-cap derived
from the TokenMeter's fixed density (4 characters per token), applied here so a
pathological row cannot produce a large payload across the process boundary.
The authoritative budget is applied on the harness side, measured with
`@deepseek-ai/dsh-token-meter` — the project's own pricing, which is a 4
characters/token heuristic and **not** the DeepSeek tokenizer's exact count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from learning_module import value

#: Heading of the whole block; states that this is reference material.
HEADER = '[Historical Experience — Reference Only]'

#: Delimiter closing every entry. The harness budget step drops whole trailing
#: entries, so it needs a boundary it can split on without guessing the format.
ENTRY_SEPARATOR = '---'

#: The sentence that tells the model this is evidence, not an instruction.
FOOTER = 'Note: the above is historical reference only. Judge the current situation independently.'

#: Heading for entries whose observed reward was positive.
SUCCESS_TITLE = '✅ Successful approaches for reference:'

#: Heading for entries whose observed reward was not positive.
AVOID_TITLE = '⚠️ Approaches to avoid (they performed poorly historically):'

#: Heading used when every entry shares one polarity and is negative.
ONLY_AVOID_TITLE = '⚠️ Approaches to avoid (historically ineffective):'

#: Maximum entries rendered, per the phase specification.
MAX_ENTRIES = 3

#: Character pre-cap per entry: 50 tokens at the TokenMeter's 4 chars/token.
ENTRY_CHAR_CAP = 200

#: Character pre-cap for the whole block: 200 tokens at 4 chars/token, minus
#: room for the header and footer that must survive any truncation.
TOTAL_CHAR_CAP = 700

#: Fixed-density characters per token used by the harness's TokenMeter.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class RenderedContext:
    """The rendered reference text plus what rendering decided."""

    #: The text, or `''` when nothing is injectable or the budget allowed none.
    text: str
    #: Entries the renderer kept.
    kept: int
    #: Entries dropped to fit the budget.
    dropped: int
    #: True when the character pre-cap, not the retrieval, decided the content.
    truncated: bool


def render(candidates: Sequence[value.Candidate]) -> RenderedContext:
    """Render retrieved candidates as labelled reference text.

    @param candidates - ranked candidates, best first.
    @returns the text and the accounting of what fitted.
    """
    if not candidates:
        return RenderedContext('', 0, 0, False)

    successes = [candidate for candidate in candidates if _polarity(candidate) > 0]
    avoids = [candidate for candidate in candidates if _polarity(candidate) <= 0]
    title_lines: list[str] = [HEADER]
    if successes and avoids:
        title_lines.append(f'Similar states appeared {_total(candidates)} times before.')
    elif successes:
        title_lines.append(f'Similar states appeared {_total(candidates)} times before, and every recorded outcome was good.')
    else:
        title_lines.append(f'Similar states appeared {_total(candidates)} times before, and they did not go well.')

    lines = list(title_lines)
    kept = 0
    dropped = 0
    truncated = False

    for group_title, group in ((SUCCESS_TITLE, successes), (ONLY_AVOID_TITLE if not successes else AVOID_TITLE, avoids)):
        if not group:
            continue
        group_lines: list[str] = []
        for candidate in group:
            if kept >= MAX_ENTRIES:
                dropped += 1
                continue
            entry = _entry(candidate)
            if len('\n'.join([*lines, *group_lines, entry, FOOTER])) > TOTAL_CHAR_CAP:
                # An entry that does not fit under the pre-cap is skipped rather
                # than ending the group: a later, smaller entry may still fit,
                # and the harness drops whatever remains over budget.
                dropped += 1
                truncated = True
                continue
            group_lines.append(entry)
            kept += 1
        if group_lines:
            # The heading is added only when at least one entry survived.
            lines.append('')
            lines.append(group_title)
            lines.extend(group_lines)

    if kept == 0:
        return RenderedContext('', 0, dropped, truncated)
    lines.append('')
    lines.append(FOOTER)
    return RenderedContext('\n'.join(lines), kept, dropped, truncated)


def _polarity(candidate: value.Candidate) -> int:
    """Classify one candidate as a success or an avoidance example.

    The stored value decides, not the arithmetic mean: the value is the
    loss-averse expectation the retriever ranks on, and the phase requires the
    label to match what the model is shown.

    @param candidate - one ranked candidate.
    @returns `1` for a positive value, `-1` otherwise.
    """
    return 1 if candidate.stored_value > 0 else -1


def _total(candidates: Sequence[value.Candidate]) -> int:
    """Total observations behind the rendered entries.

    @param candidates - the rendered candidates.
    @returns the summed observation count.
    """
    return sum(candidate.total_count for candidate in candidates)


def _entry(candidate: value.Candidate) -> str:
    """Render one entry from structured fields only.

    @param candidate - one ranked candidate.
    @returns the entry text, closed by {@link ENTRY_SEPARATOR}.
    """
    entry = '\n'.join((
        f'Action: {candidate.action_category}',
        f'Historical average reward: {candidate.average_reward:+.2f}',
        f'Confidence: {candidate.confidence:.2f} ({candidate.success_count} good /'
        f' {candidate.fail_count} bad of {candidate.total_count})',
        ENTRY_SEPARATOR,
    ))
    if len(entry) > ENTRY_CHAR_CAP:
        # Degrade to the fields that carry the signal rather than truncating
        # mid-number, which would misstate the evidence.
        entry = '\n'.join((
            f'Action: {candidate.action_category}',
            f'Historical average reward: {candidate.average_reward:+.2f}',
            f'Confidence: {candidate.confidence:.2f}',
            ENTRY_SEPARATOR,
        ))
    return entry


def pre_cap_tokens(text: str) -> int:
    """Price text under the same fixed density the harness uses.

    This is the TokenMeter's 4 characters/token heuristic, kept here only so a
    payload is bounded before it crosses the process boundary; the budget that
    enforces the phase's limit is measured on the harness side.

    @param text - rendered text.
    @returns the heuristic token count.
    """
    return -(-len(text) // CHARS_PER_TOKEN)
