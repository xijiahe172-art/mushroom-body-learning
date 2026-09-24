"""The two-call state encoder.

The agent loop has exactly two instrumentation points, and the encoder uses
both rather than merging them:

* `begin_state`, at the pre-LLM point, extracts only what exists before the
  model answers — `error_type`, `tool_context`, `file_context`. There is no
  action yet, so it neither reads nor invents `action_category`.
* `complete_state`, at the post-action point, runs once the model's calls have
  settled, and only then derives `action_category` and `action_detail`.

`state_summary` is display-only: it is built from the same structured fields, is
never read back for retrieval, and must never reach a model request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from learning_module import classify, validation

#: Keys a hook payload may carry.
RESULTS_KEY = 'results'
PRIOR_RESULTS_KEY = 'priorResults'
TOOL_NAMES_KEY = 'toolNames'
FILE_TARGETS_KEY = classify.TOOL_TARGETS_KEY
MODIFIED_FILES_KEY = 'modifiedFiles'


@dataclass(frozen=True)
class PreState:
    """What the pre-LLM point can observe."""

    #: Failure kind left by the previous settled step.
    error_type: str
    #: Tools the upcoming request offers.
    tool_context: str
    #: Files the upcoming step can touch.
    file_context: str
    #: Sorted file targets, kept for the post-action comparison.
    files: tuple[str, ...]
    #: Display-only one-liner.
    summary: str


@dataclass(frozen=True)
class PostState:
    """What the post-action point adds once the calls settled."""

    #: Concrete action category; the retrieval key's third component.
    action_category: str
    #: Most recent call description, for debugging only.
    action_detail: str
    #: Files the step wrote or edited.
    modified_files: tuple[str, ...]
    #: Failure kind left by this step's actions.
    error_type: str
    #: Whether this step wrote or edited a test file.
    touch_test_file: bool = False
    #: Changed lines this step's file-mutating calls produced.
    churn: int = 0


def _mapping(value: Any) -> Mapping[str, Any]:
    """Read one payload field as a mapping, tolerating absent or odd values."""
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    """Read one payload field as a sequence, tolerating absent or odd values."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def begin_state(payload: Mapping[str, Any]) -> PreState:
    """Extract the pre-LLM state of one step.

    The state is what the decision is made *against*: the failure the previous
    step left, the tools that step actually used, and the files those calls
    touched. Declared tool availability is not state — it is identical for
    every step of a session, so keying on it would collapse unrelated
    experiences into one row.

    @param payload - the pre-LLM hook payload.
    @returns the extracted state, with no action fields.
    """
    prior = [_mapping(entry) for entry in _sequence(payload.get(PRIOR_RESULTS_KEY))]
    targets = tuple(sorted({
        target
        for result in prior
        for target in classify.file_targets({'arguments': _mapping(result.get('arguments'))})
    }))
    error_type = classify.step_error_kind(prior)
    usage = tuple(classify.tool_usage(result) for result in prior)
    tool_context = classify.tool_context(usage) if prior else classify.NO_CONTEXT
    file_context = classify.file_context(targets)
    declared = [name for name in _sequence(payload.get(TOOL_NAMES_KEY)) if isinstance(name, str)]
    return PreState(
        error_type=error_type,
        tool_context=tool_context,
        file_context=file_context,
        files=targets,
        summary=_summarize(error_type, tool_context, file_context, declared),
    )


def complete_state(payload: Mapping[str, Any]) -> PostState:
    """Derive the action fields of a step that has finished executing.

    @param payload - the post-action hook payload.
    @returns the action fields plus the failure state the actions left.
    """
    results = [_mapping(entry) for entry in _sequence(payload.get(RESULTS_KEY))]
    modified: list[str] = []
    category = 'other'
    detail = ''
    churn = 0
    for index, result in enumerate(results):
        name = str(result.get('name', ''))
        arguments = _mapping(result.get('arguments'))
        if index == 0:
            category = classify.action_category(name, arguments)
            detail = classify.action_detail(name, arguments)
        if result.get('mutates'):
            for path in classify.file_targets({'arguments': arguments}):
                modified.append(path)
            churn += validation.diff_lines(name, arguments)
    return PostState(
        action_category=category,
        action_detail=detail,
        modified_files=tuple(sorted(set(modified))),
        error_type=classify.step_error_kind(results),
        touch_test_file=classify.touches_test_file(modified),
        churn=churn,
    )


def _summarize(error_type: str, tool_context: str, file_context: str, declared: Sequence[str]) -> str:
    """Build the display-only summary from structured fields.

    The declared tool count is appended for debugging only. It is deliberately
    outside `state_tool_context`, which carries what the previous step used.

    @param error_type - extracted failure kind.
    @param tool_context - rendered tools the previous step used.
    @param file_context - rendered file context.
    @param declared - tools the request offers.
    @returns a bounded one-line summary carrying no model text.
    """
    return f'error={error_type} used={tool_context} files={file_context} offered={len(declared)}'[:200]


def assert_summary_is_structural(summary: str) -> None:
    """Guard the rule that the summary never carries model text.

    @param summary - a candidate summary.
    @raises ValueError - when the summary is longer than its display bound.
    """
    if len(summary) > 200:
        raise ValueError('state_summary exceeds its display bound')
