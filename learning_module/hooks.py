"""Hook handlers for the two agent-loop instrumentation points.

Phase 1 records experience in shadow mode: the pre-LLM handler stores the
structural state, the post-action handler derives the action, evaluates a
reward, and upserts the experience row. Nothing here feeds a model request —
`get_experience_context` stays `None` until a later phase is allowed to inject.

Every failure is contained: the agent loop calls into this module and must
survive any error it produces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from learning_module import (
    classify,
    context as context_rendering,
    modes,
    retrieve,
    reward,
    schema,
    state as state_encoder,
    store,
)

#: Hooks the agent loop calls. `pre-llm` fires before a model request is
#: streamed; `post-action` fires once a step's tool calls have all settled.
HOOK_NAMES = ('pre-llm', 'post-action')

#: Hook that captures state, and the hook that records its outcome.
PRE_LLM = 'pre-llm'
POST_ACTION = 'post-action'


@dataclass(frozen=True)
class HookOutcome:
    """What one hook call did, for the reply envelope and the logs."""

    #: Whether the module handled the payload without an error.
    ok: bool = True
    #: Mode the call ran under.
    mode: str = modes.OFF
    #: Hook that was served.
    hook: str = ''
    #: Whether anything was persisted.
    recorded: bool = False
    #: Store path that was used.
    database: str = ''
    #: Reward recorded by a post-action call.
    reward: float | None = None
    #: Evaluator layer that produced the reward.
    reward_source: str = ''
    #: Explanation of the reward, for logs and manual review.
    rationale: str = ''
    #: Action category chosen for the step.
    action_category: str = ''
    #: Phase 4 flag this observation raised, or `''`.
    flagged_for_review: str = ''
    #: Whether Phase 4 validation discounted the reward.
    discounted: bool = False
    #: Failure kind observed before the action.
    error_before: str = ''
    #: Failure kind observed after the action.
    error_after: str = ''
    #: Error text when the call failed.
    error: str = ''

    def to_envelope(self) -> dict[str, Any]:
        """Render the JSON reply the bridge reads.

        @returns the reply envelope, without empty optional fields.
        """
        envelope: dict[str, Any] = {'ok': self.ok, 'mode': self.mode, 'hook': self.hook}
        if self.recorded:
            envelope['recorded'] = True
            envelope['database'] = self.database
        if self.reward is not None:
            envelope['reward'] = self.reward
            envelope['reward_source'] = self.reward_source
            envelope['rationale'] = self.rationale
            envelope['action_category'] = self.action_category
            envelope['error_before'] = self.error_before
            envelope['error_after'] = self.error_after
            envelope['discounted'] = self.discounted
            if self.flagged_for_review:
                envelope['flagged_for_review'] = self.flagged_for_review
        if self.error:
            envelope['error'] = self.error
        return envelope


def handle_hook(
    payload: Mapping[str, Any],
    *,
    mode: str | None = None,
    database: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Handle one hook payload and return the reply envelope.

    Never raises: an invalid payload, a broken store, or an internal error each
    produce `ok: False` plus a stderr diagnostic instead.

    Args:
        payload: hook data sent by the bridge, including the hook name.
        mode: mode override; defaults to the switch's value.
        database: store path override; defaults to the switch's value.

    Returns:
        The reply envelope: `ok`, `mode`, `hook`, and any error text.
    """
    resolved_mode = mode if mode is not None else modes.current_mode()
    hook = payload.get('hook')
    if hook not in HOOK_NAMES:
        modes.report(f'error: unknown hook {hook!r}; expected one of {"/".join(HOOK_NAMES)}')
        return HookOutcome(ok=False, mode=resolved_mode, error=f'unknown hook {hook!r}').to_envelope()
    if resolved_mode == modes.OFF:
        # Disabled learning writes no diagnostic: the harness's stderr belongs
        # to the product, so an off module stays silent.
        return HookOutcome(mode=resolved_mode, hook=hook).to_envelope()
    try:
        outcome = serve(
            hook,
            payload,
            resolved_mode,
            database if database is not None else modes.database_path(),
        )
    except Exception as error:  # noqa: BLE001 -- the hook boundary contains every failure
        modes.report(f'error: {hook} failed: {error!r}')
        return HookOutcome(ok=False, mode=resolved_mode, hook=hook, error=str(error)).to_envelope()
    return outcome.to_envelope()


def serve(
    hook: str,
    payload: Mapping[str, Any],
    mode: str,
    database: str | os.PathLike[str],
) -> HookOutcome:
    """Run one hook against the experience store.

    @param hook - hook name.
    @param payload - hook payload.
    @param mode - resolved mode.
    @param database - store path.
    @returns the outcome, including what was persisted.
    """
    path = schema.ensure_schema(database)
    session_id = str(payload.get('sessionId', ''))
    turn = _int(payload.get('turn'))
    step = _int(payload.get('step'))
    if hook == PRE_LLM:
        pre = state_encoder.begin_state(payload)
        store.save_pending_state(path, session_id, turn, step, store.StateSnapshot(
            error_type=pre.error_type,
            tool_context=pre.tool_context,
            file_context=pre.file_context,
            summary=pre.summary,
        ))
        modes.report(
            f'hook called: {hook} mode={mode} session={session_id} turn={turn} step={step}'
            f' state({pre.summary})'
        )
        return HookOutcome(mode=mode, hook=hook, recorded=True, database=os.fspath(path))

    post = state_encoder.complete_state(payload)
    pending = store.take_pending_state(path, session_id, turn, step)
    snapshot = pending or store.StateSnapshot(
        error_type='none', tool_context='-', file_context='-', summary='',
    )
    verdict = reward.evaluate(
        snapshot.error_type,
        post.error_type,
        _results(payload),
        modified_files=post.modified_files,
        observed_files=_observed_files(snapshot.file_context),
        churn=post.churn,
        touch_test_file=post.touch_test_file,
    )
    recorded = store.record_outcome(
        path,
        snapshot,
        post.action_category,
        _debug_detail(session_id, post.action_detail),
        verdict.reward,
        verdict.source,
        # The verdict already validated and discounted this reward, and the
        # harness was told that number; the store keeps it, and receives the
        # bases so the flag it writes stays explainable.
        final_reward=verdict.reward,
        base_reward=verdict.base,
        churn=post.churn,
        touch_test_file=post.touch_test_file,
    )
    modes.report(
        f'hook called: {hook} mode={mode} session={session_id} turn={turn} step={step}'
        f' action={post.action_category} reward={verdict.reward:+.2f}'
        f' ({"update" if recorded.updated else "insert"} n={recorded.total_count}'
        f' avg={recorded.average_reward:+.3f}) :: {verdict.rationale}'
    )
    return HookOutcome(
        mode=mode,
        hook=hook,
        recorded=True,
        database=os.fspath(path),
        reward=verdict.reward,
        reward_source=verdict.source,
        rationale=verdict.rationale,
        action_category=post.action_category,
        error_before=snapshot.error_type,
        error_after=post.error_type,
        flagged_for_review=verdict.flag,
        discounted=verdict.discounted,
    )


def get_experience_context(
    payload: Mapping[str, Any],
    database: str | os.PathLike[str] | None = None,
    *,
    now: Any = None,
) -> str | None:
    """Retrieve experience text for one state, or `None` when none is trusted.

    Active mode is the only mode that returns text: `off` records nothing and
    `shadow` learns without influencing a decision, so both return `None`. The
    payload is the same one the pre-LLM hook receives, so the state is resolved
    by the same code path and the two can never disagree.

    Only structured fields reach the returned text; see `context.render`.

    @param payload - state-bearing hook payload (tool names, file targets, prior results).
    @param database - store path; defaults to the configured one.
    @param now - reference time for decay; defaults to the current UTC time.
    @returns the injectable experience text, or `None`.
    """
    if modes.current_mode() != modes.ACTIVE:
        return None
    try:
        pre = state_encoder.begin_state(payload)
        path = schema.ensure_schema(store.resolve_database(database))
        result = retrieve.candidates_for(path, pre.error_type, now=now)
        rendered = context_rendering.render(result.entries)
        if rendered.text == '':
            return None
        modes.report(
            f'context: state(error={pre.error_type} used={pre.tool_context})'
            f' searched={",".join(result.error_types_searched)}'
            f' entries={rendered.kept} dropped={rendered.dropped}'
            f' pre_cap_tokens={context_rendering.pre_cap_tokens(rendered.text)}'
            + (' (character pre-cap applied)' if rendered.truncated else '')
        )
        return rendered.text
    except Exception as error:  # noqa: BLE001 -- the caller is the agent loop
        modes.report(f'error: retrieval failed: {error!r}')
        return None


def _int(value: Any) -> int:
    """Coerce one payload number, defaulting to zero."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _results(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Read the settled results list from a post-action payload."""
    entries = payload.get(state_encoder.RESULTS_KEY)
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, Mapping)]


def _debug_detail(session_id: str, detail: str) -> str:
    """Prefix a call description with its session, for the manual spot check.

    `action_detail` is overwritten debug data that never participates in the
    identity key or the aggregates, so carrying the origin here costs nothing
    and lets a recorded reward be traced back to the session log it came from.

    @param session_id - the session that produced the observation.
    @param detail - the description of the call.
    @returns the description prefixed with a short session origin.
    """
    short = session_id.removeprefix('session-')[:8]
    return f'session={short} {detail}' if short else detail


def _observed_files(file_context: str) -> tuple[str, ...]:
    """Recover the file paths one pre-hook recorded, for the unrelated-file check.

    @param file_context - the rendered `kind:basename` context.
    @returns the file names that were observable in that step.
    """
    if not file_context or file_context == classify.NO_CONTEXT:
        return ()
    names: list[str] = []
    for entry in file_context.split(','):
        _, _, name = entry.partition(':')
        if name:
            names.append(name)
    return tuple(names)
