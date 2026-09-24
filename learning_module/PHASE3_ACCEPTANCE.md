# Phase 3 acceptance report — injection into an independent, labelled prompt section

Scope: `get_experience_context()` stops being a shadow-only no-op and returns the
rendered reference text; the harness injects it through
`ctx.systemPrompt.context({ name: 'agent-loop:learning' })`, so it becomes a
durable user-role snapshot in the session log. No other phase's behaviour is
changed, and no Agent Loop core logic is modified: the module is reached through
the two instrumentation points Phase 0 fixed, plus the context provider seam.

> **Token counting caveat.** Every token number in this report is measured with
> the project's own pricing, `@deepseek-ai/dsh-token-meter` (`estimateContent`,
> `CHARS_PER_TOKEN = 4`). That is a **4 characters/token estimate, not the
> DeepSeek tokenizer's exact count.** No exact local tokenizer exists anywhere in
> this repository (no tiktoken / tokenizers / sentencepiece in the JS or Python
> dependency sets), so this heuristic is the closest available implementation of
> the ceiling. The same caveat is written in `learning_module/context.py`,
> `learning_module/cli.py` (`config` command), the harness JSDoc in
> `packages/core/agent-loop/src/learning-bridge.ts`, and
> `packages/core/agent-loop/tests/learning-acceptance.spec.ts`. Because it is an
> estimate, a real tokenizer could price the same block above or below 200; the
> budget is enforced on the estimate and the block is far enough under the
> ceiling (worst observed 143) that a modest underestimate would not breach it.

## 1. Injection format and the off/active prompt diff

Command that produced the comparison (real subprocess per mode, same seeded
store, same mock script):

```
npx vitest run packages/core/agent-loop/tests/learning-acceptance.spec.ts
```

`packages/core/agent-loop/tests/learning-acceptance.spec.ts` →
`phase 3: prompt diff between off and active` asserts, on the two final requests
of two otherwise identical runs:

| check | off | active |
| --- | --- | --- |
| request body contains `Historical Experience` | no | yes |
| request body contains `Reference Only` | no | yes |
| request body contains `Approaches to avoid` | no | yes |
| session log holds a snapshot carrying the block | 0 snapshots | > 0 snapshots |

The active-mode delta is exactly the block below, appended as its own section of
the runtime-context snapshot (the `Current runtime context. This snapshot
supersedes earlier runtime-context snapshots.` message). Nothing else in the
request changes; off mode is byte-identical to the pre-Phase-3 request because
the provider returns `''` and an empty context contribution renders no section.

Injected text, verbatim, as the model received it:

```
[Historical Experience — Reference Only]
Similar states appeared 8 times before.

✅ Successful approaches for reference:
Action: edit_file
Historical average reward: +0.84
Confidence: 0.75 (5 good / 0 bad of 5)
---

⚠️ Approaches to avoid (they performed poorly historically):
Action: run_test
Historical average reward: -0.63
Confidence: 0.45 (0 good / 3 bad of 3)
---

Note: the above is historical reference only. Judge the current situation independently.
```

"Reference, not instruction" is carried by three separate devices, none of them
implicit: the header `[Historical Experience — Reference Only]`, the warning
heading `Approaches to avoid`, and the closing sentence
`Note: the above is historical reference only. Judge the current situation
independently.` Success and failure are separated by heading and marker
(`✅` / `⚠️`), not by sign alone, and the polarity is decided by the stored
loss-averse value (`context._polarity`), the same quantity the ranking uses.

## 2. A strongly negative experience is labelled "avoid", never recommended

`phase 3: module contract` seeds six observations of `run_test` in the
`test_failure` / `pwsh:pytest` state with rewards `-0.8, -1.0, -0.8, -1.0, -0.8,
-1.0` through `store.record_outcome`, then runs the real CLI:

- text contains `⚠️ Approaches to avoid`;
- text contains `run_test` and `-0.90`;
- text does **not** contain `✅`, and does not match `/recommended/i`;
- when every entry is negative the heading degrades to
  `⚠️ Approaches to avoid (historically ineffective)` and the headline reads
  `... and they did not go well.`

## 3. No free-text field reaches the prompt

`context.py` renders only `action_category`, `average_reward`, `confidence`, and
the `success/fail/total` counts. The structured fields the module stores but must
never show — `state_summary`, `action_detail`, `state_file_context`, session
identity — are absent by construction, and the check is behavioural, not a code
reading:

- `phase 3: module contract` → `carries no free-text field` seeds a row whose
  `summary` is `seeded for acceptance` and whose `action_detail` is
  `session=seed0001 pwsh pytest`, and asserts none of `session=seed0001`,
  `seeded for acceptance`, `error=test_failure`, `pwsh pytest` appears in the
  injected text;
- across the 10 real active-mode sessions, no injected block contained a path, a
  session id, a command line, or a source snippet. Verified by reading all 35
  blocks (`.learning-tools/reference_review.ts`).

`learning_module/tests/test_phase2.py` additionally asserts
`state.assert_summary_is_structural()` and the retriever's structured-only
projection, so a `state_summary` can never be ranked on either.

## 4. Token budget, measured with the project's pricing

Measured with `@deepseek-ai/dsh-token-meter`'s `estimateContent`
(**4 characters/token estimate, not the DeepSeek tokenizer's exact count** — see
the caveat at the top).

Static ceilings, all enforced on the complete emitted value:

| limit | value | enforced in |
| --- | --- | --- |
| entries per block | 3 | `context.MAX_ENTRIES`, `learning-bridge.CONTEXT_MAX_ENTRIES` |
| estimate per entry | 50 tokens | `applyTokenBudget` in `learning-bridge.ts` |
| estimate per block | 200 tokens | `applyTokenBudget` in `learning-bridge.ts` |
| character pre-cap | 200/entry, 700/block | `context.ENTRY_CHAR_CAP`, `context.TOTAL_CHAR_CAP` |

Oversized input degrades instead of failing: `context._entry()` first drops the
per-entry breakdown line, and `learning-bridge.applyTokenBudget` then drops whole
trailing entries separated by `---`. A block that cannot fit even one entry
yields no injection at all (verified: `withholds the block when not even one
entry fits`). Truncation
never raises, so a pathological row cannot turn into an agent failure.

Measured on real traffic — all 35 blocks injected across the 10 active-mode
sessions of item 5, plus the seeded acceptance blocks:

```
task-01: 3 block(s), tokens 119/143/119, entries 2/3/2
task-02: 5 block(s), tokens 119/143/119/119/143, entries 2/3/2/2/3
task-03: 3 block(s), tokens 143/143/143, entries 3/3/3
task-04: 4 block(s), tokens 143/143/119/119, entries 3/3/2/2
task-05: 3 block(s), tokens 119/90/119, entries 2/1/2
task-06: 4 block(s), tokens 119/143/119/119, entries 2/3/2/2
task-07: 4 block(s), tokens 119/143/119/119, entries 2/3/2/2
task-08: 2 block(s), tokens 143/119, entries 3/2
task-09: 3 block(s), tokens 119/143/143, entries 2/3/3
task-10: 4 block(s), tokens 143/143/143/143, entries 3/3/3/3

blocks measured: 35
worst block: 143 tokens (554 chars of module text), ceiling 200
blocks at or over the 200-token ceiling: 0
```

The binding gate is the per-entry ceiling, not the block ceiling: 3 entries ×
50 = 150 < 200, so a 3-entry block can never reach 200 unless individual entries
exceed their own ceiling, which the budget step prevents. The block ceiling
exists as a safety net for a longer header/footer or a future entry count.

## 5. Five tasks, and whether the model referenced the experience reasonably

Ten tasks were run in `active` mode (not five) against a store seeded for the
states those tasks reach, then all ten sessions were reviewed by reading the
injected block and the reasoning and tool calls that followed it. Driver:
`.learning-tools/run_tasks.ts`; reviewer: `.learning-tools/reference_review.ts`
(prints the timeline; no keyword matching); pricing:
`.learning-tools/measure_injection_tokens.ts`.

Run facts: 10/10 tasks `status=0`, 24.6–46.8 s each; the module's stderr is empty
in every run (0 lines matching `learning` in `runs.log`), so the injection never
failed a task. Every session received the block at every step after its first
settled step — 35 blocks in total, 2–5 per session.

### What the model did with the block

| session | block said | model did | read as |
| --- | --- | --- | --- |
| 01 | avoid `run_test` (-0.63) | ran `pytest` at t1s3 and t1s5, justified by the task ("as requested") | advice declined, task followed |
| 02 | avoid `run_test` | ran `pytest` 2×; **no** mention of the block in any reasoning line | advice declined |
| 03 | avoid `run_test` and `edit_file` | ran `pytest` 2×, edited once | advice declined |
| 04 | `read_file` success +0.68; avoid `run_test`(-0.87)/`edit_file`(+0.17) | re-read `geometry.py` before re-editing, then ran `pytest` 3× successfully | read-before-retry matched a success entry; test advice declined |
| 05 | avoid `run_test` | ran `python check_config.py` 2×, re-read before re-editing | advice declined |
| 06 | avoid `run_test`/`edit_file` | ran `pytest` 2×, edited once | advice declined |
| 07 | same | ran `pytest` 2×, edited once | advice declined |
| 08 | avoid `run_test` | ran the test, it passed, **correctly changed nothing** | advice declined, and no gratuitous edit |
| 09 | avoid `run_test`/`run_shell` | ran the script 4× and probed the environment, re-read before `edit` | advice declined |
| 10 | avoid `run_test` | ran `pytest` 2×, edited once | advice declined |

**No case of blind obedience was found.** No session wasted a call on a step the
block discouraged in a way the task did not require, and no session cited the
block as an authority. The one vocabulary hit in all ten sessions
(`avoid the earlier cache-path warnings`, task-02 t1s7) refers to pytest's cache
warning, not to the block.

**But the honest reading is not "the model used the experience well" either: it
ignored the block.** A scan of every reasoning line in all ten sessions for
`historical|experience|reference|avoid|past session|advice` produced exactly one
hit, and that hit was unrelated. The model never acknowledged that historical
evidence existed, never explained why it departed from it, and never used it to
skip a redundant step. The observable influence of the block on these ten runs
was zero.

Two properties of the injected content explain the zero influence, and one of
them is a real defect:

1. **The block contradicts itself about the same action.** In task-04 the same
   block listed `run_test` under *"Successful approaches for reference"*
   (+1.00, 0.45, 3 good/0 bad) **and** under *"Approaches to avoid"* (-0.47,
   0.64, 1 good/6 bad of 7). The cause is structural, not a bug in the renderer:
   the experience identity is `(state_error_type, state_tool_context,
   action_category)`, but `retrieve._query()` selects on `state_error_type`
   alone, so `read | run_test` (value -0.79 over 10 observations) and the
   `none | ...` rows compete in the same ranked list and the same block. The
   model cannot use advice that says both "do this" and "avoid this".
2. **The reward band measures the step's outcome, not the action's soundness.**
   `run_test` scores `-0.57` in the `none | read` state because a test that
   reports a failure is a "worse than expected" step — yet running the test is
   exactly what those tasks require, and `run_test` in `test_failure | pwsh:pytest`
   scores `-0.87` for the same reason. Rendering "-0.63" next to
   `Action: run_test` therefore tells the model to avoid the task's required
   next move.
3. **Sanity check on the alternative explanation.** The model could have ignored
   the block because it is small or because the task text dominates; both are
   true, and the observation cannot separate them. What the evidence does rule
   out is obedience: nothing in the ten transcripts is explained by the block.

This is reported rather than fixed: items 1 and 2 are properties of the retrieval
key and the reward definition, both of which Phase 1/2 fixed and the brief forbids
me to change unilaterally. **Ruling taken: do not change them now; carry both
findings into Phase 4.** Nothing in Phase 3 is modified by this note, and no
frozen parameter moved. Candidate fixes if Phase 4 authorises them — restrict
retrieval to the full identity triple, exclude the current session's own rows from
its own retrieval, or split the reward band by whether the step moved the task
forward.

### What was not measured

No off-mode control group was run for these ten tasks, so this section reports
the absence of any trace of influence, not a measured behavioural delta between
off and active. The deterministic part of the claim — the block is present in the
prompt and never mentioned — is verified directly from the session logs.

## Fail-silent checks (Phase 0 invariant, re-verified in active mode)

| failure | behaviour |
| --- | --- |
| SQLite unavailable / corrupt | `get_experience_context` returns `None`; agent continues |
| module directory or Python missing | `learningContext` returns `''`; bridge logs once in active/shadow only |
| retriever raises | caught inside the hook; `''` injection, error logged |
| renderer raises | same |
| budget drops everything | `''` injection, no error |
| CLI exits non-zero / prints non-JSON | `readContextText` returns `''` |
| `DSH_LEARNING_MODULE` unset or unknown | mode resolves to `off`; provider returns `''` with no subprocess |

Evidence: `packages/core/agent-loop/tests/learning-context.spec.ts` (`contains a
failing module at the injection point`), `learning-hooks.spec.ts` (the module's
hook failures are logged and swallowed), `learning-bridge.spec.ts` (`never throws
when the CLI overruns its budget`), and the empty module stderr across all ten
real runs.

## Gates re-run for this phase

| gate | result |
| --- | --- |
| `python -m pytest learning_module/tests -q` | 128 passed, 46 subtests passed |
| `python learning_module/acceptance_phase2.py` | 26/26 checks passed |
| `npx vitest run packages/core/agent-loop/tests/learning-*.spec.ts` | 4 files passed, 49 tests passed (`learning-acceptance` 5, `learning-context` 12, `learning-bridge` 24, `learning-hooks` 8) |
| `npx tsc -b packages/core/agent-loop/tsconfig.json` | exit 0 |
| `pnpm run test:snapshot` | 34 failed / 75 passed / 7 skipped — identical to the pre-change baseline; the 34 are the host's `unknown tool "bash"` failures, and 0 stderr lines come from the Learning Module |
| `pnpm run test` (unit) | failures are a subset of the pre-change baseline set |

## Files this phase touched

New: `learning_module/context.py`, and the Phase 3 module-level cases added to
`learning_module/tests/test_phase2.py` (`InjectionGateTests`, plus the renderer
assertions in `RetrieverTests`), `packages/core/agent-loop/tests/learning-context.spec.ts`,
`packages/core/agent-loop/tests/learning-acceptance.spec.ts`,
`.learning-tools/reference_review.ts`, `.learning-tools/measure_injection_tokens.ts`,
`.learning-tools/store_survey.py`, this report.

Changed: `learning_module/hooks.py` (`get_experience_context` renders; CLI
`context` command), `learning_module/cli.py` (budget caveat in `config`),
`learning_module/retrieve.py` (`context_for` removed, replaced by the renderer),
`learning_module/README.md`, `packages/core/agent-loop/src/learning-bridge.ts`
(`learningContext`, `applyTokenBudget`, context command), `packages/core/agent-loop/src/index.ts`
(the context provider seam and `pendingStep`), `packages/core/agent-loop/package.json`
and `tsconfig.json` (TokenMeter dependency), `docs/subsystems/core.md`,
`docs/subsystems/core.zh.md` (regenerated catalog).
