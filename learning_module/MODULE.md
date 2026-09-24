# Learning Module

Optional experience-based learning for the DeepSeek Harness agent loop.

The module is **off by default** and is never allowed to become a single point
of failure: a missing interpreter, a corrupt SQLite store, a retriever error, or
any internal exception is reported on stderr and dropped. The agent loop
continues a turn either way.

## Layout

| Path | Role |
| --- | --- |
| `cli.py` | Subprocess entry point (`hook`, `context`, `init-db`, `config`, `flagged`). JSON on stdin, JSON envelope on stdout, exit status always `0` for a handled command. |
| `modes.py` | The `DSH_LEARNING_MODULE` switch and the SQLite location. |
| `schema.py` | The `experiences` and `pending_states` tables and their indexes. |
| `classify.py` | Deterministic classification: action categories, error kinds, file kinds, shell command tokens. |
| `state.py` | The two-call state encoder (pre-LLM state, post-action fields). |
| `store.py` | The transient handoff, the identity-keyed outcome upsert, and the flagged-row query. |
| `reward.py` | The three-layer reward evaluator. |
| `validation.py` | The anti-cheat layer: test-tampering discount and the small-change spot-check flag. |
| `value.py` | Value update, decay, and confidence — the fixed constants live here. |
| `retrieve.py` | Structured retrieval and its relaxation map. |
| `context.py` | Renders retrieved experience as the labelled, budgeted reference block. |
| `backfill_values.py` | Surveys and backfills the value column for rows written before it existed. |
| `explain_retrieval.py` | Prints what a state would inject, without injecting. |
| `acceptance_phase2.py` | One check per Phase 2 acceptance criterion. |
| `acceptance_phase4.py` | One check per Phase 4 acceptance criterion. |
| `hooks.py` | Handlers for the two agent-loop instrumentation points, plus the retrieval gate. |
| `tests/` | Acceptance tests (`python -m unittest discover -s tests -t ..`), plus the audit and spot-check helpers used by the phase reports. |

The two call sites live in the harness, because the loop's instrumentation
points are the only consumers of this module's interface:
`packages/core/agent-loop/src/agent.ts` calls
`packages/core/agent-loop/src/learning-bridge.ts`, which runs this CLI and owns
the fail-silent rule.

## Switch

| Value | Effect |
| --- | --- |
| unset / `off` (default) | Nothing runs and nothing is written. The harness emits no output for a hook at all, because the agent loop's stderr is authoritative product output that assembled-application snapshots compare byte for byte. |
| `shadow` | Hooks extract state, evaluate reward, and record experience. Nothing reaches a model request. |
| `active` | Same recording as `shadow`, plus injection: `get_experience_context()` returns the labelled reference block, which the harness adds to the runtime-context snapshot. |

Any other value is reported and treated as `off`.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `DSH_LEARNING_MODULE` | `off` | Mode switch. |
| `DSH_LEARNING_DB` | `~/.dsh/learning/experiences.db` | SQLite experience store. |
| `DSH_LEARNING_MODULE_HOME` | repository `learning_module/` | Module directory, for out-of-tree installs. |
| `DSH_LEARNING_PYTHON` | `python` (Windows) / `python3` | Interpreter running `cli.py`. |

## Instrumentation points

| Hook | Fires | Payload |
| --- | --- | --- |
| `pre-llm` | After the request is composed and before its stream starts. | `sessionId`, `turn`, `step`, `toolNames`, `fileTargets`, `priorResults` |
| `post-action` | After every tool call of a step committed its result. | `sessionId`, `turn`, `step`, `results[]` (`callId`, `name`, `isError`, `error`, `text`, `arguments`, `mutates`) |

`priorResults` carries the previous settled step, which is what the state
describes: the failure it left, the tools it used, the files it touched.

## Experience identity and reward

One experience is `(state_error_type, state_tool_context, action_category)`.
`record_outcome` updates that row's counters and rolling mean reward, and stores
`action_detail` as the most recent debug reference without aggregating it.
`state_tool_context` records what the previous step *used* (`pwsh:pytest`,
`read`), never the request's declared tool list — the declared list is identical
for every step of a session and would collapse unrelated experiences into one.

`action_category` names concrete practices so retrieval can tell them apart:
`read_file`, `search_files`, `edit_file`, `write_file`, `run_test`, `run_build`,
`install_deps`, `run_shell`, `fetch_url`, `web_search`, `track_task`,
`delegate_agent`, `inspect_image`, `other`.

Reward comes from three layers, in priority order, and never from a model
judging itself:

1. **Deterministic** — a build/typecheck failure, a test run's own verdict, or
   actions that succeeded with the prior failure cleared.
2. **State comparison** — the structural failure before versus after: cleared
   is an improvement, unchanged is ineffective, new or changed is a worsening.
3. **Static penalties** — modifying files outside the step's observed set,
   modifying a test file, or an excessive number of calls in one step. The
   combined penalty is capped at `-0.1`, so it can never outweigh a correctness
   verdict.

Bands: `+1.0` solved, `+0.6` improved, `+0.2` partial, `0` unchanged, `-0.3`
ineffective, `-0.8` worsened, `-1.0` severe.

## Reward validation (anti-cheat)

`validation.py` reads two deterministic signals from the step and applies one
rule each. Both are needed because a reward that measures "the tests pass" can be
earned by editing the tests, and because a large reward for a tiny change is
worth a look even when it is honest.

| Rule | Condition | Effect |
| --- | --- | --- |
| Test tampering | the step wrote or edited a test file, the deterministic layer reported a passing test run, and the reward is positive | reward `x 0.3`, `flagged_for_review = 1`, `flagged_reason = 'test_tampering'` |
| Too-easy win | reward `> 0.8` and the step's changed lines are `< 3`, without a test file | `flagged_for_review = 1`, `flagged_reason = 'small_change_high_reward'`; the reward is **not** changed and nothing is blocked |

Deliberate asymmetries: a negative reward is never discounted (making a failure
look less bad would hide the evidence the module exists to keep); a flag never
blocks execution and never removes a row from retrieval; and the discount lands
on the final reward, after the static penalties, so the stored number is the one
a reader sees — `+1.0` with a `-0.05` test-file penalty becomes `0.285`, not
`0.3`.

Changed lines are the model's own edit, read from the call's arguments: `edit`
costs `|lines(old_string) - lines(new_string)|`, `write` costs the lines it
wrote, and `str_replace_editor` follows its `command`. The module never re-reads
the workspace, so the count needs no harness change and no filesystem access.

`flag` is decided once per observation. The first reason recorded on a row
survives later observations, so a row flagged for tampering is never rewritten
to look like a merely small change.

### Reviewing what was flagged

```sh
python cli.py flagged                                  # every flagged row
python cli.py flagged --flag test_tampering             # one rule
python cli.py flagged --limit 20                        # bounded
```

Each entry carries the identity triple, the recorded reward, `action_detail`,
and a plain-language `reason`. The query returns flagged rows only, and reports
an empty list for a store that has no `experiences` table yet.

## Retrieval and value

`retrieve.candidates_for()` answers one state with ranked experience or nothing:

1. exact `state_error_type` match;
2. relax to `retrieve.ADJACENT_ERROR_KINDS` only while the exact match yields
   fewer than three rows — and a neighbour never displaces an exact match, it
   only fills what the exact kind could not supply;
3. decay each candidate's stored mean: `effective_value = average_reward *
   exp(-ln(2) * age_days / 30)`, skipping anything under `0.05` in absolute value;
4. drop candidates whose confidence is under `0.2`;
5. rank by `effective_value * (1 + 0.2 * confidence)` in absolute value and take
   three;
6. withhold everything when the best of those three is under `0.3` confidence.

The injected text carries the observed mean reward, the observation counts, and
the action category — never the internal endowment-weighted score, and never
`state_summary`.

`value.py` owns the fixed constants: `alpha = 0.3`, `lambda = 2.5`, a 30-day
half-life, and the confidence schedule
`min(0.9, 0.15 * min(total, 6)) * max(2 * |success_rate - 0.5|, 0.2)`. The value
update is loss-averse: a reward below the stored value moves the value
`alpha * lambda` of the way, and a first observation always uses the base rate
because it has no expectation to fall short of.

## Stored value

Each row carries two central tendencies:

| Column | Meaning |
| --- | --- |
| `average_reward` | Plain arithmetic mean of every observed reward. Diagnostic and display only. |
| `value` | The loss-averse EMA. This is what retrieval decays and ranks on. |
| `rewards` | The observed reward sequence (JSON), so `value` stays reconstructible from the row. |
| `flagged_for_review` | `1` when a human should look at this row. |
| `flagged_reason` | Which rule flagged it (`test_tampering`, `test_file_edit`, `small_change_high_reward`), or `''`. |

The mean, the EMA, and the reward sequence all carry the *discounted* reward when
validation applied one, so a row cannot be read two ways. `flagged_for_review`
keeps the `0`/`1` meaning Phase 0 gave it; `flagged_reason` is the Phase 4 column
beside it that says which rule fired.

`record_outcome` advances `value` on every write, reading the pair's current
value as the expectation and treating a first observation as having none.

A store created before the `value` column existed is upgraded in place by
`ensure_schema` (nullable `ALTER TABLE ADD COLUMN`, existing rows and columns
untouched). `backfill_values.py` then reports and fills those rows:

```sh
python backfill_values.py <db>            # survey only
python backfill_values.py <db> --apply    # write
```

A row with a recorded reward sequence is recomputed through the real EMA. A row
without one **cannot** be recomputed — loss aversion depends on the order and
size of the individual rewards, and inventing a plausible sequence would
fabricate results that were never observed — so it is seeded with its arithmetic
mean and counted separately in the report. From the next observation on, every
row follows the real EMA.

`explain_retrieval.py` prints what a state would inject, and why, without
injecting anything.

## Phase status

Phase 0 (infrastructure): the tables, the mode switch, and the two instrumented
call sites.

Phase 1 (state extraction and recording, shadow mode): the two-call encoder, the
experience upsert, and the reward evaluator. A step that ends a session leaves
exactly one unclaimed `pending_states` row per session; that is bounded by
design, because such a step never reaches the post-action hook.

Phase 2 (retrieval and value computation): the retriever and the value engine,
the `value`/`rewards` columns, and the in-place store upgrade with its backfill
script. `get_experience_context()` became real but still returned `None` outside
`active` mode.

Phase 3 (injection): `context.py` renders the retrieved experience as a labelled
reference block, and the harness adds it to the runtime-context snapshot through
`ctx.systemPrompt.context({ name: 'agent-loop:learning' })`, so the injected text
is also a durable session record. The block is capped at 3 entries, 50 tokens per
entry, and 200 tokens per block, measured with `@deepseek-ai/dsh-token-meter` —
a **4 characters/token estimate, not the DeepSeek tokenizer's exact count**; no
exact local tokenizer exists in this repository. Phase 3's acceptance report
(`PHASE3_ACCEPTANCE.md`) records what ten real active-mode sessions showed: the
block was present in every one, and the model never referred to it.

Phase 4 (reward validation): the anti-cheat layer above, its `flagged_reason`
column, the flagged-row query, and `acceptance_phase4.py`. The test-file signal
also required widening `classify._TEST_PATH`: it previously missed the
`test_*.py` convention, so `test_stats.py` in a workspace root classified as
ordinary source and tampering could not be detected at all.
