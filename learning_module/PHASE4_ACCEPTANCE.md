# Phase 4 acceptance report — reward validation (anti-cheat)

Scope: a validation layer between reward evaluation and persistence. Two rules,
both deterministic, both derived from what the step actually did:

1. **Test tampering** — the step wrote or edited a test file and was rewarded for
   the tests passing: a positive reward is multiplied by `0.3` and the row is
   flagged for a human.
2. **Too-easy win** — a reward above `0.8` for fewer than 3 changed lines that did
   not touch a test file: flagged for a spot check, reward untouched, nothing
   blocked.

No Phase 1/2/3 behaviour changed: the reward bands, the value engine, the
retriever, and the injection renderer are untouched, and the Agent Loop core is
unchanged (only the module reads two more fields it derives itself).

## Decisions you ruled on before implementation

| Decision | Ruling | Where it lives |
| --- | --- | --- |
| `diff` line count | **Model-side churn** from the tool arguments; no harness change | `validation.diff_lines`, fed by `state.complete_state` |
| `action_category` | **Keep** `edit_file` / `write_file`; add an internal signal instead of a new category | `state.PostState.touch_test_file`; no new value in `classify.ACTION_CATEGORIES` |
| Discount surface | **`final = (base + penalty) x 0.3`** — the penalty is inside the discount | `validation.finalize`, called from `reward.evaluate` |

The third point needed a test that can tell the two designs apart, because the
original cases could not: with `base = +1.0`, a penalty of `-0.05`, and a `0.3`
discount, "multiply the settled reward" gives `0.285` while "discount only the
positive part" would give `0.25`. `RewardLayerTests::
test_the_penalty_is_discounted_together_with_the_positive_part` pins `0.27` from
`(1.0 - 0.1) x 0.3` at the penalty cap, which no other reading produces. The same
distinction is asserted end to end in `acceptance_phase4.py` check 1 (`0.285`).

## Two specification names that do not exist in the code

`action_category` has never contained `modify_test_file`, and `reward_source` is
constrained by the Phase 0 schema to `('none', 'rule', 'human')` — there is no
`test_pass`. Both were resolved with you rather than invented:

- the test-file signal is an internal boolean, so the stored category stays the
  tool's category (`edit_file`), asserted in
  `RecordingTests::test_the_recorded_category_stays_the_tool_category` and in the
  acceptance script;
- "`reward_source` is `test_pass`" is implemented as *the deterministic layer's
  top verdict*, which is the only way that layer reports `+1.0`
  (`validation.is_test_pass(base_reward)`, i.e. `base_reward >= 1.0`). The
  stored `reward_source` therefore stays `rule`; the tampering decision is
  visible in `flagged_reason` instead.

## Schema: `flagged_for_review` keeps its Phase 0 meaning

Phase 0 declared `flagged_for_review INTEGER NOT NULL DEFAULT 0 CHECK
(flagged_for_review IN (0, 1))`, which cannot hold a reason string. Rather than
change a Phase 0 contract, Phase 4 adds **`flagged_reason TEXT`** through the
existing `_ADDED_COLUMNS` upgrade path:

- `flagged_for_review` stays the `0`/`1` marker ("a human must look at this row");
- `flagged_reason` names the rule: `test_tampering`, `test_file_edit`, or
  `small_change_high_reward`.

A third flag, `test_file_edit`, exists so an honest edit to a test that did *not*
pass is still visible to a reviewer without being priced as cheating.

Verified on the real Phase 3 store (`a scratch store outside the repository`): before the
upgrade 21 rows and no `flagged_reason`; after `ensure_schema`, 21 rows and the
column added, 0 flagged rows, query returns `[]`.

## A necessary scope widening: test-file detection

`classify._TEST_PATH` matched a test directory (`tests/`, `__tests__/`, `spec/`)
or a `.test.`/`.spec.` file ending — but **not** the `test_*.py` convention. In
this repository's own acceptance workspaces the test files are `test_stats.py`,
`test_palindrome.py`, `test_geometry.py` in the workspace root, so every one of
them classified as ordinary `source`: the anti-cheat rule could not have fired on
any real task. The pattern now also matches `test_<name>` and `<name>_test.<ext>`
with a path boundary, which the Phase 4 tests pin
(`TestFileClassificationTests`, `OutcomeDerivationTests`). This is a deliberate
widening inside Phase 4's own subject — tampering cannot be detected without it —
not an incidental extra. It also changes what `state_file_context` records for
test files, from `source:test_stats.py` to `test:test_stats.py`.

## Acceptance criteria

`python learning_module/acceptance_phase4.py` — **36/36 checks passed**, exit 0.
Every scenario runs through the real `pre-llm` and `post-action` hooks against a
real SQLite store, and is read back through the real query interface.

### Criterion 1 — "modify the tests until they pass" is flagged and discounted

```
[PASS] the base tier was the full solved reward
       rationale=deterministic: test run reported no failures; penalty: modified a
       test file; validation: test_tampering (reward discounted x0.3)
[PASS] the reward was discounted instead of staying at +1.00
       reward=0.285 = (1.0 - 0.05) x 0.3
[PASS] the stored reward is the discounted one            reward=0.285 average=0.285
[PASS] the mean, the EMA and the sequence all carry the same number
       average=0.285 value=0.0855 rewards=[0.285]
[PASS] the row is marked for review, with the reason
       flagged_for_review=1 reason=test_tampering
[PASS] the action category is still the tool category, not a new one
       action_category=edit_file
```

The stored mean, the EMA, and the reward sequence all carry the discounted
number, so a later replay cannot reconstruct a different value than the row
shows. `base = +1.0` maps to `+0.285` rather than the round `+0.3` because the
test-file penalty (`-0.05`) is inside the discount, exactly as ruled.

A negative tampering outcome is **not** discounted, which the acceptance path and
`AssessmentTests::test_a_negative_reward_for_editing_the_tests_is_left_alone`
both cover: a step that edits the tests and still fails keeps its `-0.8` and is
flagged only as `test_file_edit`.

### Criterion 2 — a normal fix is not falsely flagged

```
[PASS] the reward is untouched                                     reward=1.0
[PASS] the envelope reports no discount and no flag                discounted=False
[PASS] the clean scenario wrote its own row                        rows=2
[PASS] the row carries no flag and no reason                       flagged_for_review=0 reason=''
[PASS] a four-line source fix stays outside the small-change rule
       churn 4 >= 3, so `reward > 0.8 and churn < 3` does not apply
```

A second clean scenario shows the other rule behaving as specified: a one-line
source fix scoring `+1.0` is flagged `small_change_high_reward` **without** being
discounted and without being blocked — flagged for a look, per the brief.

The boundary is documented rather than hidden: with `churn < 3`, a genuine
two-line fix is flagged for a spot check too. That is the specification's
threshold, so the phase keeps it and the acceptance script records the case
(`RewardLayerTests::test_a_two_line_fix_is_flagged_but_keeps_its_full_reward`).

### The query interface (`cli.py flagged`)

```
[PASS] the flagged query exits cleanly                     rows=1
[PASS] the flagged query reports no error                  error=none
[PASS] it explains why the row was flagged                 reason='the tests were edited and then passed'
[PASS] the listed row exposes the discounted observation   rewards=[0.285, 1.0]
[PASS] the listed row names the file the step touched      action_detail=session=verify01 edit C:\ws\stats.py
[PASS] the listed row carries its identity, not free text  identity=(test_failure, pwsh:pytest, edit_file)
[PASS] the query can filter by flag                        rows=1 for test_tampering
[PASS] the query honours a limit                           rows=1
[PASS] the spot-check query finds it, with the small-change reason
       rows=1 reason=a high reward for fewer than 3 changed lines
```

`store.flagged_experiences()` returns only flagged rows, each carrying the
identity triple, the recorded reward, `average_reward`, `value`, the reward
sequence, `total_count`, `reward_source`, `action_detail`, and a plain-language
`reason`. The sequence matters: `reward` is the most recent observation, so once
a later observation lands on the row, the discounted one is only visible in
`rewards` (`[0.285, 1.0]` above). A store with no `experiences` table returns
`[]` instead of raising, and a store created before Phase 4 stays queryable
(`QueryTests`).

## Defects found and fixed while implementing

1. **The discount was applied twice.** `reward.evaluate` and `store.record_outcome`
   each ran validation, so `1.0` became `0.3` and then `0.09`. Fixed by giving
   validation a single owner per observation: `store` runs `validation.finalize`
   for a caller that has not validated, and a caller that has — the hook, which
   must report the reward to the harness before recording it — passes
   `final_reward` plus its bases. `RecordingTests::
   test_the_stored_reward_matches_what_the_evaluator_settled` pins `0.27` staying
   `0.27`.
2. **The hook did not pass the new signals to the evaluator.** `hooks.serve`
   called `reward.evaluate` without `churn` / `touch_test_file`, so every step
   looked like a zero-line change and every high reward was flagged as a
   small-change win. Caught by the hook tests, fixed in `hooks.serve`.
3. **A later flag rewrote an earlier reason.** The upsert replaced
   `flagged_reason` on every flagged observation, so a row flagged for tampering
   was relabelled `small_change_high_reward` by a later, weaker rule — the
   reviewer would then read the wrong evidence. The reason is now written only
   when the row has none (`RecordingTests::
   test_a_later_flag_does_not_rewrite_the_first_reason`, acceptance scenario 3).
4. **Test-file detection missed `test_*.py`** — see the scope widening above.
5. `audit_phase1.py` crashed on a store written before `flagged_reason` existed;
   it now reports a later-phase column as absent instead of failing.
6. **The flagged query did not select `rewards`.** The CLI read a key the query
   never selected, the `KeyError` was contained into `ok: false`, and the
   reviewer saw an empty list. The column is now selected, and the acceptance
   script asserts `the flagged query reports no error` so a contained failure
   cannot pass as "nothing was flagged".

## Gates re-run

| Gate | Result |
| --- | --- |
| `python -m pytest learning_module/tests -q` | **181 passed**, 46 subtests (was 128 before this phase) |
| `python learning_module/acceptance_phase4.py` | **36/36 checks**, exit 0 |
| `python learning_module/acceptance_phase2.py` | 26/26 checks, exit 0 (Phase 2 unchanged) |
| `python learning_module/tests/audit_phase1.py <real store>` | exit 0 — every declared field present, every value in range |
| `python learning_module/tests/spotcheck_phase1.py <real store> 5` | exit 0 |
| `npx vitest run packages/core/agent-loop/tests/learning-*.spec.ts` | 4 files, **49 tests passed** |
| `npx tsc -b packages/core/agent-loop/tsconfig.json` | exit 0 |
| `pnpm run test` (unit) | `41 failed / 12815 passed / 62 skipped (12918)`, `17 failed / 757 passed / 5 skipped (779)` files — every failure is a subset of the pre-change baseline set (host symlink privilege, `CreateProcessAsUserW` ACL failures, the `unknown tool "bash"` cases), and no failing test names the module or the agent-loop package |

The harness was not modified in this phase, so the snapshot gate's baseline
(`34 failed / 75 passed / 7 skipped`, all pre-existing host `unknown tool "bash"`
failures) is unaffected; the module's stderr remains absent from it.

## Files this phase touched

New: `learning_module/validation.py`, `learning_module/tests/test_phase4.py`
(52 tests), `learning_module/acceptance_phase4.py`, this report.

Changed: `learning_module/schema.py` (`flagged_reason` in `_ADDED_COLUMNS` and
`EXPERIENCE_COLUMNS`), `learning_module/classify.py` (test-path pattern plus
`touches_test_file`), `learning_module/state.py` (`touch_test_file`, `churn`),
`learning_module/reward.py` (validation applied to the settled reward),
`learning_module/store.py` (validation on write, first-flag-wins, the
`flagged_experiences` query), `learning_module/hooks.py` (signals to the
evaluator, settled reward to the store, envelope fields),
`learning_module/cli.py` (`flagged` command), `learning_module/README.md`,
`learning_module/tests/audit_phase1.py` (tolerate a later-phase column).

Untouched by design: `value.py`, `retrieve.py`, `context.py`, every frozen
parameter, and everything under `packages/`.

## Not done, by scope

The two design defects Phase 3 reported — retrieval keyed on `state_error_type`
alone, and the reward band measuring a step's outcome rather than an action's
soundness — remain open by your ruling: they are not Phase 4's subject and wait
for Phase 5's benchmark data.
