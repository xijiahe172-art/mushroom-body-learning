# Phase 5 benchmark report — off vs shadow vs active

Suite: 16 fixed tasks. Run records are written to a scratch directory you choose; the numbers below

Every number below is computed from the run records by
`.learning-tools/benchmark_report.ts`; the runner measured each run from its own
session transcript. A task counts as solved when its own test file passes in the
final workspace. Four arms were run: `off`, `off-repeat` (a second `off` run, to
measure sampling noise), `shadow`, and `active`.

## Arm active

| metric | value |
| --- | --- |
| runs | 16 |
| task success rate | 100.0% |
| mean tool calls | 8.44 |
| mean tokens | 70153 |
| repeated mistakes | 25 / 30 occurrences = 83.3% |
| repeated mistakes, the model's own steps only | 11 / 14 = 78.6% |
| recovery speed (steps, 16 recovered runs) | 1.06 |
| regression rate | 6.3% |
| crashed runs | 0 |
| runs that edited the graded test file | 0 |
| runs that received the injected block | 15 |
| mean wall time (s) | 33.7 |

## Arm off

| metric | value |
| --- | --- |
| runs | 16 |
| task success rate | 100.0% |
| mean tool calls | 8.63 |
| mean tokens | 68861 |
| repeated mistakes | 27 / 32 occurrences = 84.4% |
| repeated mistakes, the model's own steps only | 11 / 16 = 68.8% |
| recovery speed (steps, 16 recovered runs) | 1.00 |
| regression rate | 6.3% |
| crashed runs | 0 |
| runs that edited the graded test file | 0 |
| runs that received the injected block | 0 |
| mean wall time (s) | 28.5 |

## Arm off-repeat

| metric | value |
| --- | --- |
| runs | 16 |
| task success rate | 100.0% |
| mean tool calls | 7.94 |
| mean tokens | 62479 |
| repeated mistakes | 21 / 30 occurrences = 70.0% |
| repeated mistakes, the model's own steps only | 9 / 14 = 64.3% |
| recovery speed (steps, 16 recovered runs) | 1.25 |
| regression rate | 12.5% |
| crashed runs | 0 |
| runs that edited the graded test file | 0 |
| runs that received the injected block | 0 |
| mean wall time (s) | 27.6 |

## Arm shadow

| metric | value |
| --- | --- |
| runs | 16 |
| task success rate | 100.0% |
| mean tool calls | 8.44 |
| mean tokens | 69013 |
| repeated mistakes | 25 / 33 occurrences = 75.8% |
| repeated mistakes, the model's own steps only | 11 / 17 = 64.7% |
| recovery speed (steps, 16 recovered runs) | 1.13 |
| regression rate | 12.5% |
| crashed runs | 0 |
| runs that edited the graded test file | 0 |
| runs that received the injected block | 0 |
| mean wall time (s) | 33.4 |

## Verdict on "recording has zero side effects"

Shadow's 54 differences do not exceed the 58 that two `off` runs show, so recording experience is within sampling noise of having no effect, which is what
`DSH_LEARNING_MODULE=off` promises and what "zero side effects" means for a sampled model.

Noise floor: 58 field differences over 16 runs (off vs off-repeat). Off vs shadow: 54.

## Group A (off) vs Group B (shadow)

Comparable runs: 16.

**54 difference(s) found:**

- 01-sign-error: transcriptHash cd1dd8e4 != 5eef350b
- 01-sign-error: toolCalls 9 != 8
- 01-sign-error: tokens 73963 != 73344
- 02-stub-return: transcriptHash 7c24d8fb != 627948d6
- 02-stub-return: tokens 54315 != 72478
- 02-stub-return: stepErrors assertion_failure,none,none,assertion_failure,none,none != assertion_failure,none,none,assertion_failure,none,none,none,none
- 03-empty-input: transcriptHash 686a364c != 303cb49c
- 03-empty-input: tokens 74181 != 72032
- 04-missing-name: transcriptHash 1d6fce89 != ba273e12
- 04-missing-name: toolCalls 7 != 6
- 04-missing-name: tokens 53291 != 45914
- 04-missing-name: stepErrors collection_error,none,none,collection_error,none,none != collection_error,collection_error,none,none,none
- 05-json-comma: transcriptHash 8f7b56f1 != 21853117
- 05-json-comma: toolCalls 10 != 9
- 05-json-comma: tokens 83343 != 73040
- 05-json-comma: stepErrors assertion_failure,none,none,syntax_error,none,none,none,none,none != assertion_failure,none,none,syntax_error,none,none,none,none
- 06-import-name: transcriptHash f0de466c != a508bff1
- 06-import-name: toolCalls 8 != 6
- 06-import-name: tokens 73124 != 42678
- 06-import-name: stepErrors collection_error,none,none,collection_error,none,none,none,none != collection_error,none,none,none,none
- 07-sort-key: transcriptHash 6195f32e != 6bc25252
- 07-sort-key: toolCalls 8 != 6
- 07-sort-key: tokens 72197 != 44962
- 07-sort-key: stepErrors assertion_failure,none,none,assertion_failure,none,none,none,none != assertion_failure,assertion_failure,none,none,none
- 08-type-error: transcriptHash 55c8c845 != f3adf3c6
- 08-type-error: tokens 51981 != 69290
- 08-type-error: stepErrors assertion_failure,none,assertion_failure,assertion_failure,none,none != assertion_failure,none,assertion_failure,assertion_failure,none,none,none,none
- 09-path-resolution: transcriptHash add316ca != d53d0f4e
- 09-path-resolution: toolCalls 12 != 11
- 09-path-resolution: tokens 88985 != 76172
- 09-path-resolution: stepErrors assertion_failure,none,none,assertion_failure,none,none,none,none != assertion_failure,none,none,assertion_failure,none,import_error,none
- 10-index-error: transcriptHash 49e79d1e != 9a6595da
- 10-index-error: toolCalls 8 != 10
- 10-index-error: tokens 74267 != 77009
- 11-inverted-branch: transcriptHash b8a30798 != 20e4dfe1
- 11-inverted-branch: tokens 73719 != 54720
- 11-inverted-branch: stepErrors assertion_failure,none,none,assertion_failure,none,none,none,none != assertion_failure,none,none,assertion_failure,none,none
- 12-env-precedence: transcriptHash d8316350 != a858e0b8
- 12-env-precedence: toolCalls 6 != 9
- 12-env-precedence: tokens 43214 != 73613
- 12-env-precedence: stepErrors assertion_failure,none,none,none,none != assertion_failure,none,none,assertion_failure,none,none,none,none
- 13-blocking-io: transcriptHash 33fbca27 != 8f34b596
- 13-blocking-io: toolCalls 11 != 13
- 13-blocking-io: tokens 76507 != 137422
- 13-blocking-io: stepErrors timeout,none,none,timeout,none,none,none,none != timeout,none,none,timeout,none,none,none,none,none,none,none,none,none
- 14-recursion-depth: transcriptHash c6d2cc3c != d200da10
- 14-recursion-depth: tokens 75481 != 78730
- 14-recursion-depth: stepErrors assertion_failure,none,none,assertion_failure,none,none,none,none != assertion_failure,none,assertion_failure,none,none,none,none,none
- 15-trailing-comma: transcriptHash 158121c8 != 7f9d733b
- 15-trailing-comma: tokens 57656 != 57574
- 16-regex-dot: transcriptHash 9d206e07 != a9f2cc6d
- 16-regex-dot: toolCalls 9 != 7
- 16-regex-dot: tokens 75553 != 55228
- 16-regex-dot: stepErrors assertion_failure,none,none,assertion_failure,none,none,none,none != assertion_failure,none,none,assertion_failure,none,none

A difference here is not automatically a module defect: the model is sampled
and the harness sets no temperature, so two runs of the *same* arm can differ.
The repeat-run comparison below separates module effects from sampling noise.

### Sampling noise floor (off vs off-repeat): 58 difference(s) over 16 runs
- 01-sign-error: transcriptHash cd1dd8e4 != 2b5505eb
- 01-sign-error: toolCalls 9 != 8
- 01-sign-error: tokens 73963 != 70831
- 02-stub-return: transcriptHash 7c24d8fb != e41a623e
- 02-stub-return: tokens 54315 != 73067
- 02-stub-return: stepErrors assertion_failure,none,none,assertion_failure,none,none != assertion_failure,none,none,assertion_failure,none,none,none,none
- 03-empty-input: transcriptHash 686a364c != af124f37
- 03-empty-input: toolCalls 8 != 7
- 03-empty-input: tokens 74181 != 53411
- 03-empty-input: stepErrors assertion_failure,none,none,assertion_failure,none,none,none,none != assertion_failure,none,none,assertion_failure,none,none
- 04-missing-name: transcriptHash 1d6fce89 != 0f044efe
- 04-missing-name: toolCalls 7 != 8

## Group A (off) vs Group C (active)

| metric | A (off) | C (active) | C relative to A |
| --- | --- | --- | --- |
| task success rate | 100.0% | 100.0% | 0.0 pp |
| mean tool calls | 8.63 | 8.44 | 2.2% fewer |
| mean tokens | 68861 | 70153 | -1.9% fewer |
| repeated-mistake rate | 84.4% | 83.3% | 1.2% lower |
| repeated-mistake rate, model's own steps only | 68.8% | 78.6% | -14.3% lower |
| recovery speed | 1.00 | 1.06 | -6.3% faster |
| regression rate | 6.3% | 6.3% | 0.0% lower |

### The phase's core question

Repeated-mistake occurrences (seeded failure included): A 27/32, C 25/30.
Repeated-mistake occurrences (model's own steps only): A 11/16, C 11/14.

Improvement, seeded reading: **1.2%**. Model-only reading: **-14.3%**. Floor for a positive verdict: 10%.

**The Learning Module is not shown to be effective on this benchmark.** The
improvement is 1.2% on the seeded reading and -14.3% on the model-only reading, both below the 10% floor,
so the honest conclusion is that these 16 tasks do not demonstrate the mechanism
paying for itself. See the limitations section for what would change that.

## Per-task results

| task | class | active solved | off solved | off-repeat solved | shadow solved | active calls | off calls | off-repeat calls | shadow calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-sign-error | wrong operator | yes | yes | yes | yes | 8 | 9 | 8 | 8 |
| 02-stub-return | unimplemented function | yes | yes | yes | yes | 8 | 8 | 8 | 8 |
| 03-empty-input | missing edge case | yes | yes | yes | yes | 8 | 8 | 7 | 8 |
| 04-missing-name | undefined/import error | yes | yes | yes | yes | 9 | 7 | 8 | 6 |
| 05-json-comma | malformed data file | yes | yes | yes | yes | 11 | 10 | 11 | 9 |
| 06-import-name | name mismatch | yes | yes | yes | yes | 7 | 8 | 8 | 6 |
| 07-sort-key | wrong sort order | yes | yes | yes | yes | 8 | 8 | 6 | 6 |
| 08-type-error | type error | yes | yes | yes | yes | 8 | 8 | 5 | 8 |
| 09-path-resolution | wrong path resolution | yes | yes | yes | yes | 11 | 12 | 10 | 11 |
| 10-index-error | index arithmetic | yes | yes | yes | yes | 8 | 8 | 5 | 10 |
| 11-inverted-branch | inverted condition | yes | yes | yes | yes | 8 | 8 | 5 | 8 |
| 12-env-precedence | precedence bug | yes | yes | yes | yes | 8 | 6 | 9 | 9 |
| 13-blocking-io | blocking I/O assumption | yes | yes | yes | yes | 9 | 11 | 13 | 13 |
| 14-recursion-depth | unbounded recursion | yes | yes | yes | yes | 8 | 9 | 8 | 9 |
| 15-trailing-comma | trailing comma | yes | yes | yes | yes | 10 | 9 | 8 | 9 |
| 16-regex-dot | regex escaping | yes | yes | yes | yes | 6 | 9 | 8 | 7 |

### Failure kinds observed in arm active

| failure kind | occurrences | repeats |
| --- | --- | --- |
| assertion_failure | 25 | 23 |
| collection_error | 3 | 2 |
| syntax_error | 1 | 0 |
| timeout | 1 | 0 |

### Failure kinds observed in arm off

| failure kind | occurrences | repeats |
| --- | --- | --- |
| assertion_failure | 25 | 23 |
| collection_error | 4 | 3 |
| timeout | 2 | 1 |
| syntax_error | 1 | 0 |

### Failure kinds observed in arm off-repeat

| failure kind | occurrences | repeats |
| --- | --- | --- |
| assertion_failure | 24 | 19 |
| collection_error | 3 | 2 |
| import_error | 1 | 0 |
| syntax_error | 1 | 0 |
| timeout | 1 | 0 |

### Failure kinds observed in arm shadow

| failure kind | occurrences | repeats |
| --- | --- | --- |
| assertion_failure | 26 | 23 |
| collection_error | 3 | 1 |
| timeout | 2 | 1 |
| syntax_error | 1 | 0 |
| import_error | 1 | 0 |

## Limitations of this benchmark

- The suite is 16 small single-bug workspaces. Every one is solvable in a few
  steps, so there is little room for experience to save work, and the metric
  most favourable to the module — avoiding a mistake it has already seen — needs
  the same failure to recur across tasks.
- One run per arm per task, plus the `off-repeat` run used as the noise floor.
  Comparing off against shadow therefore has a measured baseline for "how much
  do two identical runs differ", but comparing off against active has one sample
  per task: a 6% difference in any rate is inside what the noise floor shows.
- The harness sets no temperature, so runs are not reproducible byte for byte.
- Token counts come from provider usage, summed over every step of a run.
- The active arm received the injected block in 15 of 16 runs; the first run of
  the suite starts against an empty store, so there is nothing to retrieve yet.
