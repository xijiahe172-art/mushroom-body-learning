# Mushroom Body Learning Module

Experience-based learning for coding agents: an optional module that lets an agent
form a behavioural tendency — "what is worth trying here" — from its own past
steps, instead of starting every task from zero.

The loop it implements:

```
State → Action → Reward → Experience → Value Update → Retrieval → (next State)
```

## What it is, and what it is not

**It is** a small, dependency-free Python module (with SQLite for storage) that
observes an agent loop at two instrumentation points, derives a structural state
before each model call and an evaluated reward after each settled action, stores
that as experience, and — in `active` mode — injects a labelled, budgeted
reference block back into the prompt. It is built to never be a single point of
failure: a missing interpreter, a corrupt store, a retrieval error, or any
internal exception is logged and dropped, and the agent keeps working.

**It is not** a simulation of the fruit-fly nervous system. Nothing here models
neurons, synapses, or the mushroom body's anatomy. The inspiration is one
abstract idea from that literature — that behaviour can be shaped by associating
states with outcomes, without any model of the world — and this project is an
engineering implementation of that idea for an agent loop. If you came looking
for a spiking-network model or a connectome, this is not it.

## Current status: validated, but not yet proven effective

Be clear about what is and is not established.

| Claim | Status |
| --- | --- |
| The module never breaks an agent turn (SQLite down, retriever error, encoder error, budget overflow) | **Verified.** Every failure path returns "no injection" and logs. |
| Recording experience has no side effect on decisions (`shadow` mode) | **Verified.** Off vs shadow differ no more than two `off` runs differ from each other. |
| Anti-cheating works (rewarding an agent for editing the tests) | **Verified.** Test-file edits that pass are discounted and flagged for review. |
| Injection respects a hard token budget | **Verified**, measured with the harness's own pricing (a 4-characters/token estimate, not an exact tokenizer). |
| **It reduces repeated mistakes** | **Not demonstrated.** |

That last row is the honest headline. A 16-task fixed benchmark, run across four
arms (`off`, `off-repeat`, `shadow`, `active`, 64 real runs), found:

- task success rate: 100% in every arm (the tasks are small; there is a ceiling);
- repeated-mistake rate: **84.4% → 83.3%** (1.2% relative improvement, against a
  10% bar) with the fixture's own seeded failure included, and **68.8% → 78.6%**
  (14.3% *worse*) counting only the model's own steps;
- mean tokens: **1.9% higher** with the module active;
- recovery speed: slightly slower.

So the mechanism is implemented and safe, but **it has not been shown to pay for
itself.** The full data, including the sampling-noise floor and the per-task
tables, is in [`PHASE5_BENCHMARK.md`](learning_module/PHASE5_BENCHMARK.md), and the five-phase
summary with the recommendation is in [`FINAL_REPORT.md`](learning_module/FINAL_REPORT.md). Two
known design defects found during validation are documented there as well: the
retriever keys on the failure kind alone (so one action can appear in both the
"successful" and "avoid" groups of the same block), and the reward measures
whether a step advanced the task rather than whether the action was sensible (so
running the tests to observe a failure is scored negative).

**Default is `off`.** `shadow` mode is safe to enable for data collection: it
records experience and never influences a decision.

## Quick start

Requires Python 3.11+ (standard library only) and SQLite. No third-party
packages, no network access.

```sh
git clone https://github.com/xijiahe172-art/mushroom-body-learning-module.git
cd mushroom-body-learning-module/learning_module

# Create the store and see the resolved configuration.
DSH_LEARNING_MODULE=off python cli.py config
DSH_LEARNING_MODULE=shadow python cli.py init-db

# Run the test suite.
python -m pytest tests -q
```

The module talks JSON over stdin/stdout, so it can be driven by hand:

```sh
# Capture the state before a model call.
echo '{"hook":"pre-llm","sessionId":"s1","turn":1,"step":1,
       "toolNames":["read","pwsh"],"fileTargets":[{"name":"read","parameters":["file_path"]}],
       "priorResults":[]}' | DSH_LEARNING_MODULE=shadow python cli.py hook

# Record what the step's calls settled to.
echo '{"hook":"post-action","sessionId":"s1","turn":1,"step":1,
       "results":[{"callId":"c1","name":"pwsh","isError":true,
                   "text":"1 failed, 2 passed in 0.10s",
                   "arguments":{"command":"python -m pytest -q"},"mutates":false}]}' \
  | DSH_LEARNING_MODULE=shadow DSH_LEARNING_DB=experiences.db python cli.py hook

# Inspect what a state would retrieve, without injecting anything.
python explain_retrieval.py experiences.db --all

# List the rows the anti-cheating layer flagged for a human.
python cli.py flagged
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DSH_LEARNING_MODULE` | `off` | Mode: `off`, `shadow`, or `active`. Any other value is reported and treated as `off`. |
| `DSH_LEARNING_DB` | `~/.dsh/learning/experiences.db` | SQLite store path. |
| `DSH_LEARNING_MODULE_HOME` | the module directory | For out-of-tree installs. |
| `DSH_LEARNING_PYTHON` | `python` (Windows) / `python3` | Interpreter the bridge invokes. |

### Modes

| Mode | Records experience | Influences the model |
| --- | --- | --- |
| `off` | no | no |
| `shadow` | yes | no |
| `active` | yes | yes — injects the reference block |

## Integrating with another agent loop

The module's interface is deliberately two calls plus one query, so a new host
only has to implement a small bridge:

| Call | When | What it must carry |
| --- | --- | --- |
| `pre-llm` hook | after the request is composed, before it streams | session/turn/step, the tools the request offers, their declared file arguments, and the **previous** settled step's results |
| `post-action` hook | after every tool call of a step has settled | the step's calls with name, arguments, `isError`, result text, and whether the call mutates its file arguments |
| `context` query | during prompt assembly, in `active` mode only | the same state payload as the pre-LLM hook |

A bridge must also own the fail-silent rule: the module may never fail a turn, so
a missing interpreter, a non-zero exit, or unparseable output must all degrade to
"no injection" plus a diagnostic.

[`integrations/deepseek-harness/learning-bridge.ts`](integrations/deepseek-harness/learning-bridge.ts)
is the reference implementation of exactly that, for **DeepSeek Harness**. Read
it as a worked example of the three calls, the token budget, and the failure
containment — it is not a generic adapter layer, and it is not part of this
module's public interface. See that directory's README for the specifics and the
caveats.

## Documentation

| File | Contents |
| --- | --- |
| [`MODULE.md`](learning_module/MODULE.md) | The module's own reference: schema, retrieval and value formulas, the injected format, and the anti-cheating rules. |
| [`FINAL_REPORT.md`](learning_module/FINAL_REPORT.md) | Five-phase summary: per-phase test results, benchmark data, and the recommendation on enabling `active`. |
| [`PHASE3_ACCEPTANCE.md`](learning_module/PHASE3_ACCEPTANCE.md) | What injection actually looked like in ten real active-mode sessions. |
| [`PHASE4_ACCEPTANCE.md`](learning_module/PHASE4_ACCEPTANCE.md) | The anti-cheating layer, its acceptance criteria, and the defects found while building it. |
| [`PHASE5_BENCHMARK.md`](learning_module/PHASE5_BENCHMARK.md) | The 64-run benchmark: six metrics per arm, the noise floor, and per-task tables. |

## License

MIT — see [LICENSE](LICENSE).

The DeepSeek Harness excerpt under `integrations/` is a copy of code from that
project and carries its own license; it is included only as an integration
reference.
