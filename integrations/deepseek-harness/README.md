# DeepSeek Harness integration

This directory is **not part of the module**. It is a reference implementation
showing how the Mushroom Body Learning Module is wired into one specific agent
loop, kept here because the wiring — where to instrument, what to carry in each
payload, and how to contain failure — is the part that does not generalise by
itself.

`learning-bridge.ts` is a verbatim copy of
`packages/core/agent-loop/src/learning-bridge.ts` from
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness). It is
**internal harness code**, not a published API of that project, and it is covered
by that project's license rather than this repository's MIT license.

## What it demonstrates

| Concern | How the bridge handles it |
| --- | --- |
| Where to instrument | Two calls, `pre-llm` and `post-action`, mapping onto the two hook names the module serves. |
| What to send | The previous settled step's results (state), and this step's settled calls (action), including whether each call mutates the file its arguments name. |
| The third call | `context`, made during prompt assembly, `active` mode only — the retrieval query, not an instrumentation point. |
| Token budget | `applyTokenBudget` drops whole trailing entries against a per-entry and a per-block ceiling, priced with `@deepseek-ai/dsh-token-meter`. |
| Failure containment | Every exported function is total: a missing interpreter, a non-zero exit, unparseable JSON, or an internal error becomes a stderr diagnostic and a normal return, never a thrown turn. |
| Mode switch | `DSH_LEARNING_MODULE` (`off` / `shadow` / `active`), read per call. |

## What a host must implement

For another agent loop, the shape is the same three calls with different plumbing:

1. a hook after the request is composed that reports the session/turn/step, the
   tools the request offers, their declared file arguments, and the previous
   settled step's results;
2. a hook after the tool calls of a step have all settled that reports each call
   with its name, arguments, error flag, result text, and whether it mutates a
   file;
3. a query during prompt assembly that sends the same state payload and inserts
   the returned text into the prompt — as a section of the existing runtime
   context, not as a new message kind, so the injected text stays replayable from
   the session log.

The third point is the one worth copying deliberately: because the text becomes a
durable record of the request, what the model saw and what the log holds cannot
drift apart.

## Using it

The file imports `node:child_process`, `node:fs`, `node:path`, `node:url`, and
`@deepseek-ai/dsh-token-meter`, and resolves the module directory relative to its
own location. To reuse it outside DeepSeek Harness, take the three call shapes and
the budget step, and replace the token-meter import with your own pricing.
