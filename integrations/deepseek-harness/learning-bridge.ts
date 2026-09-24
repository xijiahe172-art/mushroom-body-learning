/**
 * Transport for the Learning Module's two agent-loop instrumentation points.
 *
 * This file carries no learning behavior: the module itself (experience store,
 * mode switch, hook handling) lives in `learning_module/` at the repository
 * root and runs as a Python subprocess. Only this adapter sits inside the
 * agent-loop package, because the loop's instrumentation points are the sole
 * consumers of the module's interface.
 *
 * Every exported function is total: the Learning Module may never become a
 * single point of failure for the harness, so a missing interpreter, a failing
 * CLI, unparseable output, or an internal error all end in a diagnostic on
 * stderr and a normal return.
 *
 * @module dsh-agent-loop/learning-bridge
 */

import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { estimateContent } from '@deepseek-ai/dsh-token-meter/src/estimate.ts'

/** Module switch carried by `DSH_LEARNING_MODULE`. */
export const LEARNING_MODULE_ENV = 'DSH_LEARNING_MODULE'

/** Environment override for the Learning Module directory. */
export const LEARNING_MODULE_HOME_ENV = 'DSH_LEARNING_MODULE_HOME'

/** Environment override for the Python interpreter running the module. */
export const LEARNING_MODULE_PYTHON_ENV = 'DSH_LEARNING_PYTHON'

/** Mode the switch selects. */
export type LearningMode = 'off' | 'shadow' | 'active'

/** Hooks the agent loop calls, each at one instrumentation point. */
export type LearningHookName = 'pre-llm' | 'post-action'

/**
 * Commands the module's CLI serves. `hook` carries every instrumentation point
 * (the hook name travels in the payload); `context` is not an instrumentation
 * point but the retrieval query the prompt-context provider makes, one call per
 * assembly, in active mode only.
 */
export type CliCommand = 'hook' | 'context'

/** CLI command serving the instrumentation points. */
const HOOK_COMMAND: CliCommand = 'hook'

/** CLI command serving the retrieval query. */
const CONTEXT_COMMAND: CliCommand = 'context'

/** One settled tool call reported by the post-action instrumentation point. */
export interface LearningToolOutcome {
  /** Identity of the model's tool call. */
  callId: string
  /** Tool the call targeted. */
  name: string
  /** Whether the tool reported a failure. */
  isError: boolean
  /** Failure identity the tool reported, when it reported one. */
  error?: { name: string; code: string }
  /** Text the model-facing result carried. */
  text?: string
  /** Raw call arguments as the model produced them, for classification. */
  arguments?: Record<string, unknown>
  /** Whether the tool writes or edits the files its arguments name. */
  mutates?: boolean
}

/** One tool the step's request offers, with its declared file parameters. */
export interface LearningToolTargets {
  /** Tool name. */
  name: string
  /** Names of the tool's declared parameters that carry a file path. */
  parameters: string[]
}

/**
 * Parameter names that carry a file path in a tool's declared schema. The
 * Learning Module classifies by these names, so a tool whose file argument is
 * named differently contributes no file context rather than a wrong one.
 */
export const FILE_ARGUMENT_NAMES: readonly string[] = ['file_path', 'path', 'files', 'paths', 'target']

/** Tools whose successful call writes or edits a file. */
export const FILE_MUTATING_TOOLS: readonly string[] = ['write', 'edit', 'str_replace_editor']

/** Per-hook data the agent loop hands the Learning Module. */
export interface LearningHookPayload {
  /** Identity of the session whose step fired the hook. */
  sessionId: string
  /** Turn number of the firing step. */
  turn: number
  /** Step number within `turn`. */
  step: number
  /** Tools the step's request offers; present for pre-LLM hook calls. */
  toolNames?: string[]
  /** File arguments declared by those tools; present for pre-LLM hook calls. */
  fileTargets?: LearningToolTargets[]
  /** Settled calls of the previous step, which set the state before this one. */
  priorResults?: LearningToolOutcome[]
  /** Settled calls of the step; present for post-action hook calls. */
  results?: LearningToolOutcome[]
}

/** Resolved execution paths of the Learning Module. */
export interface LearningModulePaths {
  /** Directory holding the module's Python sources. */
  home: string
  /** Python CLI script the bridge invokes. */
  cli: string
  /** Interpreter used to run {@link LearningModulePaths.cli}. */
  python: string
}

/** Call-site overrides; the agent loop passes none. */
export interface LearningHookOverrides {
  /** CLI script to run instead of the resolved one. */
  cli?: string
  /** Interpreter to run it with instead of the resolved one. */
  python?: string
  /** Round-trip ceiling instead of the module default. */
  timeoutMs?: number
}

/** Modes the switch accepts; anything else is a misconfiguration. */
const MODES: readonly LearningMode[] = ['off', 'shadow', 'active']

/** Wall-clock ceiling for one hook round trip. */
const HOOK_TIMEOUT_MS = 5_000

/** Diagnostic prefix, so hook output is greppable in agent logs. */
const LOG_PREFIX = '[learning-module]'

/** Maximum experience entries the injected text may carry. */
export const CONTEXT_MAX_ENTRIES = 3

/**
 * Per-entry token ceiling, priced by the TokenMeter heuristic. The comparison
 * is strict, so a priced count of 50 already exceeds it.
 */
export const CONTEXT_ENTRY_TOKEN_BUDGET = 50

/** Total token ceiling for the injected text, priced the same way. */
export const CONTEXT_TOTAL_TOKEN_BUDGET = 200

/**
 * Candidate module directories, most specific first: the built tree
 * (`lib/index.js` is one directory below the repository root) before the
 * source tree (`src/learning-bridge.ts` is four directories below it).
 */
function candidateHomes(): string[] {
  const here = dirname(fileURLToPath(import.meta.url))
  return [
    join(here, '..', '..', '..', '..', 'learning_module'),
    join(here, '..', '..', 'learning_module'),
  ]
}

/**
 * Report one failure without letting it escape. A broken stderr must not turn a
 * learning-module problem into an agent failure.
 *
 * Only failures reach this: a healthy hook writes nothing from the harness
 * side, because the agent loop's stderr is authoritative product output that
 * assembled-application snapshots compare byte for byte. Per-hook logging
 * belongs to the module and happens in `learning_module` when it is enabled.
 * @param message - diagnostic text.
 */
function report(message: string): void {
  try {
    process.stderr.write(`${LOG_PREFIX} ${message}\n`)
  } catch {
    // A closed or broken stderr is not a reason to fail an agent turn.
  }
}

/**
 * Resolve the interpreter used when neither the environment nor a call-site
 * override names one.
 * @returns the platform's conventional Python 3 command.
 */
export function defaultPython(): string {
  return process.platform === 'win32' ? 'python' : 'python3'
}

/**
 * Read the configured mode. An unset switch is `off`; an unrecognized value is
 * reported and also resolves to `off`, because no unknown deployment choice may
 * silently enable learning.
 * @returns the active mode.
 */
export function learningMode(): LearningMode {
  const raw = process.env[LEARNING_MODULE_ENV]
  if (raw === undefined || raw === '') return 'off'
  if ((MODES as readonly string[]).includes(raw)) return raw as LearningMode
  report(`error: ${LEARNING_MODULE_ENV} must be one of ${MODES.join('/')}, got ${JSON.stringify(raw)}; treating it as off`)
  return 'off'
}

/**
 * Resolve where the module's Python sources and interpreter live.
 * @param overrides - test-only path overrides.
 * @returns the resolved paths, honouring the environment overrides.
 */
export function resolveLearningModulePaths(overrides: LearningHookOverrides = {}): LearningModulePaths {
  const override = process.env[LEARNING_MODULE_HOME_ENV]
  const candidates = override === undefined || override === '' ? candidateHomes() : [override]
  const home = candidates.find(candidate => existsSync(join(candidate, 'cli.py'))) ?? candidates[0]!
  const pythonOverride = overrides.python ?? process.env[LEARNING_MODULE_PYTHON_ENV]
  const python = pythonOverride !== undefined && pythonOverride !== '' ? pythonOverride : defaultPython()
  return { home, cli: overrides.cli ?? join(home, 'cli.py'), python }
}

/**
 * Hand one CLI call to the Python module over stdin.
 * @param command - CLI command to run.
 * @param envelope - JSON payload to hand the CLI.
 * @param resolved - execution paths for this call.
 * @returns the parsed reply, or `undefined` when the module did not answer.
 */
function invokeCli(command: CliCommand, envelope: object, resolved: LearningModulePaths & { timeoutMs: number }): unknown {
  const [interpreter, ...interpreterArgs] = resolved.python.split(' ')
  const result = spawnSync(interpreter!, [...interpreterArgs, resolved.cli, command], {
    input: JSON.stringify(envelope),
    encoding: 'utf8',
    timeout: resolved.timeoutMs,
    windowsHide: true,
  })
  // Both failure kinds share one report: `spawnSync` leaves `status` null on a
  // spawn error and `stderr` undefined on any failure, so neither branch has a
  // distinct diagnostic to offer.
  if (result.error !== undefined || result.status !== 0) {
    report(`error: ${command} could not complete ${resolved.cli}: ${result.error?.message ?? ''} ${result.stderr ?? ''}`.trim())
    return undefined
  }
  // The guard above owns every case that leaves `stdout` undefined: a success
  // status always carries the decoded stream.
  const stdout = result.stdout.trim()
  if (stdout === '') {
    report(`error: ${command} produced no reply`)
    return undefined
  }
  try {
    return JSON.parse(stdout) as unknown
  } catch {
    report(`error: ${command} produced unparseable output: ${stdout.slice(0, 200)}`)
    return undefined
  }
}

/**
 * Ask the module for the experience text this state may inject.
 *
 * The text is measured with `@deepseek-ai/dsh-token-meter`, the project's own
 * pricing. That pricing is a **4 characters per token heuristic**, not the
 * DeepSeek tokenizer's exact count — the harness has no exact local tokenizer,
 * and this is the closest available implementation of the phase's token budget.
 * Entries are dropped from the tail until the text fits the per-entry and total
 * ceilings; dropping is silent, never an error.
 *
 * @param payload - state-bearing payload (tool names, file targets, prior results).
 * @param overrides - test-only call overrides.
 * @returns the text to inject, or `undefined` when nothing is injectable.
 */
export function learningContext(
  payload: LearningHookPayload,
  overrides: LearningHookOverrides = {},
): string | undefined {
  // Only active mode may influence a decision; off and shadow answer with no
  // text so a caller cannot inject one by accident.
  if (learningMode() !== 'active') return undefined
  try {
    const seam = globalThis.__DSH_LEARNING_CONTEXT_TEST__
    const reply = typeof seam === 'function'
      ? { text: seam(payload) }
      : invokeCli(CONTEXT_COMMAND, payload, {
        ...resolveLearningModulePaths(overrides),
        timeoutMs: overrides.timeoutMs ?? HOOK_TIMEOUT_MS,
      })
    const text = readContextText(reply)
    if (text === undefined) return undefined
    // The budget applies to whatever text arrives, from the module or a test
    // seam: it is a safety gate on model-visible content, not a courtesy.
    const budgeted = applyTokenBudget(text, (candidate) => estimateContent([{ type: 'text', text: candidate }]))
    if (budgeted === undefined) {
      report('context: withheld — the rendered text exceeded the token budget with no entry to drop')
      return undefined
    }
    if (budgeted !== text) report('context: truncated to fit the token budget')
    return budgeted
  } catch (error: unknown) {
    report(`error: context failed: ${error instanceof Error ? error.message : String(error)}`)
    return undefined
  }
}

/** Read the injected text out of a CLI reply, tolerating every absent case. */
function readContextText(reply: unknown): string | undefined {
  if (typeof reply !== 'object' || reply === null) return undefined
  const { text } = reply as { text?: unknown }
  return typeof text === 'string' && text !== '' ? text : undefined
}

/** One labelled entry of the rendered text, with the provenance for rebuilding it. */
interface ContextEntry {
  /** Group heading introducing this entry, or `''` when it continues a group. */
  heading: string
  /** The entry's own body lines (the `Action:` line onward). */
  lines: string[]
}

/** The separator closing every entry. */
const CONTEXT_ENTRY_SEPARATOR = '---'

/** The first line of an entry body; the split keys on it rather than headings. */
const CONTEXT_ENTRY_OPEN = 'Action: '

/** The closing line every rendered text carries, kept for the format contract. */
const CONTEXT_FOOTER = 'Note: the above is historical reference only.'

/**
 * Drop trailing entries until the text fits the ceilings.
 *
 * The header and the closing note always survive: they carry the labelling that
 * keeps the block from reading as instruction. Only complete entries are
 * removed, and a group's heading goes with its last entry.
 *
 * @param text - the rendered text.
 * @param price - token pricing for a candidate text.
 * @returns the fitted text, or `undefined` when not even one entry fits.
 */
function applyTokenBudget(text: string, price: (candidate: string) => number): string | undefined {
  const { prefix, entries, footer } = splitContext(text)
  const fits = (candidate: string, kept: readonly ContextEntry[]): boolean =>
    price(candidate) < CONTEXT_TOTAL_TOKEN_BUDGET
    && kept.every(entry => price(entryBody(entry).join('\n')) < CONTEXT_ENTRY_TOKEN_BUDGET)
  if (fits(text, entries)) return text
  for (let count = entries.length; count > 0; count -= 1) {
    const kept = entries.slice(0, count)
    const candidate = rebuildContext(prefix, kept, footer)
    if (fits(candidate, kept)) return candidate
  }
  return undefined
}

/** One entry's body with its closing separator, as priced and re-emitted. */
function entryBody(entry: ContextEntry): string[] {
  return [...entry.lines, CONTEXT_ENTRY_SEPARATOR]
}

/**
 * Split rendered text into its header, labelled entries, and closing note.
 *
 * The renderer closes every entry with {@link CONTEXT_ENTRY_SEPARATOR} and opens
 * every entry body with {@link CONTEXT_ENTRY_OPEN}, so the split is exact: the
 * first segment carries the header plus the first entry's body, later segments
 * carry one body each, and the last segment holds the closing note.
 *
 * @param text - the rendered text.
 * @returns the parts a rebuild needs.
 */
function splitContext(text: string): { prefix: string[]; entries: ContextEntry[]; footer: string } {
  const raw = text.split(`\n${CONTEXT_ENTRY_SEPARATOR}\n`)
  const blocks = [...raw]
  const tail = blocks.pop() ?? ''
  if (blocks.length === 0) return { prefix: [], entries: [], footer: tail.trim() }

  const bodyStart = (block: string): number => {
    const lines = block.split('\n')
    const open = lines.findIndex(line => line.startsWith(CONTEXT_ENTRY_OPEN))
    return open < 0 ? 0 : open
  }

  const first = blocks.shift() ?? ''
  const firstSplit = bodyStart(first)
  const firstLines = first.split('\n')
  const prefix = firstSplit <= 0 ? [] : firstLines.slice(0, firstSplit)
  const bodies = firstSplit <= 0 ? [...blocks] : [firstLines.slice(firstSplit).join('\n'), ...blocks]

  const entries: ContextEntry[] = []
  let heading = ''
  for (const body of bodies) {
    const lines = body.split('\n')
    // A heading introduced since the previous body belongs to this entry.
    const introduced = lines.find(line => /^(?:✅|⚠️)/.test(line))
    if (introduced !== undefined) heading = introduced
    const bodyLines = lines.filter(line => line !== introduced && line.trim() !== '')
    entries.push({ heading, lines: bodyLines })
    heading = ''
  }
  const footer = tail.trim()
  if (footer !== '' && !footer.startsWith(CONTEXT_FOOTER)) {
    // The closing note is what marks the block as reference rather than
    // instruction, so an unexpected one is worth a diagnostic.
    report(`context: unexpected closing note ${JSON.stringify(footer.slice(0, 60))}`)
  }
  return { prefix, entries, footer }
}

/**
 * Rebuild the text from a surviving header, entries, and the closing note.
 *
 * A group heading is re-emitted from the entry that introduced it, so the
 * rebuild is byte-identical to the renderer's own output for the entries that
 * survive.
 *
 * @param prefix - header lines.
 * @param entries - surviving entries, in order.
 * @param footer - closing note, or `''`.
 * @returns the rebuilt text.
 */
function rebuildContext(prefix: readonly string[], entries: readonly ContextEntry[], footer: string): string {
  const lines: string[] = [...prefix]
  for (const entry of entries) {
    if (entry.heading !== '') lines.push('', entry.heading)
    lines.push(...entryBody(entry))
  }
  if (footer !== '') lines.push('', footer)
  return lines.join('\n')
}

/**
 * Fire one Learning Module hook. Never throws and never blocks the caller on
 * module failure: every error is reported and dropped.
 * @param hook - hook name.
 * @param payload - hook data; the hook name and mode are added here.
 * @param overrides - test-only call overrides; the agent loop passes none.
 */
export function learningHook(
  hook: LearningHookName,
  payload: LearningHookPayload,
  overrides: LearningHookOverrides = {},
): void {
  const mode = learningMode()
  const context = `${hook} mode=${mode} session=${payload.sessionId} turn=${payload.turn} step=${payload.step}`
  // Disabled learning adds no harness output at all: a healthy hook reports
  // through the module, and only failures below reach stderr.
  if (mode === 'off') return
  try {
    const seam = globalThis.__DSH_LEARNING_TEST__
    if (typeof seam === 'function') {
      seam(hook, payload)
      return
    }
    // The hook name travels inside the payload; `hook` is the CLI command that
    // serves every instrumentation point, and `context` is the retrieval query.
    invokeCli(HOOK_COMMAND, { hook, mode, ...payload }, {
      ...resolveLearningModulePaths(overrides),
      timeoutMs: overrides.timeoutMs ?? HOOK_TIMEOUT_MS,
    })
  } catch (error: unknown) {
    // `report` swallows its own write failure, so re-reporting here terminates.
    report(`error: ${context} failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}

declare global {
  /** Test-only seam installed in-process to observe hook calls without Python. */
  var __DSH_LEARNING_TEST__: ((hook: LearningHookName, payload: LearningHookPayload) => void) | undefined
  /** Test-only seam that answers a context request without Python. */
  var __DSH_LEARNING_CONTEXT_TEST__: ((payload: LearningHookPayload) => string) | undefined
}
