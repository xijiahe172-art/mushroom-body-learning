"""Deterministic classification of a step's action and error state.

Phase 1 stores what an action *was* so the Phase 2 retriever can tell concrete
practices apart ("check the traceback" versus "rerun the program"). Every
function here is a pure function of data the agent loop already carries — tool
names, declared file parameters, settled tool results — so the same action
always lands in the same bucket, on any host, with no LLM call.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

#: Action categories. Keys are the stored `action_category` values; values are
#: the human-readable meaning kept for docs, logs, and tests.
ACTION_CATEGORIES: Mapping[str, str] = {
    'read_file': 'read a file',
    'search_files': 'search file contents or paths',
    'inspect_image': 'inspect an image attachment',
    'write_file': 'create or overwrite a file',
    'edit_file': 'edit an existing file in place',
    'run_shell': 'run a shell command',
    'run_test': 'run a test suite or test command',
    'run_build': 'run a build, typecheck, or lint command',
    'install_deps': 'install or update dependencies',
    'run_program': 'run a program, script, or service',
    'fetch_url': 'fetch a URL',
    'web_search': 'search the web',
    'track_task': 'write the task list',
    'delegate_agent': 'delegate to a subagent or workflow',
    'other': 'an action this vocabulary does not name',
}

#: Error kinds. `none` means the step observed no failure at all. The kinds
#: below the runner verdicts describe the *form* of the failure, because two
#: failures that both read "tests failed" still need different fixes: a
#: collection error is repaired by the import line, an assertion failure by the
#: logic under test.
ERROR_KINDS: Mapping[str, str] = {
    'none': 'no failure observed',
    'tool_error': 'a tool reported a failure',
    'tool_not_found': 'the requested tool does not exist',
    'command_failed': 'a shell command exited non-zero',
    'test_failure': 'a test run reported failures',
    'test_collection_error': 'a test run failed while collecting tests',
    'build_failure': 'a build, typecheck, or lint run failed',
    'import_error': 'a module or name could not be imported',
    'syntax_error': 'source or data could not be parsed',
    'dependency_error': 'an executable or tool the command needs is missing',
    'file_not_found': 'a path the command needs does not exist',
    'warning': 'a run succeeded but emitted a warning',
    'timeout': 'an action timed out',
    'permission_denied': 'an action was denied by policy',
    'aborted': 'an action was aborted before it ran',
}

#: File kinds, used for the `state_file_context` field.
FILE_KINDS = ('source', 'test', 'config', 'doc', 'other')

_TEST_COMMAND = re.compile(r'(?:^|[\s;&|])(?:pnpm|npm|yarn|npx|node|python|py|pytest|vitest|jest|cargo|go|dotnet)\b[^\n]*\b(?:test|spec|check)\b', re.IGNORECASE)
_TEST_COMMAND_ALT = re.compile(r'(?:^|[\s;&|])(?:pytest|vitest|jest)\b', re.IGNORECASE)
_BUILD_COMMAND = re.compile(r'\b(?:tsc|build|lint|typecheck|oxlint|eslint|cargo\s+build|make|msbuild)\b', re.IGNORECASE)
_INSTALL_COMMAND = re.compile(r'\b(?:install|add|update|restore|ci)\b', re.IGNORECASE)

# A path is a test file when it sits in a test directory, carries a test file
# ending, or is named by the two conventions every language's test runners use.
# The bare `test_*` prefix matters: `test_stats.py` in the workspace root is the
# shape the acceptance tasks produce, and without it the anti-cheat layer cannot
# see that a step edited the tests.
_TEST_PATH = re.compile(
    r'(?:^|[/\\])(?:tests?|__tests__|spec)(?:[/\\]|$)'
    r'|\.(?:test|spec)\.[A-Za-z0-9]+$'
    r'|(?:^|[/\\])test_[^/\\]*$'
    r'|(?:^|[/\\])[^/\\]*_test\.[A-Za-z0-9]+$',
    re.IGNORECASE,
)
_CONFIG_PATH = re.compile(r'(?:^|[/\\])[^/\\]*\.(?:json|ya?ml|toml|ini|cfg|env)$|tsconfig[^/\\]*\.json$|package\.json$', re.IGNORECASE)
_DOC_PATH = re.compile(r'\.(?:md|markdown|txt|rst)$', re.IGNORECASE)
_SOURCE_PATH = re.compile(r'\.(?:ts|tsx|js|jsx|mjs|cjs|py|rs|go|java|cs|c|cc|cpp|h|hpp|rb|sh|ps1|psm1)$', re.IGNORECASE)

#: Test-runner outcome markers, matched against a shell tool's result text.
_TEST_SUMMARY = re.compile(
    r'(?P<passed>\d+)\s+passed|(?P<failed>\d+)\s+failed|'
    r'(?P<ok>\bOK\b)|(?P<errors>\d+)\s+error',
    re.IGNORECASE,
)

#: Markers that a result reported a compile, type, or lint failure.
_BUILD_FAILURE_TEXT = re.compile(
    r'error ts\d+|compilation error|cannot find module|syntax error|type error|build failed',
    re.IGNORECASE,
)

#: Markers that a shell command exited non-zero.
_NONZERO_EXIT_TEXT = re.compile(r'exit code [1-9]|exit status [1-9]|non-zero exit|command failed')

#: Failure-form markers, matched against the tail of one result's text, where a
#: runner prints its verdict. Each names a distinct repair, so each is its own
#: kind rather than one "something failed".
_COLLECTION_ERROR_TEXT = re.compile(r'ERROR collecting|errors during collection|collection error', re.IGNORECASE)
_IMPORT_ERROR_TEXT = re.compile(r'ModuleNotFoundError|ImportError|cannot find module|is not defined', re.IGNORECASE)
_SYNTAX_ERROR_TEXT = re.compile(r'SyntaxError|JSONDecodeError|Expecting value|Expecting .{0,3} delimiter|IndentationError', re.IGNORECASE)
#: A missing executable, as shells report it. The bare "no such file or
#: directory" wording belongs to file errors instead, so it is not listed here.
_DEPENDENCY_ERROR_TEXT = re.compile(
    r'command not found|is not recognized as an internal or external command|'
    r'no such command|not installed|is not installed',
    re.IGNORECASE,
)
_FILE_NOT_FOUND_TEXT = re.compile(
    r'FileNotFoundError|No such file or directory|No such file|cannot find the path|does not exist',
    re.IGNORECASE,
)
_WARNING_TEXT = re.compile(r'\bWarning\b|warnings? summary', re.IGNORECASE)

#: Characters of a result's text that carry the verdict.
_TEXT_TAIL = 4_000

#: Tool-name to action mapping. Names follow the harness tool inventory; a
#: name outside this table classifies as `other` rather than guessing.
_TOOL_ACTIONS: Mapping[str, str] = {
    'read': 'read_file',
    'read_image': 'inspect_image',
    'write': 'write_file',
    'edit': 'edit_file',
    'str_replace_editor': 'edit_file',
    'glob': 'search_files',
    'grep': 'search_files',
    'ls': 'search_files',
    'pwsh': 'run_shell',
    'bash': 'run_shell',
    'shell': 'run_shell',
    'web_fetch': 'fetch_url',
    'web_search': 'web_search',
    'todo_write': 'track_task',
    'task': 'delegate_agent',
    'subagent': 'delegate_agent',
}

#: Shell tools whose command line decides between the run_* categories.
_SHELL_TOOLS = frozenset({'pwsh', 'bash', 'shell'})

#: File parameters the harness tools declare, in the order they are read.
_FILE_PARAMETERS = ('file_path', 'path', 'files', 'paths', 'target')

#: Key carrying a tool's declared file parameters into the hook payload.
TOOL_TARGETS_KEY = 'fileTargets'


def _as_text(value: Any) -> str:
    """Render one argument value as a short, stable string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [str(entry) for entry in value if isinstance(entry, (str, int, float, bool))]
        return ' '.join(parts)
    return ''


#: Renderer value for "this state carries no such context".
NO_CONTEXT = 'none'


def file_targets(tool: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract the declared file arguments of one tool registration entry.

    @param tool - one `fileTargets` entry: tool name plus its declared arguments.
    @returns sorted, de-duplicated file targets.
    """
    targets: set[str] = set()
    arguments = tool.get('arguments')
    if isinstance(arguments, Mapping):
        for name in _FILE_PARAMETERS:
            text = _as_text(arguments.get(name))
            if text:
                targets.add(text)
    return tuple(sorted(targets))


def file_kind(path: str) -> str:
    """Classify one file path for `state_file_context`.

    @param path - a file path or one of its prefixes.
    @returns one of {@link FILE_KINDS}.
    """
    if _TEST_PATH.search(path):
        return 'test'
    if _CONFIG_PATH.search(path):
        return 'config'
    if _DOC_PATH.search(path):
        return 'doc'
    if _SOURCE_PATH.search(path):
        return 'source'
    return 'other'


def touches_test_file(paths: Iterable[str]) -> bool:
    """Whether any of these paths is a test file.

    @param paths - file paths a step wrote or edited.
    @returns `True` when at least one path classifies as a test file.
    """
    return any(file_kind(path) == 'test' for path in paths if path)


def file_context(paths: Iterable[str]) -> str:
    """Render the observable file context of one step.

    @param paths - file targets the step can touch.
    @returns a sorted, comma-joined list of `kind:basename` entries, or `-`.
    """
    entries = {
        f'{file_kind(path)}:{path.replace(chr(92), "/").rsplit("/", 1)[-1]}'
        for path in paths
        if path
    }
    return ','.join(sorted(entries)) if entries else '-'


def tool_context(tool_names: Iterable[str]) -> str:
    """Render the tool context of one step.

    @param tool_names - tools that produced the state being described.
    @returns a sorted, comma-joined list of tool names, or {@link NO_CONTEXT}.
    """
    names = sorted({name for name in tool_names if name})
    return ','.join(names) if names else NO_CONTEXT


#: Shell command words that name the tool doing the work, not the thing worked
#: on. They carry no retrieval signal, so the command token skips them.
_SHELL_WRAPPERS = frozenset({'python', 'python3', 'py', 'node', 'npx', 'pnpm', 'npm', 'yarn', 'uv', 'uvx', 'env', 'time'})


def tool_usage(result: Mapping[str, Any]) -> str:
    """Render one settled call as a `tool` or `tool:token` descriptor.

    A bare tool name describes a whole session ("pwsh" ran something), which
    collapses unrelated situations onto one experience. The command token
    restores the distinction between "ran the tests" and "reran the program"
    without inventing a category the action vocabulary does not name.

    @param result - one settled call with `name` and optional `arguments`.
    @returns the descriptor, or {@link NO_CONTEXT} when the call names no tool.
    """
    name = str(result.get('name', '')).strip()
    if not name:
        return NO_CONTEXT
    arguments = result.get('arguments')
    command = ''
    if isinstance(arguments, Mapping):
        command = str(arguments.get('command', '')).strip()
    if not command:
        return name
    words = [word for word in re.split(r'\s+', command) if word]
    # Flags (`-m`, `-c`, `--fix`) and interpreter wrappers name no work of their
    # own, so the first remaining word is what the command actually ran.
    stripped = [word for word in words if word not in _SHELL_WRAPPERS and not word.startswith('-')]
    token = (stripped or words or [''])[0].strip('"\'')
    return f'{name}:{token}' if token else name


def shell_category(command: str) -> str:
    """Classify one shell command line into a concrete action category.

    @param command - the command line as the model wrote it.
    @returns the `run_test`/`run_build`/`install_deps`/`run_program`/`run_shell` category.
    """
    if _TEST_COMMAND.search(command) or _TEST_COMMAND_ALT.search(command):
        return 'run_test'
    if _BUILD_COMMAND.search(command):
        return 'run_build'
    if _INSTALL_COMMAND.search(command):
        return 'install_deps'
    return 'run_shell'


def action_category(tool_name: str, arguments: Mapping[str, Any] | None = None) -> str:
    """Classify one executed tool call into a concrete action category.

    @param tool_name - the tool the model called.
    @param arguments - the call's arguments, when known.
    @returns one of {@link ACTION_CATEGORIES}.
    """
    if tool_name in _SHELL_TOOLS:
        command = _as_text((arguments or {}).get('command'))
        return shell_category(command) if command else 'run_shell'
    return _TOOL_ACTIONS.get(tool_name, 'other')


def action_detail(tool_name: str, arguments: Mapping[str, Any] | None = None) -> str:
    """Render the most recent debug reference for one action.

    @param tool_name - the tool the model called.
    @param arguments - the call's arguments, when known.
    @returns a bounded, whitespace-collapsed description of the call.
    """
    pieces = [tool_name or 'unknown']
    for key in ('command', *_FILE_PARAMETERS):
        text = _as_text((arguments or {}).get(key))
        if text:
            pieces.append(text)
            break
    return ' '.join(' '.join(pieces).split())[:200]


def step_error_kind(results: Iterable[Mapping[str, Any]]) -> str:
    """Classify the failure state left by one settled step.

    A failing test run is a failure even though the tool itself succeeded (a
    shell command that reports `1 failed` exits with a result the harness does
    not flag), so every result is classified, not only the flagged ones.

    @param results - settled calls of the step, each with `name`, `isError`, and `error`.
    @returns one of {@link ERROR_KINDS}.
    """
    kind = 'none'
    for result in results:
        candidate = _error_kind_of(result)
        # The first failure is the one the next step reacts to.
        if candidate != 'none' and kind == 'none':
            kind = candidate
    return kind


def _error_kind_of(result: Mapping[str, Any]) -> str:
    """Classify one settled tool result.

    Failure codes are authoritative when present; result text, then the raw
    JSON arguments, back them up, because a shell tool reports most failures
    only through its exit status and output. The text forms are ordered most
    specific first, so a collection error is never absorbed by the generic
    test-failure verdict it also contains.
    """
    error = result.get('error')
    code = ''
    name = ''
    if isinstance(error, Mapping):
        code = str(error.get('code', '')).upper()
        name = str(error.get('name', '')).upper()
    text = ' '.join(str(result.get('text', '')).split())
    tail = text[-_TEXT_TAIL:]
    arguments = result.get('arguments')
    command = ''
    if isinstance(arguments, Mapping):
        command = str(arguments.get('command', ''))
    haystack = f'{tail}\n{command}'.lower()
    failed = bool(result.get('isError'))

    if 'TEST_FAILED' in code or re.search(r'\b\d+ failed\b|\bfailed\b.*\bpassed\b', haystack):
        # A runner that never collected the tests reports a failure form the
        # next step repairs differently.
        if _COLLECTION_ERROR_TEXT.search(tail):
            return 'test_collection_error'
        if _IMPORT_ERROR_TEXT.search(tail):
            return 'import_error'
        return 'test_failure'
    if 'BUILD_FAILED' in code or _BUILD_FAILURE_TEXT.search(haystack):
        return 'build_failure'
    if _COLLECTION_ERROR_TEXT.search(tail):
        return 'test_collection_error'
    if _IMPORT_ERROR_TEXT.search(tail) and failed:
        return 'import_error'
    if 'TOOL_NOT_FOUND' in code or re.search(r'unknown tool|no such tool', haystack):
        return 'tool_not_found'
    if _SYNTAX_ERROR_TEXT.search(tail):
        return 'syntax_error'
    if 'TIMEOUT' in code or 'timed out' in haystack:
        return 'timeout'
    if 'PERMISSION' in code or 'permission denied' in haystack or 'access denied' in haystack:
        return 'permission_denied'
    if 'ABORT' in code or name == 'ABORTERROR':
        return 'aborted'
    if _DEPENDENCY_ERROR_TEXT.search(tail):
        return 'dependency_error'
    if _FILE_NOT_FOUND_TEXT.search(tail):
        return 'file_not_found'
    if not failed:
        # A successful call that reported no failure contributes nothing; a
        # warning is the one thing a successful call can still report.
        return 'warning' if _WARNING_TEXT.search(tail) else 'none'
    if re.search(r'\bnot found\b', haystack):
        return 'tool_not_found'
    if _NONZERO_EXIT_TEXT.search(haystack):
        return 'command_failed'
    return 'tool_error'


def test_outcome(results: Iterable[Mapping[str, Any]]) -> tuple[bool, bool]:
    """Read the test-run verdict out of a settled step's results.

    @param results - settled calls of the step.
    @returns `(ran, passed)`: whether a test command ran, and whether it passed.
    """
    ran = False
    failed = False
    for result in results:
        if action_category(str(result.get('name', '')), result.get('arguments')) != 'run_test':
            continue
        ran = True
        text = str(result.get('text', ''))
        summary = _TEST_SUMMARY.search(text)
        if result.get('isError') or (summary is not None and summary.group('failed')) or re.search(r'\bFAILED\b', text):
            failed = True
    return ran, ran and not failed


def looks_like_build_failure(result: Mapping[str, Any]) -> bool:
    """Whether one settled result reports a build, typecheck, or lint failure.

    @param result - one settled call.
    @returns `True` when the result reports a compile/type/lint failure.
    """
    if not result.get('isError'):
        return False
    text = str(result.get('text', '')).lower()
    return bool(re.search(r'error ts\d+|compilation error|cannot find module|syntax error|type error', text))
