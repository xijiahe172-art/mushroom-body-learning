"""Command line entry point for the Learning Module.

Invoked as a subprocess (`python cli.py <command>`); a JSON payload arrives on
stdin and a JSON reply leaves on stdout. Exit status is always 0 for a handled
command, because the harness treats a non-zero status as module failure and the
module reports its own errors in the envelope instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# `python learning_module/cli.py` puts learning_module/ rather than the
# repository root on sys.path, so the package needs the root to be importable.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from learning_module import (  # noqa: E402
    context as context_rendering,
    hooks,
    modes,
    schema,
    store,
)


def read_payload() -> dict[str, Any]:
    """Read the JSON payload from stdin, or an empty mapping when there is none."""
    raw = sys.stdin.read() if not sys.stdin.isatty() else ''
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError('hook payload must be a JSON object')
    return parsed


def command_hook() -> dict[str, Any]:
    """Handle one hook payload."""
    try:
        payload = read_payload()
    except Exception as error:  # noqa: BLE001 -- malformed stdin is a reported failure, not a crash
        modes.report(f'error: unreadable hook payload: {error!r}')
        return {'ok': False, 'mode': modes.current_mode(), 'error': str(error)}
    return hooks.handle_hook(payload)


def command_init_db() -> dict[str, Any]:
    """Create the experience store and report where it landed."""
    path = schema.ensure_schema(modes.database_path())
    modes.report(f'schema ready at {path}')
    return {'ok': True, 'mode': modes.current_mode(), 'database': os.fspath(path)}


def command_context() -> dict[str, Any]:
    """Render the experience text the harness may inject for one state."""
    if modes.current_mode() != modes.ACTIVE:
        # Only active mode may influence a decision; the other modes answer with
        # no text so a caller cannot accidentally inject one.
        return {'ok': True, 'mode': modes.current_mode(), 'text': None}
    try:
        payload = read_payload()
    except Exception as error:  # noqa: BLE001 -- malformed stdin is a reported failure, not a crash
        modes.report(f'error: unreadable context payload: {error!r}')
        return {'ok': False, 'mode': modes.current_mode(), 'text': None, 'error': str(error)}
    return {'ok': True, 'mode': modes.current_mode(), 'text': hooks.get_experience_context(payload)}


def command_config() -> dict[str, Any]:
    """Report the resolved switch, store location, and context budget."""
    return {
        'ok': True,
        'mode': modes.current_mode(),
        'database': os.fspath(modes.database_path()),
        'hooks': list(hooks.HOOK_NAMES),
        'context_budget': {
            'max_entries': context_rendering.MAX_ENTRIES,
            'entry_char_cap': context_rendering.ENTRY_CHAR_CAP,
            'total_char_cap': context_rendering.TOTAL_CHAR_CAP,
            'pricing': 'TokenMeter 4 chars/token heuristic; not the official tokenizer',
        },
    }


def command_flagged(argv: list[str] | None = None) -> dict[str, Any]:
    """List the rows Phase 4 flagged for the manual spot check.

    @param argv - the command's own arguments, without the command name.
    @returns the flagged rows with the reason each was flagged.
    """
    parser = argparse.ArgumentParser(prog='learning_module flagged')
    parser.add_argument('--flag', default=None, help='restrict to one flag name')
    parser.add_argument('--limit', type=int, default=100, help='maximum rows to report')
    args = parser.parse_args(argv)
    rows = store.flagged_experiences(flag=args.flag, limit=args.limit)
    return {
        'ok': True,
        'mode': modes.current_mode(),
        'database': os.fspath(modes.database_path()),
        'flags': list(store.FLAG_REASONS),
        'flagged': [
            {
                'id': row['id'],
                'flag': row['flagged_reason'],
                'reason': row['reason'],
                'state_error_type': row['state_error_type'],
                'state_tool_context': row['state_tool_context'],
                'action_category': row['action_category'],
                'action_detail': row['action_detail'],
                'reward': row['reward'],
                'average_reward': row['average_reward'],
                'value': row['value'],
                # The discounted observation is only visible in the sequence once
                # a later observation has overwritten `reward`, so the reviewer
                # gets the whole recorded history.
                'rewards': row['rewards'],
                'total_count': row['total_count'],
                'reward_source': row['reward_source'],
                'last_used_at': row['last_used_at'],
            }
            for row in rows
        ],
    }


#: Commands that read a JSON payload on stdin, and the flag `flagged` reads its
#: options from argv instead.
COMMANDS: dict[str, Any] = {
    'hook': command_hook,
    'context': command_context,
    'init-db': command_init_db,
    'config': command_config,
}


def main(argv: list[str] | None = None) -> int:
    """Run one CLI command and print its envelope."""
    parser = argparse.ArgumentParser(prog='learning_module', description='Learning Module CLI')
    parser.add_argument('command', choices=sorted([*COMMANDS, 'flagged']))
    args, rest = parser.parse_known_args(argv)
    try:
        reply = command_flagged(rest) if args.command == 'flagged' else COMMANDS[args.command]()
    except Exception as error:  # noqa: BLE001 -- the CLI never fails the harness that called it
        modes.report(f'error: {args.command} failed: {error!r}')
        reply = {'ok': False, 'mode': modes.current_mode(), 'error': str(error)}
    sys.stdout.write(json.dumps(reply) + '\n')
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
