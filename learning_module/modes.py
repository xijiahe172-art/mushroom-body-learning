"""The `DSH_LEARNING_MODULE` switch and the module's shared diagnostics.

`off` is the default and the safe answer for anything unexpected: an unknown
value must never silently enable learning.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variable carrying the module mode.
LEARNING_MODULE_ENV = 'DSH_LEARNING_MODULE'

#: Environment variable overriding the SQLite file location.
LEARNING_DB_ENV = 'DSH_LEARNING_DB'

#: Mode in which the module records nothing and hooks are pure logs.
OFF = 'off'

#: Mode in which the module observes and learns but never changes agent behavior.
SHADOW = 'shadow'

#: Mode in which the module may contribute to agent decisions.
ACTIVE = 'active'

MODES = (OFF, SHADOW, ACTIVE)

#: Directory holding the optional On-Disk Learning module, under the user's home.
_DATABASE_RELATIVE_PATH = Path('.dsh') / 'learning' / 'experiences.db'


def report(message: str) -> None:
    """Write one diagnostic to stderr, never raising into the caller."""
    try:
        sys.stderr.write(f'[learning-module] {message}\n')
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 -- a broken stderr must not fail an agent turn
        pass


def current_mode() -> str:
    """Return the configured mode, defaulting to `off` for unset or unknown values."""
    raw = os.environ.get(LEARNING_MODULE_ENV)
    if raw is None or raw == '':
        return OFF
    if raw in MODES:
        return raw
    report(f'error: {LEARNING_MODULE_ENV} must be one of {"/".join(MODES)}, got {raw!r}; treating it as {OFF}')
    return OFF


def database_path() -> Path:
    """Return the SQLite file backing the experience store."""
    override = os.environ.get(LEARNING_DB_ENV)
    if override:
        return Path(override)
    return Path.home() / _DATABASE_RELATIVE_PATH
