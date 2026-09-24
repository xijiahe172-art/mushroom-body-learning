"""Learning Module: experience storage and hook handling for the agent loop.

The module is optional infrastructure. Nothing in this package may raise into
the harness: every entry point reports failures on stderr and returns normally.
"""

from learning_module.hooks import handle_hook
from learning_module.modes import (
    ACTIVE,
    LEARNING_MODULE_ENV,
    OFF,
    SHADOW,
    current_mode,
)

__all__ = [
    'ACTIVE',
    'LEARNING_MODULE_ENV',
    'OFF',
    'SHADOW',
    'current_mode',
    'handle_hook',
]
