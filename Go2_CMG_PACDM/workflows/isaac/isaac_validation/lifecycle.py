"""Deterministic application-owned cleanup, independent of the Isaac runtime."""
from __future__ import annotations
import sys


def cleanup_actions(actions):
    """Attempt every named cleanup, even when an earlier cleanup fails.

    Returns durable diagnostics; the caller preserves any original exception
    and otherwise raises when cleanup failed. No failure becomes a success.
    """
    steps, errors = [], []
    for name, action in actions:
        try:
            action()
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            steps.append(dict(name=name, passed=False, error=message))
            print(f"[Error] [validation.cleanup] {message}", file=sys.stderr, flush=True)
        else:
            steps.append(dict(name=name, passed=True))
    return dict(passed=not errors, steps=steps, errors=errors)
