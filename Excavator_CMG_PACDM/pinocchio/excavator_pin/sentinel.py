"""Fail-closed MuJoCo sentinel for the Pinocchio-only backend.

Several preserved source modules contain a module-level ``import mujoco`` even
when the functions used here never call MuJoCo.  The validated environment does
not install MuJoCo.  Instead, this sentinel satisfies those module-level
imports and raises immediately if any MuJoCo attribute is requested.  Because the
sentinel is not a package, ``import mujoco.<submodule>`` fails at once; a
meta-path finder that rejects every ``mujoco.*`` name is kept as a second guard.
The runner records the number of attempted accesses; a single access makes the
validation fail.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types


class MuJoCoAccessError(RuntimeError):
    """Raised whenever the Pinocchio backend requests a MuJoCo facility."""


class _Sentinel(types.ModuleType):
    accesses: list[str] = []

    def __getattr__(self, name):
        # Python's import machinery may probe dunder attributes; they are not
        # MuJoCo functionality and must not be recorded as physics use.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        _Sentinel.accesses.append(name)
        raise MuJoCoAccessError(f'MuJoCo attribute {name!r} requested inside the Pinocchio-only backend')


class _BlockMuJoCoSubmodules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'mujoco' and fullname != 'mujoco':
            raise ImportError('MuJoCo submodules are blocked in the Pinocchio-only backend')
        return None


_INSTALLED = False


def install():
    """Install the sentinel before any preserved source module is imported."""
    global _INSTALLED
    existing = sys.modules.get('mujoco')
    if existing is not None and not isinstance(existing, _Sentinel):
        raise RuntimeError('A real MuJoCo module was imported before the Pinocchio backend started')
    if not _INSTALLED:
        sys.meta_path.insert(0, _BlockMuJoCoSubmodules())
        sys.modules['mujoco'] = _Sentinel('mujoco')
        _INSTALLED = True
    return sys.modules['mujoco']


def report():
    module = sys.modules.get('mujoco')
    real_spec = None
    try:
        # find_spec on the top-level name bypasses sys.modules only if absent;
        # query the path finders directly for an installed distribution.
        for finder in sys.meta_path:
            if isinstance(finder, _BlockMuJoCoSubmodules):
                continue
            spec = getattr(finder, 'find_spec', lambda *a, **k: None)('mujoco', None)
            if spec is not None:
                real_spec = spec.origin
                break
    except Exception:  # pragma: no cover - diagnostic only
        real_spec = 'unknown'
    return dict(sentinel_installed=isinstance(module, _Sentinel),
                attempted_mujoco_attribute_accesses=list(_Sentinel.accesses),
                attempted_access_count=len(_Sentinel.accesses),
                real_mujoco_distribution_found_on_path=real_spec,
                real_mujoco_submodules_loaded=sorted(k for k in sys.modules
                                                      if k.startswith('mujoco.')))
