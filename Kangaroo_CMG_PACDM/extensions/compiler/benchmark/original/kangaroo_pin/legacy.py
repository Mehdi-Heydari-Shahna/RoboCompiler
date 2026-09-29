"""MuJoCo-free access to the accepted Kangaroo v22 definitions.

The original v22 modules import MuJoCo at module level because they also
contain native MuJoCo adapters.  The Pinocchio backend needs only their
NumPy/PACDM definitions (graph adapter, reconstructed CMG builder, source
dynamics and the contact reference).  This loader parses the unchanged files
in ``original_v22/`` and executes exactly the named top-level definitions, in
their original text, inside a namespace that supplies the same imports minus
MuJoCo.  Nothing is copied or edited: every executed segment is a verbatim
slice of an original v22 file, and its SHA-256 is recorded.

Pure modules without a MuJoCo import (``pacdm``, ``source_dynamics``,
``source_bias``, ``constraint_solvers``, ``import_full_model``) are imported
normally from ``original_v22/``.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'original_v22'

# Modules whose import would pull in MuJoCo; their needed names are served
# from the namespaces built below instead.
PROBLEM_MODULES = {'mujoco', 'mechanism_backend', 'reconstructed_model',
                   'whole_body_dynamics', 'contact_reference', 'native_model',
                   'whole_body_motion', 'contact_task'}

PLAN = [
    ('mechanism_backend.py', ['array', 'fmt', 'Mechanism']),
    ('reconstructed_model.py', ['make', 'UniversalMechanism', 'CutGraph',
                                'whole_body', 'polish']),
    ('whole_body_dynamics.py', ['FloatingSource', 'PinFloating', 'add_drive_terms']),
    ('contact_task.py', ['DEFAULTS', 'HISTORY_COLUMNS', 'external_push']),
    ('contact_reference.py', ['DURATION', 'FOOT_NAMES', 'smooth', 'pelvis_target',
                              'foot_corners', 'build_reference']),
]

_CACHE = {}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _node_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _ensure_path():
    text = str(SOURCE)
    if text not in sys.path:
        sys.path.insert(0, text)


def load(reference_root: Path | None = None):
    """Return a namespace object holding the accepted definitions.

    ``reference_root`` replaces ``contact_reference.ROOT`` so a regenerated
    reference is written to a caller-chosen folder rather than the original
    package.  All other module-level values keep their original meaning.
    """
    key = str(reference_root) if reference_root is not None else None
    if key in _CACHE:
        return _CACHE[key]
    _ensure_path()
    served = {}
    record = {}
    namespaces = {}
    for filename, names in PLAN:
        path = SOURCE / filename
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        tree = ast.parse(text, filename=str(path))
        ns = {'__builtins__': builtins, '__name__': 'kangaroo_pin.accepted.' + path.stem,
              '__file__': str(path)}
        # Replay the file's own import statements, except MuJoCo-bearing modules.
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split('.')[0]
                    if base in PROBLEM_MODULES:
                        continue
                    module = __import__(alias.name)
                    ns[alias.asname or base] = (module if alias.asname is None
                                                else sys.modules[alias.name])
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ''
                if module_name.split('.')[0] in PROBLEM_MODULES:
                    for alias in node.names:
                        if alias.name in served:
                            ns[alias.asname or alias.name] = served[alias.name]
                    continue
                module = __import__(module_name, fromlist=[a.name for a in node.names])
                for alias in node.names:
                    ns[alias.asname or alias.name] = getattr(module, alias.name)
        # Module-level constants referenced by the extracted definitions.
        if filename == 'reconstructed_model.py':
            ns['BASE'] = SOURCE
        if filename == 'contact_reference.py':
            ns['ROOT'] = Path(reference_root) if reference_root is not None else SOURCE
        if filename == 'contact_task.py':
            ns['ROOT'] = SOURCE
        found = {}
        for node in tree.body:
            name = _node_name(node)
            if name in names:
                segment = ast.get_source_segment(text, node)
                if segment is None or segment not in text:
                    raise RuntimeError(f'Could not extract verbatim {filename}:{name}')
                exec(compile(segment, f'{path}:{name}', 'exec'), ns)
                found[name] = dict(first_line=node.lineno, last_line=node.end_lineno,
                                   segment_sha256=_sha(segment.encode('utf-8')))
        missing = set(names) - set(found)
        if missing:
            raise RuntimeError(f'{filename} lacks {sorted(missing)}')
        for name in names:
            served[name] = ns[name]
        namespaces[filename] = ns
        record[filename] = dict(file_sha256=_sha(raw), extracted=found)
    import pacdm
    import source_dynamics
    import source_bias
    import constraint_solvers
    import import_full_model
    pure = {}
    for module in (pacdm, source_dynamics, source_bias, constraint_solvers, import_full_model):
        file = Path(module.__file__).resolve()
        if file.parent != SOURCE.resolve():
            raise RuntimeError(f'{module.__name__} was not imported from original_v22')
        pure[file.name] = _sha(file.read_bytes())
    if 'mujoco' in sys.modules:
        raise RuntimeError('MuJoCo was imported while loading the accepted definitions')

    class Accepted:
        pass

    accepted = Accepted()
    for name, value in served.items():
        setattr(accepted, name, value)
    accepted.PACDM = pacdm.PACDM
    accepted.PointGraph = pacdm.PointGraph
    accepted.pacdm = pacdm
    accepted.SourceBiasDynamics = source_bias.SourceBiasDynamics
    accepted.SourceDynamics = source_dynamics.SourceDynamics
    accepted.reduce_kinematics = constraint_solvers.reduce_kinematics
    accepted.solve_reduced = constraint_solvers.solve_reduced
    accepted.solve_kkt = constraint_solvers.solve_kkt
    accepted.UP = import_full_model.UP
    accepted.record = dict(extracted=record, imported_whole=pure)
    accepted.namespaces = namespaces
    _CACHE[key] = accepted
    return accepted


def provenance_record():
    """Hashes of every accepted file/segment used, for the evidence bundle."""
    return load().record
