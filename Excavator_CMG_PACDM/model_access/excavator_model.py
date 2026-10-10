"""Locate the excavator model, which is supplied separately from this repository.

The repository contains the excavator code but no excavator model.  The model
(physical graph and source records, MuJoCo and visual descriptions with their
meshes, machine parameters and reference runs) is kept in a separate folder
and given to the commands at run time, with the option ``--model FOLDER`` or
with the environment variable ``ROBOCOMPILER_EXCAVATOR_MODEL``.  Commands that
need no model (the hash checks, the unit tests that do not use the model) run
without it.

Layout of the model folder
--------------------------
``excavator_parameters.json``
    Machine parameters read by the code: bucket, undercarriage and track
    geometry, cylinder and travel-drive hydraulics, effort ranges.
every other file
    At the path the code reads inside ``Excavator_CMG_PACDM``, for example
    ``v26/source/Excavator_RoboIR_full_body_v21/data/accepted_cmg_v04.json``.
    The preserved copies of the soil study (``pinocchio/original/<copy>/`` and
    ``isaac_sim/source/<copy>/``) read the files of the top-level study, so
    their prefix is not repeated in the model folder.

The code accepts any model with the same body, joint and port identifiers.

``python excavator_model.py FOLDER`` checks a model folder against the
``MODEL_SHA256.json`` it contains (when present) and lists missing files.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ENV = 'ROBOCOMPILER_EXCAVATOR_MODEL'
PARAMETERS = 'excavator_parameters.json'
MANIFEST = 'MODEL_SHA256.json'
PACKAGE = Path(__file__).resolve().parents[1]   # Excavator_CMG_PACDM
# Preserved copies of the soil study; they read the files of the top-level study.
COPIES = ('pinocchio/original/RoboCompiler_Excavator_Soil_v01',
          'isaac_sim/source/RoboCompiler_Excavator_Soil_v01')
SOURCE = 'v26/source/Excavator_RoboIR_full_body_v21'   # source study: data/ and assets/
HELP = (f'The excavator model is not part of this repository. Give its folder with '
        f'--model FOLDER or the environment variable {ENV}=FOLDER.')

_cache = {}


class ModelUnavailable(FileNotFoundError):
    """The excavator model folder, or a file in it, is not available."""


def _check_folder(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ModelUnavailable(f'Excavator model folder not found: {folder}. {HELP}')
    if not (folder / PARAMETERS).is_file():
        raise ModelUnavailable(f'{folder} is not an excavator model folder: {PARAMETERS} is missing. {HELP}')
    return folder


def set_root(folder):
    """Use FOLDER as the model; also exported to child processes through the environment."""
    folder = _check_folder(folder)
    os.environ[ENV] = str(folder)
    _cache.clear()
    return folder


def use(folder):
    """Apply a ``--model`` option of a command: set the model when FOLDER is given, else keep
    the environment variable. An invalid folder ends the command with a message."""
    if not folder:
        return None
    try:
        return set_root(folder)
    except ModelUnavailable as exc:
        raise SystemExit(f'error: {exc}') from None


def require():
    """For commands that need the model: its folder, or end the command with a message."""
    try:
        return root()
    except ModelUnavailable as exc:
        raise SystemExit(f'error: {exc}') from None


def add_argument(parser):
    """Add the ``--model FOLDER`` option to the argparse parser of a command.

    The command then also ends with a message, not a traceback, when the model
    or a file of it is not available."""
    parser.add_argument('--model', metavar='FOLDER', default=None,
                        help=f'excavator model folder (default: environment variable {ENV})')
    _report_missing_model()
    return parser


def _report_missing_model():
    """Print an uncaught ModelUnavailable as a one-line error (the exit status stays 1)."""
    previous = sys.excepthook
    if getattr(previous, 'excavator_model', False):
        return

    def hook(kind, value, traceback):
        if isinstance(value, ModelUnavailable):
            print(f'error: {value}', file=sys.stderr)
        else:
            previous(kind, value, traceback)
    hook.excavator_model = True
    sys.excepthook = hook


def root(required=True):
    """The model folder, or None when it is not given and not required."""
    value = os.environ.get(ENV, '').strip()
    if not value:
        if required:
            raise ModelUnavailable(HELP)
        return None
    if value not in _cache:
        _cache.clear()
        _cache[value] = _check_folder(value)
    return _cache[value]


def available():
    """True when a model folder is given and valid."""
    try:
        return root(required=False) is not None
    except ModelUnavailable:
        return False


def path(relative, must_exist=True):
    """Path of a model file given relative to the model folder."""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Model paths are relative to the model folder: {relative}')
    target = root() / relative
    if must_exist and not target.exists():
        raise ModelUnavailable(f'{relative.as_posix()} is not in the excavator model folder {root()}')
    return target


def relative(package_path):
    """Model-relative name of a file the code would read inside Excavator_CMG_PACDM."""
    try:
        rel = Path(package_path).resolve().relative_to(PACKAGE)
    except ValueError:
        raise ValueError(f'{package_path} is not inside {PACKAGE}') from None
    parts = rel.parts
    for copy in COPIES:
        prefix = Path(copy).parts
        if parts[:len(prefix)] == prefix:
            parts = parts[len(prefix):]
            break
    return Path(*parts)


def mirror(package_path, must_exist=True):
    """Model file standing in for PACKAGE_PATH, a path inside Excavator_CMG_PACDM."""
    return path(relative(package_path), must_exist)


def locate(base, value):
    """VALUE, a path relative to BASE as written in a model file, located on this machine.

    Model files refer to other files by their place in the package layout (the Isaac
    manifest, for example, names isaac_sim/source/<copy>/outputs/<case>/scene.xml).
    For a BASE inside the model folder VALUE is therefore resolved in the package
    layout; a package path that is not present is looked up in the model folder.
    """
    base = Path(base).resolve()
    folder = root(required=False) if available() else None
    if folder is not None and base.is_relative_to(folder):
        base = PACKAGE / base.relative_to(folder)
    target = (base / value).resolve()
    if not target.exists() and folder is not None and target.is_relative_to(PACKAGE):
        target = mirror(target, must_exist=False)
    return target


def source(relative_to_source, must_exist=True):
    """Model file of the source study, e.g. ``source('data/accepted_cmg_v04.json')``."""
    return path(Path(SOURCE) / relative_to_source, must_exist)


def parameters():
    """All machine parameters (the parsed ``excavator_parameters.json``; do not modify)."""
    folder = root()
    key = ('parameters', str(folder))
    if key not in _cache:
        with open(folder / PARAMETERS, encoding='utf-8') as stream:
            _cache[key] = json.load(stream)
    return _cache[key]


def parameter(*keys):
    """One entry of ``excavator_parameters.json``, e.g. ``parameter('bucket', 'lip_local_m')``."""
    value = parameters()
    for depth, key in enumerate(keys):
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            name = '/'.join(map(str, keys[:depth + 1]))
            raise ModelUnavailable(f'{PARAMETERS} in {root()} has no entry {name}') from None
    return value


def files(folder):
    """Files of a model folder, as sorted POSIX paths relative to it."""
    folder = Path(folder)
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts)


def sha256(file):
    digest = hashlib.sha256()
    with open(file, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check(folder):
    """Compare a model folder with its own MODEL_SHA256.json (if present)."""
    folder = _check_folder(folder)
    present = [name for name in files(folder) if name != MANIFEST]
    report = dict(folder=str(folder), files=len(present), manifest=(folder / MANIFEST).is_file(),
                  missing=[], mismatched=[], unlisted=[])
    if report['manifest']:
        listed = json.loads((folder / MANIFEST).read_text(encoding='utf-8'))
        report['missing'] = sorted(set(listed) - set(present))
        report['unlisted'] = sorted(set(present) - set(listed))
        report['mismatched'] = sorted(name for name in set(listed) & set(present)
                                      if sha256(folder / name) != listed[name])
    report['passed'] = not (report['missing'] or report['mismatched'])
    return report


def write_manifest(folder):
    """Write MODEL_SHA256.json for a model folder (for the owner of the model)."""
    folder = _check_folder(folder)
    listed = {name: sha256(folder / name) for name in files(folder) if name != MANIFEST}
    (folder / MANIFEST).write_text(json.dumps(listed, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return len(listed)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check an excavator model folder.')
    parser.add_argument('folder', nargs='?', help=f'model folder (default: {ENV})')
    parser.add_argument('--write-manifest', action='store_true', help=f'write {MANIFEST} for the folder')
    args = parser.parse_args()
    try:
        target = args.folder or root()
        if args.write_manifest:
            print(f'{write_manifest(target)} files listed in {MANIFEST}')
        result = check(target)
    except ModelUnavailable as exc:
        raise SystemExit(f'error: {exc}') from None
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
