"""Standard-library, source-checked updater used by the standalone 23.0.3 patch.

The generated standalone file embeds the payload; this module alone does not
change any project. Results, environments, and other projects are never edited.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import uuid
import zlib


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe patch path: {relative}')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root) or target.is_symlink():
        raise ValueError(f'Patch path escapes the project or is a symlink: {relative}')
    return target


def inspect_patch(root: Path, payload: dict) -> tuple[dict[str, bytes], list[str]]:
    root = root.resolve()
    if not (root/'run_isaac.py').is_file() or not (root/'data/whole_body_cmg.json').is_file():
        raise ValueError('This is not the Kangaroo project folder. Use --project "path to Kangaroo_IsaacSim_CMG_Port".')
    replacements = {}
    for name, encoded in payload['files'].items():
        data = base64.b64decode(encoded, validate=True)
        if digest(data) != payload['target_hashes'][name]:
            raise ValueError(f'Patch payload integrity failure: {name}')
        replacements[name] = data
    errors = []
    for name in set(payload['base_hashes']) | set(replacements):
        target = safe_path(root, name)
        current = digest(target.read_bytes()) if target.is_file() else None
        allowed = {payload['base_hashes'].get(name)}
        if name in replacements:
            allowed.add(payload['target_hashes'][name])
        if current not in allowed:
            errors.append(name)
    if errors:
        raise ValueError('Project files differ from the 23.0.2/23.0.3 release. '
                         'No files were changed. Preserve custom edits and use the full ZIP instead.\n' + '\n'.join(sorted(errors)))
    changed = [n for n,d in replacements.items() if not safe_path(root,n).is_file()
               or safe_path(root,n).read_bytes()!=d]
    return replacements, sorted(changed)


def apply_patch(root: Path, payload: dict, *, dry_run: bool = False) -> dict:
    root = root.resolve()
    replacements, changed = inspect_patch(root, payload)
    if dry_run or not changed:
        return {'status':'DRY_RUN' if dry_run else 'ALREADY_INSTALLED', 'changed_files':changed, 'backup':None}
    lock = root/'.kangaroo_patch.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError('Another update may be running (.kangaroo_patch.lock exists); no files were changed.')
    os.close(fd)
    backup = root/'_patch_backups'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    originals = {}; written = []
    try:
        backup.mkdir(parents=True, exist_ok=False)
        for name in changed:
            target = safe_path(root,name)
            originals[name] = target.read_bytes() if target.exists() else None
            if originals[name] is not None:
                saved = backup/name; saved.parent.mkdir(parents=True,exist_ok=True)
                saved.write_bytes(originals[name])
        # Do not overwrite a concurrent editor's changes after the first audit.
        inspect_patch(root,payload)
        for name in changed:
            target = safe_path(root,name); target.parent.mkdir(parents=True,exist_ok=True)
            temp = target.with_name(target.name+'.patch-'+uuid.uuid4().hex)
            try:
                temp.write_bytes(replacements[name]); os.replace(temp,target)
            finally:
                if temp.exists():temp.unlink()
            written.append(name)
        (backup/'patch_record.json').write_text(json.dumps({'version':payload['version'],
            'changed_files':changed,'new_files':[n for n,d in originals.items() if d is None]},indent=2))
        return {'status':'INSTALLED','changed_files':changed,'backup':str(backup)}
    except BaseException:
        for name in reversed(written):
            target=safe_path(root,name)
            if originals[name] is None: target.unlink(missing_ok=True)
            else: target.write_bytes(originals[name])
        raise
    finally:
        lock.unlink(missing_ok=True)


def patch_main(payload: dict, argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='Install the source-checked Kangaroo 23.0.3 solver/reporting repair')
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true',help='Check files without changing anything')
    parser.add_argument('--run',action='store_true',help='After updating and offline tests, run full nominal and guarded convergence checks')
    args=parser.parse_args(argv)
    if args.dry_run and args.run:parser.error('--dry-run and --run cannot be combined')
    root=args.project.resolve()
    try:
        result=apply_patch(root,payload,dry_run=args.dry_run)
    except (OSError,ValueError) as exc:
        print('UPDATE STOPPED:',exc,file=sys.stderr);return 2
    print(f'Kangaroo {payload["version"]}: {result["status"]}',flush=True)
    print('Project:',root,flush=True)
    print('Files:',len(result['changed_files']),flush=True)
    if result['backup']:print('Previous files backed up to:',result['backup'],flush=True)
    print('No packages were installed or replaced. Existing results were preserved.',flush=True)
    if args.dry_run:return 0
    # Verify the complete project, not just the patched files.
    code=subprocess.call([sys.executable,str(root/'tools/verify_package.py')],cwd=root)
    if code:return code
    if not args.run:
        print('Run: python run_isaac.py --verify-native --headless --no-visuals',flush=True)
        return 0
    print('Running offline checks; a failed check prevents the native launch.',flush=True)
    code=subprocess.call([sys.executable,str(root/'run_isaac.py'),'--offline-test'],cwd=root)
    if code:return code
    print('Starting full native verification. Any failed nominal gate stops the additional runs.',flush=True)
    return subprocess.call([sys.executable,str(root/'run_isaac.py'),'--verify-native','--headless','--no-visuals'],cwd=root)
