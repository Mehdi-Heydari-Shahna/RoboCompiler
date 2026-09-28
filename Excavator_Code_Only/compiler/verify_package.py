#!/usr/bin/env python
"""Check the delivered package against PACKAGE_MANIFEST.json (SHA-256 of every file).

    python verify_package.py            # exit code 0 = every listed file present and unchanged

Files created later by the user (virtual environments, new result directories,
__pycache__) are reported as extra files but do not fail the check.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
IGNORED = ('__pycache__', '.venv', '.pytest_cache')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    files = manifest['files']
    missing = [k for k in files if not (ROOT / k).is_file()]
    changed = [k for k in files if (ROOT / k).is_file() and sha(ROOT / k) != files[k]]
    present = {str(f.relative_to(ROOT)).replace('\\', '/') for f in ROOT.rglob('*')
               if f.is_file() and not any(part in IGNORED for part in f.parts)}
    extra = sorted(present - set(files) - {'PACKAGE_MANIFEST.json'})
    print(json.dumps(dict(listed=len(files), missing=missing, changed=changed, extra_files=len(extra),
                          extra_examples=extra[:10]), indent=2))
    ok = not missing and not changed
    print('PACKAGE OK' if ok else 'PACKAGE CHECK FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
