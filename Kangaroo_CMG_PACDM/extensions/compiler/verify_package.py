#!/usr/bin/env python
"""Verify every file against PACKAGE_MANIFEST.json (standard library only).

Usage:  python verify_package.py
"""
import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    bad = [name for name, digest in manifest['files'].items()
           if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    listed = set(manifest['files']) | {'PACKAGE_MANIFEST.json'}
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                   if p.is_file() and p.relative_to(root).as_posix() not in listed
                   and '__pycache__' not in p.parts)
    print(json.dumps({'passed': not bad, 'files': len(manifest['files']), 'mismatched_or_missing': bad,
                      'unlisted_files_present': extra[:20], 'unlisted_count': len(extra)}, indent=2))
    return 0 if not bad else 2


if __name__ == '__main__':
    sys.exit(main())
