#!/usr/bin/env python3
"""Check the repository SHA-256 manifest using the Python standard library."""
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    failed = []
    for name, digest in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failed.append(name)
    if failed:
        raise SystemExit('Missing or changed files: ' + ', '.join(failed))
    print(f'Integrity verified: {len(manifest)} files.')


if __name__ == '__main__':
    main()
