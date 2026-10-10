"""Check the shipped file hashes without importing Isaac or any dependency."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys


def verify(root: Path) -> list[str]:
    root = Path(root).resolve()
    manifest = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Package hash manifest is empty or invalid')
    problems = []
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Invalid path in package hash manifest')
        if not path.is_file():
            problems.append(f'Missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f'Changed: {name}')
    return problems


def main() -> int:
    try:
        problems = verify(Path(__file__).resolve().parent)
    except (OSError, ValueError, TypeError) as exc:
        print(f'Package integrity FAILED: {exc}', file=sys.stderr)
        return 1
    if problems:
        print('Package integrity FAILED. Extract a fresh v4 folder without mixing versions.')
        print('\n'.join(problems))
        return 1
    print('Package integrity PASS. This is a file check, not a simulation result.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
