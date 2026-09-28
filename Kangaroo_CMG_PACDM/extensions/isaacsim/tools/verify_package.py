#!/usr/bin/env python
"""Verify delivered file bytes with the standard library only."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text(encoding='utf-8'))
    failures=[]
    for name,digest in manifest['files'].items():
        f=ROOT/name
        if not f.is_file():failures.append('MISSING '+name);continue
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=digest:failures.append('CHANGED '+name)
    for message in failures:print(message)
    print(f"{len(manifest['files'])} delivered files checked; {len(failures)} mismatches.")
    print('Integrity check only: this is not Isaac Sim performance validation.')
    return 1 if failures else 0

if __name__=='__main__':raise SystemExit(main())
