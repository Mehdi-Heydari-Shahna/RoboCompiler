"""Standard-library-only file writing that tolerates transient Windows locks.

OneDrive, Windows Search and antivirus scanners can briefly hold a handle on a
file that was just written. os.replace() then raises PermissionError
(WinError 5/32). A previous suite lost the complete wrench/initial_offset case
to exactly this while updating last_checkpoint.json. These helpers retry the
atomic replace and, if the lock persists, fall back to a direct overwrite.
They import nothing from Isaac, USD or NumPy at module level.
"""
from pathlib import Path
import json
import os
import sys
import time

_RETRIES = 40


def _replace_with_retry(tmp, path, retries=_RETRIES):
    last = None
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return None
        except OSError as exc:  # PermissionError is a subclass
            last = exc
            time.sleep(min(.02 * (attempt + 1), .25))
    return last


def write_text(path, text, required=True):
    """Write text atomically when possible; never leave a stale *.tmp behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    error = None
    try:
        tmp.write_text(text, encoding='utf-8')
        error = _replace_with_retry(tmp, path)
    except OSError as exc:
        error = exc
    if error is None:
        return True
    # Last resort for a persistent lock on the destination: overwrite in place.
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        error = None
    except OSError as exc:
        error = exc
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    if error is None:
        return True
    if required:
        raise error
    print(f'[PANDA] WARNING: could not write {path}: {error!r}', file=sys.stderr, flush=True)
    return False


def write_json(path, value, required=True):
    return write_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n', required)


def save_npz(path, arrays, required=True):
    """np.savez_compressed through a temporary file and a retried replace."""
    import numpy as np  # local import keeps this module stdlib-only at import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    error = None
    try:
        with open(tmp, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
        error = _replace_with_retry(tmp, path)
    except OSError as exc:
        error = exc
    if error is not None:
        try:
            with open(path, 'wb') as handle:
                np.savez_compressed(handle, **arrays)
            error = None
        except OSError as exc:
            error = exc
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    if error is None:
        return True
    if required:
        raise error
    print(f'[PANDA] WARNING: could not write {path}: {error!r}', file=sys.stderr, flush=True)
    return False


def remove(path):
    """Best-effort delete (a locked stale partial file must not fail a completed run)."""
    path = Path(path)
    for attempt in range(_RETRIES):
        try:
            if path.exists():
                path.unlink()
            return True
        except OSError:
            time.sleep(min(.02 * (attempt + 1), .25))
    print(f'[PANDA] WARNING: could not remove {path}', file=sys.stderr, flush=True)
    return False
