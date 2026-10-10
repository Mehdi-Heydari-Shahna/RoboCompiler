"""Parent-observed process and native-log evidence, with no Isaac imports.

case.json is written BEFORE SimulationApp.close. Its execution_exit_code is
not the OS exit status. Neither completed=True nor an encoded video can
cancel a failed process, a missing observation, or a logged native error.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

_ERROR = re.compile(r'\[(?:error|fatal)\]|Windows fatal exception|Fatal Python error|'
                    r'^Traceback \(most recent call last\):|Segmentation fault', re.I)
_WARNING = re.compile(r'Mixing USD builds|A different OpenUSD|ABI/singleton', re.I)


def json_record(path):
    try:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def exit_code_text(code):
    if type(code) is not int:
        return repr(code)
    return f'{code} (0x{code & 0xffffffff:08X})' if code > 255 or code < -255 else str(code)


def inspect_native_log(path, *, max_issues=50):
    """Scan the *entire* log but bound stored excerpts. Warnings aren't errors."""
    path = Path(path)
    issues, warnings, count, lines, size = [], [], 0, 0, 0
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for lines, raw in enumerate(stream, 1):
                digest.update(raw)
                size += len(raw)
                text = raw.decode('utf-8', errors='replace').rstrip()
                if _ERROR.search(text):
                    count += 1
                    category = ('renderer' if 'render' in text.lower() or 'replicator' in text.lower()
                                else 'native_crash' if 'fatal' in text.lower() or 'segmentation' in text.lower()
                                else 'runtime_error')
                    if len(issues) < max_issues:
                        issues.append(dict(line=lines, category=category, text=text[:2000]))
                elif _WARNING.search(text) and len(warnings) < 10:
                    warnings.append(dict(line=lines, text=text[:2000]))
    except OSError as exc:
        return dict(passed=False, available=False, error=str(exc), issues=[], issue_count=0,
                    warnings=[], sha256=None, path=str(path))
    return dict(passed=size > 0 and count == 0, available=size > 0, lines=lines,
                issues=issues, issue_count=count, omitted_issues=max(0, count-len(issues)),
                warnings=warnings, sha256=digest.hexdigest(), path=str(path))


def log_errors(log):
    if not log['available']:
        return ['No readable, nonempty native process log']
    result = [f"Native log line {item['line']}: {item['text']}" for item in log['issues']]
    if log.get('omitted_issues'):
        result.append(f"{log['omitted_issues']} additional native errors in log")
    return result


def _finished_after(entry, metadata):
    try:
        start = datetime.fromisoformat(str(metadata['started_utc']).replace('Z', '+00:00'))
        before_close = datetime.fromisoformat(str(metadata['finished_utc']).replace('Z', '+00:00'))
        finish = datetime.fromisoformat(str(entry['finished_utc']).replace('Z', '+00:00'))
        return all(t.tzinfo is not None for t in (start, before_close, finish)) and finish >= before_close >= start
    except (KeyError, TypeError, ValueError):
        return False


def case_process_evidence(root, case, metadata):
    root = Path(root)
    manifest = json_record(root/'suite_manifest.json')
    raw_entries = manifest.get('cases', [])
    entries = [item for item in raw_entries if isinstance(item, dict) and item.get('case') == case] if isinstance(raw_entries, list) else []
    entry = entries[0] if len(entries) == 1 else {}
    requested = manifest.get('requested_cases', [])
    native_log = inspect_native_log(root/'logs'/f'{case}.log')
    checks = {
        'parent_manifest_present': bool(manifest),
        'parent_run_identity_matches': bool(metadata.get('run_id')) and manifest.get('run_id') == metadata.get('run_id'),
        'case_requested_by_parent': isinstance(requested, list) and case in requested,
        'parent_duration_matches': (type(manifest.get('duration_s')) in (int, float)
                                    and manifest['duration_s'] == metadata.get('duration_s')),
        'one_parent_case_observation': len(entries) == 1,
        'parent_case_completed': entry.get('status') == 'completed',
        'observed_process_exit_zero': type(entry.get('process_exit_code')) is int and entry['process_exit_code'] == 0,
        'effective_exit_zero': type(entry.get('exit_code')) is int and entry['exit_code'] == 0,
        'parent_execution_errors_empty': isinstance(entry.get('execution_errors'), list) and not entry['execution_errors'],
        'parent_observation_after_execution': _finished_after(entry, metadata),
        'preclose_python_exit_zero': type(metadata.get('execution_exit_code')) is int and metadata['execution_exit_code'] == 0,
        'native_log_present': native_log['available'],
        'native_log_error_free': native_log['passed'],
    }
    # New records explicitly acknowledge application-owned cleanup. Old native
    # records remain inspectable, but never bypass the parent/log checks.
    if metadata.get('schema_version', 1) == 2 or 'resource_cleanup' in metadata:
        cleanup = metadata.get('resource_cleanup', {})
        if not isinstance(cleanup, dict):
            cleanup = {}
        steps = cleanup.get('steps', [])
        checks['application_resources_released'] = (cleanup.get('passed') is True
            and isinstance(steps, list) and bool(steps)
            and all(isinstance(step, dict) and step.get('passed') is True for step in steps)
            and cleanup.get('errors') == [])
    shutdown = metadata.get('application_shutdown', {})
    if isinstance(shutdown, dict) and shutdown.get('state') == 'exception':
        checks['no_python_shutdown_exception'] = False
    errors = [name for name, passed in checks.items() if not passed]
    if type(entry.get('process_exit_code')) is int and entry['process_exit_code'] != 0:
        errors.append('Observed process exit: ' + exit_code_text(entry['process_exit_code']))
    errors.extend(log_errors(native_log))
    return dict(passed=all(checks.values()), checks=checks, errors=errors, log=native_log,
                process_exit_code=entry.get('process_exit_code'),
                process_exit_description=exit_code_text(entry.get('process_exit_code')),
                parent_case_status=entry.get('status'),
                scope='Parent observation after native shutdown; pre-close JSON alone is insufficient.')
