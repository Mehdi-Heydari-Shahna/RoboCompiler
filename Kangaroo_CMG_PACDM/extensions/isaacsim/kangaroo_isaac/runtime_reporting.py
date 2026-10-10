"""Keep current physics failures separate from compatibility and historical logs."""
from __future__ import annotations
import json
from pathlib import Path
import re

_TGS = re.compile(
    r'Detected an articulation at /World/Kangaroo/Bodies/base_link with more than 4 velocity '
    r'iterations being added to a TGS scene\.\s*The related behavior changed recently, '
    r'please consult the changelog\. This warning will only print once\.')


def classify_physics_log(text: str, *, high_tgs_iterations: bool = True) -> dict:
    result = {'failures': [], 'compatibility_notices': [], 'historical_crash_notices': []}
    for line in text.splitlines():
        match = re.search(r'\[(Warning|Error|Fatal)\]\s*\[([^]]+)\]\s*(.*)$', line)
        if not match:
            continue
        severity, plugin, message = match.groups()
        if plugin == 'carb.crashreporter-breakpad.plugin' and message.startswith('[previous crash]'):
            result['historical_crash_notices'].append(line)
            continue
        if not plugin.startswith(('omni.physx', 'omni.physics', 'omni.usdphysics')):
            continue
        if (severity == 'Warning' and plugin == 'omni.physx.plugin' and high_tgs_iterations
                and _TGS.fullmatch(message)):
            result['compatibility_notices'].append(line)
        else:
            result['failures'].append(line)
    return result


def make_failure_report(folder, case: str, exit_code: int) -> None:
    folder = Path(folder)
    error = {}
    for name in ('error.json', 'supervisor_error.json'):
        path = folder/name
        if path.exists():
            error.update(json.loads(path.read_text(encoding='utf-8')))
    phase = error.get('failed_phase', error.get('status', 'NO_COMPLETED_NATIVE_RESULT'))
    message = error.get('error', 'The native process did not commit a completed result. Inspect console.log.')
    lines = [f'# Native case: {case}', '', '**NOT COMPLETED — no native validation pass.**', '',
             f'Process exit code: `{exit_code}`.', f'Last recorded phase: `{phase}`.', '',
             '## Failure', '', str(message), '',
             'The configuration, controller, and task have not been validated by this failed run.',
             'See `startup_diagnostics.json`, `error.json`, and `console.log` when present.', '',
             'The separate offline audit tests source mathematics and software, not native stability or speed.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def authoritative_exit_code(result: dict, raw_code: int) -> int:
    """A process exiting normally is not evidence that the task passed.

    Some Kit shutdown paths return OS code 0 despite a failed validation result.
    Keep the raw code separately and fail closed on missing or conflicting data.
    Full source certification remains separate from this functional exit code.
    """
    if raw_code not in (0,3):return abs(raw_code) or 2
    if result.get('engine')!='Isaac Sim / PhysX' or result.get('completed') is not True:return 2
    validation=result.get('validation',{})
    status=validation.get('functional_status')
    if result.get('physics_log',{}).get('failures'):return 3
    if status not in ('FUNCTIONAL_GATES_PASSED','EXPECTED_FAILURE_OBSERVED'):return 3
    gates=validation.get('gates',[])
    if not gates or any(g.get('status')!='PASS' for g in gates):return 3
    # Retain nonzero even if a nominally passing report conflicts with raw code 3.
    return raw_code
