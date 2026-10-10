"""Recover progress after a native process failure; an interrupted run is never marked successful."""
from __future__ import annotations
import json
import math
from pathlib import Path
from measurement_store import atomic_json


def recover_status(output, launcher, native_code, assembly=None):
    output = Path(output)
    status_path = output/'status.json'
    notes = []
    try:
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if not isinstance(status, dict):
            raise ValueError('status is not an object')
    except (OSError, ValueError) as exc:
        status = {}
        notes.append('Missing or unreadable terminal status: ' + str(exc))
    checkpoint = (assembly or {}).get('last_committed_status') or {}
    if not checkpoint:
        try:
            checkpoint = json.loads((output/'progress.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            checkpoint = {}
    finished = status.get('status') in ('NATIVE_RUN_COMPLETED_UNASSESSED', 'NATIVE_MISSION_INCOMPLETE',
                                       'NATIVE_MISSION_FAILED', 'PREFLIGHT_COMPLETED_NO_DYNAMIC_VALIDATION')
    if not finished and checkpoint:
        count = checkpoint.get('native_steps_completed')
        old = status.get('native_steps_completed')
        if isinstance(count, int) and (not isinstance(old, int) or count >= old):
            # Keep a genuine exception report, but use the last durable progress.
            status = {**checkpoint, **{k:v for k,v in status.items()
                      if k in ('error', 'traceback', 'finished_utc')}}
        notes.append('Latest durable checkpoint is a progress lower bound, not proof of a completed run')
    if not finished:
        status['status'] = 'NATIVE_PROCESS_INCOMPLETE'
        status['validation_passed'] = False
        status['step_count_scope'] = 'last_observed_or_committed_progress_not_verified_exit_total'
    status.setdefault('native_steps_completed', None)
    status.setdefault('simulated_duration', None)
    status.setdefault('dt', launcher.get('physics_dt'))
    status.setdefault('requested_duration', launcher.get('duration_requested_s'))
    status.setdefault('stop_when_mission_done', launcher.get('full_mission_requested', False))
    status['video_requested'] = bool(launcher.get('video_requested', False))
    status['native_process_exit_code'] = native_code
    # Surviving per-step diagnostic rows can establish a later lower bound.
    lower = None
    last_metrics = None
    try:
        with (output/'bridge_diagnostics.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # A hard crash can leave one truncated tail line.
                stamp = row.get('state_time')
                if isinstance(stamp, (int, float)) and math.isfinite(stamp):
                    lower = max(lower or 0., float(stamp))
                response = row.get('bridge_response') or {}
                if response.get('metrics'):
                    last_metrics = response['metrics']
    except OSError:
        pass
    if lower is not None:
        if not finished and status.get('native_steps_completed') == 0 and lower > 0:
            status['native_steps_completed'] = None
            status['simulated_duration'] = None
            notes.append('Discarded a zero-step placeholder contradicted by surviving native diagnostic rows')
        status['observed_native_time_lower_bound_s'] = lower
        dt = status.get('dt')
        if isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0:
            status['observed_native_steps_lower_bound'] = round(lower/dt)
    if last_metrics is not None:
        status['last_observed_mission_metrics'] = last_metrics
    if notes:
        status['recovery_notes'] = notes
    atomic_json(status_path, status)
    return status
