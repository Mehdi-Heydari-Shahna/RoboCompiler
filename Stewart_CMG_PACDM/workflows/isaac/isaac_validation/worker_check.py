"""A successful OS exit does not prove a successful Isaac worker."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def check_worker(output,name,run_id,exit_code,duration,dt):
    output=Path(output); failures=[]; result=dict(case=name,exit_code=exit_code,passed=False)
    if exit_code!=0:failures.append(f'Worker exited with code {exit_code}')
    path=output/f'{name}.failure.json'
    if path.exists():
        try:failures.append('Worker failure artifact: '+json.loads(path.read_text(encoding='utf-8')).get('error','unspecified'))
        except (OSError,ValueError,TypeError):failures.append('Unreadable worker failure artifact')
    try:
        saved=json.loads((output/f'{name}.json').read_text(encoding='utf-8'))
        if saved.get('run_id')!=run_id or saved.get('name')!=name or saved.get('completed') is not True:
            failures.append('Missing completion or mismatched case/run identity in metrics')
        with np.load(output/f'{name}.npz',allow_pickle=False) as z:
            stamp=z['run_id']
            if stamp.shape!=() or stamp.dtype.kind!='U' or stamp.item()!=run_id:
                failures.append('Mismatched NPZ run identity')
            n=round(duration/dt)+1
            if z['time'].shape!=(n,) or not np.allclose(z['time'],np.arange(n)*dt,rtol=0,atol=1e-8):
                failures.append('Incomplete physical recording')
            for key in ('q','velocity','force','wrench','engine_time','body_transforms','body_velocities'):
                if len(z[key])!=n or not np.all(np.isfinite(z[key])):
                    failures.append('Incomplete/nonfinite '+key)
            if not np.allclose(z['engine_time'],z['time'],rtol=0,atol=2e-7):
                failures.append('Engine clock does not match recorded samples')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        failures.append('Missing or invalid completed case evidence: '+str(exc))
    result.update(passed=not failures,errors=failures)
    return result


def smoke_checks(output,run_id,video_requested):
    """Score a 2s diagnostic without pretending a docking gate can be evaluated."""
    from .scoring import score_case
    from .model_checks import physical_case_cmg
    root=Path(__file__).resolve().parents[1]; output=Path(output)
    try:
        with np.load(output/'nominal.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        score=score_case(d,physical_case_cmg(root,'nominal'),'nominal',.002,2.,True)
        checks={k:v for k,v in score['checks'].items() if k not in
                    ('full_mission','docking_position','docking_orientation')}
        if video_requested:
            meta=json.loads((output/'video_metadata.json').read_text(encoding='utf-8'))
            # Short clips cannot satisfy the full 22s video coverage gate.
            video_path=output/'Stewart_IsaacSim.mp4'
            good=(meta.get('run_id')==run_id and meta.get('source')=='live PhysX physics'
                  and not meta.get('error') and not meta.get('failure') and meta.get('valid_file') is True
                  and meta.get('capture_cadence_ok') is True and meta.get('encoded_frames')==meta.get('frames')
                  and meta.get('sim_start_s',1)<=.04 and meta.get('sim_end_s',0)>=1.96
                  and video_path.is_file() and video_path.stat().st_size>0)
            checks['diagnostic_video']=dict(passed=good,detail='Short clip only, not full video acceptance.')
        return dict(passed=bool(checks) and all(x['passed'] for x in checks.values()),checks=checks)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return dict(passed=False,error=str(exc),checks={})
