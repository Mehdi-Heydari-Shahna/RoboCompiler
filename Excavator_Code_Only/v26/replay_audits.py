"""Reaudit saved trajectories with the complete native global-wrench ledger.

No integration or physical-trace mutation occurs. Original reports and audits
are preserved; only audit metadata in the simulator JSON is revised. Run this
before tracked_task_audit.py, whose report hash must describe the final metadata.
"""
from project import ROOT,RESULTS,save_json
import argparse
import hashlib
import json
import shutil
import time
import zipfile
import numpy as np
import mujoco
import pinocchio
from tracked_audit import CompiledTreeAudit,AUDIT_REVISION
from native_solve import accurate_forward
from tracked_task_audit import validate_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physics_hash(report):
    physics={key:value for key,value in report.items() if key not in ('audit_passed','audit_replay')}
    return hashlib.sha256(json.dumps(physics,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def replay(name):
    start=time.perf_counter()
    report_path=RESULTS/f'{name}.json';audit_path=RESULTS/f'{name}_audit.json'
    model_path=RESULTS/'assets'/f'{name}.xml';trace_path=RESULTS/f'{name}.npz'
    report=json.loads(report_path.read_text());old=json.loads(audit_path.read_text())
    before_physics_hash=physics_hash(report)
    initial_hashes={path.relative_to(ROOT).as_posix():sha(path) for path in [model_path,trace_path]}
    with zipfile.ZipFile(trace_path) as archive:
        bad=archive.testzip()
        if bad is not None:raise ValueError('Trace CRC failure: '+bad)
    with np.load(trace_path,allow_pickle=False) as archive:
        trace={key:archive[key] for key in ['time','qpos','qvel','ctrl']}
        for key in ['act','qfrc_applied','xfrc_applied']:
            if key in archive:trace[key]=archive[key]
    m=mujoco.MjModel.from_xml_path(str(model_path));validate_trace(m,trace)
    if abs(float(trace['time'][-1])-report['duration_s'])>1e-9:
        raise ValueError('Trace does not cover the reported duration')
    if not report.get('completed') or report.get('base_propulsion_wrench') is not False:
        raise ValueError('Expected completed trajectory with zero externally applied propulsion')
    # Preserve the exact originally published metadata for these trace bytes.
    archive_dir=ROOT/'outputs/momentum_final_diagnostic/original_audits'/name/sha(trace_path)[:12]
    archive_dir.mkdir(parents=True,exist_ok=True)
    original_report=archive_dir/'simulation_report.json';original_audit=archive_dir/'dynamics_audit.json'
    if not original_report.exists():shutil.copy2(report_path,original_report)
    if not original_audit.exists():shutil.copy2(audit_path,original_audit)
    original=json.loads(original_audit.read_text())
    marks=[float(snapshot['time_s']) for snapshot in original['snapshots']]
    if not marks or marks[0]!=0. or abs(marks[-1]-report['duration_s'])>1e-9:
        raise ValueError('Original audit schedule must include beginning and end')
    backend=CompiledTreeAudit(m);data=mujoco.MjData(m);snapshots=[]
    for mark in marks:
        k=int(np.argmin(abs(trace['time']-mark)))
        if abs(float(trace['time'][k])-mark)>1e-9:raise ValueError('Audit time has no exact trace sample')
        mujoco.mj_resetData(m,data);data.time=mark
        for key in ['qpos','qvel','ctrl']:getattr(data,key)[:]=trace[key][k]
        if m.na:
            if 'act' not in trace:raise ValueError('Activation state missing')
            data.act[:]=trace['act'][k]
        for key in ['qfrc_applied','xfrc_applied']:
            if key in trace:getattr(data,key)[:]=trace[key][k]
        accuracy=accurate_forward(m,data)
        snapshot=backend.audit_snapshot(data)
        snapshot['numerical_replay_solver']=accuracy
        snapshots.append(snapshot)
    covariance=backend.frame_covariance(trace['qpos'][-1],trace['qvel'][-1])
    passed=all(item['passed'] for item in snapshots) and covariance['passed']
    for relative,expected in initial_hashes.items():
        if sha(ROOT/relative)!=expected:raise RuntimeError('Physical input changed during audit replay: '+relative)
    report['audit_passed']=passed
    report['audit_replay']=dict(revision=AUDIT_REVISION,method='Native forward replay and independent Pinocchio at original recorded audit timestamps',
        physical_trace_unchanged=True,physical_report_fields_sha256=before_physics_hash,
        original_report=original_report.relative_to(ROOT).as_posix(),original_audit=original_audit.relative_to(ROOT).as_posix(),
        reason='Complete global wrench balance includes measured finite-gap soft-equality couples; raw ideal-closure residuals remain explicit. No physical trajectory, actuator, contact, or integrator change.')
    if physics_hash(report)!=before_physics_hash:raise RuntimeError('Attempted physics report modification')
    save_json(report_path,report)
    input_paths=[Path for Path in [ROOT/'tracked_audit.py',ROOT/'replay_audits.py',ROOT/'native_solve.py',ROOT/'native_step.py',
                                  model_path,trace_path,report_path,original_report,original_audit]]
    hashes={path.relative_to(ROOT).as_posix():sha(path) for path in input_paths}
    proof=dict(passed=passed,revision=AUDIT_REVISION,name=name,
        snapshots=snapshots,frame_covariance=covariance,input_sha256=hashes,
        versions=dict(mujoco=mujoco.__version__,pinocchio=pinocchio.__version__),
        physical_trace_unchanged=True,physical_report_fields_sha256=before_physics_hash,
        audit_times_unchanged=True,original_audit_passed=original.get('passed'),
        elapsed_s=time.perf_counter()-start,
        scope='Independent compiled-tree backend and complete native force accounting. Equality global couples are finite-gap soft-constraint artifacts; exact ideal joint momentum conservation is not claimed. Original raw residuals and original reports retained.')
    save_json(audit_path,proof)
    summary=dict(name=name,passed=passed,snapshots=len(snapshots),
        maximum_raw_ideal_angular_relative=max(s.get('momentum_accounting',{}).get('raw_ideal_angular_balance_relative',0.) for s in snapshots),
        maximum_complete_angular_relative=max(s['metrics'].get('angular_momentum_rate_relative',0.) for s in snapshots),
        maximum_analytic_complete_angular_relative=max(s['metrics'].get('analytic_angular_momentum_rate_relative',0.) for s in snapshots),
        elapsed_s=proof['elapsed_s'])
    print(json.dumps(summary,indent=2),flush=True)
    if not passed:
        print(json.dumps([dict(time_s=s['time_s'],failed={k:v for k,v in s['metrics'].items() if not s['checks'][k]}) for s in snapshots if not s['passed']],indent=2),flush=True)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('name')
    args=parser.parse_args();result=replay(args.name)
    raise SystemExit(0 if result['passed'] else 1)
