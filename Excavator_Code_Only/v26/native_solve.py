"""Residual-controlled native MuJoCo forward solves, without state projection.

MuJoCo's Newton optimizer can stop on objective improvement at floating-point
roundoff before a requested force-balance accuracy is reached. This module
checks that accuracy explicitly and permits at most two native solve restarts.
Only qacc_warmstart (a numerical initial guess) changes between attempts. The
positions, velocities, activations, controls, applied forces, and time do not.
The caller subsequently advances with the official native integrator.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import mujoco
from native_step import checked_forward, validate_model

FORCE_TOLERANCE=1e-8
PHYSICAL_FIELDS=('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied',
                 'mocap_pos','mocap_quat','eq_active','userdata')


def force_residual(model,data):
    """Registered infinity-norm balance, independently evaluated from native M."""
    ma=np.zeros(model.nv)
    mujoco.mj_mulM(model,data,ma,data.qacc)
    residual=ma-data.qfrc_smooth-data.qfrc_constraint
    scale=max(1.,float(np.max(abs(ma),initial=0)),float(np.max(abs(data.qfrc_bias),initial=0)))
    value=float(np.max(abs(residual),initial=0))/scale
    if not np.isfinite(value):
        raise FloatingPointError('Nonfinite native force-balance residual')
    return value


def accurate_forward(model,data):
    """Run native forward dynamics, retrying numerical initialization if needed.

    Returns residuals and restart counts. A remaining residual above 1e-8 or
    any mutation of physical inputs raises; no inaccurate solution is accepted.
    Model options are never changed. In particular, this is not an alternate
    integrator or a claim of bitwise equivalence to a single unconverged solve.
    """
    validate_model(model)
    if model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        raise ValueError('Residual-controlled restart was validated with native Newton only')
    before={key:getattr(data,key).copy() for key in PHYSICAL_FIELDS}
    before_time=float(data.time)
    checked_forward(model,data)
    history=[force_residual(model,data)]
    retries=cold=0
    if history[-1]>FORCE_TOLERANCE:
        # Keep the current native solution, not its previous-step initializer.
        data.qacc_warmstart[:]=data.qacc
        checked_forward(model,data)
        history.append(force_residual(model,data));retries+=1
    if history[-1]>FORCE_TOLERANCE:
        data.qacc_warmstart[:]=0.
        checked_forward(model,data)
        history.append(force_residual(model,data));retries+=1;cold+=1
    changed=[key for key,value in before.items() if not np.array_equal(value,getattr(data,key))]
    if float(data.time)!=before_time:changed.append('time')
    if changed:
        raise RuntimeError('Native forward changed physical inputs: '+', '.join(changed))
    if history[-1]>FORCE_TOLERANCE:
        raise RuntimeError(f'Native force solve did not converge after {retries} restarts: {history}')
    return dict(initial_residual=history[0],final_residual=history[-1],
                retry_count=retries,cold_count=cold,residual_history=history)


def verify(model_path,checkpoint_path,trace_path,output):
    """Reproduce 121 initial-guess cases and 151 states, checking input identity."""
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    model_path=Path(model_path);checkpoint_path=Path(checkpoint_path);trace_path=Path(trace_path)
    m=mujoco.MjModel.from_xml_path(str(model_path));validate_model(m)
    p=np.load(checkpoint_path,allow_pickle=False);rng=np.random.default_rng(51921)
    records=[]
    def check(d,label):
        before={key:getattr(d,key).copy() for key in PHYSICAL_FIELDS};t=float(d.time)
        options=(int(m.opt.iterations),float(m.opt.tolerance),int(m.opt.ls_iterations),float(m.opt.ls_tolerance))
        result=accurate_forward(m,d)
        unchanged=all(np.array_equal(v,getattr(d,key)) for key,v in before.items()) and t==float(d.time)
        after=(int(m.opt.iterations),float(m.opt.tolerance),int(m.opt.ls_iterations),float(m.opt.ls_tolerance))
        result.update(case=label,physical_inputs_unchanged=unchanged,solver_options_unchanged=options==after)
        records.append(result)
    for i in range(121):
        d=mujoco.MjData(m)
        for key in ['qpos','qvel','ctrl']:getattr(d,key)[:]=p[key]
        d.time=float(p['time'])
        if i==0:d.qacc_warmstart[:]=p['qacc_warmstart']
        elif i<=60:d.qacc_warmstart[:]=(i/30)*p['qacc_warmstart']
        else:d.qacc_warmstart[:]=p['qacc_warmstart']+rng.normal(0,10,size=m.nv)
        check(d,f'checkpoint_warmstart_{i:03d}')
    trace=np.load(trace_path,allow_pickle=False)
    selected=np.unique(np.linspace(0,len(trace['time'])-1,151).astype(int))
    # Keep the exact validation inputs independent of later mission reruns.
    state_path=out/'validation_states.npz'
    temporary=out/'validation_states.npz.tmp'
    with open(temporary,'wb') as stream:
        np.savez_compressed(stream,**{key:trace[key][selected] for key in ['time','qpos','qvel','ctrl']})
    temporary.replace(state_path)
    for i,k in enumerate(selected):
        d=mujoco.MjData(m)
        for key in ['qpos','qvel','ctrl']:getattr(d,key)[:]=trace[key][k]
        d.time=float(trace['time'][k])
        check(d,f'sampled_state_{i:03d}')
    passed=all(r['final_residual']<=FORCE_TOLERANCE and r['physical_inputs_unchanged'] and r['solver_options_unchanged'] for r in records)
    project_root=Path(__file__).resolve().parent
    def relative_name(path):
        try:return path.resolve().relative_to(project_root).as_posix()
        except ValueError:raise ValueError('Reproducible solver validation inputs must be inside the project')
    hashes={relative_name(path):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),model_path,checkpoint_path,state_path]}
    report=dict(status='PASS' if passed else 'FAIL',passed=passed,mujoco_version=mujoco.__version__,
        checkpoint_cases=121,sampled_states=151,checks=len(records),tolerance=FORCE_TOLERANCE,
        maximum_initial_residual=max(r['initial_residual'] for r in records),
        maximum_final_residual=max(r['final_residual'] for r in records),
        retried_cases=sum(r['retry_count']>0 for r in records),total_native_restarts=sum(r['retry_count'] for r in records),
        total_cold_restarts=sum(r['cold_count'] for r in records),physical_inputs_unchanged=all(r['physical_inputs_unchanged'] for r in records),
        solver_options_unchanged=all(r['solver_options_unchanged'] for r in records),input_sha256=hashes,
        elapsed_s=time.perf_counter()-start,records=records,
        scope='Native solver convergence component only. Sampled states do not establish trajectory/task acceptance. No integration, physical state projection, force insertion, or solver-option mutation.',
        primary_source='https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_solver.c')
    temp=out/'report.json.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(out/'report.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','input_sha256']},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--trace',required=True)
    p.add_argument('--output',default='outputs/native_solver_accuracy')
    a=p.parse_args();result=verify(a.model,a.checkpoint,a.trace,a.output)
    raise SystemExit(0 if result['passed'] else 1)
