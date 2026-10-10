"""MuJoCo 3.3.7 implicitfast pipeline with one forward solve per timestep.

This module calls the official native integrator; it implements no integration,
state projection, custom dynamics, or altered solver. It is useful when diagnostics
already require a full forward solve before advancing that exact state/control.

The tagged primary source (inspected, not inferred from the older documentation
example) is https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c
Sections: mj_step; mj_implicit; mj_implicitSkip; mj_advance. In this version mj_step
runs checkPos/checkVel, forward, checkAcc, optional compareFwdInv, then mj_implicit.
The integrator itself advances native activations, positions, velocities, time,
plugin state and warmstart acceleration. No manual state update is required.

Caller contract: call checked_forward after setting state, controls and external
forces; perform read-only diagnostics; then advance_after_forward exactly once.
Do not change controls/forces/state/model or call inverse dynamics in between.
This deliberately narrow wrapper requires implicitfast, MuJoCo 3.3.7, no native
plugins and no Python control/passive callbacks. Diagnostics timers differ from
mj_step, but physical state, warmstart and sensor outputs are preserved.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import mujoco

SUPPORTED_VERSION='3.3.7'
SOURCE_URL='https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/src/engine/engine_forward.c'


def validate_model(model):
    if mujoco.__version__ != SUPPORTED_VERSION:
        raise RuntimeError('native_step was verified only against MuJoCo '+SUPPORTED_VERSION)
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError('native_step requires implicitfast; no integrator is substituted')
    if model.nplugin:
        raise ValueError('Native plugin callbacks are outside this audited pipeline scope')
    for name in ('get_mjcb_control','get_mjcb_passive'):
        if hasattr(mujoco,name) and getattr(mujoco,name)() is not None:
            raise ValueError('Stateful callbacks are outside this audited pipeline scope')


def checked_forward(model,data):
    """The pre-integrator checked forward segment of native mj_step.

    Model suitability is checked once with validate_model before entering a loop.
    """
    mujoco.mj_checkPos(model,data)
    mujoco.mj_checkVel(model,data)
    mujoco.mj_forward(model,data)


def advance_after_forward(model,data):
    """Consume the existing full forward solution using the native integrator.

    PRECONDITION: checked_forward has run for the current, unchanged inputs. No
    manual activation, time, warmstart, position or velocity assignment occurs.
    """
    mujoco.mj_checkAcc(model,data)
    if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_FWDINV):
        mujoco.mj_compareFwdInv(model,data)
    mujoco.mj_implicit(model,data)


def _array_errors(a,b,model=None):
    # Complete publicly exposed numeric mjData arrays except timers (call counts)
    # and solver arena pointer-backed containers (contact structs are checked via
    # their numeric attributes below). These arrays include sensors, forces,
    # accelerations, activations, warmstart, equality state, mocap and integrator
    # factorization workspaces, not only qpos/qvel.
    result={}
    for name in dir(a):
        if name.startswith('_'):continue
        # efc_AR is a dual-solver workspace and is not initialized by Newton.
        if name.startswith('efc_AR') and model is not None and not mujoco.mj_isDual(model):continue
        try:
            x=getattr(a,name);y=getattr(b,name)
        except (AttributeError,ValueError):continue
        if isinstance(x,np.ndarray) and x.dtype.kind in 'biufc' and x.shape==y.shape:
            result[name]=float(np.max(np.abs(x.astype(float)-y.astype(float)),initial=0))
    result['time']=abs(float(a.time-b.time))
    result['ncon']=abs(int(a.ncon)-int(b.ncon))
    result['nefc']=abs(int(a.nefc)-int(b.nefc))
    result['warning_number']=max((abs(x.number-y.number) for x,y in zip(a.warning,b.warning)),default=0)
    if a.ncon==b.ncon:
        for name in ('dist','pos','frame','friction','solref','solimp','geom','efc_address'):
            try:
                x=np.asarray(getattr(a.contact,name));y=np.asarray(getattr(b.contact,name))
                result['contact_'+name]=float(np.max(abs(x.astype(float)-y.astype(float)),initial=0))
            except (AttributeError,ValueError):pass
    return result


def verify(xml_path, output='outputs/native_step_validation', steps=200, checkpoint=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(xml_path));validate_model(m)
    a=mujoco.MjData(m);b=mujoco.MjData(m)
    if checkpoint is not None:
        state=np.load(checkpoint)
        a.qpos[:]=state['qpos'];a.qvel[:]=state['qvel']
    mujoco.mj_forward(m,a);b=copy.copy(a)
    maxima={};checks=[];native_elapsed=0.;split_elapsed=0.
    for k in range(steps):
        # Identical, deterministic bounded torques excite all native ports. This
        # test checks numerical pipeline parity, not the operational controller.
        ctrl=50*np.sin(.03*k+np.arange(m.nu))
        a.ctrl[:]=ctrl;b.ctrl[:]=ctrl
        t=time.perf_counter();mujoco.mj_step(m,a);native_elapsed+=time.perf_counter()-t
        t=time.perf_counter();checked_forward(m,b);advance_after_forward(m,b);split_elapsed+=time.perf_counter()-t
        errors=_array_errors(a,b,m)
        for name,value in errors.items():maxima[name]=max(maxima.get(name,0),value)
        if k in (0,49,199,steps-1):
            checks.append(dict(step=k+1,max_abs_difference=max(errors.values()),
                               qpos=errors.get('qpos'),qvel=errors.get('qvel'),
                               warmstart=errors.get('qacc_warmstart'),time=errors.get('time')))
    # Independent one-step test from a fully contact-active state, comparing a
    # native mj_step against consuming an existing solved forward result.
    c=mujoco.MjData(m);e=mujoco.MjData(m)
    c=copy.copy(a);e=copy.copy(a)
    mujoco.mj_step(m,c)
    checked_forward(m,e);advance_after_forward(m,e)
    one=_array_errors(c,e,m)
    # Benchmark the actual use case: diagnostics require an initial full solve.
    # The redundant path forward+step is compared with forward+native advance.
    repetitions=200
    duplicated_elapsed=0.;reuse_elapsed=0.;reuse_maxima={};reuse_checks=[]
    c=copy.copy(a);e=copy.copy(a)
    for k in range(repetitions):
        ctrl=50*np.sin(.03*k+np.arange(m.nu));c.ctrl[:]=ctrl;e.ctrl[:]=ctrl
        t=time.perf_counter();checked_forward(m,c);mujoco.mj_step(m,c);duplicated_elapsed+=time.perf_counter()-t
        t=time.perf_counter();checked_forward(m,e);advance_after_forward(m,e);reuse_elapsed+=time.perf_counter()-t
        reuse_errors=_array_errors(c,e,m)
        for name,value in reuse_errors.items():reuse_maxima[name]=max(reuse_maxima.get(name,0),value)
        if k in (0,49,199):reuse_checks.append(dict(step=k+1,max_abs_difference=max(reuse_errors.values()),qpos=reuse_errors['qpos'],qvel=reuse_errors['qvel'],warmstart=reuse_errors['qacc_warmstart']))
    result=dict(status='PASS' if max(maxima.values())==0 and max(one.values())==0 and max(reuse_maxima.values())==0 else 'FAIL',
        mujoco_version=mujoco.__version__,model=str(xml_path),steps=steps,
        compared_numeric_arrays=len(maxima),checkpoints=checks,maximum_array_differences=maxima,
        one_step_after_contact_max_difference=max(one.values()),
        independent_trajectories_bit_identical=all(v==0 for v in maxima.values()),
        duplicate_forward_vs_reuse_max_difference=max(reuse_maxima.values()),
        duplicate_forward_vs_reuse_checkpoints=reuse_checks,
        excluded_uninitialized_arrays=['efc_AR* (Newton is a primal solver; these dual-solver buffers are unused)'],
        timings=dict(native_step_s=native_elapsed,checked_forward_and_advance_s=split_elapsed,
                     duplicate_forward_and_step_s=duplicated_elapsed,reused_forward_and_advance_s=reuse_elapsed,
                     reuse_speedup=duplicated_elapsed/reuse_elapsed),
        official_source=SOURCE_URL,
        native_integrator='mj_implicit (implicitfast selected in model)',
        manual_state_updates=False,
        scope='MuJoCo3.3.7 implicitfast, unchanged current forward inputs, no callbacks/plugins; timers/call counts intentionally differ.')
    source=Path('/tmp/mujoco337_engine_forward.c')
    if source.exists():result['official_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    root=Path(__file__).resolve().parent
    inputs=[Path(__file__),Path(xml_path)]
    if checkpoint is not None:inputs.append(Path(checkpoint))
    result['input_sha256']={path.resolve().relative_to(root).as_posix():
        hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='maximum_array_differences'},indent=2),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml',required=True);parser.add_argument('--output',default='outputs/native_step_validation')
    parser.add_argument('--steps',type=int,default=200);parser.add_argument('--checkpoint')
    args=parser.parse_args();r=verify(args.xml,args.output,args.steps,args.checkpoint)
    if r['status']!='PASS':raise SystemExit(1)
