"""Force-driven MuJoCo mission; no pose assignment after initialization."""
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
import mujoco
from .model import compile_mujoco, rotation, save_cmg
from .reference import DURATION


def disturbance(t, scale=1.):
    wrench=np.zeros(6) # world force xyz, then world torque xyz at platform COM
    pulses=[(5.5,.35,[80,0,0,0,0,0]),(11.2,.35,[0,120,0,0,0,0]),(14.2,.4,[0,0,0,12,0,0])]
    for start,duration,value in pulses:
        if start<=t<=start+duration: wrench+=np.array(value)*np.sin(np.pi*(t-start)/duration)
    return scale*wrench


def run_case(cmg, reference_path, output, name='nominal', timestep=.002, disturbance_scale=1., feedforward=True):
    started=time.perf_counter(); output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    save_cmg(cmg,output/f'{name}.cmg.json')
    xml=compile_mujoco(cmg,output/f'{name}.xml',timestep)
    model=mujoco.MjModel.from_xml_path(str(xml.resolve()));data=mujoco.MjData(model)
    with np.load(reference_path,allow_pickle=False) as z: ref={k:z[k] for k in z.files}
    qs=CubicSpline(ref['time'],ref['q']); vs=CubicSpline(ref['time'],ref['velocity']); fs=CubicSpline(ref['time'],ref['feedforward_force'])
    ids=cmg['coordinate_ids']; joint_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,k) for k in ids])
    qadr=model.jnt_qposadr[joint_ids]; vadr=model.jnt_dofadr[joint_ids]
    active=np.array([ids.index(k) for k in cmg['independent_ids']])
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'platform')
    tip=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'tip_{i}') for i in range(6)])
    top=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'top_{i}') for i in range(6)])
    data.qpos[qadr]=ref['q'][0];data.qvel[:]=0;mujoco.mj_forward(model,data)
    n=round(DURATION/timestep)+1
    keys=dict(time=np.empty(n),q=np.empty((n,24)),velocity=np.empty((n,24)),target_pose=np.empty((n,6)),
              force=np.empty((n,6)),wrench=np.empty((n,6)),pose_error_m=np.empty(n),angle_error_rad=np.empty(n),
              closure_error_m=np.empty(n),actuator_error_m=np.empty((n,6)),power_W=np.empty(n),energy_J=np.empty(n))
    kp=cmg['actuation']['length_kp_N_m'];kd=cmg['actuation']['length_kd_N_s_m'];limit=cmg['actuation']['force_limit_N']
    saturated=0;max_unclipped=0.;max_solver_iterations=0
    for k in range(n):
        t=min(k*timestep,DURATION)
        # Forward computes current geometry; mj_step alone leaves some derived
        # quantities at its pre-integration state, so refresh before recording.
        mujoco.mj_forward(model,data)
        target=qs(t); target_v=vs(t)
        force=(fs(t) if feedforward else np.zeros(6))+kp*(target[active]-data.qpos[qadr[active]])+kd*(target_v[active]-data.qvel[vadr[active]])
        max_unclipped=max(max_unclipped,float(np.max(abs(force))));saturated+=int(np.any(abs(force)>limit))
        force=np.clip(force,-limit,limit);wrench=disturbance(t,disturbance_scale)
        data.ctrl[:]=force;data.xfrc_applied[:]=0;data.xfrc_applied[body]=wrench
        R=rotation(data.qpos[qadr][:6]); Rtarget=rotation(target[:6])
        keys['time'][k]=t;keys['q'][k]=data.qpos[qadr];keys['velocity'][k]=data.qvel[vadr]
        keys['target_pose'][k]=target[:6];keys['force'][k]=force;keys['wrench'][k]=wrench
        keys['pose_error_m'][k]=np.linalg.norm(data.qpos[qadr][:3]-target[:3])
        keys['angle_error_rad'][k]=np.linalg.norm(Rotation.from_matrix(Rtarget.T@R).as_rotvec())
        keys['closure_error_m'][k]=np.max(np.linalg.norm(data.site_xpos[tip]-data.site_xpos[top],axis=1))
        keys['actuator_error_m'][k]=data.qpos[qadr[active]]-target[active]
        keys['power_W'][k]=force@data.qvel[vadr[active]]
        mujoco.mj_energyPos(model,data);mujoco.mj_energyVel(model,data);keys['energy_J'][k]=np.sum(data.energy)
        max_solver_iterations=max(max_solver_iterations,int(np.max(data.solver_niter)))
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)): raise RuntimeError(f'{name}: non-finite state at {t}')
        if k<n-1: mujoco.mj_step(model,data)
    np.savez_compressed(output/f'{name}.npz',**keys,coordinate_ids=np.array(ids))
    joint_lower=np.array([next(j for j in cmg['joints'] if j['id']==i)['limits']['lower'] for i in ids])
    joint_upper=np.array([next(j for j in cmg['joints'] if j['id']==i)['limits']['upper'] for i in ids])
    dock=keys['time']>=21.
    metrics=dict(name=name,timestep_s=timestep,samples=n,duration_s=DURATION,elapsed_s=time.perf_counter()-started,
                 payload_mass_kg=next(b['mass_kg'] for b in cmg['bodies'] if b['id']=='payload'),
                 rms_position_error_m=float(np.sqrt(np.mean(keys['pose_error_m']**2))),
                 max_position_error_m=float(np.max(keys['pose_error_m'])),
                 max_orientation_error_deg=float(np.rad2deg(np.max(keys['angle_error_rad']))),
                 docking_max_position_error_m=float(np.max(keys['pose_error_m'][dock])),
                 docking_max_orientation_error_deg=float(np.rad2deg(np.max(keys['angle_error_rad'][dock]))),
                 max_closure_error_m=float(np.max(keys['closure_error_m'])),
                 max_actuator_force_N=float(np.max(abs(keys['force']))),max_requested_force_N=max_unclipped,
                 saturation_samples=saturated,min_joint_limit_margin=float(min(np.min(keys['q']-joint_lower),np.min(joint_upper-keys['q']))),
                 positive_actuator_work_J=float(np.trapezoid(np.maximum(keys['power_W'],0),keys['time'])),
                 max_solver_iterations=max_solver_iterations,disturbance_scale=disturbance_scale,
                 feedforward=feedforward,warning_counts={str(mujoco.mjtWarning(i)):int(x.number) for i,x in enumerate(data.warning) if x.number})
    return metrics
