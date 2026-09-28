"""Fresh CPU dynamics with the production force controller, mapping and guards.

Outputs are deliberately separated from results_isaac. This runner never
imports or impersonates Isaac modules and cannot grant native Isaac acceptance.
"""
from __future__ import annotations
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .rigid_body import RigidStewart
from isaac_validation.control import ReferenceController
from isaac_validation.mechanics import StateMapping
from isaac_validation.model_checks import physical_case_cmg, audit_case
from isaac_validation.protocol import RuntimeGuard, validate_request
from isaac_validation.scoring import score_case

ENGINE = 'Independent CPU Newton-Euler / exact six-UPS geometry (NOT Isaac Sim)'


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run_cpu_case(root, output, name, run_id, dt=.002, duration=22., model_audit=True):
    validate_request(name,dt,duration)
    root,output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cmg = physical_case_cmg(root,name)
    controller = ReferenceController(cmg,root/'baseline/reference.npz',name)
    physics = RigidStewart(cmg)
    mapping = StateMapping(cmg,physics.names,physics.com)
    guard = RuntimeGuard(cmg)
    initial = controller.initial_q
    p = initial[:3].copy(); R = Rotation.from_euler('ZYX',initial[3:6]).as_matrix()
    velocity = np.zeros(6)
    n = round(duration/dt)+1
    shapes = {'time':(), 'engine_time':(), 'q':(24,), 'velocity':(24,), 'target_pose':(6,),
              'force':(6,), 'requested_force':(6,), 'wrench':(6,), 'pose_error_m':(),
              'angle_error_rad':(), 'closure_error_m':(), 'joint_error_m':(),
              'joint_error_rad':(), 'actuator_error_m':(6,), 'power_W':(),
              'body_transforms':(19,7), 'body_velocities':(19,6),
              'dynamics_residual':(), 'force_mapping_residual':()}
    data = {k:np.empty((n,)+shape) for k,shape in shapes.items()}
    recorded = 0; started = time.perf_counter(); min_mass_eigenvalue = float('inf')
    try:
        for i in range(n):
            t = i*dt
            kin = physics.kinematics(p,R)
            poses,velocities = physics.body_arrays(kin,velocity)
            state = mapping.read(poses,velocities)
            target,_,requested,force,wrench = controller.evaluate(t,state['q'],state['velocity'])
            pos_error = float(np.linalg.norm(p-target[:3]))
            angle_error = float((Rotation.from_euler('ZYX',target[3:6]).inv()*Rotation.from_matrix(R)).magnitude())
            body_force,body_torque = mapping.forces(state,force,wrench)
            generalized = (np.einsum('bij,bi->j',kin['Jv'],body_force)
                           + np.einsum('bij,bi->j',kin['Jw'],body_torque))
            expected = kin['JL'].T@force+wrench
            force_residual = float(np.max(abs(generalized-expected)))
            if force_residual > 1e-8:
                raise RuntimeError(f'Actuator/body force mapping inconsistent at {t}: {force_residual}')
            M,bias = physics.terms(p,R,velocity,kin)
            eig = float(np.linalg.eigvalsh(M)[0]); min_mass_eigenvalue=min(min_mass_eigenvalue,eig)
            if eig <= 0:
                raise RuntimeError('Non-positive rigid-body mass matrix')
            acceleration = np.linalg.solve(M,generalized-bias)
            values = dict(time=t,engine_time=t,q=state['q'],velocity=state['velocity'],target_pose=target[:6],
                          force=force,requested_force=requested,wrench=wrench,pose_error_m=pos_error,
                          angle_error_rad=angle_error,closure_error_m=state['closure_error_m'],
                          joint_error_m=state['joint_error_m'],joint_error_rad=state['joint_error_rad'],
                          actuator_error_m=state['q'][mapping.active]-target[mapping.active],
                          power_W=force@state['velocity'][mapping.active],body_transforms=poses,
                          body_velocities=velocities,dynamics_residual=np.max(abs(M@acceleration+bias-generalized)),
                          force_mapping_residual=force_residual)
            for key,value in values.items(): data[key][i] = value
            recorded = i+1
            guard.check(name,state,pos_error,t)
            if i % max(1,round(2/dt)) == 0:
                print(f'CPU {name}: {t:5.2f}s position {1000*pos_error:.4f} mm',flush=True)
            if i < n-1:
                # These are integrated physical variables, never target commands.
                velocity = velocity+dt*acceleration
                p = p+dt*velocity[:3]
                R = Rotation.from_rotvec(dt*velocity[3:]).as_matrix()@R
        metrics = score_case(data,cmg,name,dt,duration,controller.feedforward)
        metrics.update(engine=ENGINE,run_id=run_id,completed=True,
                       elapsed_wall_s=time.perf_counter()-started,
                       min_platform_mass_eigenvalue=min_mass_eigenvalue,
                       max_dynamics_residual=float(np.max(data['dynamics_residual'])),
                       max_force_mapping_residual=float(np.max(data['force_mapping_residual'])),
                       native_isaac_execution=False,
                       reference_used_only_for_controller=True,
                       python_version=platform.python_version(),numpy_version=np.__version__,
                       integration='Semi-implicit Euler in platform world twist; SO(3) exponential update; exact passive geometry')
        write_json(output/f'{name}.json',metrics)
    finally:
        evidence = {k:v[:recorded] for k,v in data.items()}
        evidence.update(body_names=np.array(physics.names),body_coms=physics.com,
                        coordinate_ids=np.array(cmg['coordinate_ids']),run_id=np.array(run_id),engine=np.array(ENGINE))
        np.savez_compressed(output/f'{name}.npz',**evidence)
    if model_audit and abs(duration-22.) < 1e-9:
        audit = audit_case(evidence,cmg,initial)
        audit['scope'] = ('The unchanged sampled CMG/PACDM model auditor applied to independently integrated '
                          'CPU states, NOT PhysX body states. Shared model inputs; no native Isaac certification.')
        audit.update(engine=ENGINE,run_id=run_id,native_isaac_execution=False)
        write_json(output/f'{name}.model_validation.json',audit)
        metrics['sampled_model_audit_passed'] = audit['passed']
        write_json(output/f'{name}.json',metrics)
    return metrics
