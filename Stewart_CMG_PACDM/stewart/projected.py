"""Direct PACDM reduced forward dynamics with six independent strut states.

MuJoCo is not used to advance this rollout. Pinocchio supplies the unconstrained
tree M/h; the unchanged PACDM graph supplies closure, N and curvature. Strut
lengths follow force-driven semi-implicit Euler, then PACDM assembles passives.
"""
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from vendor.pacdm_original import PointGraph,PACDM,skew
from .model import inverse_seed,rotation,save_cmg,compile_mujoco
from .pin_backend import PinBackend
from .simulation import disturbance
from .reference import DURATION


def run_projected(cmg,reference_path,output,timestep=.002):
    start=time.perf_counter();out=Path(output)
    with np.load(reference_path,allow_pickle=False) as z: ref={k:z[k] for k in z.files}
    qs=CubicSpline(ref['time'],ref['q']);vs=CubicSpline(ref['time'],ref['velocity']);fs=CubicSpline(ref['time'],ref['feedforward_force'])
    seed=inverse_seed(cmg,cmg['geometry']['nominal_pose']);g=PointGraph(cmg,seed);solver=PACDM(g);pin=PinBackend(cmg)
    q,acquire=solver.acquire(seed[g.active],g.lift(seed))
    if not acquire['success']:raise RuntimeError(str(acquire))
    N,mi=solver.mapping(q);ld=np.zeros(6);n=round(DURATION/timestep)+1
    keys=dict(time=np.empty(n),q=np.empty((n,g.nt)),velocity=np.empty((n,g.nt)),target_pose=np.empty((n,6)),
              force=np.empty((n,6)),wrench=np.empty((n,6)),pose_error_m=np.empty(n),angle_error_rad=np.empty(n),closure_error_m=np.empty(n),
              power_W=np.empty(n),reduced_equation_residual=np.empty(n))
    kp=cmg['actuation']['length_kp_N_m'];kd=cmg['actuation']['length_kd_N_s_m'];limit=cmg['actuation']['force_limit_N']
    fallback=0;saturated=0
    for i in range(n):
        t=i*timestep;target=qs(t);target_v=vs(t);v=N@ld
        force=fs(t)+kp*(target[g.active]-q[g.active])+kd*(target_v[g.active]-ld)
        saturated+=int(np.any(abs(force)>limit));force=np.clip(force,-limit,limit)
        wrench=disturbance(t)
        h=1e-5/max(1.,np.linalg.norm(v));J=g.residual(q)[1]
        Jdot=(g.residual(q+h*v)[1]-g.residual(q-h*v)[1])/(2*h)
        rows=np.array(mi['rows']);curvature=np.zeros(g.n)
        curvature[g.passive]=-np.linalg.solve(J[np.ix_(rows,g.passive)],(Jdot@v)[rows])
        M=pin.mass(q[:g.nt]);bias=pin.bias(q[:g.nt],v[:g.nt]);P=N[:g.nt]
        poses,E=g.poses(q);X=poses['platform'];S=E['platform'][:,:g.nt]
        W=np.vstack((S[3:]-skew(X[:3,3])@S[:3],S[:3]))
        external=P.T@W.T@wrench
        Mr=P.T@M@P;br=P.T@(bias+M@curvature[:g.nt])
        ldd=np.linalg.solve(Mr,force+external-br)
        keys['time'][i]=t;keys['q'][i]=q[:g.nt];keys['velocity'][i]=v[:g.nt]
        keys['target_pose'][i]=target[:6];keys['force'][i]=force;keys['wrench'][i]=wrench
        keys['pose_error_m'][i]=np.linalg.norm(q[:3]-target[:3])
        keys['angle_error_rad'][i]=np.linalg.norm(Rotation.from_matrix(rotation(target[:6]).T@rotation(q[:6])).as_rotvec())
        keys['closure_error_m'][i]=max(np.linalg.norm(poses[c['body1']][:3,3]+poses[c['body1']][:3,:3]@c['point1_m']-poses[c['body2']][:3,3]-poses[c['body2']][:3,:3]@c['point2_m']) for c in g.cuts)
        keys['power_W'][i]=force@ld;keys['reduced_equation_residual'][i]=np.max(abs(Mr@ldd+br-force-external))
        if i<n-1:
            ld+=timestep*ldd;next_l=q[g.active]+timestep*ld
            prediction=q[g.passive]+N[g.passive]@(next_l-q[g.active])
            candidate,ci=solver.correct(next_l,prediction,None,rows,maxiter=8)
            if not ci['success']:
                fallback+=1;candidate,ci=solver.acquire(next_l,q)
            if not ci['success']:raise RuntimeError(f'PACDM dynamic assembly at {t}: {ci}')
            q=candidate;N,mi=solver.mapping(q)
            if not mi['success']:raise RuntimeError(f'PACDM dynamic tangent at {t}: {mi}')
        if i%2500==0: print(f'PACDM reduced dynamics: {t:.1f}/{DURATION:.1f} s',flush=True)
    np.savez_compressed(out/'pacdm.npz',**keys,coordinate_ids=np.array(cmg['coordinate_ids']))
    # XML is for replay of saved states only; it is not used by this integrator.
    compile_mujoco(cmg,out/'pacdm.xml',timestep);save_cmg(cmg,out/'pacdm.cmg.json')
    dock=keys['time']>=21
    metrics=dict(name='pacdm',integrator='Semi-implicit Euler in six strut lengths; PACDM assembles dependent coordinates at every step.',
                 timestep_s=timestep,samples=n,elapsed_s=time.perf_counter()-start,
                 rms_position_error_m=float(np.sqrt(np.mean(keys['pose_error_m']**2))),max_position_error_m=float(max(keys['pose_error_m'])),
                 max_orientation_error_deg=float(np.rad2deg(max(keys['angle_error_rad']))),
                 docking_max_position_error_m=float(max(keys['pose_error_m'][dock])),
                 docking_max_orientation_error_deg=float(np.rad2deg(max(keys['angle_error_rad'][dock]))),
                 max_closure_error_m=float(max(keys['closure_error_m'])),max_actuator_force_N=float(np.max(abs(keys['force']))),
                 max_reduced_equation_residual_N=float(max(keys['reduced_equation_residual'])),
                 saturation_samples=saturated,fallback_count=fallback)
    return metrics
