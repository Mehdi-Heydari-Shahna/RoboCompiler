"""Force-driven six-coordinate PACDM integration using only Pinocchio dynamics.

No recorded MuJoCo states are consumed. No target pose is imposed after the
initial feasible assembly. The legacy PACDM algorithm is retained verbatim.
"""
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from vendor.pacdm_original import PointGraph, PACDM, skew
from .model import inverse_seed, rotation, save_cmg
from .pin_backend import PinBackend
from .reference import DURATION


def disturbance(t, scale=1.):
    """World force then world moment at the platform origin, SI units."""
    wrench = np.zeros(6)
    for start, duration, value in [(5.5,.35,[80,0,0,0,0,0]),
                                 (11.2,.35,[0,120,0,0,0,0]),
                                 (14.2,.4,[0,0,0,12,0,0])]:
        if start <= t <= start+duration:
            wrench += np.asarray(value)*np.sin(np.pi*(t-start)/duration)
    return scale*wrench


def run_case(cmg, reference_path, output, name='nominal', timestep=.002,
             feedforward=True, disturbance_scale=1., duration=DURATION):
    started = time.perf_counter()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    if timestep <= 0 or duration <= 0 or not np.isclose(round(duration/timestep)*timestep,duration):
        raise ValueError('Positive duration must be divisible by timestep.')
    with np.load(reference_path, allow_pickle=False) as z:
        ref = {k:z[k] for k in z.files}
    if not np.array_equal(ref['coordinate_ids'],cmg['coordinate_ids']):
        raise ValueError('Reference and model coordinate order differ.')
    qs = CubicSpline(ref['time'],ref['q']); vs = CubicSpline(ref['time'],ref['velocity'])
    fs = CubicSpline(ref['time'],ref['feedforward_force'])
    seed = inverse_seed(cmg,cmg['geometry']['nominal_pose'])
    graph = PointGraph(cmg,seed); solver = PACDM(graph); backend = PinBackend(cmg)
    state, acquire = solver.acquire(seed[graph.active],graph.lift(seed))
    if not acquire['success']: raise RuntimeError(f'Initial assembly failed: {acquire}')
    mapping, info = solver.mapping(state)
    if not info['success']: raise RuntimeError(f'Initial tangent failed: {info}')
    ld = np.zeros(6); n = round(duration/timestep)+1
    arrays = {k:np.empty((n,*s)) for k,s in {
        'time':(), 'q':(graph.nt,), 'q_augmented':(graph.n,), 'velocity':(graph.nt,),
        'acceleration':(graph.nt,), 'length_acceleration':(6,), 'target_pose':(6,),
        'force':(6,), 'wrench':(6,), 'pose_error_m':(), 'angle_error_rad':(),
        'closure_error_m':(), 'augmented_closure_residual':(), 'velocity_closure_residual':(),
        'acceleration_closure_residual':(), 'tangent_residual':(), 'reduced_equation_residual':(),
        'power_W':(), 'external_power_W':(), 'energy_J':(), 'reduced_mass_min_eigenvalue':(),
        'mapping_rcond':(), 'rank_full':(), 'rank_passive':(), 'actuator_error_m':(6,)
    }.items()}
    kp = cmg['actuation']['length_kp_N_m']; kd = cmg['actuation']['length_kd_N_s_m']
    limit = cmg['actuation']['force_limit_N']; fallback = 0; saturation = 0; requested_peak = 0.
    for i in range(n):
        t = i*timestep; target = qs(t); target_v = vs(t); velocity = mapping@ld
        requested = (fs(t) if feedforward else np.zeros(6)) + kp*(target[graph.active]-state[graph.active]) + kd*(target_v[graph.active]-ld)
        requested_peak = max(requested_peak,float(np.max(np.abs(requested))))
        saturation += int(np.any(np.abs(requested)>limit)); force = np.clip(requested,-limit,limit)
        wrench = disturbance(t,disturbance_scale)
        phi,J,_ = graph.residual(state); step = 1e-5/max(1.,np.linalg.norm(velocity))
        Jdot = (graph.residual(state+step*velocity)[1]-graph.residual(state-step*velocity)[1])/(2*step)
        rows = np.asarray(info['rows']); curvature = np.zeros(graph.n)
        curvature[graph.passive] = -np.linalg.solve(J[np.ix_(rows,graph.passive)],(Jdot@velocity)[rows])
        mass = backend.mass(state[:graph.nt]); bias = backend.bias(state[:graph.nt],velocity[:graph.nt])
        P = mapping[:graph.nt]; poses, twists = graph.poses(state)
        X = poses['platform']; S = twists['platform'][:,:graph.nt]
        W = np.vstack((S[3:]-skew(X[:3,3])@S[:3],S[:3]))
        external = P.T@W.T@wrench
        mr = P.T@mass@P; br = P.T@(bias+mass@curvature[:graph.nt])
        ldd = np.linalg.solve(mr,force+external-br); acceleration = mapping@ldd+curvature
        if not all(np.all(np.isfinite(x)) for x in [state,velocity,acceleration,force,mr]):
            raise RuntimeError(f'{name}: non-finite result at {t}s')
        a = arrays
        a['time'][i] = t; a['q'][i] = state[:graph.nt]; a['q_augmented'][i] = state
        a['velocity'][i] = velocity[:graph.nt]; a['acceleration'][i] = acceleration[:graph.nt]
        a['length_acceleration'][i] = ldd; a['target_pose'][i] = target[:6]
        a['force'][i] = force; a['wrench'][i] = wrench
        a['pose_error_m'][i] = np.linalg.norm(state[:3]-target[:3])
        a['angle_error_rad'][i] = np.linalg.norm(Rotation.from_matrix(rotation(target[:6]).T@rotation(state[:6])).as_rotvec())
        a['closure_error_m'][i] = max(np.linalg.norm(poses[c['body1']][:3,3]+poses[c['body1']][:3,:3]@c['point1_m']-poses[c['body2']][:3,3]-poses[c['body2']][:3,:3]@c['point2_m']) for c in graph.cuts)
        a['augmented_closure_residual'][i] = np.max(np.abs(phi))
        a['velocity_closure_residual'][i] = np.max(np.abs(J@velocity))
        a['acceleration_closure_residual'][i] = np.max(np.abs(J@acceleration+Jdot@velocity))
        a['tangent_residual'][i] = np.max(np.abs(J@mapping))
        a['reduced_equation_residual'][i] = np.max(np.abs(mr@ldd+br-force-external))
        a['power_W'][i] = force@ld; a['external_power_W'][i] = wrench@W@velocity[:graph.nt]
        a['energy_J'][i] = backend.energy(state[:graph.nt],velocity[:graph.nt])['total_J']
        a['reduced_mass_min_eigenvalue'][i] = np.linalg.eigvalsh(mr).min()
        for key in ['rank_full','rank_passive']: a[key][i] = info[key]
        a['mapping_rcond'][i] = info['rcond']; a['actuator_error_m'][i] = state[graph.active]-target[graph.active]
        if i < n-1:
            ld = ld+timestep*ldd
            next_length = state[graph.active]+timestep*ld
            predicted = state[graph.passive]+mapping[graph.passive]@(next_length-state[graph.active])
            candidate, correction = solver.correct(next_length,predicted,None,rows,maxiter=8)
            if not correction['success']:
                fallback += 1; candidate,correction = solver.acquire(next_length,state)
            if not correction['success']: raise RuntimeError(f'{name}: assembly failed at {t}: {correction}')
            state = candidate; mapping,info = solver.mapping(state)
            if not info['success']: raise RuntimeError(f'{name}: tangent/rank failed at {t}: {info}')
        if i % max(1,round(2/timestep)) == 0:
            print(f'{name}: {t:.1f}/{duration:.1f} s',flush=True)
    ids = cmg['coordinate_ids']; joints = {j['id']:j for j in cmg['joints']}
    lower = np.asarray([joints[j]['limits']['lower'] for j in ids]); upper = np.asarray([joints[j]['limits']['upper'] for j in ids])
    revolute = [i for i,j in enumerate(ids) if joints[j]['type']=='revolute']
    prismatic = [i for i,j in enumerate(ids) if joints[j]['type']=='prismatic']
    margins = np.minimum(arrays['q']-lower,upper-arrays['q'])
    dock = arrays['time']>=max(0.,duration-1.)
    metrics = dict(name=name,engine='Pinocchio 3.8.0 tree dynamics + unchanged PACDM reduction',
        integrator='Semi-implicit Euler in six independent leg lengths, PACDM passive assembly every step',
        timestep_s=timestep,duration_s=duration,samples=n,elapsed_s=time.perf_counter()-started,
        feedforward=bool(feedforward),disturbance_scale=disturbance_scale,
        payload_mass_kg=next(b['mass_kg'] for b in cmg['bodies'] if b['id']=='payload'),
        rms_position_error_m=float(np.sqrt(np.mean(arrays['pose_error_m']**2))),
        max_position_error_m=float(arrays['pose_error_m'].max()),
        max_orientation_error_deg=float(np.rad2deg(arrays['angle_error_rad'].max())),
        docking_max_position_error_m=float(arrays['pose_error_m'][dock].max()),
        docking_max_orientation_error_deg=float(np.rad2deg(arrays['angle_error_rad'][dock].max())),
        max_closure_error_m=float(arrays['closure_error_m'].max()),
        max_actuator_force_N=float(np.max(np.abs(arrays['force']))),max_requested_force_N=requested_peak,
        saturation_samples=saturation,fallback_count=fallback,
        min_prismatic_limit_margin_m=float(margins[:,prismatic].min()),
        min_revolute_limit_margin_rad=float(margins[:,revolute].min()),
        min_reduced_mass_eigenvalue=float(arrays['reduced_mass_min_eigenvalue'].min()),
        min_mapping_rcond=float(arrays['mapping_rcond'].min()),
        min_rank_full=int(arrays['rank_full'].min()),min_rank_passive=int(arrays['rank_passive'].min()),
        all_finite=all(bool(np.all(np.isfinite(x))) for x in arrays.values()),
        positive_actuator_work_J=float(np.trapezoid(np.maximum(arrays['power_W'],0),arrays['time'])),
        energy_work_defect_J=float(arrays['energy_J'][-1]-arrays['energy_J'][0]-np.trapezoid(arrays['power_W']+arrays['external_power_W'],arrays['time'])))
    for k in ['augmented_closure_residual','velocity_closure_residual','acceleration_closure_residual','tangent_residual','reduced_equation_residual']:
        metrics['max_'+k] = float(arrays[k].max())
    np.savez_compressed(out/f'{name}.npz',**arrays,coordinate_ids=np.asarray(ids),
                        dynamics_backend=np.asarray('Pinocchio 3.8.0 + PACDM'),timestep_s=np.asarray(timestep))
    save_cmg(cmg,out/f'{name}.cmg.json')
    return metrics
