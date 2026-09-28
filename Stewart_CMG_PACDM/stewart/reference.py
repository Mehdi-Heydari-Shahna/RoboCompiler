"""Six-axis mission assembled and differentiated by the PACDM core."""
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from vendor.pacdm_original import PointGraph, PACDM
from .model import inverse_seed
from .pin_backend import PinBackend

DURATION=22.0


def phase(t):
    if t<2: return 'Initial hold'
    if t<9: return 'Helical inspection sweep'
    if t<17: return 'Figure-eight / disturbance rejection'
    if t<20: return 'Precision docking'
    return 'Dock hold'


def task_pose(t):
    """xyz,yaw,pitch,roll. Smooth finite motion, safely inside chart bounds."""
    p=np.array([0.,0.,.60,0.,0.,0.])
    if 2<=t<9:
        s=(t-2)/7; e=np.sin(np.pi*s)**4; a=4*np.pi*s
        p+=e*np.array([.060*np.cos(a),.060*np.sin(a),.060*(s-.5),
                       .20*np.sin(a),.14*np.cos(a),.13*np.sin(2*a)])
    elif 9<=t<17:
        s=(t-9)/8; e=np.sin(np.pi*s)**4; a=4*np.pi*s
        p+=e*np.array([.060*np.sin(a),.045*np.sin(2*a),.035*np.cos(a),
                       .20*np.sin(a),.13*np.cos(a),.15*np.sin(2*a)])
    elif t>=17:
        s=min(1.,(t-17)/3); e=10*s**3-15*s**4+6*s**5
        p+=e*np.array([.028,-.026,.030,.14,-.07,.08])
    return p


def build_reference(cmg, path, interval=.02):
    started=time.perf_counter()
    times=np.linspace(0,DURATION,round(DURATION/interval)+1)
    targets=np.array([task_pose(t) for t in times])
    commanded=np.array([inverse_seed(cmg,p) for p in targets])
    graph=PointGraph(cmg,commanded[0]); solver=PACDM(graph)
    lengths=commanded[:,graph.active]
    q,tracking=solver.track(lengths,graph.lift(commanded[0]))
    length_spline=CubicSpline(times,lengths)
    ld=length_spline(times,1); ldd=length_spline(times,2)
    pin=PinBackend(cmg)
    velocity=[]; acceleration=[]; forces=[]; maps=[]; conditions=[]
    residuals=[]; tangent=[]; acceleration_residual=[]; mass_eigen=[]
    for k,state in enumerate(q):
        N,mi=solver.mapping(state)
        if not mi['success']: raise RuntimeError(f'PACDM map at {k}: {mi}')
        v=N@ld[k]
        h=1e-5/max(1.,np.linalg.norm(v))
        J=graph.residual(state)[1]
        Jdot=(graph.residual(state+h*v)[1]-graph.residual(state-h*v)[1])/(2*h)
        rows=np.array(mi['rows']); curvature=np.zeros(graph.n)
        curvature[graph.passive]=-np.linalg.solve(J[np.ix_(rows,graph.passive)],(Jdot@v)[rows])
        a=N@ldd[k]+curvature
        P=N[:graph.nt]; tau=pin.inverse(state[:graph.nt],v[:graph.nt],a[:graph.nt])
        force=P.T@tau
        Mr=P.T@pin.mass(state[:graph.nt])@P
        mass_eigen.append(np.linalg.eigvalsh(Mr).min())
        velocity.append(v[:graph.nt]);acceleration.append(a[:graph.nt]);forces.append(force);maps.append(P)
        conditions.append(1/mi['rcond']);residuals.append(mi['residual_inf']);tangent.append(mi['tangent_residual'])
        acceleration_residual.append(np.max(abs(J@a+Jdot@v)))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,time=times,q=q[:,:graph.nt],q_augmented=q,velocity=velocity,acceleration=acceleration,
                        lengths=lengths,length_velocity=ld,length_acceleration=ldd,feedforward_force=forces,
                        mapping=maps,target_pose=targets,coordinate_ids=np.array(cmg['coordinate_ids']),
                        closure_residual=residuals,tangent_residual=tangent,acceleration_residual=acceleration_residual,
                        selected_block_condition=conditions,reduced_mass_min_eigenvalue=mass_eigen)
    return dict(samples=len(times),interval_s=interval,elapsed_s=time.perf_counter()-started,
                fallback_count=tracking['fallback_count'],initial_acquisition=tracking['initial_acquisition'],
                max_closure_residual=float(max(residuals)),max_tangent_residual=float(max(tangent)),
                max_acceleration_residual=float(max(acceleration_residual)),
                max_target_pose_reconstruction_error=float(np.max(abs(q[:,:6]-targets))),
                min_reduced_mass_eigenvalue=float(min(mass_eigen)),
                max_selected_block_condition=float(max(conditions)),
                max_feedforward_force_N=float(np.max(np.abs(forces))))
