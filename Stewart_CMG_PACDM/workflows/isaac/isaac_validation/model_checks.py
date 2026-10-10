"""Fresh CMG/PACDM checks against measured simulator states, not reference replay.

Only NumPy/SciPy and the unchanged original PointGraph/PACDM are used here.
Newton-Euler body summation is a second dynamics implementation, independent of
PhysX and of the archived Pinocchio mass/bias calculation. It shares CMG data
and the source graph, so it is not an independent identification of real inertia.
The gates below are declared engineering tolerances, not universal error bounds.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from vendor.pacdm_original import PointGraph, PACDM, skew

# Versioned before any new Isaac execution. Do not fit these to failed runs.
MODEL_LIMITS = dict(body_position_m=1e-4, body_orientation_rad=np.deg2rad(.05),
                    velocity_m_s=.01, velocity_rad_s=.05,
                    force_residual_abs_N=2., force_residual_relative=.10,
                    closure=1e-8, tangent=1e-8,
                    archived_feedforward_consistency_N=1e-4)
# Interior disturbance samples plus an evenly spaced, deterministic mission grid.
SAMPLE_TIMES = np.unique(np.r_[np.arange(.5,22.,.5),[5.65,11.35,14.4]])


def physical_case_cmg(root: Path, name: str) -> dict:
    cmg=json.loads((Path(root)/'data/stewart.cmg.json').read_text())
    if name=='heavy_payload':
        body=next(b for b in cmg['bodies'] if b['id']=='payload')
        ratio=14./body['mass_kg']; body['mass_kg']=14.
        body['inertia_kg_m2']=(np.asarray(body['inertia_kg_m2'])*ratio).tolist()
    return cmg


def _body_jacobians(body, poses, twists):
    T=poses[body['id']]; E=twists[body['id']]
    c=T[:3,3]+T[:3,:3]@np.asarray(body['com_m'])
    return E[3:]-skew(c)@E[:3], E[:3]


class ReducedModel:
    """PointGraph tangent + rigid-body Newton-Euler reduced mass and bias."""
    def __init__(self,cmg,seed):
        self.cmg=cmg; self.graph=PointGraph(cmg,np.asarray(seed))
        self.solver=PACDM(self.graph)
        self.bodies=[b for b in cmg['bodies'] if b['mass_kg']>0 and b['id']!=cmg['root_body']]

    def assemble(self,measured_q):
        g=self.graph
        q,acquisition=self.solver.acquire(np.asarray(measured_q)[g.active],g.lift(measured_q))
        if not acquisition['success']:
            raise ValueError('PACDM acquisition failed: '+str(acquisition))
        N,info=self.solver.mapping(q)
        if not info['success']:
            raise ValueError('PACDM mapping failed: '+str(info))
        return q,N,info

    def terms(self,measured_q,length_velocity):
        g=self.graph; q,N,info=self.assemble(measured_q)
        v=N@np.asarray(length_velocity); h=1e-5/max(1.,np.linalg.norm(v))
        J=g.residual(q)[1]
        Jdot=(g.residual(q+h*v)[1]-g.residual(q-h*v)[1])/(2*h)
        rows=np.asarray(info['rows']); curvature=np.zeros(g.n)
        curvature[g.passive]=-np.linalg.solve(J[np.ix_(rows,g.passive)],(Jdot@v)[rows])
        P,E=g.poses(q); Pp,Ep=g.poses(q+h*v); Pm,Em=g.poses(q-h*v)
        H=np.zeros((6,6)); bias=np.zeros(6)
        gravity=np.asarray(self.cmg['gravity_m_s2'])
        for body in self.bodies:
            Jv,Jw=_body_jacobians(body,P,E)
            Jvp,Jwp=_body_jacobians(body,Pp,Ep)
            Jvm,Jwm=_body_jacobians(body,Pm,Em)
            Bv=Jv@N; Bw=Jw@N; R=P[body['id']][:3,:3]
            I=R@np.asarray(body['inertia_kg_m2'])@R.T; m=body['mass_kg']
            omega=Jw@v
            a0=Jv@curvature+(Jvp-Jvm)@v/(2*h)
            alpha0=Jw@curvature+(Jwp-Jwm)@v/(2*h)
            H+=m*Bv.T@Bv+Bw.T@I@Bw
            bias+=Bv.T@(m*(a0-gravity))+Bw.T@(I@alpha0+np.cross(omega,I@omega))
        platform=P['platform']; twist=E['platform']
        W=np.vstack((twist[3:]-skew(platform[:3,3])@twist[:3],twist[:3]))@N
        if not np.all(np.isfinite(H)) or np.linalg.eigvalsh(H)[0]<=0:
            raise ValueError('Reduced mass is not finite positive definite')
        return dict(q=q,N=N,velocity=v,curvature=curvature,mass=H,bias=bias,
                    wrench_map=W,poses=P,twists=E,info=info)


def check_reference(root):
    """CPU algebra check against archived Pinocchio feedforward, NOT Isaac."""
    root=Path(root); cmg=physical_case_cmg(root,'nominal')
    with np.load(root/'baseline/reference.npz',allow_pickle=False) as z:
        ref={k:z[k] for k in z.files}
    model=ReducedModel(cmg,ref['q'][0]); errors=[]
    for k in np.unique(np.linspace(0,len(ref['time'])-1,13,dtype=int)):
        a=model.terms(ref['q'][k],ref['length_velocity'][k])
        f=a['mass']@ref['length_acceleration'][k]+a['bias']
        errors.append(float(np.max(abs(f-ref['feedforward_force'][k]))))
    limit=MODEL_LIMITS['archived_feedforward_consistency_N']
    return dict(passed=bool(max(errors)<=limit),samples=len(errors),max_force_difference_N=max(errors),
                limit_N=limit,scope='CPU Newton-Euler algebra versus archived Pinocchio feedforward; not a new physics rollout.')


def audit_case(data,cmg,seed):
    """Recompute sampled geometry, tangent velocity, and applied-force balance.

    ldd is the forward difference over [t_k,t_k+dt], matched to the zero-order
    held force at k. This tests local dynamics, not a fitted controller error.
    The reported relative/absolute tolerance allows finite-step solver error.
    """
    result=dict(passed=False,status='UNVERIFIED',thresholds=MODEL_LIMITS,
                sample_schedule_s=SAMPLE_TIMES.tolist(),samples=[],checks={})
    try:
        t=np.asarray(data['time'],float); count=len(t)
        if t.shape!=(count,) or count<3 or not np.all(np.isfinite(t)) or abs(t[-1]-22.)>1e-7:
            raise ValueError('A complete 22 s recording is required')
        dt=float(t[1]-t[0])
        if dt<=0 or not np.allclose(np.diff(t),dt,atol=2e-7,rtol=0):
            raise ValueError('Invalid sampling times')
        names=[str(x) for x in data['body_names']]
        expected={'platform'}|{f'leg_{i}_{part}' for i in range(6) for part in ['yoke','barrel','rod']}
        if len(names)!=19 or set(names)!=expected:
            raise ValueError('Raw body names are missing, duplicated, or unexpected')
        for key,shape in [('body_transforms',(count,19,7)),('body_velocities',(count,19,6)),
                          ('body_coms',(19,3)),('q',(count,24)),('velocity',(count,24)),
                          ('force',(count,6)),('wrench',(count,6))]:
            if np.shape(data[key])!=shape or not np.all(np.isfinite(data[key])):
                raise ValueError('Invalid raw schema: '+key)
        model=ReducedModel(cmg,seed); active=model.graph.active
        force_residual=[]; force_scale=[]; p_errors=[]; r_errors=[]; v_errors=[]; w_errors=[]
        closure=[]; tangent=[]; mass_min=[]; ranks=[]
        for target in SAMPLE_TIMES:
            k=int(round(target/dt))
            if k<0 or k+1>=count or abs(t[k]-target)>dt/2+1e-8:
                raise ValueError('Missing scheduled sample')
            a=model.terms(data['q'][k],data['velocity'][k,active])
            ldd=(data['velocity'][k+1,active]-data['velocity'][k,active])/dt
            lhs=a['mass']@ldd+a['bias']
            rhs=data['force'][k]+a['wrench_map'].T@data['wrench'][k]
            residual=lhs-rhs; force_residual.append(residual); force_scale.append(rhs)
            pe=[]; re=[]; ve=[]; we=[]
            for i,name in enumerate(names):
                actual=data['body_transforms'][k,i]
                R=Rotation.from_quat(actual[3:].copy()).as_matrix()
                T=a['poses'][name]; E=a['twists'][name]
                com=T[:3,3]+T[:3,:3]@data['body_coms'][i]
                expected_v=(E[3:]-skew(com)@E[:3])@a['velocity']
                expected_w=E[:3]@a['velocity']
                pe.append(np.linalg.norm(T[:3,3]-actual[:3]))
                re.append(Rotation.from_matrix(T[:3,:3].T@R).magnitude())
                ve.append(np.linalg.norm(expected_v-data['body_velocities'][k,i,:3]))
                we.append(np.linalg.norm(expected_w-data['body_velocities'][k,i,3:]))
            p_errors+=pe; r_errors+=re; v_errors+=ve; w_errors+=we
            info=a['info']; closure.append(info['residual_inf']);tangent.append(info['tangent_residual'])
            mass_min.append(float(np.linalg.eigvalsh(a['mass'])[0]))
            ranks.append(info['rank_full']==36 and info['rank_passive']==36)
            result['samples'].append(dict(time_s=float(t[k]),force_residual_N=residual.tolist(),
                max_body_position_error_m=float(max(pe)),max_body_orientation_error_rad=float(max(re)),
                max_body_velocity_error_m_s=float(max(ve)),max_body_angular_velocity_error_rad_s=float(max(we)),
                min_mass_eigenvalue=mass_min[-1],rank_full=info['rank_full'],rank_passive=info['rank_passive']))
        rms=np.sqrt(np.mean(np.square(force_residual),axis=0))
        scale=np.sqrt(np.mean(np.square(force_scale),axis=0))
        limits=MODEL_LIMITS['force_residual_abs_N']+MODEL_LIMITS['force_residual_relative']*scale
        checks=result['checks']
        def bounded(key,value,limit,unit):
            checks[key]=dict(passed=bool(value<=limit),value=float(value),limit=float(limit),unit=unit)
        bounded('body_position',max(p_errors),MODEL_LIMITS['body_position_m'],'m')
        bounded('body_orientation',max(r_errors),MODEL_LIMITS['body_orientation_rad'],'rad')
        bounded('body_velocity',max(v_errors),MODEL_LIMITS['velocity_m_s'],'m/s')
        bounded('body_angular_velocity',max(w_errors),MODEL_LIMITS['velocity_rad_s'],'rad/s')
        bounded('full_closure',max(closure),MODEL_LIMITS['closure'],'mixed SE3 component')
        bounded('tangent_residual',max(tangent),MODEL_LIMITS['tangent'],'mixed')
        checks['rank_and_positive_mass']=dict(passed=bool(all(ranks) and min(mass_min)>0),value=min(mass_min),unit='mixed-coordinate mass eigenvalue')
        for i in range(6):bounded(f'leg_{i+1}_force_balance_rms',rms[i],limits[i],'N')
        result.update(passed=bool(all(x['passed'] for x in checks.values())),sample_count=len(result['samples']),
            force_residual_rms_N=rms.tolist(),force_residual_limits_N=limits.tolist(),
            scope='Sampled CMG/PACDM assembly, tangent velocity, and Newton-Euler force balance against native PhysX bodies. No hardware or all-state guarantee.')
        result['status']='PASS' if result['passed'] else 'FAIL'
    except Exception as exc:
        result.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
    return result
