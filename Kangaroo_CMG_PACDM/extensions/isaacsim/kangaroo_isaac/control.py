"""The v22 feedforward-plus-PD law with the physical per-port force limits.

PACDM provides the closed-manifold reference and feedforward. Feedback uses
actual native joint measurements. This is NOT an online PACDM dynamics solve.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, asdict
import numpy as np
from scipy.interpolate import CubicSpline
from .model import ROOT, finite
from .contact_model import CONTACT_MODELS, SOURCE_STIFFNESS_PER_S2, SOURCE_DAMPING_PER_S

@dataclass(frozen=True)
class Config:
    duration_s: float=10.
    dt_s: float=.000025
    control_period_s: float=.001
    sample_period_s: float=.005
    render_period_s: float=1/30
    drop_height_m: float=.05
    push_force_N: float=50.
    push_start_s: float=6.7
    push_duration_s: float=.25
    friction: float=.8
    actuator_time_constant_s: float=.002
    command_slew_N_s: float=1500000.
    kp_N_m: float=300000.
    kd_N_s_m: float=4000.
    # Numerical solver settings, NOT tuned or natively validated controller gains.
    # 23.0.5 default: PGS 64/8 (see contact_model.py and docs/CONTACT_COMPLIANCE_REPAIR.md).
    # The 23.0.4 profile 'tgs_32_1' remains selectable with --solver-profile.
    solver_position_iterations: int=64
    solver_velocity_iterations: int=8
    solver_type: str='PGS'
    external_forces_every_iteration: bool=False
    solver_profile: str='pgs_64_8'
    physx_threads: int=0
    contact_offset_m: float=.001
    rest_offset_m: float=0.
    feedforward_contact_mode: str='contact_envelope'
    # Sole/floor contact law. Default: the source MuJoCo soft contact mapped to a
    # PhysX implicit acceleration spring (see contact_model.py). 'rigid' is the
    # legacy 23.0.4 contact, kept only for controlled comparison.
    contact_model: str='source_compliance'
    contact_stiffness_per_s2: float=SOURCE_STIFFNESS_PER_S2
    contact_damping_per_s: float=SOURCE_DAMPING_PER_S
    no_contact: bool=False
    no_loops: bool=False
    passive: bool=False
    no_feedforward: bool=False

    def validate(self):
        for k,v in asdict(self).items():
            if isinstance(v,(float,int)) and not np.isfinite(v):raise ValueError(f'Nonfinite {k}')
        if min(self.duration_s,self.dt_s,self.control_period_s,self.sample_period_s,self.actuator_time_constant_s,self.push_duration_s,self.render_period_s)<=0:
            raise ValueError('Time intervals must be positive')
        if self.duration_s>10. or min(self.friction,self.drop_height_m,self.contact_offset_m)<0:
            raise ValueError('Invalid duration, friction or geometry settings')
        if self.contact_offset_m<=self.rest_offset_m:raise ValueError('Contact offset must exceed rest offset')
        if min(self.kp_N_m,self.kd_N_s_m,self.command_slew_N_s)<0:raise ValueError('Gains and slew limit must be nonnegative')
        for name,low in (('solver_position_iterations',1),('solver_velocity_iterations',0),('physx_threads',0)):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or not low<=value<=255:
                raise ValueError(f'{name} must be an integer in [{low},255]')
        if self.solver_type not in ('TGS','PGS'):raise ValueError('Unknown solver type')
        if not isinstance(self.external_forces_every_iteration,bool):raise ValueError('External-force flag must be boolean')
        if self.solver_profile not in SOLVER_PROFILES:raise ValueError('Unknown solver profile')
        if self.feedforward_contact_mode not in ('contact_envelope','force_threshold'):
            raise ValueError('Unknown feedforward contact mode')
        if self.contact_model not in CONTACT_MODELS:raise ValueError('Unknown contact model')
        if self.contact_model=='source_compliance' and not (
                0<self.contact_stiffness_per_s2<float('inf') and 0<self.contact_damping_per_s<float('inf')):
            raise ValueError('Compliant contact stiffness and damping must be positive and finite')
        for name in ('duration_s','control_period_s','sample_period_s'):
            ratio=getattr(self,name)/self.dt_s
            if abs(ratio-round(ratio))>1e-7:raise ValueError(f'{name} must be an integer multiple of dt')
        return self


# Fixed, named choices; never change gains, friction, trajectory, or gates to pass.
# 'pgs_64_8' is the 23.0.5 default (PGS keeps the sole friction anchors from
# oscillating; 64 doubles to 128 <= 255 for the solver_check case). 'tgs_32_1'
# is the 23.0.4 default. The legacy profile reproduces the reported 23.0.2 setup.
SOLVER_PROFILES={
    'pgs_64_8':dict(solver_type='PGS',solver_position_iterations=64,solver_velocity_iterations=8,external_forces_every_iteration=False),
    'tgs_32_1':dict(solver_type='TGS',solver_position_iterations=32,solver_velocity_iterations=1,external_forces_every_iteration=True),
    'tgs_64_1':dict(solver_type='TGS',solver_position_iterations=64,solver_velocity_iterations=1,external_forces_every_iteration=True),
    'tgs_64_0':dict(solver_type='TGS',solver_position_iterations=64,solver_velocity_iterations=0,external_forces_every_iteration=True),
    'pgs_128_8':dict(solver_type='PGS',solver_position_iterations=128,solver_velocity_iterations=8,external_forces_every_iteration=False),
    'legacy_128_32':dict(solver_type='TGS',solver_position_iterations=128,solver_velocity_iterations=32,external_forces_every_iteration=False),
}


def case_config(name,dt=.000025,solver_profile='pgs_64_8',physx_threads=0,feedforward_contact_mode='contact_envelope',
                contact_model='source_compliance'):
    if solver_profile not in SOLVER_PROFILES:raise ValueError(f'Unknown solver profile {solver_profile}')
    base=Config(dt_s=dt,solver_profile=solver_profile,physx_threads=physx_threads,
                feedforward_contact_mode=feedforward_contact_mode,contact_model=contact_model,
                **SOLVER_PROFILES[solver_profile])
    variants={
        'nominal':{},
        'refined':{'dt_s':dt/2},
        'solver_check':{'solver_position_iterations':min(255,2*base.solver_position_iterations)},
        'low_friction':{'friction':.4},
        'higher_drop_push':{'drop_height_m':.08,'push_force_N':80.},
        'slow_actuators':{'actuator_time_constant_s':.004},
        'no_feedforward':{'no_feedforward':True},
        'no_contact':{'duration_s':.3,'no_contact':True},
        'no_loops':{'duration_s':.3,'no_loops':True},
        'passive':{'duration_s':1.,'passive':True},
    }
    if name not in variants:raise ValueError(f'Unknown case {name}')
    return replace(base,**variants[name]).validate()

CASES=('nominal','refined','low_friction','higher_drop_push','slow_actuators','no_feedforward','no_contact','no_loops','passive')


class Reference:
    def __init__(self,path=None):
        with np.load(path or ROOT/'data/contact_reference.npz',allow_pickle=False) as a:
            self.data={k:a[k].copy() for k in a.files}
        a=self.data;t=a['t']
        if len(t)!=401 or abs(t[0])>1e-12 or abs(t[-1]-10)>1e-12 or np.any(np.diff(t)<=0):
            raise ValueError('Unexpected source reference clock')
        for k,v in a.items():finite(v,name='reference '+k)
        self.u=CubicSpline(t,a['u'],axis=0,extrapolate=False)
        self.force=CubicSpline(t,a['force'],axis=0,extrapolate=False)
        self.base=CubicSpline(t,a['base'],axis=0,extrapolate=False)
        self.rotvec=CubicSpline(t,a['rotvec'],axis=0,extrapolate=False)

    def prepare(self,cfg):
        t=np.arange(round(cfg.duration_s/cfg.control_period_s)+1)*cfg.control_period_s
        return self.u(t),self.u(t,1),self.force(t)


class Controller:
    def __init__(self,model,ref,cfg):
        self.model,self.ref,self.cfg=model,ref,cfg.validate()
        self.targets,self.veltargets,self.feedforward=ref.prepare(cfg)
        self.command=np.zeros(12);self.force=np.zeros(12)
        self.control_stride=round(cfg.control_period_s/cfg.dt_s)
        self.saturation_updates=0;self.update_count=0
        self.alpha=-np.expm1(-cfg.dt_s/cfg.actuator_time_constant_s)
        self.last_raw=np.zeros(12)

    def update_command(self,k,q,qd,in_contact):
        if k%self.control_stride:return self.command.copy()
        i=k//self.control_stride
        target=self.targets[min(i,len(self.targets)-1)]
        vd=self.veltargets[min(i,len(self.targets)-1)]
        f=self.feedforward[min(i,len(self.targets)-1)] if in_contact and not self.cfg.no_feedforward else np.zeros(12)
        raw=f+self.cfg.kp_N_m*(target-finite(q,(12,),'motor q'))+self.cfg.kd_N_s_m*(vd-finite(qd,(12,),'motor v'))
        if self.cfg.passive:raw=np.zeros(12)
        bounded=np.clip(raw,self.model.bounds[:,0],self.model.bounds[:,1])
        self.saturation_updates+=int(np.any(abs(raw-bounded)>1e-9));self.update_count+=1
        max_delta=self.cfg.command_slew_N_s*self.cfg.control_period_s
        self.command+=np.clip(bounded-self.command,-max_delta,max_delta)
        self.last_raw=raw.copy()
        return self.command.copy()

    def advance_filter(self):
        # Called AFTER the physical step: native effort during that step is
        # force(t_k), matching the causal filter state in the source pipeline.
        self.force+=self.alpha*(self.command-self.force)
        if not np.isfinite(self.force).all():raise FloatingPointError('Nonfinite actuator state')
        return self.force.copy()


def external_push(t,cfg):
    a=(t-cfg.push_start_s)/cfg.push_duration_s
    return np.array([0.,cfg.push_force_N*np.sin(np.pi*a)**2 if 0<a<1 else 0.,0.])
