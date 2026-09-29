"""Smooth task-space gait, assembled and differentiated using PACDM."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from .model import load_model
from .contact import ContactGraph

DURATION=26.
PERIOD=.8
SWING=.28
HURDLES=[(.48,.025),(.86,.035)]
PHASES=[(0,2,'Balance and push'),(2,8,'Precision trot'),(8,14,'Clear the rails'),(14,21,'Turn and crouch'),(21,24,'Exit'),(24,26,'Recover and dock')]

def smooth(s):
    return 10*s**3-15*s**4+6*s**5

def ground_height(x,y):
    return max([h for center,h in HURDLES if abs(x-center)<=.025 and abs(y)<.6]+[0.])

def body_spline():
    ts=[0,2,8,14,16,18,21,24,26]
    values=[[0,0,.32,0,0,0],[0,0,.32,0,0,0],
            [.46,0,.32,0,0,0],[1.20,.05,.32,.25,0,0],
            [1.35,.10,.26,.35,0,0],[1.58,.16,.26,.35,0,0],
            [1.96,.25,.26,.35,0,0],[2.20,.30,.32,.35,0,0],[2.20,.30,.32,.35,0,0]]
    return BPoly.from_derivatives(ts,[[v,np.zeros(6),np.zeros(6)] for v in values])

def target_trajectory(time):
    """Prescribed base6 and four foot centers, with C2 swing trajectories."""
    time=np.asarray(time);body=body_spline()
    offsets=np.array([[.1934,.142,0],[.1934,-.142,0],[-.1934,.142,0],[-.1934,-.142,0]])
    feet=np.tile(offsets[None],(len(time),1,1));feet[:,:,2]=.022
    fv=np.zeros_like(feet);fa=np.zeros_like(feet);stance=np.ones((len(time),4),bool)
    starts=offsets.copy();starts[:,2]=.022
    for k,start in enumerate(np.arange(2.,23.61,.4)):
        pair=[0,3] if k%2==0 else [1,2]
        end=start+SWING;landing_body=body(min(end+(PERIOD-SWING)/2,24.))
        R=Rotation.from_euler('ZYX',landing_body[3:6]).as_matrix()
        for leg in pair:
            target=landing_body[:3]+R@offsets[leg]
            target[2]=.022+ground_height(*target[:2])
            origin=starts[leg].copy();delta=target-origin
            mask=(time>=start)&(time<end);s=(time[mask]-start)/SWING
            h=smooth(s);hd=(30*s*s-60*s**3+30*s**4)/SWING
            hdd=(60*s-180*s*s+120*s**3)/SWING**2
            bump=64*s**3*(1-s)**3
            bd=64*(3*s*s-12*s**3+15*s**4-6*s**5)/SWING
            bdd=64*(6*s-36*s*s+60*s**3-30*s**4)/SWING**2
            feet[mask,leg]=origin+h[:,None]*delta
            fv[mask,leg]=hd[:,None]*delta;fa[mask,leg]=hdd[:,None]*delta
            feet[mask,leg,2]+=.085*bump;fv[mask,leg,2]+=.085*bd;fa[mask,leg,2]+=.085*bdd
            feet[time>=end,leg]=target;stance[mask,leg]=False;starts[leg]=target
    active=np.c_[body(time),feet.reshape(len(time),12)]
    av=np.c_[body(time,nu=1),fv.reshape(len(time),12)]
    aa=np.c_[body(time,nu=2),fa.reshape(len(time),12)]
    return active,av,aa,stance

def make_reference(root,dt=.01,path=None):
    """Assemble the PACDM reference and save it to path (default: root/data/reference.npz)."""
    root=Path(root);cmg=load_model();seed=np.asarray(cmg['q_reference'])
    graph=ContactGraph(cmg,seed);time=np.linspace(0,DURATION,round(DURATION/dt)+1)
    active,av,aa,stance=target_trajectory(time)
    qs=[];vs=[];acc=[];maps=[];infos=[]
    for k,t in enumerate(time):
        q,N,info=graph.solve_feet(active[k,:6],active[k,6:].reshape(4,3),seed if k==0 else None)
        v=N@av[k];_,J=graph.kinematics.points_and_jacobians(q);J=J.reshape(12,18)
        eps=1e-5
        Jp=graph.kinematics.points_and_jacobians(q+eps*v)[1].reshape(12,18)
        Jm=graph.kinematics.points_and_jacobians(q-eps*v)[1].reshape(12,18)
        a=np.r_[aa[k,:6],np.linalg.solve(J[:,6:],aa[k,6:]-J[:,:6]@aa[k,:6]-(Jp-Jm)@v/(2*eps))]
        qs.append(q);vs.append(v);acc.append(a);maps.append(N);infos.append(info)
    path=root/'data/reference.npz' if path is None else Path(path)
    np.savez_compressed(path,time=time,q=qs,v=vs,a=acc,N=maps,active=active,active_v=av,active_a=aa,feet=active[:,6:].reshape(-1,4,3),stance=stance)
    summary=dict(samples=len(time),duration_s=DURATION,dt_s=dt,constraint_rank=12,physical_coordinates=18,virtual_coordinates=12,
                 max_residual_inf=max(x['residual_inf'] for x in infos),max_tangent_residual=max(x['tangent_residual'] for x in infos),
                 min_rcond=min(x['rcond'] for x in infos),fallback_count=sum(x['continuation_fallback'] for x in infos))
    (root/'results/reference.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
