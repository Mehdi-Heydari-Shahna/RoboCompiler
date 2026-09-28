"""Contact-task references solved through the unchanged PACDM mapping.

Foot poses are targets for inverse kinematics, never welded to the world.
Only the twelve physical force ports are commanded during simulation.
"""
from pathlib import Path
import json, time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from reconstructed_model import whole_body, CutGraph, polish
from pacdm import PACDM, skew
from whole_body_dynamics import FloatingSource, add_drive_terms

ROOT = Path(__file__).resolve().parent
DURATION = 10.0
FOOT_NAMES = ['left_ankle_roll', 'right_ankle_roll']

def smooth(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 - 15*x + 6*x*x)

def pelvis_target(t):
    # z, lateral shift, yaw. Smooth position/velocity/acceleration at transitions.
    knots = [0, 1.4, 2.6, 3.5, 4.5, 5.3, 6.4, DURATION]
    values = np.array([[0,0,0], [0,0,0], [-.12,0,0], [-.12,.045,.055],
                       [-.12,-.045,-.055], [-.12,0,0], [0,0,0], [0,0,0]])
    i = min(len(knots)-2, max(0, np.searchsorted(knots, t, side='right')-1))
    a = smooth((t-knots[i])/(knots[i+1]-knots[i]))
    z, y, yaw = (1-a)*values[i] + a*values[i+1]
    return np.array([0., y, z]), Rotation.from_euler('z', yaw).as_matrix()

def foot_corners(poses):
    out=[]
    Rg=Rotation.from_quat([-.173648,0,0,.984808]).as_matrix()
    for body in FOOT_NAMES:
        T=poses[body]
        for a in [-.045,.045]:
            for b in [-.105,.105]:
                local=np.array([0,-.005,.04])+Rg@np.array([a,-.0125,b])
                out.append((body,local,T[:3,3]+T[:3,:3]@local))
    return out

def build_reference():
    start=time.time();c=whole_body();g=CutGraph(c,np.array(c['initial_seed']));solver=PACDM(g)
    q=np.load(ROOT/'data/configurations.npz')['home'].copy()
    q=polish(g,q);P,_=g.poses(q);corners=foot_corners(P)
    height=-min(x[2][2] for x in corners)
    fixed={name:P[name].copy() for name in FOOT_NAMES}
    ts=np.arange(round(DURATION/.025)+1)*.025
    Q=[];positions=[];rotations=[];maxres=0.;maxpose=0.;mincond=1.;iterations=0
    for k,t in enumerate(ts):
        p,R=pelvis_target(t);target={}
        for name in FOOT_NAMES:
            T=np.eye(4);T[:3,:3]=R.T@fixed[name][:3,:3]
            T[:3,3]=R.T@(fixed[name][:3,3]-p);target[name]=T
        for iteration in range(12):
            N,info=solver.mapping(q)
            if not info['success']:raise RuntimeError(info)
            P,E=g.poses(q);error=[];J=[]
            for name in FOOT_NAMES:
                error.extend(target[name][:3,3]-P[name][:3,3])
                error.extend(Rotation.from_matrix(target[name][:3,:3]@P[name][:3,:3].T).as_rotvec())
                J.append(np.vstack([E[name][3:]-skew(P[name][:3,3])@E[name][:3],E[name][:3]])@N)
            error=np.array(error);J=np.vstack(J)
            if np.max(abs(error))<1e-9:break
            du=np.linalg.solve(J,error)
            du*=min(1.,.006/max(1e-12,np.max(abs(du))))
            candidate,ci=solver.correct(q[g.active]+du,q[g.passive]+N[g.passive]@du,None,np.array(info['rows']),maxiter=12)
            if not ci['success']:raise RuntimeError({'time':float(t),'ik_iteration':iteration,**ci})
            q=polish(g,candidate);iterations+=1
        else:raise RuntimeError('Foot IK did not converge')
        maxres=max(maxres,info['residual_inf']);maxpose=max(maxpose,float(max(abs(error))));mincond=min(mincond,info['rcond'])
        Q.append(q.copy());positions.append(p+[0,0,height]);rotations.append(Rotation.from_matrix(R).as_rotvec())
        if k%60==0:print('Contact PACDM reference',k,'/',len(ts),flush=True)
    Q=np.array(Q);positions=np.array(positions);rotations=np.array(rotations)
    qs=CubicSpline(ts,Q[:,:76],axis=0);ps=CubicSpline(ts,positions,axis=0);rs=CubicSpline(ts,rotations,axis=0)
    f=FloatingSource(c);forces=[];cornerforces=[];minimum_normal=1e10;maxbase=0.;maxvelclosure=0.;maps=[]
    for k,t in enumerate(ts):
        R=Rotation.from_rotvec(rotations[k]).as_matrix()
        v=np.r_[ps(t,1),rs(t,1),qs(t,1)];acc=np.r_[ps(t,2),rs(t,2),qs(t,2)]
        s=f.evaluate(Q[k,:76],positions[k],R,v,[0,0,-9.81]);s=add_drive_terms(s,c)
        residual,J,_=f.internal.closure(s)
        # Native tree velocities are expressed in world coordinates here.
        N,_=solver.mapping(Q[k]);Nf=np.vstack([np.zeros((6,12)),N[:76]])
        maps.append(N[:76]);maxvelclosure=max(maxvelclosure,float(max(abs(J@v))))
        corners=foot_corners(s['poses']);jac=[]
        for body,local,point in corners:jac.append(f.internal.source._point(s,body,local)[1])
        G=np.hstack([jj[:,:6].T for jj in jac]);load=s['mass_matrix']@acc+s['bias_forces']
        # Minimum norm contact-force distribution satisfying all six base equations.
        cf=G.T@np.linalg.solve(G@G.T,load[:6]);cf=cf.reshape(8,3)
        minimum_normal=min(minimum_normal,float(min(cf[:,2])))
        external=sum(jj.T@ff for jj,ff in zip(jac,cf))
        maxbase=max(maxbase,float(max(abs((load-external)[:6]))))
        effort=np.linalg.solve(Nf.T@f.B,Nf.T@(load-external))
        forces.append(effort);cornerforces.append(cf)
    result=dict(t=ts,qaug=Q,q=Q[:,:76],base=positions,rotvec=rotations,
                u=Q[:,g.active],ud=qs(ts,1)[:,g.active],force=forces,
                tangent=maps,corner_force=cornerforces,home_height=height)
    np.savez_compressed(ROOT/'data/contact_reference.npz',**result)
    summary=dict(samples=len(ts),duration_s=DURATION,pacdm_all_rows_maximum=maxres,
                 foot_pose_maximum_error=maxpose,minimum_pacdm_rcond=mincond,
                 ik_iterations=iterations,reacquisitions=0,minimum_planned_corner_normal_N=minimum_normal,
                 maximum_base_equation_error=maxbase,maximum_interpolated_velocity_closure=maxvelclosure,
                 commanded_crouch_m=.12,commanded_lateral_shift_m=.045,commanded_yaw_rad=.055,
                 maximum_feedforward_force_N=float(np.max(np.abs(forces))),
                 actuator_travel_m=np.ptp(Q[:,g.active],axis=0).tolist(),elapsed_seconds=time.time()-start)
    (ROOT/'results/contact_reference.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary,flush=True);return result

if __name__=='__main__':build_reference()
