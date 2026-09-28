"""Independent physical witnesses and fixed Stewart perturbations.

The source CMG is used ONLY to generate matched initial conditions, prescribed
actuator commands, and independent verification. It is not read by the new
compiler and target solutions are never given to continuation correctors.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .bootstrap import PRIOR
from .compiler import from_original_cmg
from numpy_backend import NumpyTree,axis_rotation,cross_matrix
from vendor.pacdm_original import PointGraph, PACDM, exp, inv, log
from run_comparison import task_pose, pacdm_terms
from stewart.model import inverse_seed

VARIANTS = ['nominal','base_radius_minus_2pct','base_radius_plus_4pct',
            'platform_radius_minus_3pct','platform_radius_plus_4pct',
            'base_height_plus_15mm','base_anchor_offsets','platform_anchor_offsets',
            'payload_14kg','leg_inertia_and_com_edit','body_frame_reexpression',
            'record_and_actuator_permutation']


def rigid_pose(state):
    T=np.eye(4);T[:3,:3]=state['R'];T[:3,3]=state['p'];return T


def world_poses(backend,q):
    return {k:rigid_pose(v) for k,v in backend.forward(q).items()}


def set_geometry(cmg,base=None,top=None):
    geom=cmg['geometry']
    B=np.asarray(geom['base_anchors_m'] if base is None else base,float)
    A=np.asarray(geom['platform_anchors_m'] if top is None else top,float)
    p=np.asarray(geom['nominal_pose'][:3]);R=Rotation.from_euler('ZYX',geom['nominal_pose'][3:]).as_matrix()
    rotations=[]
    for i,(b,a) in enumerate(zip(B,A)):
        u=p+R@a-b;u=u/np.linalg.norm(u);z=np.array([0.,0.,1.]);v=np.cross(z,u)
        angle=np.arctan2(np.linalg.norm(v),z@u)
        R0=np.eye(3) if np.linalg.norm(v)<1e-14 else Rotation.from_rotvec(v/np.linalg.norm(v)*angle).as_matrix()
        rotations.append(R0.tolist())
        j=next(j for j in cmg['joints'] if j['id']==f'leg_{i}_u_x')
        T=np.eye(4);T[:3,:3]=R0;T[:3,3]=b;j['T_BJ']=T.tolist()
        cmg['closures'][i]['point2_m']=a.tolist()
    geom['base_anchors_m']=B.tolist();geom['platform_anchors_m']=A.tolist();geom['base_rotations']=rotations


def reexpress_body(cmg,bid,H):
    """H maps NEW body coordinates to OLD ones. Preserve physical properties."""
    Hi=inv(H)
    for j in cmg['joints']:
        if j['base_body']==bid:j['T_BJ']=(Hi@np.asarray(j['T_BJ'])).tolist()
        if j['follower_body']==bid:j['T_FJ']=(Hi@np.asarray(j['T_FJ'])).tolist()
    for c in cmg['closures']:
        for k in (1,2):
            if c[f'body{k}']==bid:c[f'point{k}_m']=(Hi@np.r_[c[f'point{k}_m'],1])[:3].tolist()
    b=next(b for b in cmg['bodies'] if b['id']==bid)
    b['com_m']=(Hi@np.r_[b['com_m'],1])[:3].tolist()
    b['inertia_kg_m2']=(H[:3,:3].T@np.asarray(b['inertia_kg_m2'])@H[:3,:3]).tolist()


def variants(nominal):
    rng=np.random.default_rng(91462)
    out=[]
    for name in VARIANTS:
        c=deepcopy(nominal);g=c['geometry'];B=np.array(g['base_anchors_m']);A=np.array(g['platform_anchors_m'])
        if name=='base_radius_minus_2pct':B[:,:2]*=.98;set_geometry(c,B,A)
        elif name=='base_radius_plus_4pct':B[:,:2]*=1.04;set_geometry(c,B,A)
        elif name=='platform_radius_minus_3pct':A[:,:2]*=.97;set_geometry(c,B,A)
        elif name=='platform_radius_plus_4pct':A[:,:2]*=1.04;set_geometry(c,B,A)
        elif name=='base_height_plus_15mm':B[:,2]+=.015;set_geometry(c,B,A)
        elif name=='base_anchor_offsets':B+=rng.uniform(-.004,.004,(6,3));set_geometry(c,B,A)
        elif name=='platform_anchor_offsets':A+=rng.uniform(-.004,.004,(6,3));set_geometry(c,B,A)
        elif name=='payload_14kg':
            b=next(b for b in c['bodies'] if b['id']=='payload');old=b['mass_kg'];b['mass_kg']=14.;b['inertia_kg_m2']=(np.array(b['inertia_kg_m2'])*14/old).tolist()
        elif name=='leg_inertia_and_com_edit':
            for b in c['bodies']:
                if b['id'].startswith('leg_'):
                    b['mass_kg']*=1.10;b['inertia_kg_m2']=(np.array(b['inertia_kg_m2'])*1.10).tolist();b['com_m'][1]+=.001
        elif name=='body_frame_reexpression':
            for bid,angles,offset in [('platform',[.16,-.09,.12],[.018,-.012,.009]),('leg_3_barrel',[-.07,.11,.08],[.01,.005,-.01]),('leg_2_rod',[.13,.04,-.08],[.006,-.007,.009])]:
                H=np.eye(4);H[:3,:3]=Rotation.from_euler('XYZ',angles).as_matrix();H[:3,3]=offset;reexpress_body(c,bid,H)
        elif name=='record_and_actuator_permutation':
            rng.shuffle(c['bodies']);rng.shuffle(c['joints']);rng.shuffle(c['closures'])
            c['independent_ids']=[c['independent_ids'][i] for i in [3,0,5,1,4,2]]
        out.append((name,c))
    return out


def make_state(compiled,source_cmg,source_backend,t):
    oldq=inverse_seed(source_cmg,task_pose(float(t)))
    W=world_poses(source_backend,oldq)
    q=compiled.seed_from_body_poses(dict(zip(source_cmg['coordinate_ids'],oldq)),W)
    return q,oldq,W


def physical_joint_gaps(source,poses):
    """Check ALL SIX physical spherical joints, including the tree joint."""
    gaps=[]
    for j in source['joints']:
        if j['type']!='spherical':continue
        a=poses[j['body_a']]@np.asarray(j['T_AJ'])
        b=poses[j['body_b']]@np.asarray(j['T_BJ'])
        gaps.append(np.linalg.norm(a[:3,3]-b[:3,3]))
    return float(max(gaps,default=0.))


def physical_acceptance(graph,backend,source,state,qa,truth_poses=None):
    if not np.all(np.isfinite(state)):
        return dict(success=False,gap_m=float('inf'),augmented_inf=float('inf'),within_bounds=False,rank=0,passive_rcond=0.,pose_error=float('inf'))
    q=state[:graph.nt];poses=world_poses(backend,q)
    gap=physical_joint_gaps(source,poses)
    residual,_,delta=graph.residual(state)
    N,mi=PACDM(graph).mapping(state)
    bounds=bool(np.all(state>=graph.lower-1e-10) and np.all(state<=graph.upper+1e-10))
    active_error=float(np.max(abs(state[graph.active]-qa)))
    err=0.
    if truth_poses is not None:
        err=max(float(np.max(abs(poses[b['id']]-truth_poses[b['id']]))) for b in source['bodies'])
    success=bool(bounds and gap<=1e-8 and active_error<=1e-12 and PACDM.physical_ok(residual,delta) and mi['success'])
    return dict(success=success,gap_m=gap,augmented_inf=float(np.max(abs(residual))),within_bounds=bounds,
                rank=int(mi['rank_full']),passive_rcond=float(mi['rcond']),pose_error=err,active_error_m=active_error)


def kkt_acceleration(backend,q,velocity,forces,wrench):
    geom=backend.geometry(q,velocity);J=geom['jacobian'];gamma=geom['acceleration_bias']
    M,b,U=backend.mass_bias(q,velocity)
    tau=geom['platform_jacobian'].T@wrench;tau[backend.active]+=forces
    K=np.block([[M,-J.T],[J,np.zeros((len(J),len(J)))]])
    sol=np.linalg.solve(K,np.r_[tau-b,-gamma]);a=sol[:backend.n]
    return a,dict(M=M,b=b,U=U,geometry=geom,tau=tau,force_residual=float(np.max(abs(K@sol-np.r_[tau-b,-gamma]))))


def reduced_acceleration(graph,backend,x,active_velocity,forces,wrench):
    N,v,c,info=pacdm_terms(graph,x,active_velocity)
    q=x[:graph.nt];geom=backend.geometry(q,v);M,b,U=backend.mass_bias(q,v)
    tau=geom['platform_jacobian'].T@wrench;tau[backend.active]+=forces
    Mr=N.T@M@N;ar=np.linalg.solve(Mr,N.T@(tau-b-M@c));a=N@ar+c
    return a,dict(N=N,v=v,c=c,M=M,b=b,U=U,Mr=Mr,ar=ar,tau=tau,geometry=geom,info=info)


def body_motion_errors(backend1,q1,v1,a1,backend2,q2,v2,a2,physical_bodies):
    S1=backend1.forward(q1,v1,acceleration=True);S2=backend2.forward(q2,v2,acceleration=True)
    velocity_error=acceleration_error=0.
    for name in physical_bodies:
        s,t=S1[name],S2[name]
        vel1=np.r_[s['Jv']@v1,s['Jw']@v1];vel2=np.r_[t['Jv']@v2,t['Jw']@v2]
        acc1=np.r_[s['a']+s['Jv']@a1,s['alpha']+s['Jw']@a1]
        acc2=np.r_[t['a']+t['Jv']@a2,t['alpha']+t['Jw']@a2]
        velocity_error=max(velocity_error,float(np.max(abs(vel1-vel2))))
        acceleration_error=max(acceleration_error,float(np.max(abs(acc1-acc2))))
    return velocity_error,acceleration_error


class ResidualOnly:
    """Fair residual-only evaluator for FD baselines; does NOT compute a Jacobian.

    It evaluates the same SE(3) equations as the unchanged vendor PointGraph.
    Equality is independently tested off the closure manifold before timing.
    """
    def __init__(self,graph):self.g=graph;self.calls=0
    def __call__(self,q):
        self.calls+=1;g=self.g;P={g.root:np.eye(4)}
        for j,E,F,k in g.edges:
            T=P[j['base_body']]@E
            if k is not None:
                axis=np.asarray(j['axis']);motion=np.eye(4)
                if j['type']=='revolute':motion[:3,:3]=axis_rotation(axis,q[k])
                else:motion[:3,3]=axis*q[k]
                T=T@motion
            P[j['follower_body']]=T@F
        rs=[]
        for i,c in enumerate(g.cuts):
            A=np.eye(4);A[:3,3]=c['point1_m'];A[:3,:3]=g.references[i]
            Tminus=P[c['body1']]@A
            for k,axis in enumerate(np.eye(3)):
                rot=np.eye(4);rot[:3,:3]=axis_rotation(axis,q[g.nt+3*i+k]);Tminus=Tminus@rot
            B=np.eye(4);B[:3,3]=c['point2_m']
            rs.append(log(inv(Tminus)@P[c['body2']]@B))
        return np.concatenate(rs)


def structural_sparsity(graph):
    """Generated conservative support; no sampled numerical-zero discovery."""
    S=np.zeros((6*graph.nc,graph.n),bool)
    for i,p in enumerate(graph.paths):
        joint_ids=set(p['root_to_body1']) ^ set(p['root_to_body2'])
        cols=[graph.ids.index(jid) for jid in joint_ids if jid in graph.ids]
        cols+=list(range(graph.nt+3*i,graph.nt+3*i+3))
        S[6*i:6*i+6,cols]=True
    return S
