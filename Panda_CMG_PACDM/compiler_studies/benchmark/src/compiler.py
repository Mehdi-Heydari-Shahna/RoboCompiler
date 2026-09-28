"""Scoped physical-tree/affine-coupling compiler for the Franka benchmark.

New extension. Original PACDM is not edited. The input contains no coordinate
indices, kinematic paths, residual/Jacobian code, or reduction/moment matrices.
Supported: rooted fixed/revolute/prismatic tree, one affine coupling of two
prismatic coordinates, one SE(3) tool site with seven upstream joints. This is
not arbitrary CAD import, automatic task choice, or a new optimal tree method.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation

class ModelError(ValueError):pass

def transform(T,name):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-11,rtol=0):raise ModelError(name+': invalid transform')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-9,rtol=0) or abs(np.linalg.det(T[:3,:3])-1)>1e-9:raise ModelError(name+': not SO(3)')
    return T

def inverse(T):
    o=np.eye(4);o[:3,:3]=T[:3,:3].T;o[:3,3]=-o[:3,:3]@T[:3,3];return o

@dataclass
class Compilation:
    cmg:dict
    plan:dict
    source:dict
    S:np.ndarray
    offset:np.ndarray
    C:np.ndarray
    q_seed:np.ndarray
    lower:np.ndarray
    upper:np.ndarray
    def physical(self,x):return self.S@np.asarray(x)+self.offset
    def reduced(self,q):return np.asarray(q)[self.plan['free_physical_indices']].copy()

def from_original(cmg,seed):
    """A one-time source-data migration, separate from the tested compiler."""
    joints=[]
    for j in cmg['joints']:
        z={k:deepcopy(v)for k,v in j.items() if k in ['id','type','axis','limits','armature','damping']}
        z.update(body_a=j['base_body'],body_b=j['follower_body'],T_AJ=j['T_BJ'],T_BJ=j['T_FJ']);joints.append(z)
    # The source equality is q1=q2, so either declared dependent choice is valid.
    c=cmg['coordinate_couplings'][0]
    if c['polycoef'] != [0.,1.,0.,0.,0.]:raise ModelError('Migration expects the supplied symmetric coupling')
    return dict(schema='physical-franka-input/1.0',name=cmg['name'],root=cmg['root_body'],gravity_m_s2=cmg['gravity_m_s2'],
        bodies=[{k:deepcopy(b[k])for k in ['id','mass_kg','com_m','inertia_kg_m2']}for b in cmg['bodies']],joints=joints,
        affine_couplings=[dict(id=c['id'],master=c['joint1'],slave=c['joint2'],multiplier=1.,offset=0.)],
        tool=deepcopy(cmg['tool']),redundancy_joint='joint3',target_reference_rotation=Rotation.from_euler('x',np.pi).as_matrix().tolist(),
        actuators=[dict(id=a['id'],transmission=a['transmission'],force_range=a['force_range'])for a in cmg['actuation']['actuators']],
        seed=dict(zip(cmg['coordinate_ids'],map(float,seed))),
        source=deepcopy(cmg['source']))

def compile_graph(data):
    s=deepcopy(data);bodies=s['bodies'];joints=s['joints'];root=s['root']
    bmap={b['id']:b for b in bodies};jmap={j['id']:j for j in joints}
    if len(bmap)!=len(bodies) or len(jmap)!=len(joints):raise ModelError('Duplicate body/joint name')
    if root not in bmap or len(joints)!=len(bodies)-1:raise ModelError('Expected a connected physical tree')
    for b in bodies:
        m=float(b['mass_kg']);c=np.asarray(b['com_m']);I=np.asarray(b['inertia_kg_m2'])
        if not np.isfinite(m) or m<0 or c.shape!=(3,) or I.shape!=(3,3) or not np.isfinite(c).all() or not np.isfinite(I).all():raise ModelError('Invalid body inertia record')
        if not np.allclose(I,I.T,atol=1e-12,rtol=0) or np.linalg.eigvalsh(I).min()<-1e-12 or (m==0 and np.any(I)):raise ModelError('Invalid mass/inertia tensor')
    adj={b:[]for b in bmap}
    for j in joints:
        a,b=j['body_a'],j['body_b']
        if a not in bmap or b not in bmap or a==b:raise ModelError('Invalid joint endpoints')
        if j['type'] not in ['fixed','revolute','prismatic']:raise ModelError('Unsupported joint type')
        transform(j['T_AJ'],j['id']);transform(j['T_BJ'],j['id'])
        if j['type']!='fixed':
            axis=np.asarray(j['axis']);lo=j['limits']['lower'];hi=j['limits']['upper']
            if axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-10:raise ModelError('Invalid axis')
            if not np.isfinite(lo+hi) or lo>=hi:raise ModelError('Invalid limits')
        adj[a].append((j,b,1));adj[b].append((j,a,-1))
    ordered=[];paths={root:[]};seen={root};used=set()
    def visit(body):
        for j,child,d in sorted(adj[body],key=lambda v:v[0]['id']):
            if j['id'] in used:continue
            if child in seen:raise ModelError('Cycle/multiple parent')
            used.add(j['id']);seen.add(child)
            e={k:deepcopy(v)for k,v in j.items()if k in ['id','type','axis','limits','armature','damping']}
            e.update(base_body=body,follower_body=child,T_BJ=deepcopy(j['T_AJ']if d==1 else j['T_BJ']),T_FJ=deepcopy(j['T_BJ']if d==1 else j['T_AJ']))
            if e['type']!='fixed':e['axis']=(d*np.asarray(e['axis'])).tolist()
            ordered.append(e);paths[child]=paths[body]+[j['id']];visit(child)
    visit(root)
    if seen!=set(bmap):raise ModelError('Disconnected tree')
    ids=[j['id']for j in ordered if j['type']!='fixed'];n=len(ids)
    if n!=9:raise ModelError('Scoped study expects nine physical coordinates')
    if set(s['seed'])!=set(ids):raise ModelError('Named seed must cover moving joints exactly')
    q=np.array([s['seed'][i]for i in ids]);lo=np.array([jmap[i]['limits']['lower']for i in ids]);hi=np.array([jmap[i]['limits']['upper']for i in ids])
    if not np.isfinite(q).all() or np.any(q<lo) or np.any(q>hi):raise ModelError('Invalid seed/bounds')
    if len(s['affine_couplings'])!=1:raise ModelError('One affine coupling required for this study')
    coupling=s['affine_couplings'][0];master,slave=coupling['master'],coupling['slave'];r=float(coupling['multiplier']);b=float(coupling['offset'])
    if master not in ids or slave not in ids or master==slave or not np.isfinite(r+b) or r==0:raise ModelError('Invalid affine coupling')
    if jmap[master]['type']!='prismatic' or jmap[slave]['type']!='prismatic':raise ModelError('Coupling units must agree (prismatic here)')
    mi,si=ids.index(master),ids.index(slave);free=[i for i in range(n)if i!=si];free_ids=[ids[i]for i in free]
    S=np.zeros((n,n-1));S[free]=np.eye(n-1);S[si,free.index(mi)]=r;c=np.zeros(n);c[si]=b
    C=np.zeros((1,n));C[0,si]=1;C[0,mi]=-r
    low=lo[free].copy();high=hi[free].copy();bounds=sorted([(lo[si]-b)/r,(hi[si]-b)/r]);k=free.index(mi)
    low[k]=max(low[k],bounds[0]);high[k]=min(high[k],bounds[1])
    if low[k]>=high[k] or abs(float((C@q)[0])-b)>1e-10:raise ModelError('Inconsistent affine bounds/seed')
    tool=s['tool'];transform(tool['T_body_tool'],'tool')
    if tool['body']not in bmap:raise ModelError('Missing tool body')
    path=paths[tool['body']];arm=[i for i in ids if i in path]
    if len(arm)!=7 or master in arm or slave in arm:raise ModelError('Expected seven-joint tool path independent of coupled fingers')
    red=s['redundancy_joint']
    if red not in arm:raise ModelError('Redundancy joint must be on the tool path')
    RR=np.asarray(s['target_reference_rotation']);T=np.eye(4);T[:3,:3]=RR;transform(T,'target chart')
    acts=s['actuators'];B=np.zeros((n,len(acts)))
    if len({a['id']for a in acts})!=len(acts)or len(acts)!=8:raise ModelError('Eight distinct actuator records required')
    for k,a in enumerate(acts):
        if not a['transmission']:raise ModelError('Empty transmission')
        for j,coef in a['transmission'].items():
            if j not in ids or not np.isfinite(coef):raise ModelError('Invalid actuator transmission')
            B[ids.index(j),k]=coef
    if np.linalg.matrix_rank(S.T@B)!=8:raise ModelError('Independent physical actuation is rank deficient')
    cmg=dict(name=s['name'],schema='franka-compiled-tree/1.0',root_body=root,bodies=bodies,joints=ordered,coordinate_ids=ids,
        independent_ids=free_ids,closures=[],q_reference=q.tolist(),gravity_m_s2=s['gravity_m_s2'],tool=tool,
        actuation=dict(actuators=acts,moment_matrix=B.tolist()),armature=[float(jmap[k].get('armature',0))for k in ids],
        damping=[float(jmap[k].get('damping',0))for k in ids],affine_couplings=s['affine_couplings'])
    plan=dict(physical_bodies=len(bodies),physical_joints=len(joints),structural_cycle_count=0,
        original_physical_coordinates=9,physical_coupling_rank=1,physical_mobility=8,
        original_augmented_coordinates=15,compiled_augmented_coordinates=14,original_passive=7,compiled_passive=6,
        original_residual_rows=12,original_rank=7,compiled_residual_rows=6,compiled_rank=6,independent_task_coordinates=8,
        free_physical_indices=free,free_physical_ids=free_ids,master_index=mi,slave_index=si,master_id=master,slave_id=slave,
        reduced_master_index=free.index(mi),reduced_redundancy_index=free_ids.index(red),redundancy_id=red,
        arm_ids=arm,tool_path=path,arm_dependency_input_indices=list(range(7)),algebraic_input_indices=[7],
        affine_lift=S.tolist(),affine_offset=c.tolist(),constraint_matrix=C.tolist(),constraint_rhs=[b],
        reduced_actuation_matrix=(S.T@B).tolist(),passive_sparsity=np.ones((6,6),int).tolist(),
        scope='Task chart and scalar physical coupling; no structural arm loop. Affine lift is generated, not discovered from sensor data.',
        input_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    return Compilation(cmg,plan,s,S,c,C,q,low,high)

def variant_inputs(source):
    out=[('nominal',deepcopy(source))]
    for scale in [.98,1.02]:
        s=deepcopy(source)
        for j in s['joints']:
            for k in ['T_AJ','T_BJ']:
                T=np.array(j[k]);T[:3,3]*=scale;j[k]=T.tolist()
        for b in s['bodies']:b['com_m']=(scale*np.array(b['com_m'])).tolist();b['inertia_kg_m2']=(scale**2*np.array(b['inertia_kg_m2'])).tolist()
        T=np.array(s['tool']['T_body_tool']);T[:3,3]*=scale;s['tool']['T_body_tool']=T.tolist();out.append((f'geometry_{scale}',s))
    for name,shift,rot in [('longer_tool',[0,0,.02],[0,0,0]),('offset_tilted_tool',[.015,0,.01],[.08,-.04,.05])]:
        s=deepcopy(source);T=np.eye(4);T[:3,3]=shift;T[:3,:3]=Rotation.from_rotvec(rot).as_matrix();s['tool']['T_body_tool']=(np.array(s['tool']['T_body_tool'])@T).tolist();out.append((name,s))
    for m in [.15,.30]:
        s=deepcopy(source);b=next(b for b in s['bodies']if b['id']==s['tool']['body']);b['mass_kg']+=m;out.append((f'hand_COM_point_payload_{m}',s))
    s=deepcopy(source)
    for b in s['bodies']:
        if b['id']!=s['root']:b['mass_kg']*=1.1;b['inertia_kg_m2']=(1.1*np.array(b['inertia_kg_m2'])).tolist()
    out.append(('moving_inertias_1p1',s))
    s=deepcopy(source);rng=np.random.default_rng(271828);Hs={}
    for b in s['bodies']:
        if b['id']==s['root']:continue
        H=np.eye(4);H[:3,:3]=Rotation.from_rotvec(rng.uniform(-.2,.2,3)).as_matrix();H[:3,3]=rng.uniform(-.006,.006,3);Hs[b['id']]=H
        b['com_m']=(H[:3,:3].T@(np.array(b['com_m'])-H[:3,3])).tolist();b['inertia_kg_m2']=(H[:3,:3].T@np.array(b['inertia_kg_m2'])@H[:3,:3]).tolist()
    for j in s['joints']:
        for bk,tk in [('body_a','T_AJ'),('body_b','T_BJ')]:j[tk]=(inverse(Hs.get(j[bk],np.eye(4)))@np.array(j[tk])).tolist()
    s['tool']['T_body_tool']=(inverse(Hs[s['tool']['body']])@np.array(s['tool']['T_body_tool'])).tolist();out.append(('body_frame_reexpression',s))
    s=deepcopy(source)
    for j in s['joints']:
        j['body_a'],j['body_b']=j['body_b'],j['body_a'];j['T_AJ'],j['T_BJ']=j['T_BJ'],j['T_AJ']
        if j['type']!='fixed':j['axis']=(-np.array(j['axis'])).tolist()
    out.append(('reversed_edge_directions',s))
    s=deepcopy(source)
    for k in ['bodies','joints','actuators']:s[k].reverse()
    out.append(('reversed_records_actuators',s))
    s=deepcopy(source);bm={b['id']:f'b_{i:02d}'for i,b in enumerate(s['bodies'])};jm={j['id']:f'j_{i:02d}'for i,j in enumerate(s['joints'])}
    for b in s['bodies']:b['id']=bm[b['id']]
    for j in s['joints']:j['id']=jm[j['id']];j['body_a']=bm[j['body_a']];j['body_b']=bm[j['body_b']]
    for a in s['actuators']:a['transmission']={jm[k]:v for k,v in a['transmission'].items()}
    for c in s['affine_couplings']:c['master']=jm[c['master']];c['slave']=jm[c['slave']]
    s['root']=bm[s['root']];s['tool']['body']=bm[s['tool']['body']];s['redundancy_joint']=jm[s['redundancy_joint']];s['seed']={jm[k]:v for k,v in s['seed'].items()};out.append(('opaque_names',s))
    assert len(out)==12
    return out
