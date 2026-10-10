"""Physical floating-tree -> CMG + conditional Cartesian-task/support plans.

New extension, not functionality attributed to the unchanged original PACDM.
Input: bodies, unordered physical edges, a floating root, actuator declarations,
point sites, numerical parameters and a named seed. No leg memberships, paths,
closure functions, Jacobian functions or per-support coordinate partitions.
Scope: one floating body-root with scalar/fixed descendants, four three-DoF
point-task branches. The shared base must be independent for four-way assembly.
No physical loops are invented; ideal support edges are separate mode data.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation

BASE_IDS = ['base_x', 'base_y', 'base_z', 'base_yaw', 'base_pitch', 'base_roll']

class ModelError(ValueError):
    pass


def rigid(value, name):
    a=np.asarray(value,float)
    if a.shape!=(4,4) or not np.all(np.isfinite(a)) or not np.allclose(a[3],[0,0,0,1],atol=1e-11,rtol=0):
        raise ModelError(f'{name}: expected finite homogeneous transform')
    r=a[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-10,rtol=0) or abs(np.linalg.det(r)-1)>1e-10:
        raise ModelError(f'{name}: rotation is not proper orthonormal')
    return a


def export_physical(cmg):
    """One-time provenance-preserving extraction from the source CMG records."""
    base_ids=cmg['free_base_chart']['coordinates']
    joints={j['id']:j for j in cmg['joints']}
    root=joints[base_ids[-1]]['follower_body']
    chart_bodies={joints[i]['base_body'] for i in base_ids}
    bodies=[deepcopy(b) for b in cmg['bodies'] if b['id'] not in chart_bodies]
    physical=[]
    for j in cmg['joints']:
        if j['id'] in base_ids: continue
        physical.append(dict(id=j['id'],type=j['type'],body_a=j['base_body'],body_b=j['follower_body'],
            T_AJ=deepcopy(j['T_BJ']),T_BJ=deepcopy(j['T_FJ']),axis=deepcopy(j['axis']),
            limits=deepcopy(j['limits']),armature=j.get('armature',0.),damping=j.get('damping',0.),
            frictionloss=j.get('frictionloss',0.),stiffness=j.get('stiffness',0.)))
    seed=dict(zip(cmg['coordinate_ids'],cmg['q_reference']))
    return dict(schema='cmg.physical-floating-point-tree/1.0',name='Go2 physical graph',
        floating_root=root,base_chart='XYZ + intrinsic ZYX',bodies=bodies,joints=physical,
        actuators=[dict(id=a['id'],joint_id=a['joint'],gear=a['gear'],limits=a['ctrl_range'])
                   for a in cmg['actuation']['actuators']],
        point_sites=[dict(id=f['id'],body=f['body'],point_m=deepcopy(f['point_m']),radius_m=f['radius_m'])
                     for f in cmg['feet']],
        seed=dict(base_pose=[seed[k] for k in base_ids],joints={j['id']:seed[j['id']] for j in physical if j['type']!='fixed'}),
        gravity_m_s2=deepcopy(cmg['gravity_m_s2']),provenance=deepcopy(cmg['source']),
        note='Feet are task/contact sites, not permanent mechanical closure joints.')


def components(supports):
    """Task adjacency is shared DEPENDENT coordinates, never topology alone."""
    todo=set(range(len(supports)));groups=[]
    while todo:
        group={min(todo)};todo-=group
        changed=True
        while changed:
            changed=False
            used=set().union(*(set(supports[i]) for i in group))
            extra={i for i in todo if set(supports[i])&used}
            if extra:group|=extra;todo-=extra;changed=True
        groups.append(sorted(group))
    return groups


@dataclass
class Compilation:
    cmg: dict
    plan: dict
    physical: dict

    def support_plan(self, sites):
        indices=[int(x) for x in sites]
        if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=len(self.cmg['feet']):
            raise ModelError('Support set must contain distinct existing point-site indices')
        passive=sorted(set().union(*(set(self.plan['point_paths'][i]['dependent_indices']) for i in indices)))
        active=sorted(set(range(len(self.cmg['coordinate_ids'])))-set(passive))
        return dict(kind='ideal_fixed_anchor_support',sites=indices,passive=passive,active=active,
            candidate_modules=components([self.plan['point_paths'][i]['dependent_indices'] for i in indices]),
            expected_point_rank=3*len(indices),expected_mobility=len(active),
            note='Numerical rank must be checked. No friction, unilateral feasibility, impacts or stability implied.')


def compile_graph(physical):
    source=deepcopy(physical)
    forbidden={'coordinate_ids','leg_indices','modules','paths','closures','jacobian','tree','cuts','independent_ids'}
    if forbidden & set(source):raise ModelError('Input must not pre-author computational paths/modules/partitions')
    root=source.get('floating_root');bodies=source.get('bodies',[]);edges=source.get('joints',[])
    bmap={b['id']:b for b in bodies}
    if not bodies or len(bmap)!=len(bodies) or root not in bmap or 'world' in bmap:
        raise ModelError('Unique physical bodies and a valid floating root are required; world is generated')
    for b in bodies:
        mass=float(b['mass_kg']);I=np.asarray(b['inertia_kg_m2'],float);c=np.asarray(b['com_m'],float)
        if not np.isfinite(mass) or mass<=0 or c.shape!=(3,) or not np.all(np.isfinite(c)) or I.shape!=(3,3) or not np.all(np.isfinite(I)):
            raise ModelError('Physical bodies require finite positive masses and finite COM/inertia')
        eigen=np.linalg.eigvalsh(I)
        if not np.allclose(I,I.T,rtol=0,atol=1e-12) or min(eigen)<=0 or max(eigen)>sum(eigen)-max(eigen)+1e-10:
            raise ModelError('Inertia must be symmetric positive definite with physical principal moments')
    if len({j['id'] for j in edges})!=len(edges) or any(j['id'] in BASE_IDS for j in edges):
        raise ModelError('Unique non-reserved physical joint IDs required')
    if len(edges)!=len(bodies)-1:raise ModelError('Supported physical mechanism must be a tree')
    adj={b:[] for b in bmap}
    for j in edges:
        a,b=j['body_a'],j['body_b']
        if a not in bmap or b not in bmap or a==b:raise ModelError('Invalid physical joint endpoints')
        if j['type'] not in ('revolute','prismatic','fixed'):raise ModelError('Unsupported descendant joint type')
        rigid(j['T_AJ'],j['id']+'.T_AJ');rigid(j['T_BJ'],j['id']+'.T_BJ')
        if j['type']!='fixed':
            axis=np.asarray(j['axis'],float);lo=j['limits']['lower'];hi=j['limits']['upper']
            if axis.shape!=(3,) or not np.all(np.isfinite(axis)) or abs(np.linalg.norm(axis)-1)>1e-10:
                raise ModelError('Axis must be a finite unit vector')
            if not np.isfinite(lo+hi) or lo>=hi:raise ModelError('Invalid finite joint limits')
        adj[a].append((j,b,1));adj[b].append((j,a,-1))
    ordered=[];seen={root};parent={};paths={root:[]}
    def walk(body):
        for j,child,direction in sorted(adj[body],key=lambda x:x[0]['id']):
            if child==parent.get(body):continue
            if child in seen:raise ModelError('Cycle or multiply connected physical graph')
            seen.add(child);parent[child]=body
            e=deepcopy(j);e['base_body']=body;e['follower_body']=child
            e['T_BJ']=deepcopy(j['T_AJ'] if direction==1 else j['T_BJ'])
            e['T_FJ']=deepcopy(j['T_BJ'] if direction==1 else j['T_AJ'])
            if j['type']!='fixed':e['axis']=(direction*np.asarray(j['axis'])).tolist()
            e.pop('body_a',None);e.pop('body_b',None);e.pop('T_AJ',None)
            ordered.append(e);paths[child]=paths[body]+[e['id']]
            walk(child)
    walk(root)
    if seen!=set(bmap):raise ModelError('Disconnected physical graph')
    seed=source['seed'];base=np.asarray(seed['base_pose'],float)
    if base.shape!=(6,) or not np.all(np.isfinite(base)) or abs(base[4])>=1.45:raise ModelError('Invalid or singular base chart seed')
    generated_bodies=[dict(id='world',mass_kg=0.,com_m=[0.,0.,0.],inertia_kg_m2=np.zeros((3,3)).tolist())]
    generated_joints=[];axes=np.vstack((np.eye(3),np.eye(3)[[2,1,0]]));parent_body='world'
    for i,jid in enumerate(BASE_IDS):
        child=f'chart_{i}' if i<5 else root
        if child in bmap and child!=root:raise ModelError('Body ID collides with a generated chart frame')
        if i<5:generated_bodies.append(dict(id=child,mass_kg=0.,com_m=[0.,0.,0.],inertia_kg_m2=np.zeros((3,3)).tolist()))
        lo,hi=(-1e6,1e6) if i<3 else (-np.pi,np.pi)
        if i==4:lo,hi=-1.45,1.45
        generated_joints.append(dict(id=jid,type='prismatic' if i<3 else 'revolute',base_body=parent_body,follower_body=child,
            axis=axes[i].tolist(),T_BJ=np.eye(4).tolist(),T_FJ=np.eye(4).tolist(),limits=dict(lower=lo,upper=hi),
            armature=0.,damping=0.,frictionloss=0.,stiffness=0.,actuated=False))
        parent_body=child
    ids=BASE_IDS+[j['id'] for j in ordered if j['type']!='fixed']
    if set(seed['joints'])!=set(ids[6:]):raise ModelError('Named seed must cover exactly the moving physical joints')
    q=np.r_[base,[seed['joints'][i] for i in ids[6:]]]
    records={j['id']:j for j in generated_joints+ordered}
    lo=np.array([records[k]['limits']['lower'] for k in ids]);hi=np.array([records[k]['limits']['upper'] for k in ids])
    if np.any(q<lo) or np.any(q>hi):raise ModelError('Seed outside declared chart/branch bounds')
    acts=source['actuators'];aid=[a['id'] for a in acts];aj=[a['joint_id'] for a in acts]
    if len(set(aid))!=len(aid) or len(set(aj))!=len(aj) or set(aj)!=set(ids[6:]):raise ModelError('Exactly one declared actuator per descendant coordinate required')
    B=np.zeros((len(ids),len(acts)))
    for k,a in enumerate(acts):
        gear=float(a['gear'])
        if not np.isfinite(gear) or gear==0:raise ModelError('Invalid actuator gear')
        B[ids.index(a['joint_id']),k]=gear
    sites=sorted(source['point_sites'],key=lambda x:x['id'])
    if len(sites)!=4 or len({s['id'] for s in sites})!=4:raise ModelError('Go2 study requires four distinct point sites')
    point_paths=[];feet=[]
    for s in sites:
        if s['body'] not in bmap or np.asarray(s['point_m']).shape!=(3,) or not np.all(np.isfinite(s['point_m'])):
            raise ModelError('Invalid point-site attachment')
        path=paths[s['body']];dep=[ids.index(j) for j in path if j in ids]
        if len(dep)!=3:raise ModelError('Supported point branches require three dependent coordinates')
        point_paths.append(dict(site=s['id'],body=s['body'],joint_path=path,dependent_indices=dep,
            dependent_ids=[ids[i] for i in dep],active_dependencies=BASE_IDS+[f"target_{s['id']}_{a}" for a in 'xyz']))
        f=deepcopy(s);f['joint_ids']=[ids[i] for i in dep];feet.append(f)
    groups=components([p['dependent_indices'] for p in point_paths])
    if len(groups)!=4 or set().union(*(set(p['dependent_indices']) for p in point_paths))!=set(range(6,len(ids))):
        raise ModelError('This executable pruned evaluator requires four disjoint dependent branches')
    for j in ordered:j['actuated']=j['id'] in aj
    cmg=dict(schema='cmg.go2.compiled-physical-tree/1.0',name=source['name'],root_body='world',physical_root=root,
        bodies=generated_bodies+deepcopy(bodies),joints=generated_joints+ordered,coordinate_ids=ids,
        independent_ids=ids,closures=[],feet=feet,q_reference=q.tolist(),gravity_m_s2=source['gravity_m_s2'],
        actuation=dict(actuators=[dict(id=a['id'],joint=a['joint_id'],gear=a['gear'],ctrl_range=a['limits']) for a in acts],moment_matrix=B.tolist()),
        free_base_chart=dict(coordinates=BASE_IDS,convention='world XYZ then intrinsic ZYX yaw pitch roll'),
        armature=[float(records[k].get('armature',0)) for k in ids])
    sparsity=np.zeros((24,12),dtype=int)
    for k,p in enumerate(point_paths):sparsity[np.ix_(np.arange(6*k+3,6*k+6),np.array(p['dependent_indices'])-6)]=1
    plan=dict(physical_bodies=len(bodies),physical_joints=len(edges),physical_cycles=0,
        physical_coordinates=len(ids),physical_actuators=len(acts),unactuated_base_coordinates=6,
        base_chart_frames=5,point_paths=point_paths,conditional_modules=groups,
        conditional_task_residual_rows=24,conditional_task_rank=12,dependent_coordinates=12,
        conditional_subproblem_sizes=[len(p['dependent_indices']) for p in point_paths],
        passive_sparsity=sparsity.tolist(),sparsity_nonzeros=int(sparsity.sum()),
        task_virtual_coordinates=12,task_augmented_coordinates=30,
        note='Four kinematic modules only conditional on prescribed base6. Full rigid-body dynamics remain coupled.',
        physical_input_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest())
    return Compilation(cmg,plan,source)


def variant_inputs(source):
    """Twelve fixed, data-only variants. No selection by measured performance."""
    out=[('nominal',deepcopy(source))]
    for scale in (.95,1.05):
        s=deepcopy(source)
        for j in s['joints']:
            for key in ('T_AJ','T_BJ'):
                t=np.array(j[key]);t[:3,3]*=scale;j[key]=t.tolist()
        for b in s['bodies']:
            b['com_m']=(np.array(b['com_m'])*scale).tolist();b['inertia_kg_m2']=(np.array(b['inertia_kg_m2'])*scale**2).tolist()
        for f in s['point_sites']:f['point_m']=(np.array(f['point_m'])*scale).tolist()
        out.append((f'geometry_scale_{scale}',s))
    # Root attachment offset: data selection derives from topology, not leg names.
    for displacement in (.012,-.012):
        s=deepcopy(source)
        for j in s['joints']:
            if j['body_a']==s['floating_root']:
                t=np.array(j['T_AJ']);t[0,3]+=np.sign(t[0,3])*displacement;j['T_AJ']=t.tolist()
        out.append((f'root_attachment_{displacement:+.3f}m',s))
    for scale in (.95,1.05):
        s=deepcopy(source)
        for f in s['point_sites']:f['point_m']=(scale*np.array(f['point_m'])).tolist()
        out.append((f'toe_offset_scale_{scale}',s))
    s=deepcopy(source)
    b=next(b for b in s['bodies'] if b['id']==s['floating_root']);m=b['mass_kg'];payload=1.5
    # Co-located point payload at the body COM changes mass, not inertia about COM.
    b['mass_kg']=m+payload
    out.append(('base_com_payload_1p5kg',s))
    s=deepcopy(source)
    for b in s['bodies']:
        if b['id']!=s['floating_root']:
            b['mass_kg']*=1.1;b['inertia_kg_m2']=(1.1*np.array(b['inertia_kg_m2'])).tolist()
    out.append(('leg_inertia_scale_1p1',s))
    s=deepcopy(source);rng=np.random.default_rng(271828)
    transforms={}
    for b in s['bodies']:
        if b['id']==s['floating_root']:continue
        H=np.eye(4);H[:3,:3]=Rotation.from_rotvec(rng.uniform(-.25,.25,3)).as_matrix();H[:3,3]=rng.uniform(-.008,.008,3)
        transforms[b['id']]=H;R,p=H[:3,:3],H[:3,3]
        b['com_m']=(R.T@(np.array(b['com_m'])-p)).tolist();b['inertia_kg_m2']=(R.T@np.array(b['inertia_kg_m2'])@R).tolist()
    for j in s['joints']:
        for bodykey,framekey in (('body_a','T_AJ'),('body_b','T_BJ')):
            if j[bodykey] in transforms:j[framekey]=(np.linalg.inv(transforms[j[bodykey]])@np.array(j[framekey])).tolist()
    for f in s['point_sites']:
        H=transforms.get(f['body'],np.eye(4));f['point_m']=(H[:3,:3].T@(np.array(f['point_m'])-H[:3,3])).tolist()
    out.append(('leg_body_frame_reexpression',s))
    s=deepcopy(source)
    for key in ('bodies','joints','actuators','point_sites'):s[key]=list(reversed(s[key]))
    out.append(('reversed_record_and_actuator_order',s))
    s=deepcopy(source);bm={b['id']:f'body_{i:02d}' for i,b in enumerate(s['bodies'])};jm={j['id']:f'joint_{i:02d}' for i,j in enumerate(s['joints'])}
    for b in s['bodies']:b['id']=bm[b['id']]
    for j in s['joints']:j['id']=jm[j['id']];j['body_a']=bm[j['body_a']];j['body_b']=bm[j['body_b']]
    for a in s['actuators']:a['joint_id']=jm[a['joint_id']]
    for i,f in enumerate(s['point_sites']):f['body']=bm[f['body']];f['id']=f'site_{i:02d}'
    s['floating_root']=bm[s['floating_root']];s['seed']['joints']={jm[k]:v for k,v in s['seed']['joints'].items()}
    out.append(('opaque_body_joint_site_names',s))
    assert len(out)==12
    return out


def support_sets():
    return [list(s) for m in (2,3,4) for s in combinations(range(4),m)]
