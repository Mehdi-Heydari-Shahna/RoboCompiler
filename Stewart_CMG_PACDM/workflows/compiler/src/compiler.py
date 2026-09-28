"""New, scoped graph-to-PACDM compiler for the Stewart experiment.

Accepts physical bodies and fixed/revolute/prismatic/spherical joints, NOT an
input tree, cuts, closure equations, or Jacobians. A deterministic Kruskal
policy keeps scalar joints before spherical joints. Only spherical chords are
currently supported. Spherical TREE joints are lowered to three XYZ chart
coordinates and two explicitly massless frames. Every physical body occurs
once. This is a new adapter; vendor PACDM is not modified.

This is not CAD import, a general arbitrary-joint compiler, URDF+, or a new
constraint-embedding algorithm. Independent actuator selection is declared by
the input, not inferred from graph cycle counts.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
from .bootstrap import ROOT
from vendor.pacdm_original import PointGraph, inv


class ModelError(ValueError):
    """A model or supported-chart precondition is invalid."""


def rigid(value, name):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4,4) or not np.all(np.isfinite(a))
        or not np.allclose(a[3], [0,0,0,1], atol=1e-10, rtol=0)
        or not np.allclose(a[:3,:3].T@a[:3,:3], np.eye(3), atol=1e-9, rtol=0)
        or not np.isclose(np.linalg.det(a[:3,:3]),1,atol=1e-9,rtol=0)):
        raise ModelError(f'{name}: invalid proper rigid transform')
    return a


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass
class Compiled:
    cmg: dict
    plan: dict
    source: dict
    spherical_charts: list

    def seed_from_body_poses(self, scalar_values, world_poses):
        """TEST/initialization utility. Caller supplies a seed, not its target.

        Physical scalar values are preserved. Only coordinate representation
        of spherical tree joints is extracted. Used for shared initial seeds
        and independent witnesses, never to feed solved targets to continuation.
        """
        vals = dict(scalar_values)
        for item in self.spherical_charts:
            T0 = world_poses[item['parent']] @ np.asarray(item['T_parent_joint'])
            T1 = world_poses[item['child']] @ np.asarray(item['T_child_joint'])
            relative = inv(T0) @ T1
            angles = Rotation.from_matrix(relative[:3,:3]).as_euler('XYZ')
            vals.update(zip(item['coordinate_ids'], angles))
        return np.array([vals[k] for k in self.cmg['coordinate_ids']],float)

    def graph(self, seed):
        return PointGraph(self.cmg, seed)


def validate_source(source):
    if source.get('schema') != 'cmg.physical-joint-graph/1.0':
        raise ModelError('Unsupported graph schema')
    if source.get('units') != 'SI':
        raise ModelError('Only explicit SI units are supported')
    for banned in ('tree','closures','cuts','coordinate_ids','constraint_equations','jacobian'):
        if banned in source:
            raise ModelError(f'Physical input must not supply generated field {banned}')
    bodies = {b['id']: b for b in source['bodies']}
    joints = {j['id']: j for j in source['joints']}
    if len(bodies)!=len(source['bodies']) or len(joints)!=len(source['joints']):
        raise ModelError('Duplicate body or joint ID')
    if source['root_body'] not in bodies:
        raise ModelError('Missing root body')
    for bid,b in bodies.items():
        m=float(b['mass_kg']); c=np.asarray(b['com_m'],float); I=np.asarray(b['inertia_kg_m2'],float)
        if not np.isfinite(m) or m<0 or c.shape!=(3,) or not np.all(np.isfinite(c)):
            raise ModelError(f'{bid}: invalid mass or center of mass')
        if I.shape!=(3,3) or not np.all(np.isfinite(I)) or not np.allclose(I,I.T,atol=1e-12,rtol=0):
            raise ModelError(f'{bid}: invalid symmetric inertia')
        d=np.linalg.eigvalsh(I)
        if d[0]<-1e-12 or d[2]>d[0]+d[1]+1e-10 or (m==0 and np.max(abs(I))>1e-12):
            raise ModelError(f'{bid}: inadmissible rigid-body inertia')
    adjacency={b:[] for b in bodies}
    for jid,j in joints.items():
        a,b=j['body_a'],j['body_b']
        if a not in bodies or b not in bodies or a==b:
            raise ModelError(f'{jid}: invalid body endpoints')
        if j['type'] not in ('fixed','revolute','prismatic','spherical'):
            raise ModelError(f'{jid}: unsupported joint type')
        rigid(j['T_AJ'],jid+'.T_AJ'); rigid(j['T_BJ'],jid+'.T_BJ')
        if j['type'] in ('revolute','prismatic'):
            axis=np.asarray(j['axis'],float)
            if axis.shape!=(3,) or not np.all(np.isfinite(axis)) or abs(np.linalg.norm(axis)-1)>1e-9:
                raise ModelError(f'{jid}: joint axis must be unit length')
            lo,hi=j['limits']['lower'],j['limits']['upper']
            if not np.isfinite(lo) or not np.isfinite(hi) or lo>=hi:
                raise ModelError(f'{jid}: invalid bounds')
        adjacency[a].append(b);adjacency[b].append(a)
    seen={source['root_body']};queue=deque(seen)
    while queue:
        for b in adjacency[queue.popleft()]:
            if b not in seen:seen.add(b);queue.append(b)
    if len(seen)!=len(bodies):raise ModelError('Disconnected physical graph')
    active=[a['joint_id'] for a in source['actuators']]
    if len(active)!=len(set(active)) or not active:
        raise ModelError('Actuator selection must be nonempty and unique')
    for jid in active:
        if jid not in joints or joints[jid]['type'] not in ('revolute','prismatic'):
            raise ModelError('An actuator must refer to a scalar physical joint')
    return bodies,joints,active


def compile_graph(source, preferred_spherical_tree=None):
    """Select a physical spanning tree and emit a PointGraph-compatible CMG.

    Default policy is fixed before benchmarking: scalar-first then lexical ID.
    preferred_spherical_tree is for exhaustive tree-choice sensitivity tests;
    it is never selected from measured runtimes.
    """
    source=deepcopy(source)
    bodies,joints,active=validate_source(source)
    if preferred_spherical_tree is not None and (preferred_spherical_tree not in joints or joints[preferred_spherical_tree]['type']!='spherical'):
        raise ModelError('Preferred spherical tree edge does not exist')
    parent={b:b for b in bodies}
    def find(b):
        while parent[b]!=b:parent[b]=parent[parent[b]];b=parent[b]
        return b
    def priority(j):
        return (int(j['type']=='spherical'),int(j['id']!=preferred_spherical_tree),j['id'])
    tree=[];chords=[]
    for j in sorted(joints.values(),key=priority):
        a,b=find(j['body_a']),find(j['body_b'])
        if a!=b:parent[a]=b;tree.append(j)
        else:chords.append(j)
    if len(tree)!=len(bodies)-1:raise ModelError('Tree construction failed')
    if any(j['type']!='spherical' for j in chords):
        raise ModelError('This scoped compiler requires every chord to be spherical; scalar-loop lowering is not implemented')
    if any(jid not in {j['id'] for j in tree} for jid in active):
        raise ModelError('Actuated joint was excluded from the tree')
    root=source['root_body'];adjacency={b:[] for b in bodies}
    for j in tree:
        adjacency[j['body_a']].append((j['body_b'],j,1))
        adjacency[j['body_b']].append((j['body_a'],j,-1))
    queue=deque([root]);seen={root};oriented=[];physical_parent={}
    while queue:
        a=queue.popleft()
        for b,j,sign in sorted(adjacency[a],key=lambda z:z[1]['id']):
            if b in seen:continue
            oriented.append((a,b,j,sign));physical_parent[b]=(a,j['id'],sign)
            seen.add(b);queue.append(b)
    lowered=[];body_records=[deepcopy(bodies[k]) for k in sorted(bodies)]
    ids=[];chart_info=[];expanded={}
    for a,b,j,sign in oriented:
        E=np.array(j['T_AJ'] if sign==1 else j['T_BJ'])
        F=np.array(j['T_BJ'] if sign==1 else j['T_AJ'])
        if j['type']=='spherical':
            names=[f'__{j["id"]}_chart_{k}' for k in ('x','y','z')]
            middles=[f'__{j["id"]}_frame_{k}' for k in ('x','y')]
            if any(k in bodies for k in middles) or any(k in joints for k in names):
                raise ModelError('Generated chart identifier collides with a physical identifier')
            for k in middles:
                body_records.append(dict(id=k,mass_kg=0.,com_m=[0.,0.,0.],inertia_kg_m2=np.zeros((3,3)).tolist(),computational_frame=True))
            nodes=[a]+middles+[b]
            for k,name in enumerate(names):
                lowered.append(dict(id=name,type='revolute',base_body=nodes[k],follower_body=nodes[k+1],
                    axis=np.eye(3)[k].tolist(),T_BJ=(E if k==0 else np.eye(4)).tolist(),
                    T_FJ=(F if k==2 else np.eye(4)).tolist(),limits=dict(lower=-np.pi,upper=np.pi),
                    source_joint=j['id'],computational_coordinate=True))
                ids.append(name)
            chart_info.append(dict(joint_id=j['id'],parent=a,child=b,T_parent_joint=E.tolist(),T_child_joint=F.tolist(),coordinate_ids=names))
            expanded[j['id']]=names
        else:
            rec=dict(id=j['id'],type=j['type'],base_body=a,follower_body=b,
                     T_BJ=E.tolist(),T_FJ=F.tolist(),source_joint=j['id'])
            if j['type']!='fixed':
                rec['axis']=(sign*np.asarray(j['axis'])).tolist();rec['limits']=deepcopy(j['limits']);ids.append(j['id'])
            else:rec['axis']=[0.,0.,1.]
            lowered.append(rec);expanded[j['id']]=[j['id']] if j['type']!='fixed' else []
    cuts=[dict(id=j['id'],body1=j['body_a'],body2=j['body_b'],point1_m=np.array(j['T_AJ'])[:3,3].tolist(),
               point2_m=np.array(j['T_BJ'])[:3,3].tolist(),source_joint=j['id']) for j in sorted(chords,key=lambda j:j['id'])]
    def tree_path(start,goal):
        todo=deque([start]);prev={start:None}
        while todo:
            u=todo.popleft()
            if u==goal:break
            for v,j,sign in adjacency[u]:
                if v not in prev:prev[v]=(u,j['id'],sign);todo.append(v)
        path=[];v=goal
        while v!=start:
            u,jid,sign=prev[v];path.append(dict(joint_id=jid,sign=sign));v=u
        return path[::-1]
    cycles=[]
    for j in sorted(chords,key=lambda j:j['id']):
        path=tree_path(j['body_a'],j['body_b'])
        support=[k for item in path for k in expanded[item['joint_id']]]
        cycles.append(dict(chord=j['id'],tree_path=path,passive_tree_support=[k for k in support if k not in active],
                           auxiliary_coordinates=3,scalar_augmented_residuals=6))
    # Merge shared passive SUPPORT sets only; numerical rank remains separate.
    modules=[]
    for cycle in cycles:
        support=set(cycle['passive_tree_support']);members=[cycle['chord']]
        changed=True
        while changed:
            changed=False
            for i,(olds,oldm) in enumerate(modules):
                if support & olds:
                    support |= olds;members+=oldm;modules.pop(i);changed=True;break
        modules.append((support,members))
    cmg=dict(schema='cmg.stewart.point-closures/1.0',id=source['id']+'_compiled',root_body=root,units='SI',
             bodies=body_records,joints=lowered,closures=cuts,coordinate_ids=ids,independent_ids=active,
             gravity_m_s2=source.get('gravity_m_s2',[0.,0.,-9.81]),task_body=source.get('task_body','platform'))
    expected_coordinates=sum(3 if j['type']=='spherical' else int(j['type']!='fixed') for j in tree)
    if expected_coordinates!=len(ids):raise AssertionError('Incorrect scalar lowering count')
    plan=dict(input_sha256=canonical_hash(source),physical_bodies=len(bodies),physical_joints=len(joints),
        physical_cycles=len(joints)-len(bodies)+1,tree_edges=[j['id'] for j in tree],
        chord_edges=[j['id'] for j in sorted(chords,key=lambda j:j['id'])],
        policy='scalar-first Kruskal; spherical lexical tie-break; only spherical chords supported',
        preferred_spherical_tree=preferred_spherical_tree,physical_tree_coordinates=len(ids),
        augmented_coordinates=len(ids)+3*len(cuts),augmented_residual_rows=6*len(cuts),
        point_residual_rows=3*len(cuts),declared_actuators=len(active),
        dimensional_mobility=len(ids)-3*len(cuts),
        numerical_rank_status='not_checked_until_a_configuration_is_supplied',
        generated_massless_frames=len(body_records)-len(bodies),cycles=cycles,
        candidate_modules=[dict(chords=sorted(m),shared_passive_support=sorted(s)) for s,m in modules],
        generated_model_sha256=canonical_hash(cmg),manual_tree_labels_in_input=0,manual_closure_functions_in_input=0)
    return Compiled(cmg,plan,source,chart_info)


def from_original_cmg(cmg):
    """Stewart-specific DATA adapter, not part of the topology compiler.

    Removes the original explicit free-platform coordinate chain. Those five
    intermediate zero-inertia frames and six joints are computational devices,
    not physical parts of the six-UPS mechanism. Original point closures become
    physical spherical joint records. No body or inertial property is inferred.
    """
    pose_ids=set(cmg['coordinate_ids'][:6])
    pose_joints=[j for j in cmg['joints'] if j['id'] in pose_ids]
    if len(pose_joints)!=6:raise ModelError('Expected the supplied six-coordinate platform chart')
    chart_bodies={j['follower_body'] for j in pose_joints}-{'platform'}
    if len(chart_bodies)!=5:raise ModelError('Unexpected source platform chart')
    outb=[deepcopy(b) for b in cmg['bodies'] if b['id'] not in chart_bodies]
    outj=[]
    for j in cmg['joints']:
        if j['id'] in pose_ids:continue
        r=dict(id=j['id'],type=j['type'],body_a=j['base_body'],body_b=j['follower_body'],
               T_AJ=deepcopy(j['T_BJ']),T_BJ=deepcopy(j['T_FJ']))
        if j['type']!='fixed':r.update(axis=deepcopy(j['axis']),limits=deepcopy(j['limits']))
        outj.append(r)
    for c in cmg['closures']:
        A=np.eye(4);B=np.eye(4);A[:3,3]=c['point1_m'];B[:3,3]=c['point2_m']
        outj.append(dict(id=c['id'],type='spherical',body_a=c['body1'],body_b=c['body2'],T_AJ=A.tolist(),T_BJ=B.tolist()))
    return dict(schema='cmg.physical-joint-graph/1.0',id='stewart_6ups_physical',units='SI',
                root_body=cmg['root_body'],task_body='platform',bodies=outb,joints=outj,
                actuators=[dict(joint_id=k,effort_type='force',unit='N') for k in cmg['independent_ids']],
                gravity_m_s2=deepcopy(cmg['gravity_m_s2']),
                provenance='Physical data copied from supplied Stewart CMG; artificial free-platform chain removed by a documented Stewart-specific adapter')
