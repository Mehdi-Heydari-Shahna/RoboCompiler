"""Physical closed-chain graph -> spanning tree, revolute cuts, fundamental
cycles, mobility, completed coordinate partition and generated loop modules.

New extension; not functionality attributed to the unchanged original PACDM
core. Input: bodies, *undirected* physical joints (loop-closing joints are not
marked), actuators, a named seed, a numerical branch window and named partition
requests (which physical coordinates are commanded). Not supplied and rejected
if present: spanning tree, cut joints, coordinate order, independent/dependent
split, loop paths, closure or Jacobian functions, modules.

Scope: one fixed ground (``world``, generated), revolute/prismatic/fixed joints,
loops that can be cut at passive revolute joints (the residual form of the
original revolute-cut adapter). Mobility, partition completion, module
separability and module rank are decided numerically at a closed configuration.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from . import bootstrap  # noqa: F401
from pacdm import PACDM, rank, select, skew  # unchanged original core

GROUND = 'world'
FORBIDDEN = {'tree', 'tree_ids', 'tree_joint_ids', 'cuts', 'cut_joint_ids', 'coordinate_ids',
             'independent_ids', 'dependent_ids', 'modules', 'paths', 'closures', 'jacobian', 'loops'}
RANK_TOL = 1e-10      # PACDM's relative singular-value rank tolerance
RCOND_MIN = 1e-10     # PACDM's conditioning gate


class ModelError(ValueError):
    pass


def rigid(value, name):
    a = np.asarray(value, float)
    if a.shape != (4, 4) or not np.all(np.isfinite(a)) or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-11, rtol=0):
        raise ModelError(f'{name}: expected a finite homogeneous transform')
    r = a[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-9, rtol=0) or abs(np.linalg.det(r) - 1) > 1e-9:
        raise ModelError(f'{name}: rotation is not proper orthonormal')
    return a


def _inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def export_physical(cmg, name='Excavator physical graph'):
    """One-time, provenance-preserving extraction from the source CMG records."""
    ref = next(r for r in cmg['reference_poses'] if r['id'] == 'working_reconstruction')
    seed = dict(zip(ref['coordinate_ids'], ref['q_SI']))
    bodies = [dict(id=b['id'], name=b.get('name', ''), mass_kg=b['mass_kg'], com_m=deepcopy(b['com_m']),
                   inertia_com_kg_m2=deepcopy(b['inertia_com_kg_m2']))
              for b in cmg['bodies'] if b['kind'] == 'rigid_body']
    joints = [dict(id=j['id'], type=j['type'], body_a=j['base_body'], body_b=j['follower_body'],
                   T_AJ=deepcopy(j['T_BJ']), T_BJ=deepcopy(j['T_FJ']), axis=deepcopy(j['axis']))
              for j in cmg['joints']]
    actuators = [dict(id=a['id'], joint_id=a['drive_joint_id'], type=a['type'], unit=a['effort_unit'],
                      input_status=a['input_status']) for a in cmg['actuators']]
    moving = [j['id'] for j in joints if j['type'] != 'fixed']
    return dict(
        schema='cmg.physical-closed-chain/1.0', name=name, ground=GROUND, bodies=bodies, joints=joints,
        actuators=actuators, seed=dict(joints={k: float(seed[k]) for k in moving}),
        branch_window=dict(revolute_rad=3.0, prismatic_m=1.8), gravity_m_s2=[0., 0., -9.81],
        partition_requests=dict(joint_space=['q23', 'q7', 'q4', 'q0', 'q1', 'q21', 'q22'],
                                cylinder_space=['q23', 'p3', 'p5', 'p2', 'p1', 'q21', 'q22']),
        provenance=dict(source=cmg.get('id'), schema=cmg.get('schema'),
                        seed='reference_poses/working_reconstruction',
                        branch_window='the +/-3 rad and +/-1.8 m numerical branch guards of the original revolute-cut adapter',
                        partition_requests='joint_space = the original online controller independent set; '
                                           'cylinder_space = slew, boom cylinder p3, stick cylinder p5, bucket '
                                           'cylinder p2, tilt cylinder p1 (connected channel), rotator and the '
                                           'passive pin q22'),
        note='Loop-closing joints are ordinary joints here: no tree, cut, coordinate order or module is supplied.')


def validate(physical):
    source = physical
    bad = FORBIDDEN & set(source)
    if bad:
        raise ModelError(f'Input must not pre-author computational structure: {sorted(bad)}')
    bodies = source.get('bodies', [])
    bmap = {b['id']: b for b in bodies}
    if not bodies or len(bmap) != len(bodies) or GROUND in bmap:
        raise ModelError('Unique physical bodies required; the ground frame is generated')
    for b in bodies:
        mass = float(b['mass_kg'])
        com = np.asarray(b['com_m'], float)
        inertia = np.asarray(b['inertia_com_kg_m2'], float)
        if not np.isfinite(mass) or mass <= 0 or com.shape != (3,) or not np.all(np.isfinite(com)) \
                or inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ModelError(f'{b["id"]}: finite positive mass and finite COM/inertia required')
        eig = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
        if not np.allclose(inertia, inertia.T, rtol=0, atol=1e-10) or eig.min() <= 0 \
                or eig.max() > eig.sum() - eig.max() + 1e-9:
            raise ModelError(f'{b["id"]}: inertia must be symmetric positive definite with physical moments')
    joints = source.get('joints', [])
    jmap = {j['id']: j for j in joints}
    if len(jmap) != len(joints):
        raise ModelError('Unique physical joint IDs required')
    for j in joints:
        if j['type'] not in ('revolute', 'prismatic', 'fixed'):
            raise ModelError(f'{j["id"]}: unsupported joint type {j["type"]}')
        a, b = j['body_a'], j['body_b']
        if a == b or (a not in bmap and a != GROUND) or (b not in bmap and b != GROUND):
            raise ModelError(f'{j["id"]}: invalid joint endpoints')
        rigid(j['T_AJ'], j['id'] + '.T_AJ')
        rigid(j['T_BJ'], j['id'] + '.T_BJ')
        if j['type'] != 'fixed':
            axis = np.asarray(j['axis'], float)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)) or abs(np.linalg.norm(axis) - 1) > 1e-10:
                raise ModelError(f'{j["id"]}: axis must be a finite unit vector')
    moving = {j['id'] for j in joints if j['type'] != 'fixed'}
    if set(source['seed']['joints']) != moving:
        raise ModelError('Named seed must cover exactly the moving physical joints')
    if not all(np.isfinite(float(v)) for v in source['seed']['joints'].values()):
        raise ModelError('Seed values must be finite')
    window = source['branch_window']
    if not (float(window['revolute_rad']) > 0 and float(window['prismatic_m']) > 0):
        raise ModelError('Branch window must be positive')
    acts = source.get('actuators', [])
    if len({a['id'] for a in acts}) != len(acts) or any(a['joint_id'] not in moving for a in acts):
        raise ModelError('Actuators need unique IDs and must drive moving physical joints')
    if len({a['joint_id'] for a in acts}) != len(acts):
        raise ModelError('At most one declared actuator per joint')
    return bmap, jmap, moving


@dataclass
class Compilation:
    cmg: dict
    plan: dict
    physical: dict
    structure: dict = field(repr=False)

    def summary(self):
        return {k: self.plan[k] for k in ('partition', 'independent_ids', 'mobility', 'modules_count',
                                          'largest_dependent_block', 'cut_joint_ids')}


def _bfs_depth(bmap, joints):
    adj = {b: [] for b in list(bmap) + [GROUND]}
    for j in joints:
        adj[j['body_a']].append(j['body_b'])
        adj[j['body_b']].append(j['body_a'])
    depth = {GROUND: 0}
    queue = deque([GROUND])
    while queue:
        b = queue.popleft()
        for o in sorted(adj[b]):
            if o not in depth:
                depth[o] = depth[b] + 1
                queue.append(o)
    if len(depth) != len(adj):
        raise ModelError('Disconnected physical graph')
    return depth


def spanning_tree(bmap, joints, requested, cut_override=None):
    """Priority spanning tree: fixed, then requested, prismatic, passive revolute.

    Within a class edges are ordered by BFS depth from the ground and ID. An edge
    closing a cycle becomes a cut. ``cut_override`` (benchmark control only) puts
    the named joints last so that they become the cuts when possible.
    """
    depth = _bfs_depth(bmap, joints)
    forced = set(cut_override or [])
    parent = {b: b for b in depth}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def klass(j):
        if j['id'] in forced:
            return 9
        if j['type'] == 'fixed':
            return 0
        if j['id'] in requested:
            return 1
        return 2 if j['type'] == 'prismatic' else 3

    order = sorted(joints, key=lambda j: (klass(j), min(depth[j['body_a']], depth[j['body_b']]), j['id']))
    tree, cuts = [], []
    for j in order:
        a, b = find(j['body_a']), find(j['body_b'])
        if a == b:
            if klass(j) <= 1:
                raise ModelError(f'{j["id"]}: fixed or requested joints close a loop and cannot all be tree joints')
            cuts.append(j)
        else:
            parent[a] = b
            tree.append(j)
    if forced and not forced <= {c['id'] for c in cuts}:
        raise ModelError('Cut override is not a valid cut set')
    for c in cuts:
        if c['type'] != 'revolute':
            raise ModelError(f'{c["id"]}: loop would have to be cut at a {c["type"]} joint (unsupported)')
    return tree, cuts, depth


def components(supports):
    """Cycles sharing dependent coordinates form one module (never topology alone)."""
    todo = set(range(len(supports)))
    groups = []
    while todo:
        group = {min(todo)}
        todo -= group
        changed = True
        while changed:
            changed = False
            used = set().union(*(set(supports[i]) for i in group))
            extra = {i for i in todo if set(supports[i]) & used}
            if extra:
                group |= extra
                todo -= extra
                changed = True
        groups.append(sorted(group))
    return groups


def compile_graph(physical, partition='joint_space', *, requested=None, cut_override=None, complete=True,
                  seed_tolerance=1e-8):
    source = deepcopy(physical)
    bmap, jmap, moving = validate(source)
    if requested is None:
        if partition not in source.get('partition_requests', {}):
            raise ModelError(f'Unknown partition request {partition}')
        requested = list(source['partition_requests'][partition])
    requested = list(requested)
    if len(set(requested)) != len(requested) or any(r not in moving for r in requested):
        raise ModelError('Requested independent coordinates must be distinct moving physical joints')
    joints = source['joints']
    tree, cuts, depth = spanning_tree(bmap, joints, set(requested), cut_override)

    # Directed tree from the ground (breadth first, deterministic).
    adj = {b: [] for b in depth}
    for j in tree:
        adj[j['body_a']].append((j, j['body_b'], 1))
        adj[j['body_b']].append((j, j['body_a'], -1))
    edges, parent_edge, body_depth = [], {GROUND: None}, {GROUND: 0}
    queue = deque([GROUND])
    while queue:
        body = queue.popleft()
        for j, child, direction in sorted(adj[body], key=lambda x: x[0]['id']):
            if child in parent_edge:
                continue
            A, B = rigid(j['T_AJ'], j['id']), rigid(j['T_BJ'], j['id'])
            enter, leave = (A, _inverse(B)) if direction == 1 else (B, _inverse(A))
            axis = direction * np.asarray(j['axis'], float) if j['type'] != 'fixed' else np.zeros(3)
            parent_edge[child] = len(edges)
            body_depth[child] = body_depth[body] + 1
            edges.append(dict(id=j['id'], parent=body, child=child, kind=j['type'], direction=direction,
                              enter=enter, leave=leave, axis=axis, depth=body_depth[child]))
            queue.append(child)
    if len(parent_edge) != len(depth):
        raise ModelError('Spanning tree does not reach every body')
    tree_ids = [e['id'] for e in edges if e['kind'] != 'fixed']
    cuts = sorted(cuts, key=lambda c: c['id'])
    coordinate_ids = tree_ids + [c['id'] for c in cuts]
    index = {k: i for i, k in enumerate(coordinate_ids)}
    for e in edges:
        e['coord'] = index.get(e['id']) if e['kind'] != 'fixed' else None
        e['enter_R'], e['enter_p'] = e['enter'][:3, :3].copy(), e['enter'][:3, 3].copy()
        e['leave_R'], e['leave_p'] = e['leave'][:3, :3].copy(), e['leave'][:3, 3].copy()
        e['K'] = skew(e['axis'])
        e['K2'] = e['K'] @ e['K']

    def path(body):
        out = []
        while parent_edge[body] is not None:
            out.append(parent_edge[body])
            body = edges[parent_edge[body]]['parent']
        return out[::-1]

    cut_records = []
    for c in cuts:
        pa, pb = path(c['body_a']), path(c['body_b'])
        k = 0
        while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
            k += 1
        lca = GROUND if k == 0 else edges[pa[k - 1]]['child']
        minus, plus = pa[k:], pb[k:]
        axis = np.asarray(c['axis'], float)
        cycle = [edges[e]['coord'] for e in minus + plus if edges[e]['coord'] is not None] + [index[c['id']]]
        cut_records.append(dict(id=c['id'], base=c['body_a'], follower=c['body_b'], T_A=rigid(c['T_AJ'], c['id']),
                                T_B=rigid(c['T_BJ'], c['id']), axis=axis, K=skew(axis), K2=skew(axis) @ skew(axis),
                                coord=index[c['id']], lca=lca, minus_path=minus, plus_path=plus,
                                cycle=sorted(cycle)))

    window = source['branch_window']
    seed = np.array([float(source['seed']['joints'][k]) for k in coordinate_ids])
    span = np.array([float(window['prismatic_m']) if jmap[k]['type'] == 'prismatic' else float(window['revolute_rad'])
                     for k in coordinate_ids])
    structure = dict(root=GROUND, edges=edges, cuts=cut_records, coordinate_ids=coordinate_ids,
                     tree_coordinates=len(tree_ids), seed=seed, lower=seed - span, upper=seed + span,
                     independent=[index[r] for r in requested])
    comp = Compilation(cmg={}, plan={}, physical=source, structure=structure)

    # ---------------------------------------------------------------- numerics
    from .loop_graph import GlobalLoopGraph
    n = len(coordinate_ids)
    graph = GlobalLoopGraph(comp, independent=structure['independent'])
    r, J, D = graph.residual(seed)
    seed_residual = float(np.max(np.abs(r))) if r.size else 0.
    acquisition = None

    def complete_partition(Jc):
        m = n - rank(Jc)
        req = [index[x] for x in requested]
        if len(req) > m:
            raise ModelError(f'{len(req)} requested coordinates exceed the mobility {m}')
        others = [i for i in range(n) if i not in req]
        if rank(Jc[:, others]) < n - m:
            raise ModelError('Requested coordinates are kinematically redundant (not independent)')
        chosen = []
        if len(req) < m:
            if not complete:
                raise ModelError(f'Requested coordinates leave {m - len(req)} free mode(s); completion disabled')
            # Preference: tree joints in breadth-first order, then cut coordinates.
            for c in others:
                if len(req) + len(chosen) == m:
                    break
                keep = [i for i in others if i not in chosen and i != c]
                if rank(Jc[:, keep]) == n - m:
                    chosen.append(c)
        if len(req) + len(chosen) != m:
            raise ModelError('Partition completion failed')
        return m, req + chosen, chosen

    mobility, independent, completion = complete_partition(J)
    if seed_residual > seed_tolerance:
        # Close the declared seed with the unchanged PACDM homotopy (acquire).
        graph = GlobalLoopGraph(comp, independent=independent)
        q, info = PACDM(graph).acquire(seed[independent].copy(), seed)
        acquisition = dict(success=bool(info['success']), message=info.get('message'),
                           accepted_steps=info.get('accepted_steps'), rejected_steps=info.get('rejected_steps'))
        if not info['success']:
            raise ModelError(f'Seed could not be closed: {info.get("message")}')
        seed = q
        r, J, D = graph.residual(seed)
        mobility, independent, completion = complete_partition(J)
    structure['seed_closed'] = seed
    structure['independent'] = independent
    dependent = [i for i in range(n) if i not in independent]
    structure['dependent'] = dependent

    # Modules: cycles joined when they share dependent coordinates.
    supports = [sorted(set(c['cycle']) & set(dependent)) for c in cut_records]
    groups = components(supports)
    modules = []
    for g in groups:
        dep = sorted(set().union(*(supports[i] for i in g)))
        inp = sorted(set().union(*(set(cut_records[i]['cycle']) & set(independent) for i in g)))
        rows = np.concatenate([np.arange(6 * i, 6 * i + 6) for i in g])
        block = J[np.ix_(rows, dep)]
        rk = rank(block)
        _, rc, _ = select(block)
        modules.append(dict(cut_indices=g, cut_ids=[cut_records[i]['id'] for i in g],
                            dependent_indices=dep, dependent_ids=[coordinate_ids[i] for i in dep],
                            input_indices=inp, input_ids=[coordinate_ids[i] for i in inp],
                            lca=sorted({cut_records[i]['lca'] for i in g}), rows=int(len(rows)), rank=int(rk),
                            rcond=float(rc)))
        if rk != len(dep) or rc < RCOND_MIN:
            raise ModelError(f'Module {g} is rank deficient or ill-conditioned at the closed seed')
    if sum(len(m['dependent_indices']) for m in modules) != len(dependent):
        raise ModelError('Every dependent coordinate must belong to exactly one module')
    uncovered = [coordinate_ids[i] for i in dependent if not any(i in m['dependent_indices'] for m in modules)]
    if uncovered:
        raise ModelError(f'Dependent coordinates outside every loop: {uncovered}')
    rows_all = 6 * len(cut_records)
    sparsity = np.zeros((rows_all, len(dependent)), int)
    for ci, c in enumerate(cut_records):
        for k in c['cycle']:
            if k in dependent:
                sparsity[6 * ci:6 * ci + 6, dependent.index(k)] = 1
    _, global_rcond, _ = select(J[:, dependent])
    structure['modules'] = modules
    structure['sparsity'] = sparsity

    acts = source.get('actuators', [])
    requested_ids = list(requested)
    cmg = dict(schema='cmg.excavator.compiled-closed-chain/1.0', name=source.get('name'), root_body=GROUND,
               bodies=[dict(id=GROUND, kind='reference_frame', mass_kg=0., com_m=[0., 0., 0.],
                            inertia_com_kg_m2=np.zeros((3, 3)).tolist())]
               + [dict(b, kind='rigid_body') for b in source['bodies']],
               joints=[dict(id=j['id'], type=j['type'], base_body=j['body_a'], follower_body=j['body_b'],
                            T_BJ=j['T_AJ'], T_FJ=j['T_BJ'], axis=j['axis'], actuated=any(a['joint_id'] == j['id'] for a in acts))
                       for j in joints],
               coordinate_ids=coordinate_ids, tree_joint_ids=tree_ids, cut_joint_ids=[c['id'] for c in cut_records],
               independent_ids=[coordinate_ids[i] for i in independent],
               tree=[dict(id=e['id'], parent_body=e['parent'], child_body=e['child'], direction=e['direction'])
                     for e in edges],
               q_reference=seed.tolist(), gravity_m_s2=source['gravity_m_s2'],
               actuators=deepcopy(acts))
    plan = dict(
        partition=partition, requested_ids=requested_ids,
        completion_ids=[coordinate_ids[i] for i in completion],
        independent_ids=[coordinate_ids[i] for i in independent],
        physical_bodies=len(bmap), physical_joints=len(joints), moving_joints=len(moving),
        fixed_joints=len(joints) - len(moving), tree_coordinates=len(tree_ids), cut_joint_ids=[c['id'] for c in cut_records],
        independent_cycles=len(cut_records), augmented_coordinates=n, closure_rows=rows_all,
        closure_rank=int(rank(J)), mobility=int(mobility), dependent_coordinates=len(dependent),
        global_dependent_block=[rows_all, len(dependent)], global_rcond=float(global_rcond),
        modules_count=len(modules), module_sizes=[len(m['dependent_indices']) for m in modules],
        largest_dependent_block=max(len(m['dependent_indices']) for m in modules),
        modules=[{k: m[k] for k in ('cut_ids', 'dependent_ids', 'input_ids', 'lca', 'rows', 'rank', 'rcond')}
                 for m in modules],
        cycles=[dict(cut=c['id'], lca=c['lca'], coordinates=[coordinate_ids[i] for i in c['cycle']]) for c in cut_records],
        sparsity_nonzeros=int(sparsity.sum()), sparsity_shape=list(sparsity.shape),
        actuated_joint_ids=[a['joint_id'] for a in acts],
        seed_residual_max=seed_residual, seed_acquisition=acquisition,
        note='Loop modules are conditional on their input coordinates only; coordinates outside every '
             'cycle (here slew and rotator) never enter a closure solve. Rigid-body dynamics remain coupled.',
        physical_input_sha256=hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest())
    comp.cmg, comp.plan = cmg, plan
    return comp
