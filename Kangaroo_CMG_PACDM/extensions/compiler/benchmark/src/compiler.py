"""Physical floating graph with declared loop cuts -> CMG, PACDM partitions, modules, support plans.

New extension; not functionality attributed to the unchanged PACDM core.

Input (``inputs/physical_graph.json``): bodies with inertias, unordered physical
joint edges, a floating root, declared loop-cut records (point coincidence or
universal), actuator declarations, drive data, sole-support site declarations,
gravity and a named seed configuration.

Not supplied (generated here): tree directions and coordinate order, cut paths
and lowest common ancestors, chart columns, the independent/dependent
partition, loop modules, the passive-Jacobian sparsity pattern, support-mode
partitions and candidate support modules, and the chart-base model used by
the dynamics comparisons.

Scope: one floating root; scalar revolute/prismatic/fixed tree joints; loop
cuts of type point_coincidence (three chart coordinates) or universal (two
chart coordinates) following the accepted v22 CutGraph convention; one
actuator per actuated scalar joint; ideal sole welds as separate mode data.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation

BASE_IDS = ['base_x', 'base_y', 'base_z', 'base_yaw', 'base_pitch', 'base_roll']
FORBIDDEN = {'coordinate_ids', 'independent_ids', 'modules', 'paths', 'chart_columns', 'partitions',
             'sparsity', 'tree', 'leg_indices', 'initial_seed', 'rows'}


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


def rotation_ok(value, name):
    r = np.asarray(value, float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)) or not np.allclose(r.T @ r, np.eye(3), atol=1e-9, rtol=0) \
            or abs(np.linalg.det(r) - 1) > 1e-9:
        raise ModelError(f'{name}: expected a proper rotation matrix')
    return r


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def _motion(kind, axis, q):
    T = np.eye(4)
    if kind == 'revolute':
        K = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        T[:3, :3] = np.eye(3) + np.sin(q) * K + (1. - np.cos(q)) * (K @ K)
    elif kind == 'prismatic':
        T[:3, 3] = axis * q
    return T


# ---------------------------------------------------------------------------
# One-time transparent extraction from the original v22 CMG
# ---------------------------------------------------------------------------
def export_physical(cmg, sole_corners):
    """Provenance-preserving physical records; no paths, partitions or modules.

    ``sole_corners`` maps a foot body to its four local sole corner points (from
    the accepted v22 ``contact_reference.foot_corners``).
    """
    joints = []
    for j in cmg['joints']:
        joints.append(dict(id=j['id'], type=j['type'], body_a=j['base_body'], body_b=j['follower_body'],
                           T_AJ=deepcopy(j['T_BJ']), T_BJ=deepcopy(j['T_FJ']), axis=deepcopy(j['axis']),
                           limits=deepcopy(j.get('limits')) if j['type'] != 'fixed' else None))
    cuts = []
    for c in cmg['closures']:
        cut = dict(id=c['id'], type=c['type'], body1=c['body1'], body2=c['body2'],
                   point1_m=deepcopy(c['point1_m']), point2_m=deepcopy(c['point2_m']))
        if c['type'] == 'universal':
            cut.update(frame1_R=deepcopy(c['frame1_R']), frame2_R=deepcopy(c['frame2_R']))
        cuts.append(cut)
    seed = dict(zip(cmg['coordinate_ids'], cmg['initial_seed']))
    sites = []
    for side, body in (('left', 'left_ankle_roll'), ('right', 'right_ankle_roll')):
        corners = np.asarray(sole_corners[body], float)
        sites.append(dict(id=f'{side}_sole', body=body, frame_translation_m=corners.mean(axis=0).tolist(),
                          frame_rotation=np.eye(3).tolist(), corners_m=corners.tolist()))
    drive = {j: dict(armature=float(cmg['armature'][j]),
                     damping=float(cmg['joint_dissipation'][j]['damping']),
                     frictionloss=float(cmg['joint_dissipation'][j].get('frictionloss', 0.)))
             for j in cmg['coordinate_ids']}
    return dict(schema='cmg.physical-floating-loop-graph/1.0', name='Kangaroo v22 physical graph',
                floating_root=cmg['root_body'], bodies=deepcopy(cmg['bodies']), joints=joints, loop_cuts=cuts,
                actuators=[dict(id=a['id'], joint_id=a['joint'], gear=a['gear'],
                                force_bounds_N=deepcopy(a['force_bounds_N'])) for a in cmg['actuators']],
                drive=drive, sole_sites=sites, gravity_m_s2=[0., 0., -9.81],
                seed=dict(joints={k: float(v) for k, v in seed.items()}),
                provenance=dict(source=deepcopy(cmg['source']), extracted_from='original/original_v22/data/whole_body_cmg.json'),
                note='Loop cuts are declared physical closures (as authored in the source CMG). Sole sites are '
                     'support-mode data, not permanent mechanical closures.')


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
@dataclass
class Compilation:
    cmg: dict
    plan: dict
    physical: dict
    directed: list = field(repr=False, default_factory=list)

    # ----------------------------------------------------------------- support
    def support_plan(self, sites):
        """Chart-base partition for an ideal sole-weld set; an empty set is the floating loop-only mode."""
        names = [str(s) for s in sites]
        known = [s['id'] for s in self.cmg['sole_sites']]
        if len(set(names)) != len(names) or any(n not in known for n in names):
            raise ModelError('Support set must contain distinct existing sole-site ids')
        plan = self.plan
        nt, n_aug = plan['physical_coordinates'], plan['augmented_coordinates']
        motor_idx = plan['motor_indices']
        # chart-base layout: [base6, physical nt, cut charts] -> offset 6
        dependent = {6 + i for i in plan['passive_indices']}
        weld_supports = []
        for name in names:
            site = next(s for s in self.cmg['sole_sites'] if s['id'] == name)
            path = plan['root_paths'][site['body']]
            touched = {i for i in path['moving_indices'] if i not in motor_idx}
            motors = sorted({m for mod in plan['modules'] if set(mod['passive']) & touched for m in mod['motors']})
            dependent |= {6 + m for m in motors}
            weld_supports.append(sorted({6 + i for i in path['moving_indices']} | {6 + m for m in motors}))
        n = 6 + n_aug
        passive = sorted(dependent)
        active = sorted(set(range(n)) - dependent)
        supports = [sorted({6 + i for i in mod['passive']}) for mod in plan['modules']] + weld_supports
        return dict(kind='ideal_sole_weld_support' if names else 'floating_loops_only', sites=names,
                    coordinates=n, passive=passive, active=active,
                    candidate_modules=components(supports), expected_augmented_rank=len(passive),
                    expected_physical_constraint_rank=plan['physical_closure_rank_expected'] + 6 * len(names),
                    expected_mobility=len(active),
                    note='Numerical rank is checked at every witness. No friction, unilateral feasibility, impacts or '
                         'stability are implied.')

    def chart_cmg(self):
        """CMG with a generated world root and six scalar base-chart joints (xyz + intrinsic ZYX)."""
        c = deepcopy(self.cmg)
        root = c['root_body']
        bodies = [dict(id='world', kind='generated_frame', mass_kg=0., com_m=[0., 0., 0.],
                       inertia_com_kg_m2=np.zeros((3, 3)).tolist())]
        joints = []
        axes = [np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], np.eye(3)[2], np.eye(3)[1], np.eye(3)[0]]
        parent = 'world'
        for i, jid in enumerate(BASE_IDS):
            child = f'chart_{i}' if i < 5 else root
            if i < 5:
                bodies.append(dict(id=child, kind='generated_frame', mass_kg=0., com_m=[0., 0., 0.],
                                   inertia_com_kg_m2=np.zeros((3, 3)).tolist()))
            lo, hi = (-1e6, 1e6) if i < 3 else ((-1.45, 1.45) if i == 4 else (-np.pi, np.pi))
            joints.append(dict(id=jid, type='prismatic' if i < 3 else 'revolute', base_body=parent,
                               follower_body=child, axis=axes[i].tolist(), T_BJ=np.eye(4).tolist(),
                               T_FJ=np.eye(4).tolist(), limits=dict(lower=lo, upper=hi)))
            parent = child
        c['bodies'] = bodies + c['bodies']
        c['joints'] = joints + c['joints']
        c['root_body'] = 'world'
        c['physical_root'] = root
        c['coordinate_ids'] = BASE_IDS + c['coordinate_ids']
        c['independent_ids'] = BASE_IDS + c['independent_ids']
        c['armature'] = {**{b: 0. for b in BASE_IDS}, **c['armature']}
        c['joint_dissipation'] = {**{b: dict(damping=0., frictionloss=0.) for b in BASE_IDS}, **c['joint_dissipation']}
        c['initial_seed'] = [0.] * 6 + list(c['initial_seed'])
        c['free_base_chart'] = dict(coordinates=BASE_IDS, convention='world XYZ then intrinsic ZYX yaw pitch roll')
        return c


def components(supports):
    """Adjacency is shared DEPENDENT coordinates, never names or topology alone."""
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


def _validate_bodies(bodies):
    bmap = {}
    for b in bodies:
        if b['id'] in bmap or b['id'] == 'world' or str(b['id']).startswith('chart_'):
            raise ModelError('Unique physical body ids required; world/chart_* are generated')
        mass = float(b['mass_kg'])
        I = np.asarray(b['inertia_com_kg_m2'], float)
        c = np.asarray(b['com_m'], float)
        if not np.isfinite(mass) or mass <= 0 or c.shape != (3,) or not np.all(np.isfinite(c)) \
                or I.shape != (3, 3) or not np.all(np.isfinite(I)):
            raise ModelError('Physical bodies require finite positive mass and finite COM/inertia')
        eig = np.linalg.eigvalsh(I)
        if not np.allclose(I, I.T, rtol=0, atol=1e-12) or eig.min() <= 0 or eig.max() > eig.sum() - eig.max() + 1e-10:
            raise ModelError('Inertia must be symmetric positive definite with physical principal moments')
        bmap[b['id']] = b
    return bmap


def compile_graph(physical):
    source = deepcopy(physical)
    if FORBIDDEN & set(source):
        raise ModelError('Input must not pre-author paths, partitions, modules or coordinate orders')
    bodies = source.get('bodies', [])
    edges = source.get('joints', [])
    root = source.get('floating_root')
    bmap = _validate_bodies(bodies)
    if root not in bmap:
        raise ModelError('A valid floating root body is required')
    if len({j['id'] for j in edges}) != len(edges) or any(j['id'] in BASE_IDS for j in edges):
        raise ModelError('Unique non-reserved physical joint ids required')
    if len(edges) != len(bmap) - 1:
        raise ModelError('Physical joints must form a spanning tree; loops are declared as loop_cuts')
    adj = {b: [] for b in bmap}
    for j in edges:
        a, b = j['body_a'], j['body_b']
        if a not in bmap or b not in bmap or a == b:
            raise ModelError(f"Invalid joint endpoints at {j['id']}")
        if j['type'] not in ('revolute', 'prismatic', 'fixed'):
            raise ModelError(f"Unsupported joint type at {j['id']}")
        rigid(j['T_AJ'], j['id'] + '.T_AJ')
        rigid(j['T_BJ'], j['id'] + '.T_BJ')
        if j['type'] != 'fixed':
            axis = np.asarray(j['axis'], float)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)) or abs(np.linalg.norm(axis) - 1) > 1e-9:
                raise ModelError(f"{j['id']}: axis must be a finite unit vector")
            lo, hi = j['limits']['lower'], j['limits']['upper']
            if not np.isfinite(lo + hi) or lo >= hi:
                raise ModelError(f"{j['id']}: invalid joint limits")
        adj[a].append((j, b, 1))
        adj[b].append((j, a, -1))
    # ---- directed tree by depth-first preorder from the floating root
    ordered, parent, paths, seen = [], {}, {root: []}, {root}

    def walk(body):
        for j, child, direction in sorted(adj[body], key=lambda x: x[0]['id']):
            if child == parent.get(body):
                continue
            if child in seen:
                raise ModelError('Cycle in the physical joint graph; declare loops as loop_cuts')
            seen.add(child)
            parent[child] = body
            e = deepcopy(j)
            e['base_body'], e['follower_body'] = body, child
            e['T_BJ'] = deepcopy(j['T_AJ'] if direction == 1 else j['T_BJ'])
            e['T_FJ'] = deepcopy(j['T_BJ'] if direction == 1 else j['T_AJ'])
            if j['type'] != 'fixed':
                e['axis'] = (direction * np.asarray(j['axis'], float)).tolist()
            for k in ('body_a', 'body_b', 'T_AJ'):
                e.pop(k, None)
            ordered.append(e)
            paths[child] = paths[body] + [e['id']]
            walk(child)

    walk(root)
    if seen != set(bmap):
        raise ModelError('Disconnected physical graph')
    ids = [j['id'] for j in ordered if j['type'] != 'fixed']
    nt = len(ids)
    jrec = {j['id']: j for j in ordered}
    # ---- actuators: one per actuated scalar joint
    acts = sorted(source.get('actuators', []), key=lambda a: a['id'])
    aid = [a['id'] for a in acts]
    aj = [a['joint_id'] for a in acts]
    if not acts or len(set(aid)) != len(aid) or len(set(aj)) != len(aj) or any(j not in ids for j in aj):
        raise ModelError('Unique actuators on distinct existing scalar joints are required')
    for a in acts:
        lo, hi = a['force_bounds_N']
        if not np.isfinite(lo + hi) or lo >= hi or not np.isfinite(float(a['gear'])) or float(a['gear']) == 0.:
            raise ModelError('Invalid actuator gear or force bounds')
    motor_idx = [ids.index(j) for j in aj]
    # ---- seed (named) and drive data
    seed = source.get('seed', {}).get('joints', {})
    if set(seed) != set(ids):
        raise ModelError('Named seed must cover exactly the moving physical joints')
    q_seed = np.array([seed[i] for i in ids], float)
    lo = np.array([jrec[i]['limits']['lower'] for i in ids])
    hi = np.array([jrec[i]['limits']['upper'] for i in ids])
    if np.any(q_seed < lo) or np.any(q_seed > hi):
        raise ModelError('Seed outside the declared joint ranges')
    drive = source.get('drive', {})
    if set(drive) != set(ids):
        raise ModelError('Drive data (armature/damping) must cover exactly the moving joints')
    # ---- forward kinematics at the seed (for point-cut chart references)
    poses = {root: np.eye(4)}
    for j in ordered:
        k = j['id']
        T = poses[j['base_body']] @ np.asarray(j['T_BJ'])
        if j['type'] != 'fixed':
            T = T @ _motion(j['type'], np.asarray(j['axis']), seed[k])
        poses[j['follower_body']] = T @ _inv(np.asarray(j['T_FJ']))
    # ---- loop cuts
    cuts = sorted(source.get('loop_cuts', []), key=lambda c: c['id'])
    if len({c['id'] for c in cuts}) != len(cuts):
        raise ModelError('Unique loop-cut ids required')
    at = nt
    cut_plans = []
    for c in cuts:
        if c['body1'] not in bmap or c['body2'] not in bmap or c['body1'] == c['body2']:
            raise ModelError(f"Loop cut {c['id']}: invalid bodies")
        for key in ('point1_m', 'point2_m'):
            if np.asarray(c[key]).shape != (3,) or not np.all(np.isfinite(c[key])):
                raise ModelError(f"Loop cut {c['id']}: invalid {key}")
        if c['type'] == 'universal':
            rotation_ok(c['frame1_R'], c['id'] + '.frame1_R')
            rotation_ok(c['frame2_R'], c['id'] + '.frame2_R')
            nchart = 2
        elif c['type'] == 'point_coincidence':
            nchart = 3
        else:
            raise ModelError(f"Loop cut {c['id']}: unsupported type {c['type']}")
        p1, p2 = paths[c['body1']], paths[c['body2']]
        common = 0
        while common < min(len(p1), len(p2)) and p1[common] == p2[common]:
            common += 1
        lca = root if common == 0 else jrec[p1[common - 1]]['follower_body']
        path1, path2 = p1[common:], p2[common:]
        moving = [ids.index(j) for j in path1 + path2 if jrec[j]['type'] != 'fixed']
        charts = list(range(at, at + nchart))
        at += nchart
        reference = (poses[c['body1']][:3, :3].T @ poses[c['body2']][:3, :3]).tolist()
        cut_plans.append(dict(id=c['id'], type=c['type'], body1=c['body1'], body2=c['body2'], lca=lca,
                              path1=path1, path2=path2, moving_indices=sorted(set(moving)),
                              motor_indices=sorted(set(moving) & set(motor_idx)),
                              passive_physical=sorted(set(moving) - set(motor_idx)),
                              chart_columns=charts, chart_reference=reference))
    n_aug = at
    passive_phys = sorted(set(range(nt)) - set(motor_idx))
    covered = set().union(*(set(p['passive_physical']) for p in cut_plans)) if cut_plans else set()
    if set(passive_phys) - covered:
        missing = [ids[i] for i in sorted(set(passive_phys) - covered)]
        raise ModelError('Unactuated coordinates not determined by any loop cut: ' + ', '.join(missing[:5]))
    for p in cut_plans:
        if not p['passive_physical']:
            raise ModelError(f"Loop cut {p['id']} has no dependent physical coordinate")
    supports = [p['passive_physical'] + p['chart_columns'] for p in cut_plans]
    groups = components(supports)
    passive_idx = sorted(set(range(n_aug)) - set(motor_idx))
    modules = []
    for g in groups:
        pas = sorted(set().union(*(set(supports[i]) for i in g)))
        mot = sorted(set().union(*(set(cut_plans[i]['motor_indices']) for i in g)))
        modules.append(dict(cuts=g, cut_ids=[cut_plans[i]['id'] for i in g], passive=pas, motors=mot,
                            rows=[6 * i + r for i in g for r in range(6)],
                            passive_physical=[i for i in pas if i < nt], charts=[i for i in pas if i >= nt]))
    sparsity = np.zeros((6 * len(cut_plans), len(passive_idx)), dtype=int)
    for k, s in enumerate(supports):
        sparsity[np.ix_(np.arange(6 * k, 6 * k + 6), [passive_idx.index(i) for i in s])] = 1
    root_paths = {}
    sites = sorted(source.get('sole_sites', []), key=lambda s: s['id'])
    if len({s['id'] for s in sites}) != len(sites):
        raise ModelError('Unique sole-site ids required')
    for s in sites:
        if s['body'] not in bmap:
            raise ModelError(f"Sole site {s['id']}: unknown body")
        rotation_ok(s['frame_rotation'], s['id'] + '.frame_rotation')
        path = paths[s['body']]
        root_paths[s['body']] = dict(joint_path=path,
                                     moving_indices=[ids.index(j) for j in path if jrec[j]['type'] != 'fixed'])
    # ---- generated CMG (source-CMG schema so accepted and native backends consume it)
    out_bodies = []
    for b in bodies:
        e = deepcopy(b)
        e['kind'] = 'rigid_body'
        out_bodies.append(e)
    closures = []
    for c in cuts:
        e = dict(id=c['id'], type=c['type'], body1=c['body1'], body2=c['body2'],
                 point1_m=deepcopy(c['point1_m']), point2_m=deepcopy(c['point2_m']))
        if c['type'] == 'universal':
            e.update(frame1_R=deepcopy(c['frame1_R']), frame2_R=deepcopy(c['frame2_R']))
        closures.append(e)
    cmg = dict(schema='cmg.kangaroo.compiled-loop-graph/1.0', name=source.get('name', ''), root_body=root,
               coordinate_ids=ids, independent_ids=aj, bodies=out_bodies, joints=ordered, closures=closures,
               actuators=[dict(id=a['id'], joint=a['joint_id'], gear=a['gear'], unit='N',
                               force_bounds_N=deepcopy(a['force_bounds_N'])) for a in acts],
               armature={k: float(drive[k]['armature']) for k in ids},
               joint_dissipation={k: dict(damping=float(drive[k]['damping']),
                                          frictionloss=float(drive[k].get('frictionloss', 0.))) for k in ids},
               initial_seed=q_seed.tolist(), sole_sites=deepcopy(sites),
               gravity_m_s2=deepcopy(source.get('gravity_m_s2', [0., 0., -9.81])),
               floating_base=dict(coordinates='world translation and quaternion xyzw', actuated=False))
    physical_rows = sum(3 + (1 if p['type'] == 'universal' else 0) for p in cut_plans)
    physical_rank_expected = len(passive_phys)
    plan = dict(physical_bodies=len(bmap), physical_joints=len(edges), physical_coordinates=nt,
                floating_base_coordinates=6, physical_actuators=len(acts), unactuated_base_coordinates=6,
                loop_cuts=len(cut_plans), point_cuts=sum(p['type'] == 'point_coincidence' for p in cut_plans),
                universal_cuts=sum(p['type'] == 'universal' for p in cut_plans),
                chart_coordinates=n_aug - nt, augmented_coordinates=n_aug,
                augmented_residual_rows=6 * len(cut_plans), augmented_rank_expected=len(passive_idx),
                physical_closure_rows=physical_rows, physical_closure_rank_expected=physical_rank_expected,
                redundant_physical_rows=physical_rows - physical_rank_expected,
                motor_indices=motor_idx, passive_indices=passive_idx, cut_plans=cut_plans, modules=modules,
                module_sizes=[len(m['passive']) for m in modules],
                largest_module_block=max(len(m['passive']) for m in modules),
                passive_sparsity=sparsity.tolist(), sparsity_nonzeros=int(sparsity.sum()),
                root_paths=root_paths,
                note='Loop modules are independent given the motor coordinates because every cut is closed between '
                     'bodies below its lowest common ancestor; the floating-base dynamics remain fully coupled.',
                physical_input_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest())
    return Compilation(cmg, plan, source, ordered)


# ---------------------------------------------------------------------------
# Twelve predefined data/representation variants (no selection by results)
# ---------------------------------------------------------------------------
def variant_inputs(source):
    out = [('nominal', deepcopy(source))]
    for scale in (.95, 1.05):
        s = deepcopy(source)
        for j in s['joints']:
            for key in ('T_AJ', 'T_BJ'):
                t = np.array(j[key])
                t[:3, 3] *= scale
                j[key] = t.tolist()
            if j['type'] == 'prismatic':
                j['limits'] = dict(lower=j['limits']['lower'] * scale, upper=j['limits']['upper'] * scale)
        for b in s['bodies']:
            b['com_m'] = (np.array(b['com_m']) * scale).tolist()
            b['inertia_com_kg_m2'] = (np.array(b['inertia_com_kg_m2']) * scale ** 2).tolist()
        for c in s['loop_cuts']:
            c['point1_m'] = (np.array(c['point1_m']) * scale).tolist()
            c['point2_m'] = (np.array(c['point2_m']) * scale).tolist()
        for site in s['sole_sites']:
            site['frame_translation_m'] = (np.array(site['frame_translation_m']) * scale).tolist()
            site['corners_m'] = (np.array(site['corners_m']) * scale).tolist()
        prismatic = {j['id'] for j in s['joints'] if j['type'] == 'prismatic'}
        s['seed']['joints'] = {k: v * scale if k in prismatic else v for k, v in s['seed']['joints'].items()}
        out.append((f'geometry_scale_{scale}', s))
    # Rigid shift of every sub-mechanism attached to the floating root (loops stay closable).
    for d in (.004, -.004):
        s = deepcopy(source)
        for j in s['joints']:
            if j['body_a'] == s['floating_root']:
                t = np.array(j['T_AJ'])
                t[0, 3] += d
                j['T_AJ'] = t.tolist()
            elif j['body_b'] == s['floating_root']:
                t = np.array(j['T_BJ'])
                t[0, 3] += d
                j['T_BJ'] = t.tolist()
        for c in s['loop_cuts']:
            for side in (1, 2):
                if c[f'body{side}'] == s['floating_root']:
                    p = np.array(c[f'point{side}_m'])
                    p[0] += d
                    c[f'point{side}_m'] = p.tolist()
        out.append((f'root_attachment_{d:+.3f}m', s))
    for scale in (.95, 1.05):
        # Sole frame (and its corners) moved radially about the foot-body origin: changes the weld geometry.
        s = deepcopy(source)
        for site in s['sole_sites']:
            site['frame_translation_m'] = (scale * np.array(site['frame_translation_m'])).tolist()
            site['corners_m'] = (scale * np.array(site['corners_m'])).tolist()
        out.append((f'sole_site_offset_scale_{scale}', s))
    s = deepcopy(source)
    b = next(b for b in s['bodies'] if b['id'] == s['floating_root'])
    b['mass_kg'] = b['mass_kg'] + 1.5  # co-located point payload at the root COM
    out.append(('root_com_payload_1p5kg', s))
    s = deepcopy(source)
    for b in s['bodies']:
        if b['id'] != s['floating_root']:
            b['mass_kg'] *= 1.1
            b['inertia_com_kg_m2'] = (1.1 * np.array(b['inertia_com_kg_m2'])).tolist()
    out.append(('leg_inertia_scale_1p1', s))
    s = deepcopy(source)
    rng = np.random.default_rng(271828)
    H = {}
    for b in s['bodies']:
        if b['id'] == s['floating_root']:
            continue
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rng.uniform(-.25, .25, 3)).as_matrix()
        T[:3, 3] = rng.uniform(-.008, .008, 3)
        H[b['id']] = T
        R, p = T[:3, :3], T[:3, 3]
        b['com_m'] = (R.T @ (np.array(b['com_m']) - p)).tolist()
        b['inertia_com_kg_m2'] = (R.T @ np.array(b['inertia_com_kg_m2']) @ R).tolist()
    for j in s['joints']:
        for bodykey, framekey in (('body_a', 'T_AJ'), ('body_b', 'T_BJ')):
            if j[bodykey] in H:
                j[framekey] = (np.linalg.inv(H[j[bodykey]]) @ np.array(j[framekey])).tolist()
    for c in s['loop_cuts']:
        for side in (1, 2):
            T = H.get(c[f'body{side}'])
            if T is None:
                continue
            R, p = T[:3, :3], T[:3, 3]
            c[f'point{side}_m'] = (R.T @ (np.array(c[f'point{side}_m']) - p)).tolist()
            if c['type'] == 'universal':
                c[f'frame{side}_R'] = (R.T @ np.array(c[f'frame{side}_R'])).tolist()
    for site in s['sole_sites']:
        T = H.get(site['body'])
        if T is not None:
            R, p = T[:3, :3], T[:3, 3]
            site['frame_translation_m'] = (R.T @ (np.array(site['frame_translation_m']) - p)).tolist()
            site['frame_rotation'] = (R.T @ np.array(site['frame_rotation'])).tolist()
            site['corners_m'] = [(R.T @ (np.array(x) - p)).tolist() for x in site['corners_m']]
    out.append(('body_frame_reexpression', s))
    s = deepcopy(source)
    for key in ('bodies', 'joints', 'actuators', 'loop_cuts', 'sole_sites'):
        s[key] = list(reversed(s[key]))
    out.append(('reversed_record_and_actuator_order', s))
    s = deepcopy(source)
    bm = {b['id']: f'body_{i:02d}' for i, b in enumerate(s['bodies'])}
    jm = {j['id']: f'joint_{i:02d}' for i, j in enumerate(s['joints'])}
    for b in s['bodies']:
        b['id'] = bm[b['id']]
    for j in s['joints']:
        j['id'] = jm[j['id']]
        j['body_a'] = bm[j['body_a']]
        j['body_b'] = bm[j['body_b']]
    for i, c in enumerate(s['loop_cuts']):
        c['id'] = f'cut_{i:02d}'
        c['body1'] = bm[c['body1']]
        c['body2'] = bm[c['body2']]
    for i, a in enumerate(s['actuators']):
        a['id'] = f'actuator_{i:02d}'
        a['joint_id'] = jm[a['joint_id']]
    for i, site in enumerate(s['sole_sites']):
        site['id'] = f'site_{i:02d}'
        site['body'] = bm[site['body']]
    s['floating_root'] = bm[s['floating_root']]
    s['seed']['joints'] = {jm[k]: v for k, v in s['seed']['joints'].items()}
    s['drive'] = {jm[k]: v for k, v in s['drive'].items()}
    out.append(('opaque_body_joint_cut_names', s))
    assert len(out) == 12
    return out


def support_sets(comp):
    ids = sorted(s['id'] for s in comp.cmg['sole_sites'])
    return [list(x) for m in (1, 2) for x in combinations(ids, m)]
