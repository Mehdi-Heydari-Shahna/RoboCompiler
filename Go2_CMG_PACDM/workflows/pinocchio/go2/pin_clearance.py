"""Independent, sampled collision audit of saved Pinocchio trajectories.

This module imports neither MuJoCo nor its renderer. Source collision
primitives and MJCF defaults are read as XML; the CMG-to-Pinocchio compiler
provides all moving body transforms. Coal computes signed primitive
distances, without replacing cylinders with boxes or mesh approximations.
The audit detects unexpected intersections; it does not resolve them.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import coal
import numpy as np

from .model import _defaults, _rotation, load_model
from .pin_backend import PinBackend
from .task import HURDLES, RAIL_HALF_WIDTH

# A foot sphere may rest on the floor or on the top face of a rail. Contact with
# a rail side face or edge is not support and is audited like any other
# environment contact. Top-face support: the sphere centre lies above the rail
# top, at least RAIL_EDGE_MARGIN inside its x extent (the closest rail point is
# then on the top face, with a vertical normal).
RAIL_EDGE_MARGIN = .001


def _rail_top_support(centre, rail_index):
    x, height = HURDLES[rail_index]
    return bool(centre[2] > height and abs(centre[0] - x) <= RAIL_HALF_WIDTH - RAIL_EDGE_MARGIN
                and abs(centre[1]) < .6)


@dataclass
class Primitive:
    name: str
    body: str
    kind: str
    shape: object
    placement: np.ndarray
    contype: int = 1
    conaffinity: int = 1
    foot: bool = False


def _primitive(name, body, attrs, foot=False):
    kind = attrs.get('type', 'sphere')
    size = np.fromstring(attrs.get('size', ''), sep=' ')
    local = np.eye(4)
    local[:3, :3] = _rotation(attrs)
    local[:3, 3] = np.fromstring(attrs.get('pos', '0 0 0'), sep=' ')
    if 'fromto' in attrs:
        raise ValueError('Unexpected source fromto geometry; explicit support is required')
    if kind == 'box' and len(size) == 3:
        shape = coal.Box(*(2 * size))
    elif kind == 'sphere' and len(size) >= 1:
        shape = coal.Sphere(float(size[0]))
    elif kind == 'cylinder' and len(size) >= 2:
        shape = coal.Cylinder(float(size[0]), float(2 * size[1]))
    elif kind == 'capsule' and len(size) >= 2:
        shape = coal.Capsule(float(size[0]), float(2 * size[1]))
    elif kind == 'plane':
        # MJCF planes are infinite for contact, regardless of visual size.
        shape = coal.Halfspace(np.array([0., 0., 1.]), 0.)
    else:
        raise ValueError(f'Unsupported collision primitive {name}: {kind}, {size}')
    return Primitive(name, body, kind, shape, local,
                     int(attrs.get('contype', '1')),
                     int(attrs.get('conaffinity', '1')), foot)


def source_collision_primitives(root):
    """Read authored active collision primitives and source body filters."""
    path = Path(root) / 'upstream/unitree_go2/go2.xml'
    source = ET.parse(path).getroot()
    defaults = _defaults(source)
    robot, parents = [], {}
    ignored_visual = 0

    def walk(body, parent, inherited_class):
        nonlocal ignored_visual
        name = body.attrib['name']
        parents[name] = parent
        childclass = body.get('childclass', inherited_class)
        for index, geom in enumerate(body.findall('geom')):
            classname = geom.get('class', childclass)
            if classname not in defaults:
                raise ValueError(f'Unknown source geom default class {classname}')
            attrs = dict(defaults[classname].get('geom', {}), **geom.attrib)
            if not (int(attrs.get('contype', '1')) or
                    int(attrs.get('conaffinity', '1'))):
                ignored_visual += 1
                continue
            geom_name = geom.get('name', f'{name}/geom_{index}')
            robot.append(_primitive(geom_name, name, attrs,
                                    foot=geom_name in ('FL', 'FR', 'RL', 'RR')))
        for child in body.findall('body'):
            walk(child, name, childclass)

    for body in source.findall('worldbody/body'):
        walk(body, 'world', 'main')
    excludes = {frozenset((e.attrib['body1'], e.attrib['body2']))
                for e in source.findall('contact/exclude')}
    if source.findall('contact/pair'):
        raise ValueError('Explicit contact pairs need separate precedence handling')
    return robot, parents, excludes, ignored_visual


def course_primitives(terrain=True):
    """The exact physical course dimensions in the original simulation.py."""
    environment = [_primitive('floor', 'world', {'type': 'plane'})]
    if terrain:
        for k, (x, height) in enumerate(HURDLES):
            environment.append(_primitive(f'rail_{k}', 'world',
                {'type': 'box', 'size': f'.025 .6 {height / 2}',
                 'pos': f'{x} 0 {height / 2}'}))
        environment.append(_primitive('low_gate', 'world',
            {'type': 'box', 'size': '.025 .575 .02', 'pos': '1.65 0 .38'}))
        for y in [-.55, .55]:
            environment.append(_primitive(f'gate_post_{y}', 'world',
                {'type': 'box', 'size': '.025 .025 .18', 'pos': f'1.65 {y} .18'}))
    return environment


def _compatible(a, b):
    return bool((a.contype & b.conaffinity) or (b.contype & a.conaffinity))


def _coal_transform(pose):
    return coal.Transform3s(pose[:3, :3], pose[:3, 3])


def _distance_request():
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    request.gjk_tolerance = 1e-9
    request.epa_tolerance = 1e-9
    request.gjk_max_iterations = 256
    request.epa_max_iterations = 256
    return request


def _audit_self_checks(backend, cmg, robot):
    """Analytic primitive checks plus an injected body/floor collision."""
    request = _distance_request()
    plane = course_primitives(False)[0].shape
    identity = coal.Transform3s()
    cases = [
        ('sphere_above_floor', coal.Sphere(.1), np.array([0., 0., .3]),
         plane, .2),
        ('sphere_intersects_floor', coal.Sphere(.1), np.array([0., 0., .05]),
         plane, -.05),
        ('separated_boxes', coal.Box(.2, .2, .2), np.array([.3, 0., 0.]),
         coal.Box(.2, .2, .2), .1),
        ('overlapping_boxes', coal.Box(.2, .2, .2), np.array([.1, 0., 0.]),
         coal.Box(.2, .2, .2), -.1),
    ]
    checks = []
    for name, a, position, b, expected in cases:
        measured = float(coal.distance(a, coal.Transform3s(np.eye(3), position),
                                       b, identity, request, coal.DistanceResult()))
        checks.append(dict(name=name, expected_signed_distance_m=expected,
                           measured_signed_distance_m=measured,
                           absolute_error_m=abs(measured - expected),
                           passed=abs(measured - expected) < 1e-8))
    lowered = np.asarray(cmg['q_reference'], dtype=float).copy()
    lowered[2] -= .4
    poses = backend.poses(lowered)
    clearances = [float(coal.distance(p.shape,
                  _coal_transform(poses[p.body] @ p.placement), plane,
                  identity, request, coal.DistanceResult()))
                  for p in robot if not p.foot]
    count = sum(d < -1e-7 for d in clearances)
    checks.append(dict(name='lowered_body_negative_control',
                       injected_base_lowering_m=.4,
                       nonfoot_floor_intersections=count,
                       minimum_clearance_m=min(clearances), passed=count > 0))
    if not all(c['passed'] for c in checks):
        raise RuntimeError('Clearance implementation self-check failed: ' + str(checks))
    return {'passed': True, 'checks': checks}


def audit_clearance(root, case='nominal', stride=1):
    """Audit saved actual states and write ``{case}_clearance.json``.

    Prefer ``time_dense,q_dense`` (integration states) when available,
    otherwise use ``time,q``. Every saved sample is checked by default. A
    stride greater than one is reported and never represented as a
    continuous-time collision proof.
    Foot/floor and foot/rail-top contact is expected; foot contact with a rail
    side face or edge, all other environment contacts and all eligible self
    contacts are unexpected.
    """
    root = Path(root)
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('stride must be a positive integer')
    results = root / 'results_pinocchio'
    trajectory = results / f'{case}.npz'
    with np.load(trajectory, allow_pickle=False) as saved:
        logged_samples = len(saved['time'])
        dense = 'time_dense' in saved or 'q_dense' in saved
        if dense and not ('time_dense' in saved and 'q_dense' in saved):
            raise ValueError('Dense history requires both time_dense and q_dense')
        time_key, q_key = ('time_dense', 'q_dense') if dense else ('time', 'q')
        time, q = saved[time_key].copy(), saved[q_key].copy()
    if (time.ndim != 1 or q.shape != (len(time), 18) or not len(time)
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError('Trajectory needs increasing time and finite q[N,18]')
    metadata_path = results / f'{case}.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    terrain = bool(metadata.get('terrain', True))
    cmg = load_model(root / 'data/go2_cmg.json')
    backend = PinBackend(cmg)
    robot, parents, excludes, ignored_visual = source_collision_primitives(root)
    self_checks = _audit_self_checks(backend, cmg, robot)
    environment = course_primitives(terrain)
    pairs, excluded = [], {'same_body': 0, 'parent_child': 0,
                           'source_exclude': 0, 'bitmask': 0,
                           'expected_foot_support': 0}
    foot_rail = {}  # pair index -> rail index, audited with the top-face exemption
    for i, a in enumerate(robot):
        for j in range(i + 1, len(robot)):
            b = robot[j]
            if a.body == b.body:
                excluded['same_body'] += 1
            elif parents.get(a.body) == b.body or parents.get(b.body) == a.body:
                excluded['parent_child'] += 1
            elif frozenset((a.body, b.body)) in excludes:
                excluded['source_exclude'] += 1
            elif not _compatible(a, b):
                excluded['bitmask'] += 1
            else:
                pairs.append((i, j, 'self', f'{a.name} / {b.name}'))
    all_primitives = robot + environment
    for i, a in enumerate(robot):
        for j, b in enumerate(environment, start=len(robot)):
            if a.foot and b.name == 'floor':
                excluded['expected_foot_support'] += 1
            elif a.foot and b.name.startswith('rail_'):
                foot_rail[len(pairs)] = int(b.name.split('_')[1])
                pairs.append((i, j, b.name, f'{a.name} / {b.name}'))
            elif _compatible(a, b):
                pairs.append((i, j, b.name, f'{a.name} / {b.name}'))
            else:
                excluded['bitmask'] += 1

    indices = np.unique(np.r_[np.arange(0, len(time), stride), len(time) - 1])
    request = _distance_request()
    minima = np.full(len(pairs), np.inf)
    minimum_indices = np.zeros(len(pairs), dtype=int)
    pair_intersections = np.zeros(len(pairs), dtype=int)
    sample_intersections = 0
    total_intersections = 0
    top_support_exemptions = 0
    records = []
    tolerance = 1e-7
    for sample_index in indices:
        poses = backend.poses(q[sample_index])
        transforms = [_coal_transform(poses[p.body] @ p.placement)
                      for p in robot]
        transforms.extend(_coal_transform(p.placement) for p in environment)
        intersections_here = 0
        for k, (i, j, category, label) in enumerate(pairs):
            if k in foot_rail:
                foot = robot[i]
                centre = (poses[foot.body] @ foot.placement)[:3, 3]
                if _rail_top_support(centre, foot_rail[k]):
                    top_support_exemptions += 1
                    continue
            result = coal.DistanceResult()
            value = float(coal.distance(all_primitives[i].shape, transforms[i],
                                        all_primitives[j].shape, transforms[j],
                                        request, result))
            if not np.isfinite(value):
                raise RuntimeError(f'Nonfinite distance at {time[sample_index]}: {label}')
            if value < minima[k]:
                minima[k], minimum_indices[k] = value, sample_index
            if value < -tolerance:
                intersections_here += 1
                pair_intersections[k] += 1
                if len(records) < 100:
                    records.append({'time_s': float(time[sample_index]),
                                    'pair': label, 'category': category,
                                    'signed_clearance_m': value})
        sample_intersections += bool(intersections_here)
        total_intersections += intersections_here

    pair_reports = [{'pair': p[3], 'category': p[2],
                     'minimum_clearance_m': float(minima[k]),
                     'minimum_time_s': float(time[minimum_indices[k]]),
                     'intersection_sample_count': int(pair_intersections[k])}
                    for k, p in enumerate(pairs)]
    categories = {}
    for category in sorted({p[2] for p in pairs}):
        matched = [r for r in pair_reports if r['category'] == category]
        worst = min(matched, key=lambda r: r['minimum_clearance_m'])
        categories[category] = dict(
            minimum_clearance_m=worst['minimum_clearance_m'],
            minimum_pair=worst['pair'], minimum_time_s=worst['minimum_time_s'],
            checked_pairs=len(matched),
            intersection_sample_count=sum(r['intersection_sample_count'] for r in matched))
    worst = min(pair_reports, key=lambda r: r['minimum_clearance_m'])
    source = root / 'upstream/unitree_go2/go2.xml'
    sampled_dt = np.diff(time[indices])
    recorded_dt = np.diff(time)
    integration_dt = metadata.get('dt_s')
    covers_steps = bool(dense and stride == 1 and integration_dt is not None
                        and len(recorded_dt) and np.allclose(
                            recorded_dt, float(integration_dt), rtol=1e-6, atol=1e-10))
    report = dict(
        case=case, passed=total_intersections == 0,
        backend='CMG-to-Pinocchio body poses + Coal signed primitive distances',
        coal_version=getattr(coal, '__version__', 'unknown'),
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        trajectory_sha256=sha256(trajectory.read_bytes()).hexdigest(),
        source_collision_primitives=len(robot), source_visual_geoms_ignored=ignored_visual,
        collision_meshes=0, environment_primitives=len(environment), terrain=terrain,
        trajectory_state_keys=[time_key, q_key], dense_history_available=dense,
        saved_samples=len(time), logged_samples=logged_samples,
        checked_samples=len(indices), stride=stride,
        resolution=('every_integration_state' if covers_steps else 'saved_state_samples'),
        all_integration_states_checked=covers_steps,
        declared_integration_step_s=integration_dt,
        checked_start_s=float(time[indices[0]]), checked_end_s=float(time[indices[-1]]),
        min_sample_interval_s=float(np.min(sampled_dt)) if len(sampled_dt) else None,
        max_sample_interval_s=float(np.max(sampled_dt)) if len(sampled_dt) else None,
        saved_sample_rate_hz=float(1 / np.median(recorded_dt)) if len(time) > 1 else None,
        checked_pairs_per_sample=len(pairs), distance_evaluations=len(pairs) * len(indices),
        intersection_tolerance_m=tolerance, gjk_tolerance=1e-9, epa_tolerance=1e-9,
        minimum_clearance_m=worst['minimum_clearance_m'], minimum_pair=worst['pair'],
        minimum_time_s=worst['minimum_time_s'],
        intersection_pair_samples=total_intersections,
        samples_with_intersections=sample_intersections,
        excluded_pair_counts=excluded, category_results=categories,
        foot_rail_top_support_exemptions=top_support_exemptions,
        implementation_self_checks=self_checks,
        intersection_examples=records, pair_minima=pair_reports,
        interpretation='A sampled geometric audit, not continuous collision detection. '
                       'Source margin is not added to shapes: clearance is actual '
                       'primitive surface separation. Contact eligibility uses '
                       'contype/conaffinity, same-body and immediate-parent filtering, '
                       'and explicit source body excludes. Expected sphere foot/floor '
                       'support and sphere foot/rail-top support (foot centre above '
                       'the rail top face) are excluded; foot contact with a rail side '
                       'face or edge is audited. Other intersections invalidate this '
                       'audit and are not resolved by it.')
    (results / f'{case}_clearance.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--case', default='nominal')
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    outcome = audit_clearance(args.root, args.case, args.stride)
    print(json.dumps({k: outcome[k] for k in ('case', 'passed', 'checked_samples',
        'minimum_clearance_m', 'intersection_pair_samples', 'category_results')}, indent=2))
