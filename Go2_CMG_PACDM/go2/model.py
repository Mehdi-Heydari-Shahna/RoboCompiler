"""Authored Menagerie Go2 MJCF to an independent scalar-coordinate CMG.

The native free joint is represented by an unactuated XYZ + ZYX chart. Its
massless intermediate frames are coordinate devices, not extra robot links
or actuators. All inertial data are read from source XML; MuJoCo is never
imported here. The physical robot is a floating branched tree. Ground contact
constraints are mode-dependent and are added by the contact adapter.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE_XML = ROOT / 'upstream' / 'unitree_go2' / 'go2.xml'
CMG_PATH = ROOT / 'data' / 'go2_cmg.json'
SOURCE_COMMIT = '367e3d9884401dcf6f9c27fa69f118992539039f'
BASE_IDS = ['base_x', 'base_y', 'base_z', 'base_yaw', 'base_pitch', 'base_roll']
LEG_NAMES = ['FL', 'FR', 'RL', 'RR']
JOINT_IDS = [f'{leg}_{part}_joint' for leg in LEG_NAMES for part in ('hip', 'thigh', 'calf')]
COORDINATE_IDS = BASE_IDS + JOINT_IDS


def _numbers(value, default):
    return np.asarray(default if value is None else [float(x) for x in value.split()], float)


def transform(rotation=None, position=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if position is not None:
        result[:3, 3] = position
    return result


def _rotation(attrs):
    if 'quat' in attrs:
        q = _numbers(attrs['quat'], [1., 0., 0., 0.])
        if q.shape != (4,) or np.linalg.norm(q) == 0.:
            raise ValueError('Invalid authored quaternion')
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    if any(k in attrs for k in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
        raise ValueError('Pinned Go2 source uses quaternion orientations')
    return np.eye(3)


def _defaults(root):
    classes = {'main': {}}
    def walk(element, inherited):
        merged = deepcopy(inherited)
        for child in element:
            if child.tag != 'default':
                merged.setdefault(child.tag, {}).update(child.attrib)
        classes[element.get('class', 'main')] = merged
        for child in element.findall('default'):
            walk(child, merged)
    for element in root.findall('default'):
        walk(element, classes['main'])
    return classes


def _vector(value, n, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (n,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{label} must contain {n} finite coordinates')
    return value


def native_qpos_from_chart(q):
    """Convert 18 chart positions to native free joint wxyz + 12 hinges."""
    q = _vector(q, 18, 'q')
    quat_xyzw = Rotation.from_euler('ZYX', q[3:6]).as_quat()
    return np.r_[q[:3], quat_xyzw[[3, 0, 1, 2]], q[6:]]


def chart_from_native(qpos):
    """Canonical ZYX chart; callers must avoid pitch near +/- pi/2."""
    qpos = _vector(qpos, 19, 'qpos')
    angles = Rotation.from_quat(qpos[[4, 5, 6, 3]]).as_euler('ZYX')
    return np.r_[qpos[:3], angles, qpos[7:]]


def velocity_map(q):
    """Native v = E(q) qdot; free translation world, angular velocity body."""
    q = _vector(q, 18, 'q')
    pitch, roll = q[4:6]
    sp, cp, sr, cr = np.sin(pitch), np.cos(pitch), np.sin(roll), np.cos(roll)
    result = np.eye(18)
    result[3:6, 3:6] = [[-sp, 0., 1.], [sr * cp, cr, 0.], [cr * cp, -sr, 0.]]
    return result


def velocity_map_dot(q, v):
    """Exact time derivative of E along chart velocity v."""
    q, v = _vector(q, 18, 'q'), _vector(v, 18, 'v')
    pitch, roll = q[4:6]
    pd, rd = v[4:6]
    sp, cp, sr, cr = np.sin(pitch), np.cos(pitch), np.sin(roll), np.cos(roll)
    result = np.zeros((18, 18))
    result[3:6, 3:6] = [[-cp * pd, 0., 0.],
                        [cr * rd * cp - sr * sp * pd, -sr * rd, 0.],
                        [-sr * rd * cp - cr * sp * pd, -cr * rd, 0.]]
    return result


def convective_velocity(q, v):
    """Native acceleration contribution Edot(q,v) v (not a force)."""
    return velocity_map_dot(q, v) @ _vector(v, 18, 'v')


def chart_velocity_from_native(q, velocity):
    return np.linalg.solve(velocity_map(q), _vector(velocity, 18, 'velocity'))


def build_model(source_xml=None):
    source = SOURCE_XML if source_xml is None else Path(source_xml)
    root = ET.parse(source).getroot()
    compiler = root.find('compiler')
    if compiler is None or compiler.get('angle', 'degree') != 'radian':
        raise ValueError('Pinned source must use radian angles')
    defaults = _defaults(root)
    def attrs(element, inherited_class):
        resolved = deepcopy(defaults[element.get('class', inherited_class)].get(element.tag, {}))
        resolved.update(element.attrib)
        return resolved
    bodies, joints, coordinate_ids, feet = [], [], [], []
    def massless(name):
        bodies.append(dict(id=name, mass_kg=0., com_m=[0., 0., 0.],
                           inertia_kg_m2=np.zeros((3, 3)).tolist(),
                           inertia_source='massless coordinate chart stage'))
    massless('world')
    parent = 'world'
    axes = np.vstack((np.eye(3), np.eye(3)[[2, 1, 0]]))
    for i, jid in enumerate(BASE_IDS):
        child = f'chart_{i}' if i < 5 else 'base'
        if i < 5:
            massless(child)
        interval = [-1.e6, 1.e6] if i < 3 else [-np.pi, np.pi]
        if i == 4:
            interval = [-1.45, 1.45]
        joints.append(dict(id=jid, type='prismatic' if i < 3 else 'revolute',
                           base_body=parent, follower_body=child, axis=axes[i].tolist(),
                           T_BJ=np.eye(4).tolist(), T_FJ=np.eye(4).tolist(),
                           limits=dict(lower=interval[0], upper=interval[1]),
                           coordinate_unit='m' if i < 3 else 'rad',
                           effort_unit='N' if i < 3 else 'N m',
                           armature=0., damping=0., stiffness=0., frictionloss=0.,
                           spring_reference=0., actuated=False,
                           scope='Unactuated chart coordinate of the native free joint'))
        coordinate_ids.append(jid)
        parent = child

    def visit(element, parent, inherited_class, is_base=False):
        name = element.get('name')
        if not name:
            raise ValueError('Every physical source body must be named')
        child_class = element.get('childclass', inherited_class)
        inertial = element.find('inertial')
        if inertial is None:
            raise ValueError(f'Explicit authored inertia required at {name}')
        if 'fullinertia' in inertial.attrib:
            if 'quat' in inertial.attrib:
                raise ValueError('Full inertia orientation needs explicit handling')
            a, b, c, d, e, f = _numbers(inertial.get('fullinertia'), [])
            tensor = np.array([[a, d, e], [d, b, f], [e, f, c]])
        else:
            R = _rotation(inertial.attrib)
            diagonal = _numbers(inertial.get('diaginertia'), [])
            if diagonal.shape != (3,):
                raise ValueError('Diagonal inertia must have three components')
            tensor = R @ np.diag(diagonal) @ R.T
        bodies.append(dict(id=name, mass_kg=float(inertial.get('mass')),
                           com_m=_numbers(inertial.get('pos'), [0., 0., 0.]).tolist(),
                           inertia_kg_m2=tensor.tolist(),
                           inertia_source='explicit authored MJCF inertial',
                           source_inertial_attributes=dict(inertial.attrib)))
        moving = element.findall('joint')
        if is_base:
            if element.find('freejoint') is None or moving:
                raise ValueError('Expected one native free joint at base')
        else:
            if len(moving) != 1 or element.find('freejoint') is not None:
                raise ValueError('Expected one scalar joint per physical leg body')
            resolved = attrs(moving[0], child_class)
            kind = resolved.get('type', 'hinge')
            if kind != 'hinge' or float(resolved.get('ref', 0.)) != 0.:
                raise ValueError('Pinned Go2 source has zero-reference hinge joints')
            axis = _numbers(resolved.get('axis'), [0., 0., 1.])
            axis /= np.linalg.norm(axis)
            pivot = _numbers(resolved.get('pos'), [0., 0., 0.])
            T0 = transform(_rotation(element.attrib), _numbers(element.get('pos'), [0., 0., 0.]))
            interval = _numbers(resolved.get('range'), [-np.pi, np.pi])
            jid = resolved['name']
            joints.append(dict(id=jid, type='revolute', base_body=parent, follower_body=name,
                               axis=axis.tolist(), T_BJ=(T0 @ transform(position=pivot)).tolist(),
                               T_FJ=transform(position=pivot).tolist(),
                               limits=dict(lower=float(interval[0]), upper=float(interval[1])),
                               coordinate_unit='rad', effort_unit='N m', actuated=True,
                               armature=float(resolved.get('armature', 0.)),
                               damping=float(resolved.get('damping', 0.)),
                               frictionloss=float(resolved.get('frictionloss', 0.)),
                               stiffness=float(resolved.get('stiffness', 0.)),
                               spring_reference=float(resolved.get('springref', 0.)),
                               source_resolved_attributes=resolved))
            coordinate_ids.append(jid)
        for geom in element.findall('geom'):
            if geom.get('class') == 'foot':
                resolved = attrs(geom, child_class)
                foot_id = geom.get('name')
                feet.append(dict(id=foot_id, body=name,
                                 point_m=_numbers(resolved.get('pos'), [0., 0., 0.]).tolist(),
                                 radius_m=float(_numbers(resolved.get('size'), [0.])[0]),
                                 joint_ids=[f'{foot_id}_{part}_joint' for part in ('hip', 'thigh', 'calf')],
                                 source_geom_attributes=resolved))
        for child in element.findall('body'):
            visit(child, name, child_class)
    bases = root.find('worldbody').findall('body')
    if len(bases) != 1 or bases[0].get('name') != 'base':
        raise ValueError('Expected one floating base tree')
    visit(bases[0], None, 'main', True)
    if coordinate_ids != COORDINATE_IDS or [f['id'] for f in feet] != LEG_NAMES:
        raise ValueError('Pinned source coordinate or foot order changed')
    actuators, columns = [], []
    for element in root.find('actuator'):
        if element.tag != 'motor':
            raise ValueError('Expected source torque motors')
        resolved = attrs(element, 'main')
        column = np.zeros(18)
        gear = _numbers(resolved.get('gear'), [1.])[0]
        column[coordinate_ids.index(resolved['joint'])] = gear
        columns.append(column)
        actuators.append(dict(id=resolved['name'], joint=resolved['joint'], gear=gear,
                              ctrl_range=_numbers(resolved.get('ctrlrange'), []).tolist(),
                              source_resolved_attributes=resolved))
    home = root.find("keyframe/key[@name='home']")
    native_home = _numbers(home.get('qpos'), [])
    records = {j['id']: j for j in joints}
    cmg = dict(schema='cmg.go2.floating-branched-tree/1.0', name='Unitree Go2',
               root_body='world', units='SI', gravity_m_s2=[0., 0., -9.81],
               bodies=bodies, joints=joints, coordinate_ids=coordinate_ids,
               independent_ids=coordinate_ids, actuated_independent_ids=JOINT_IDS,
               passive_independent_ids=BASE_IDS, closures=[], coordinate_couplings=[],
               q_reference=chart_from_native(native_home).tolist(),
               native_q_reference=native_home.tolist(), feet=feet,
               source_home_control=_numbers(home.get('ctrl'), []).tolist(),
               mobility=dict(floating_base_coordinates=6, leg_coordinates=12,
                             independent_physical_constraints=0,
                             independent_coordinates=18, actuators=12),
               free_base_chart=dict(coordinates=BASE_IDS, convention='world XYZ then intrinsic ZYX yaw pitch roll',
                                    native_velocity='world translation, body angular velocity, source leg velocities',
                                    nominal_chart_pitch_limit_rad=1.45,
                                    native_source_pose=dict(bases[0].attrib)),
               actuation=dict(type='twelve_leg_joint_torque_motors_unactuated_free_base',
                              actuators=actuators, moment_matrix=np.column_stack(columns).tolist()),
               source=dict(repository='https://github.com/google-deepmind/mujoco_menagerie',
                           commit=SOURCE_COMMIT, relative_path='unitree_go2/go2.xml',
                           sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           license='BSD-3-Clause',
                           import_method='Direct authored XML only; no simulator arrays'),
               scope='Floating branched tree, 18 velocities and 12 physical motors. Contact modes create temporary constraints; the free-base chart stages are not physical actuators. Source inertias are simulation data, not hardware identification.')
    cmg['total_mass_kg'] = sum(b['mass_kg'] for b in bodies)
    for field in ('armature', 'damping', 'stiffness', 'spring_reference', 'frictionloss'):
        cmg[field] = [records[jid][field] for jid in coordinate_ids]
    return cmg


def load_model(path=None):
    return json.loads((CMG_PATH if path is None else Path(path)).read_text(encoding='utf-8'))


def save_model(cmg, path=None):
    path = CMG_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cmg, indent=2) + '\n', encoding='utf-8')
    return path
