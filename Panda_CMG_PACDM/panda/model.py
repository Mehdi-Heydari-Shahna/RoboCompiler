"""Pinned Menagerie Panda MJCF -> explicit CMG -> unchanged PACDM.

All kinematics, full inertial tensors, defaults and transmissions are parsed
from authored MJCF. This importer never imports MuJoCo or reads its compiled
arrays. The Panda arm is a serial chain. Its parallel gripper adds one scalar
coupling; it does not add a physical arm loop.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from vendor.pacdm_original import PointGraph, exp, inv, log

ROOT = Path(__file__).resolve().parents[1]
SOURCE_XML = ROOT / 'upstream' / 'franka_emika_panda' / 'panda.xml'
CMG_PATH = ROOT / 'data' / 'panda_cmg.json'
SOURCE_COMMIT = '822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7'
ARM_IDS = [f'joint{i}' for i in range(1, 8)]
FINGER_IDS = ['finger_joint1', 'finger_joint2']
INDEPENDENT_IDS = ARM_IDS + FINGER_IDS[:1]


def _numbers(value, default):
    return np.asarray(default if value is None else [float(x) for x in value.split()], float)


def transform(rotation=None, position=None):
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if position is not None:
        T[:3, 3] = position
    return T


def _rotation(attrs):
    if 'quat' in attrs:
        q = _numbers(attrs['quat'], [1., 0., 0., 0.])
        if q.shape != (4,) or np.linalg.norm(q) == 0.:
            raise ValueError('Invalid source quaternion')
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    if any(k in attrs for k in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
        raise ValueError('Pinned source adapter expects quaternion orientations')
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


def build_model(source_xml=None):
    """Compile a serial Panda + coupled parallel fingers directly from MJCF."""
    source = SOURCE_XML if source_xml is None else Path(source_xml)
    root = ET.parse(source).getroot()
    compiler = root.find('compiler')
    if compiler is None or compiler.get('angle', 'degree') != 'radian':
        raise ValueError('Expected source angular units to be radians')
    defaults = _defaults(root)
    bodies = [dict(id='world', mass_kg=0., com_m=[0., 0., 0.],
                   inertia_kg_m2=np.zeros((3, 3)).tolist(),
                   inertia_source='massless world')]
    joints, coordinate_ids = [], []

    def attrs(element, inherited_class):
        merged = deepcopy(defaults[element.get('class', inherited_class)].get(element.tag, {}))
        merged.update(element.attrib)
        return merged

    def visit(element, parent, inherited_class):
        name = element.get('name')
        if not name:
            raise ValueError('Every source body must be named')
        child_class = element.get('childclass', inherited_class)
        T0 = transform(_rotation(element.attrib), _numbers(element.get('pos'), [0., 0., 0.]))
        inertial = element.find('inertial')
        if inertial is None:
            raise ValueError(f'Explicit authored inertial required at {name}')
        if 'fullinertia' in inertial.attrib:
            if 'quat' in inertial.attrib:
                raise ValueError('Full inertia with orientation requires explicit handling')
            a, b, c, d, e, f = _numbers(inertial.get('fullinertia'), [])
            tensor = np.array([[a, d, e], [d, b, f], [e, f, c]])
        else:
            R = _rotation(inertial.attrib)
            diagonal = _numbers(inertial.get('diaginertia'), [])
            if diagonal.shape != (3,):
                raise ValueError(f'Expected full or diagonal inertia at {name}')
            tensor = R @ np.diag(diagonal) @ R.T
        bodies.append(dict(id=name, mass_kg=float(inertial.get('mass')),
                           com_m=_numbers(inertial.get('pos'), [0., 0., 0.]).tolist(),
                           inertia_kg_m2=tensor.tolist(),
                           inertia_source='explicit authored MJCF inertial',
                           source_inertial_attributes=dict(inertial.attrib)))
        moving = element.findall('joint')
        if len(moving) > 1 or element.find('freejoint') is not None:
            raise ValueError('Source adapter expects at most one scalar joint per body')
        if moving:
            resolved = attrs(moving[0], child_class)
            kind = resolved.get('type', 'hinge')
            if kind not in ('hinge', 'slide') or float(resolved.get('ref', 0.)) != 0.:
                raise ValueError('Expected zero-reference hinge or slide')
            jid = resolved['name']
            axis = _numbers(resolved.get('axis'), [0., 0., 1.])
            axis /= np.linalg.norm(axis)
            pivot = _numbers(resolved.get('pos'), [0., 0., 0.])
            interval = _numbers(resolved.get('range'), [-np.pi, np.pi])
            # T_parent_child(q) = T0 Trans(pivot) Motion(axis*q) Trans(-pivot).
            joints.append(dict(id=jid, type='revolute' if kind == 'hinge' else 'prismatic',
                               base_body=parent, follower_body=name, axis=axis.tolist(),
                               T_BJ=(T0 @ transform(position=pivot)).tolist(),
                               T_FJ=transform(position=pivot).tolist(),
                               limits=dict(lower=float(interval[0]), upper=float(interval[1])),
                               coordinate_unit='rad' if kind == 'hinge' else 'm',
                               effort_unit='N m' if kind == 'hinge' else 'N',
                               armature=float(resolved.get('armature', 0.)),
                               damping=float(resolved.get('damping', 0.)),
                               stiffness=float(resolved.get('stiffness', 0.)),
                               spring_reference=float(resolved.get('springref', 0.)),
                               source_resolved_attributes=resolved))
            coordinate_ids.append(jid)
        else:
            joints.append(dict(id='fixed_' + name, type='fixed', base_body=parent,
                               follower_body=name, axis=[1., 0., 0.],
                               T_BJ=T0.tolist(), T_FJ=np.eye(4).tolist()))
        for child in element.findall('body'):
            visit(child, name, child_class)

    for element in root.find('worldbody').findall('body'):
        visit(element, 'world', 'main')
    if coordinate_ids != ARM_IDS + FINGER_IDS:
        raise ValueError('Pinned Panda source coordinate order changed')
    couplings = []
    for i, element in enumerate(root.find('equality')):
        if element.tag != 'joint':
            raise ValueError('Pinned Panda source should contain scalar finger coupling only')
        couplings.append(dict(id=element.get('name', f'coordinate_coupling_{i}'),
                              joint1=element.get('joint1'), joint2=element.get('joint2'),
                              polycoef=_numbers(element.get('polycoef'), [0., 1., 0., 0., 0.]).tolist(),
                              source_equality_index=i,
                              source_attributes=dict(element.attrib)))
    tendons = {}
    for element in root.find('tendon'):
        if element.tag != 'fixed' or any(c.tag != 'joint' for c in element):
            raise ValueError('Only fixed scalar joint tendons are supported')
        tendons[element.get('name')] = {j.get('joint'): float(j.get('coef')) for j in element}
    actuators, columns = [], []
    for element in root.find('actuator'):
        if element.tag != 'general':
            raise ValueError('Expected source general actuators')
        resolved = attrs(element, 'main')
        coefficients = ({resolved['joint']: 1.} if 'joint' in resolved
                        else tendons[resolved['tendon']])
        column = np.zeros(len(coordinate_ids))
        gear = _numbers(resolved.get('gear'), [1.])[0]
        for jid, coefficient in coefficients.items():
            column[coordinate_ids.index(jid)] = gear * coefficient
        columns.append(column)
        actuators.append(dict(id=resolved['name'], source_resolved_attributes=resolved,
                              transmission=coefficients,
                              ctrl_range=_numbers(resolved.get('ctrlrange'), []).tolist(),
                              force_range=_numbers(resolved.get('forcerange'), []).tolist(),
                              gain=_numbers(resolved.get('gainprm'), [1.]).tolist(),
                              bias=_numbers(resolved.get('biasprm'), [0., 0., 0.]).tolist()))
    home = root.find("keyframe/key[@name='home']")
    records = {j['id']: j for j in joints}
    cmg = dict(schema='cmg.panda.serial-tree-and-scalar-coupling/1.0',
               name='Franka Emika Panda, serial arm and coupled parallel gripper',
               root_body='world', units='SI', gravity_m_s2=[0., 0., -9.81],
               bodies=bodies, joints=joints, coordinate_ids=coordinate_ids,
               independent_ids=INDEPENDENT_IDS, actuated_independent_ids=INDEPENDENT_IDS,
               passive_independent_ids=[], closures=[], coordinate_couplings=couplings,
               q_reference=_numbers(home.get('qpos'), []).tolist(),
               source_reference=[0.] * len(coordinate_ids),
               source_home_control=_numbers(home.get('ctrl'), []).tolist(),
               mobility=dict(tree_coordinates=9, independent_physical_constraints=1,
                             independent_coordinates=8, actuators=8,
                             point_orientation_chart_coordinates=0),
               tool=dict(body='hand', T_body_tool=transform(position=[0., 0., .1029]).tolist(),
                         convention='Main fingertip pad midpoint; +z from hand toward fingertips'),
               actuation=dict(type='seven_joint_actuators_and_one_finger_tendon',
                              actuators=actuators, fixed_tendons=tendons,
                              moment_matrix=np.column_stack(columns).tolist()),
               source=dict(repository='https://github.com/google-deepmind/mujoco_menagerie',
                           commit=SOURCE_COMMIT, relative_path='franka_emika_panda/panda.xml',
                           sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           license='Apache-2.0',
                           import_method='Direct authored XML only; no simulator arrays'),
               scope='Serial arm plus parallel-finger scalar coupling. Task pose constraints are virtual task constraints, not structural arm loops. Source simulation model is not hardware parameter identification.')
    for field in ('armature', 'damping', 'stiffness', 'spring_reference'):
        cmg[field] = [records[jid][field] for jid in coordinate_ids]
    return cmg


def load_model(path=None):
    """Read the explicit saved CMG without loading an engine or XML parser."""
    return json.loads((CMG_PATH if path is None else Path(path)).read_text(encoding='utf-8'))


def save_model(cmg, path=None):
    destination = CMG_PATH if path is None else Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(cmg, indent=2) + '\n', encoding='utf-8')
    return destination


class PandaGraph(PointGraph):
    """Source scalar coupling as one SE(3) translation block, with no auxiliaries.

    A pure x translation encodes each polynomial scalar equation. Its derivative
    is exact, including during translation-only defect homotopy. The encoding
    adds redundant zero rows but no new physical constraint or coordinate.
    """

    def __init__(self, cmg, seed=None):
        if cmg['closures']:
            raise ValueError('PandaGraph represents the serial source, not task closures')
        super().__init__(cmg, np.asarray(cmg['q_reference'] if seed is None else seed, float))
        self.couplings = cmg['coordinate_couplings']
        self.constraint_blocks = len(self.couplings)
        if not self.constraint_blocks:
            raise ValueError('At least one scalar coupling is required by this adapter')

    def residual(self, q, defects=None):
        residuals, jacobians, deltas = [], [], []
        for k, coupling in enumerate(self.couplings):
            first, second = self.ids.index(coupling['joint1']), self.ids.index(coupling['joint2'])
            coeff = np.asarray(coupling['polycoef'])
            polynomial = np.polynomial.polynomial.polyval(q[second], coeff)
            derivative = np.polynomial.polynomial.polyval(q[second], np.arange(1, len(coeff)) * coeff[1:])
            x = np.zeros(6)
            x[3] = q[first] - polynomial
            defect = np.eye(4) if defects is None else defects[k]
            delta = inv(defect) @ exp(x)
            J = np.zeros((6, self.n))
            J[3, first], J[3, second] = 1., -derivative
            residuals.append(log(delta))
            jacobians.append(J)
            deltas.append(delta)
        return np.concatenate(residuals), np.vstack(jacobians), np.asarray(deltas)


def make_graph(cmg=None, seed=None):
    return PandaGraph(load_model() if cmg is None else cmg, seed)
