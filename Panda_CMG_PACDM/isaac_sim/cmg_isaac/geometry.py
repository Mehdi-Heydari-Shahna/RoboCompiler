"""Resolve source MJCF geometry/defaults without importing any simulator."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from panda.model import _defaults, _rotation, _numbers


def geometry_records(root):
    root = Path(root)
    xml = ET.parse(root/'upstream/franka_emika_panda/panda.xml').getroot()
    defaults = _defaults(xml)
    assets = {m.get('name', Path(m.attrib['file']).stem): m.attrib for m in xml.findall('./asset/mesh')}
    materials = {m.attrib['name']: _numbers(m.get('rgba'), [0.7,0.7,0.7,1.]).tolist()
                 for m in xml.findall('./asset/material')}
    result = []
    def walk(body, inherited):
        childclass = body.get('childclass', inherited)
        for i, geom in enumerate(body.findall('geom')):
            cls = geom.get('class', childclass)
            a = dict(defaults[cls].get('geom', {}), **geom.attrib)
            collision = not (int(a.get('contype', '1')) == 0 and int(a.get('conaffinity','1')) == 0)
            mesh = assets.get(a.get('mesh',''))
            T = np.eye(4)
            T[:3,:3] = _rotation(a)
            T[:3,3] = _numbers(a.get('pos'), [0,0,0])
            r = dict(body=body.attrib['name'], name=geom.get('name',f'geom_{i}'),
                     kind=a.get('type','sphere'), collision=collision,
                     pad=cls.startswith('fingertip_pad_collision_'), T=T.tolist(),
                     rgba=_numbers(a.get('rgba'), materials.get(a.get('material'), [.65,.65,.65,1.])).tolist())
            if mesh:
                if any(key in mesh for key in ('refpos','refquat')):
                    raise ValueError('Unexpected mesh reference transform in pinned source')
                r.update(mesh=a['mesh'], file=mesh['file'],
                         scale=_numbers(mesh.get('scale'),[1,1,1]).tolist())
            elif r['kind']=='box':
                r['halfsize'] = _numbers(a.get('size'), [0,0,0]).tolist()
            else:
                raise ValueError(f'Unsupported authored geometry: {r}')
            result.append(r)
        for child in body.findall('body'):
            walk(child, childclass)
    for body in xml.findall('./worldbody/body'):
        walk(body,'main')
    return result


def align_x(axis):
    """A proper rotation whose local +X is the CMG joint axis."""
    x = np.asarray(axis, float)
    if not np.isclose(np.linalg.norm(x),1.):
        raise ValueError('Axis is not unit length')
    seed = np.eye(3)[np.argmin(abs(x))]
    y = np.cross(seed, x); y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x,y)))


def joint_frames(joint, offset=0.):
    """USD/PhysX joint frames (local X = CMG axis).

    A nonzero revolute ``offset`` c rotates the parent frame by +c about X, so
    the native joint angle is q_native = q_cmg - c while every body pose stays
    identical. It only moves the float32 zero of the native coordinate.
    """
    T = np.eye(4)
    if joint['type'] != 'fixed':
        T[:3,:3] = align_x(joint['axis'])
    A = np.asarray(joint['T_BJ']) @ T
    if offset:
        if joint['type'] != 'revolute':
            raise ValueError('Joint-zero offsets are only defined for revolute joints')
        c, s = np.cos(offset), np.sin(offset)
        Rx = np.eye(4); Rx[1:3,1:3] = [[c,-s],[s,c]]
        A = A @ Rx
    return A, np.asarray(joint['T_FJ']) @ T


def joint_zero_offsets(cmg, reference_q, enabled=True):
    """Native joint-zero offsets (rad), one per CMG coordinate.

    PhysX stores articulation joint angles in float32 and advances them once
    per solver sub-step. Joints 4 and 6 of this route stay near -2.3 and +2.3 rad,
    where float32 spacing is 2.4e-7 rad, 8-16x coarser than near 0.1-0.4 rad.
    Authoring each revolute joint's zero at the middle of its reference range
    (rounded to 1 mrad) keeps native angles small. Physics is unchanged:
    q_cmg = q_native + offset exactly, and native FK is checked before a run.
    Prismatic finger coordinates are never offset.
    """
    offsets = np.zeros(len(cmg['coordinate_ids']))
    if not enabled:
        return offsets
    q = np.asarray(reference_q, float)
    types = {j['id']: j['type'] for j in cmg['joints']}
    for k, jid in enumerate(cmg['coordinate_ids']):
        if types[jid] == 'revolute':
            offsets[k] = round(.5*(float(q[:,k].min())+float(q[:,k].max())), 3)
    return offsets


def collision_exclusions(cmg):
    """MJCF default: exclude welded groups and adjacent welded groups."""
    names = [b['id'] for b in cmg['bodies']]
    group = {n:n for n in names}
    def find(n):
        while group[n]!=n:
            n=group[n]
        return n
    for j in cmg['joints']:
        if j['type']=='fixed':
            group[find(j['follower_body'])]=find(j['base_body'])
    adjacent = set()
    for j in cmg['joints']:
        a,b=find(j['base_body']),find(j['follower_body'])
        if a!=b:
            adjacent.add(frozenset((a,b)))
    pairs = []
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if 'world' in (a,b):
                continue
            if find(a)==find(b) or frozenset((find(a),find(b))) in adjacent:
                pairs.append((a,b))
    return pairs
