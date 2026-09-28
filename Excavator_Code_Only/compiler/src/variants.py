"""Twelve fixed, data-only variants of the physical input.

The list is fixed in code and no variant is selected or dropped by its results. One
disclosed change was made during development (METHODS.md, section 12): the pin offsets
of the loop-geometry variants were first applied along the joint axis, which made two
parallel-axis loops unclosable (the compiler rejected them, correctly); they are now
applied in the joint plane (``_shift_attachment``).
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation


def _scale_geometry(s, scale):
    for j in s['joints']:
        for key in ('T_AJ', 'T_BJ'):
            t = np.array(j[key], float)
            t[:3, 3] *= scale
            j[key] = t.tolist()
    for b in s['bodies']:
        b['com_m'] = (np.array(b['com_m']) * scale).tolist()
        b['inertia_com_kg_m2'] = (np.array(b['inertia_com_kg_m2']) * scale ** 2).tolist()
    for jid, value in s['seed']['joints'].items():
        if next(j for j in s['joints'] if j['id'] == jid)['type'] == 'prismatic':
            s['seed']['joints'][jid] = value * scale
    s['branch_window']['prismatic_m'] *= scale


def _shift_attachment(s, joint_id, side, offset):
    """Move a joint origin by ``offset`` expressed in the joint frame.

    Offsets are perpendicular to the joint axis (joint-frame x), so planar loops
    stay closable; an axial offset would make a parallel-axis loop inconsistent.
    """
    j = next(j for j in s['joints'] if j['id'] == joint_id)
    t = np.array(j[side], float)
    t[:3, 3] += t[:3, :3] @ np.asarray(offset, float)
    j[side] = t.tolist()


def variant_inputs(source):
    out = [('nominal', deepcopy(source))]
    for scale in (.95, 1.05):
        s = deepcopy(source)
        _scale_geometry(s, scale)
        out.append((f'geometry_scale_{scale}', s))
    # Loop-geometry changes (pin positions in the source body frames); the seed
    # is no longer closed and is re-closed by the compiler's PACDM acquisition.
    s = deepcopy(source)
    _shift_attachment(s, 'q24', 'T_AJ', (0.01, 0., 0.))
    _shift_attachment(s, 'q25', 'T_AJ', (0.01, 0., 0.))
    out.append(('boom_cylinder_base_pins_+0.01m', s))
    s = deepcopy(source)
    _shift_attachment(s, 'q8', 'T_AJ', (-0.01, 0., 0.))
    out.append(('stick_cylinder_base_pin_-0.01m', s))
    s = deepcopy(source)
    for jid in ('q14', 'q17'):
        _shift_attachment(s, jid, 'T_AJ', (0.005, 0., 0.))
    out.append(('linkage_side_link_pins_+5mm', s))
    s = deepcopy(source)
    bucket = next(b for b in s['bodies'] if b['id'] == 'body_56')
    bucket['mass_kg'] = float(bucket['mass_kg']) + 250.   # point payload at the bucket COM
    out.append(('bucket_payload_250kg', s))
    s = deepcopy(source)
    for b in s['bodies']:
        b['mass_kg'] = float(b['mass_kg']) * 1.1
        b['inertia_com_kg_m2'] = (1.1 * np.array(b['inertia_com_kg_m2'])).tolist()
    out.append(('mass_inertia_scale_1p1', s))
    # Consistent re-expression of every body frame (rotation + translation).
    s = deepcopy(source)
    rng = np.random.default_rng(271828)
    transforms = {}
    for b in s['bodies']:
        H = np.eye(4)
        H[:3, :3] = Rotation.from_rotvec(rng.uniform(-.25, .25, 3)).as_matrix()
        H[:3, 3] = rng.uniform(-.02, .02, 3)
        transforms[b['id']] = H
        R, p = H[:3, :3], H[:3, 3]
        b['com_m'] = (R.T @ (np.array(b['com_m']) - p)).tolist()
        b['inertia_com_kg_m2'] = (R.T @ np.array(b['inertia_com_kg_m2']) @ R).tolist()
    for j in s['joints']:
        for bodykey, framekey in (('body_a', 'T_AJ'), ('body_b', 'T_BJ')):
            if j[bodykey] in transforms:
                j[framekey] = (np.linalg.inv(transforms[j[bodykey]]) @ np.array(j[framekey])).tolist()
    out.append(('body_frame_reexpression', s))
    s = deepcopy(source)
    for key in ('bodies', 'joints', 'actuators'):
        s[key] = list(reversed(s[key]))
    out.append(('reversed_record_order', s))
    s = deepcopy(source)
    for j in s['joints']:
        j['body_a'], j['body_b'] = j['body_b'], j['body_a']
        j['T_AJ'], j['T_BJ'] = j['T_BJ'], j['T_AJ']
        if j['type'] != 'fixed':
            j['axis'] = (-np.array(j['axis'])).tolist()
    out.append(('reversed_joint_storage', s))   # swapped ends and negated axis: same coordinates
    s, _ = opaque(source)
    out.append(('opaque_body_joint_names', s))
    assert len(out) == 12
    return out


def opaque(source):
    s = deepcopy(source)
    bm = {b['id']: f'b{i:02d}' for i, b in enumerate(s['bodies'])}
    jm = {j['id']: f'j{i:02d}' for i, j in enumerate(s['joints'])}
    for b in s['bodies']:
        b['id'] = bm[b['id']]
        b['name'] = ''
    for j in s['joints']:
        j['id'] = jm[j['id']]
        j['body_a'] = bm.get(j['body_a'], j['body_a'])
        j['body_b'] = bm.get(j['body_b'], j['body_b'])
    for a in s['actuators']:
        a['joint_id'] = jm[a['joint_id']]
    s['seed']['joints'] = {jm[k]: v for k, v in s['seed']['joints'].items()}
    s['partition_requests'] = {k: [jm[x] for x in v] for k, v in s['partition_requests'].items()}
    return s, dict(bodies=bm, joints=jm)
