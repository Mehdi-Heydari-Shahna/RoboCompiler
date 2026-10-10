"""Power-consistent effort ports; no simulator imports or state writes.

Inputs follow manifest body order (all non-world bodies). The returned torques
are about each body's COM, in world axes, for native PhysX force application.
Viscous joint losses are applied here once, not through an extra native drive.
"""
from __future__ import annotations
import numpy as np


def rotation_wxyz(q):
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError('Expected finite wxyz quaternions')
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError('Zero quaternion')
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1]+(3,3))


def _state(manifest, positions, quaternions_wxyz, linear_velocities, angular_velocities):
    bodies = manifest['bodies']; n = len(bodies)
    p, v, w = (np.asarray(a, dtype=float) for a in (positions, linear_velocities, angular_velocities))
    if any(a.shape != (n,3) or not np.isfinite(a).all() for a in (p,v,w)):
        raise ValueError('Invalid native body state')
    R = rotation_wxyz(quaternions_wxyz)
    if R.shape != (n,3,3): raise ValueError('Body quaternion count mismatch')
    com_delta = np.einsum('nij,nj->ni', R, np.array([b['com_local'] for b in bodies]))
    com = p + com_delta
    vcom = v + np.cross(w, com_delta)
    return p,R,v,w,com,vcom,{int(b['id']): i for i,b in enumerate(bodies)}


def joint_wrenches(manifest, positions, quaternions_wxyz, linear_velocities, angular_velocities, efforts):
    """Return (body forces, body torques about COM), world SI.

    Each prismatic pair uses one common world application point, including
    reaction moments on offset parent/child COMs. Hinge ports use torque pairs.
    Never add the rotor's reflected inertia: it is an explicit native body.
    """
    p,R,v,w,com,vcom,lookup = _state(manifest,positions,quaternions_wxyz,linear_velocities,angular_velocities)
    acts = manifest['actuators']; u = np.asarray(efforts, dtype=float)
    if u.shape != (len(acts),) or not np.isfinite(u).all(): raise ValueError('Invalid effort vector')
    per_joint = {}
    for a, value in zip(acts,u):
        for key in ('ctrlrange','forcerange'):
            bounds = a.get(key)
            if bounds is not None: value = float(np.clip(value,*bounds))
        gear = a.get('gear',1.)
        if isinstance(gear,list): gear=gear[0]
        jid=int(a['joint_id'])
        per_joint[jid] = per_joint.get(jid,0.) + float(gear)*value
    F=np.zeros_like(p); T=np.zeros_like(p)
    for j in manifest['joints']:
        kind=j['type']
        if kind not in ('hinge','slide'): continue
        child=lookup[int(j['body_id'])]; parent=lookup.get(int(j['parent_id']))
        axis=R[child]@np.asarray(j['axis_local_child'],float)
        anchor=p[child]+R[child]@np.asarray(j['anchor_local_child'],float)
        wp=np.zeros(3) if parent is None else w[parent]
        if kind=='hinge': rate=float(axis@(w[child]-wp))
        else:
            vc=v[child]+np.cross(w[child],anchor-p[child])
            vp=np.zeros(3) if parent is None else v[parent]+np.cross(wp,anchor-p[parent])
            rate=float(axis@(vc-vp))
        tau=per_joint.get(int(j['id']),0.)-float(j.get('damping',0.))*rate
        if kind=='hinge':
            T[child]+=axis*tau
            if parent is not None: T[parent]-=axis*tau
        else:
            f=axis*tau
            F[child]+=f; T[child]+=np.cross(anchor-com[child],f)
            if parent is not None:
                F[parent]-=f; T[parent]-=np.cross(anchor-com[parent],f)
    return F,T


def mechanical_observables(manifest, positions, quaternions_wxyz, linear_velocities, angular_velocities,
                           forces=None, torques=None):
    p,R,v,w,com,vcom,lookup=_state(manifest,positions,quaternions_wxyz,linear_velocities,angular_velocities)
    masses=np.array([b['mass'] for b in manifest['bodies']])
    inertia=np.array([b['inertia_diagonal'] for b in manifest['bodies']])
    RI=R@rotation_wxyz([b['inertia_quat_wxyz'] for b in manifest['bodies']])
    wi=np.einsum('nji,nj->ni',RI,w)
    kinetic=float(.5*np.sum(masses[:,None]*vcom*vcom)+.5*np.sum(inertia*wi*wi))
    gravity=np.asarray(manifest.get('gravity',[0.,0.,-9.81]))
    potential=float(-np.sum(masses*(com@gravity)))
    momentum=np.sum(masses[:,None]*vcom,axis=0)
    spin=np.einsum('nij,nj->ni',RI,inertia*wi)
    angular_momentum=np.sum(spin+np.cross(com,masses[:,None]*vcom),axis=0)
    result=dict(kinetic_energy_J=kinetic,potential_energy_J=potential,
                mechanical_energy_J=kinetic+potential,linear_momentum=momentum.tolist(),
                angular_momentum_about_world_origin=angular_momentum.tolist())
    if forces is not None and torques is not None:
        F,T=np.asarray(forces),np.asarray(torques)
        result['applied_power_W']=float(np.sum(F*vcom+T*w))
        result['applied_resultant_force_N']=F.sum(axis=0).tolist()
        result['applied_resultant_moment_Nm']=(T+np.cross(com,F)).sum(axis=0).tolist()
    return result
