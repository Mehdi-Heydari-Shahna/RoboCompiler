"""PhysX actor-pose / COM-velocity conversion to the original XYZ/ZYX chart."""
import numpy as np


def rotation(q):
    """xyzw quaternion to a body-to-world matrix."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quaternion(yaw, pitch, roll):
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    return np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])


def angular_map(q):
    yaw, pitch, roll = q[3:6]
    # World angular velocity = A @ [yaw_dot,pitch_dot,roll_dot].
    return np.array([[0, -np.sin(yaw), np.cos(yaw)*np.cos(pitch)],
                     [0, np.cos(yaw), np.sin(yaw)*np.cos(pitch)],
                     [1, 0, -np.sin(pitch)]])


def to_chart(transform, com_velocity, joints, joint_velocities, com_local):
    r = rotation(transform[3:7])
    pitch = np.arcsin(np.clip(-r[2, 0], -1, 1))
    q = np.r_[transform[:3], np.arctan2(r[1, 0], r[0, 0]), pitch,
              np.arctan2(r[2, 1], r[2, 2]), joints]
    if abs(pitch) >= 1.45:
        raise ValueError('Floating-base ZYX chart outside its declared domain')
    omega = np.asarray(com_velocity[3:6], float)
    origin_velocity = com_velocity[:3] - np.cross(omega, r @ com_local)
    v = np.r_[origin_velocity, np.linalg.solve(angular_map(q), omega), joint_velocities]
    return q, v


def from_chart(q, v, com_local):
    quat = quaternion(*q[3:6])
    omega = angular_map(q) @ v[3:6]
    com_velocity = v[:3] + np.cross(omega, rotation(quat) @ com_local)
    return np.r_[q[:3], quat], np.r_[com_velocity, omega]
