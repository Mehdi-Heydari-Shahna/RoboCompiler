"""Independent source-frame calculations for backend verification.

No MuJoCo or exporter imports. Uses a separate breadth-first traversal of the
full physical graph, including cut edges, from the accepted MATLAB coordinates.
These are numerical source references, not new physical measurements.
"""
from collections import deque
import numpy as np


def joint_motion(joint, value):
    transform = np.eye(4)
    axis = np.asarray(joint['axis'], dtype=float)
    if joint['type'] == 'revolute':
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        transform[:3, :3] = (np.eye(3) + np.sin(value)*skew
                            + (1-np.cos(value))*(skew@skew))
    elif joint['type'] == 'prismatic':
        transform[:3, 3] = axis*value
    elif joint['type'] != 'fixed':
        raise ValueError('Unsupported source joint type')
    return transform


def source_poses(cmg, coordinates):
    adjacency = {b['id']: [] for b in cmg['bodies']}
    for joint in sorted(cmg['joints'], key=lambda x: x['id']):
        adjacency[joint['base_body']].append((joint, True))
        adjacency[joint['follower_body']].append((joint, False))
    root = cmg['root_body']
    poses = {root: np.eye(4)}
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        for joint, direct in adjacency[parent]:
            child = joint['follower_body'] if direct else joint['base_body']
            if child in poses:
                continue
            relative = (np.asarray(joint['T_BJ'])
                        @ joint_motion(joint, coordinates.get(joint['id'], 0.))
                        @ np.linalg.inv(np.asarray(joint['T_FJ'])))
            poses[child] = poses[parent] @ (relative if direct else np.linalg.inv(relative))
            queue.append(child)
    if len(poses) != len(adjacency):
        raise ValueError('Disconnected reference graph')
    return poses


def pose_error(actual, expected):
    delta_rotation = expected[:3, :3].T @ actual[:3, :3]
    skew = (delta_rotation-delta_rotation.T)/2
    sine = np.linalg.norm([skew[2, 1], skew[0, 2], skew[1, 0]])
    cosine = (np.trace(delta_rotation)-1)/2
    return (float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
            float(np.arctan2(sine, cosine)))


def cylinder_attachment_records(cmg):
    """Source revolute-frame origins; no pin-center or hydraulic zero is inferred."""
    records = []
    joints = cmg['joints']
    for slide in sorted((j for j in joints if j['type']=='prismatic'), key=lambda j:j['id']):
        endpoints = []
        for body in [slide['base_body'], slide['follower_body']]:
            pins = [j for j in joints if j['type']=='revolute'
                    and body in [j['base_body'], j['follower_body']]]
            if len(pins) != 1:
                raise ValueError('Cylinder attachment identification is ambiguous')
            pin = pins[0]
            key = 'T_BJ' if pin['base_body']==body else 'T_FJ'
            endpoints.append({'body':body, 'joint':pin['id'],
                              'point':np.asarray(pin[key])[:3, 3]})
        records.append({'id':slide['id'], 'endpoints':endpoints})
    return records


def cylinder_distances(poses, records):
    result = {}
    for record in records:
        points = [(poses[e['body']] @ np.r_[e['point'], 1.])[:3]
                  for e in record['endpoints']]
        result[record['id']] = float(np.linalg.norm(points[1]-points[0]))
    return result
