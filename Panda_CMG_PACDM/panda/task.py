"""PACDM task closure for the serial Panda and its coupled fingers.

The tool-to-target edge is an imposed task constraint, not a physical arm loop.
Six massless virtual target coordinates make its differential map explicit.
The PACDM core is imported without modification.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from vendor.pacdm_original import PointGraph, PACDM, adj, exp, inv, inv_left, log


HOME = np.array([0., -.45, 0., -2.1, 0., 1.65, np.pi / 4, .04, .04])
REFERENCE_ROTATION = Rotation.from_euler("x", np.pi).as_matrix()
TOOL_TRANSFORM = np.eye(4)
TOOL_TRANSFORM[2, 3] = .1029


class TaskGraph:
    """Nine physical coordinates, six virtual coordinates and rank-seven cuts.

    Independent coordinates are target x/y/z, target local XYZ Euler angles,
    joint3 and finger_joint1. The remaining six arm joints and second finger
    are solved by PACDM. Euler angles parameterize only a small local chart
    around a downward facing nominal tool orientation.
    """

    def __init__(self, cmg, seed=HOME, tool_body="hand", tool_transform=None,
                 redundancy="joint3", reference_rotation=None):
        self.cmg = cmg
        self.physical_ids = list(cmg["coordinate_ids"])
        self.nt = len(self.physical_ids)
        if self.nt != 9:
            raise ValueError("Panda task adapter requires seven arm and two finger coordinates")
        self.n = self.nt + 6
        self.ids = self.physical_ids + ["target_" + s for s in ("x", "y", "z", "rx", "ry", "rz")]
        self.tool_body = tool_body
        self.tool_transform = np.asarray(TOOL_TRANSFORM if tool_transform is None else tool_transform).copy()
        self.reference_rotation = np.asarray(REFERENCE_ROTATION if reference_rotation is None else reference_rotation).copy()
        self.redundancy_index = self.physical_ids.index(redundancy)
        finger_names = [s for s in self.physical_ids if "finger" in s]
        if len(finger_names) != 2:
            raise ValueError("Expected two source finger joints")
        self.finger_indices = np.array([self.physical_ids.index(s) for s in finger_names], int)
        self.active = np.r_[np.arange(self.nt, self.n), self.redundancy_index, self.finger_indices[0]]
        self.passive = np.setdiff1d(np.arange(self.n), self.active)
        self.lower = np.full(self.n, -np.inf)
        self.upper = np.full(self.n, np.inf)
        records = {j["id"]: j for j in cmg["joints"]}
        for k, name in enumerate(self.physical_ids):
            self.lower[k] = records[name]["limits"]["lower"]
            self.upper[k] = records[name]["limits"]["upper"]
        # Local Euler chart: pitch stays away from its singular surface.
        self.lower[self.nt:] = [-2., -2., -.5, -1.4, -1.4, -3.1]
        self.upper[self.nt:] = [2., 2., 2., 1.4, 1.4, 3.1]
        tree_cmg = dict(cmg, closures=[], independent_ids=self.physical_ids)
        self.tree = PointGraph(tree_cmg, np.asarray(seed))
        self.nc = 2
        self.paths = [
            {"cut": "imposed_tool_target", "cut_type": "full SE3 task closure; virtual target, not physical linkage"},
            {"cut": "source_finger_coupling", "cut_type": "q_finger1 - q_finger2 = 0"},
        ]

    def poses(self, q):
        p, e = self.tree.poses(np.asarray(q)[:self.nt])
        return p, {name: np.pad(j, ((0, 0), (0, 6))) for name, j in e.items()}

    def tool_pose(self, q):
        p, _ = self.tree.poses(np.asarray(q)[:self.nt])
        return p[self.tool_body] @ self.tool_transform

    def target_coordinates(self, pose):
        pose = np.asarray(pose)
        local_rotation = self.reference_rotation.T @ pose[:3, :3]
        return np.r_[pose[:3, 3], Rotation.from_matrix(local_rotation).as_euler("XYZ")]

    def target_pose(self, q):
        x = np.asarray(q)[self.nt:] if len(q) == self.n else np.asarray(q)
        t = np.eye(4)
        t[:3, 3] = x[:3]
        t[:3, :3] = self.reference_rotation @ Rotation.from_euler("XYZ", x[3:6]).as_matrix()
        return t

    def lift(self, q):
        q = np.asarray(q)[:self.nt]
        return np.r_[q, self.target_coordinates(self.tool_pose(q))]

    augment = lift

    def _target(self, q):
        x = q[self.nt:]
        t = np.eye(4)
        t[:3, 3] = x[:3]
        t[:3, :3] = self.reference_rotation
        e = np.zeros((6, self.n))
        e[3:, self.nt:self.nt + 3] = np.eye(3)
        for i, axis in enumerate(np.eye(3)):
            twist = np.r_[axis, np.zeros(3)]
            col = self.nt + 3 + i
            e[:, col] = adj(t) @ twist
            t = t @ exp(twist * x[3 + i])
        return t, e

    def residual(self, q, defects=None):
        q = np.asarray(q)
        p, e = self.poses(q)
        target, etarget = self._target(q)
        tool = p[self.tool_body] @ self.tool_transform
        d0 = np.eye(4) if defects is None else defects[0]
        reverse = inv(target @ d0)
        delta0 = reverse @ tool
        r0 = log(delta0)
        j0 = inv_left(r0) @ adj(reverse) @ (e[self.tool_body] - etarget)

        # A translation subgroup gives an exact scalar residual and lets the
        # original PACDM homotopy/physical_ok check all six rows unchanged.
        coupling = np.eye(4)
        coupling[0, 3] = q[self.finger_indices[0]] - q[self.finger_indices[1]]
        d1 = np.eye(4) if defects is None else defects[1]
        delta1 = inv(d1) @ coupling
        r1 = log(delta1)
        j1 = np.zeros((6, self.n))
        j1[3, self.finger_indices] = [1., -1.]
        # The acquire() defect remains in the same translation subgroup.
        return np.r_[r0, r1], np.vstack([j0, j1]), np.asarray([delta0, delta1])


def make_graph(cmg, seed=HOME, **kwargs):
    return TaskGraph(cmg, seed, **kwargs)


def solve_targets(cmg, poses, gap=.08, elbow=0., seed=HOME, **kwargs):
    """Track world-frame tool poses with PACDM; gap is total finger opening.

    Returns augmented configurations, differential maps and solver metadata.
    The source finger coordinate is half the requested symmetric opening.
    """
    graph = TaskGraph(cmg, seed, **kwargs)
    n = len(poses)
    gaps = np.broadcast_to(gap, (n,))
    elbows = np.broadcast_to(elbow, (n,))
    independent = np.asarray([np.r_[graph.target_coordinates(t), elbows[i], gaps[i] / 2]
                              for i, t in enumerate(poses)])
    solver = PACDM(graph)
    states, info = solver.track(independent, graph.lift(np.asarray(seed)))
    maps = []
    conditions = []
    for q in states:
        mapping, result = solver.mapping(q)
        if not result["success"]:
            raise RuntimeError(str(result))
        maps.append(mapping)
        conditions.append(result["rcond"])
    info.update(min_rcond=float(np.min(conditions)), physical_coordinates=graph.nt,
                augmented_coordinates=graph.n, independent_coordinates=len(graph.active),
                constraint_rank=len(graph.passive))
    return states, np.asarray(maps), info


def smoothstep(s):
    s = np.clip(np.asarray(s), 0., 1.)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def mission_knots(pick=(.45, -.20), place=(.45, .20), grasp_z=.092, cruise_z=.35,
                  open_gap=.08, grasp_gap=.010):
    """Default 22-second keyed-cartridge route, all values in SI units.

    Columns are x,y,z, local rx,ry,rz, redundancy joint3, total finger gap.
    Root simulation may replace this route; no object is attached kinematically.
    """
    px, py = pick
    dx, dy = place
    def row(x, y, z, yaw=0., tilt=0., gap=open_gap):
        return [x, y, z, tilt, 0., -yaw, 0., gap]
    times = np.array([0., 1., 3., 4., 5., 8., 12., 13.5, 15., 16., 18., 19., 19.3, 20., 22.])
    rows = np.array([
        row(px, py, grasp_z + .15),
        row(px, py, grasp_z + .15),
        row(px, py, grasp_z),
        row(px, py, grasp_z),
        row(px, py, grasp_z, gap=grasp_gap),
        row(px, py, cruise_z, gap=grasp_gap),
        row(dx, dy, cruise_z, yaw=np.pi / 2, gap=grasp_gap),
        row(dx + .025, dy - .025, cruise_z + .03, yaw=np.pi / 2, tilt=np.deg2rad(20), gap=grasp_gap),
        row(dx - .025, dy + .025, cruise_z, yaw=np.pi / 2, tilt=-np.deg2rad(15), gap=grasp_gap),
        row(dx, dy, cruise_z, yaw=np.pi / 2, gap=grasp_gap),
        row(dx, dy, grasp_z + .07, yaw=np.pi / 2, gap=grasp_gap),
        row(dx, dy, grasp_z, yaw=np.pi / 2, gap=grasp_gap),
        row(dx, dy, grasp_z, yaw=np.pi / 2, gap=grasp_gap),
        row(dx, dy, grasp_z, yaw=np.pi / 2),
        row(dx, dy, grasp_z + .18, yaw=np.pi / 2),
    ])
    # Prescribed redundancy sweep while the six-dimensional tool task remains
    # fixed by its own coordinates: seven arm DoF are used explicitly.
    rows[7, 6] = .25
    rows[8, 6] = -.20
    return times, rows


def sample_knots(times, values, t):
    """Piecewise quintic interpolation with zero endpoint speed/acceleration."""
    t = np.atleast_1d(t)
    k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    duration = times[k + 1] - times[k]
    s = np.clip((t - times[k]) / duration, 0., 1.)
    delta = values[k + 1] - values[k]
    h = s ** 3 * (10 - 15 * s + 6 * s ** 2)
    dh = 30 * s ** 2 * (1 - s) ** 2 / duration
    ddh = 60 * s * (1 - s) * (1 - 2 * s) / duration ** 2
    return (values[k] + h[:, None] * delta,
            dh[:, None] * delta, ddh[:, None] * delta)


def build_reference(cmg, dt=.01, seed=HOME, knots=None, **graph_kwargs):
    """Generate q,v,a using the original PACDM map along prescribed targets.

    Acceleration curvature is solved from J a + Jdot v = 0, using a centered
    directional finite difference of the analytic J for Jdot. Independent
    target acceleration is prescribed by the quintic route.
    """
    graph = TaskGraph(cmg, seed, **graph_kwargs)
    times, values = mission_knots() if knots is None else knots
    t = np.arange(0., times[-1] + dt / 2, dt)
    active, active_v, active_a = sample_knots(np.asarray(times), np.asarray(values), t)
    active[:, -1] /= 2
    active_v[:, -1] /= 2
    active_a[:, -1] /= 2
    solver = PACDM(graph)
    states, info = solver.track(active, graph.lift(np.asarray(seed)))
    maps, rc, jacobians, rows, closure, tangent = [], [], [], [], [], []
    for q in states:
        n, result = solver.mapping(q)
        if not result["success"]:
            raise RuntimeError(str(result))
        maps.append(n)
        rc.append(result["rcond"])
        residual, jacobian, _ = graph.residual(q)
        jacobians.append(jacobian)
        rows.append(np.asarray(result["rows"], int))
        closure.append(float(np.max(np.abs(residual))))
        tangent.append(float(np.max(np.abs(jacobian @ n))))
    maps = np.asarray(maps)
    velocity = np.einsum("tij,tj->ti", maps, active_v)
    acceleration = np.einsum("tij,tj->ti", maps, active_a)
    acceleration_residual = []
    for i, (q, v, jacobian, selected) in enumerate(zip(states, velocity, jacobians, rows)):
        epsilon = 1e-6 / max(1., np.linalg.norm(v))
        jdot = (graph.residual(q + epsilon * v)[1] -
                graph.residual(q - epsilon * v)[1]) / (2 * epsilon)
        curvature = np.zeros(graph.n)
        curvature[graph.passive] = -np.linalg.solve(
            jacobian[np.ix_(selected, graph.passive)], (jdot @ v)[selected])
        acceleration[i] += curvature
        acceleration_residual.append(float(np.max(np.abs(jacobian @ acceleration[i] + jdot @ v))))
    info.update(min_rcond=float(np.min(rc)), physical_coordinates=9,
                augmented_coordinates=15, independent_coordinates=8, constraint_rank=7,
                acceleration_method="N a_active + curvature solving J a + Jdot v = 0; analytic J, centered directional finite difference Jdot",
                max_closure_residual=float(np.max(closure)),
                max_tangent_residual=float(np.max(tangent)),
                max_acceleration_constraint_residual=float(np.max(acceleration_residual)))
    tool_poses = np.asarray([graph.target_pose(q) for q in states])
    return {"time": t, "q": states[:, :9], "v": velocity[:, :9], "a": acceleration[:, :9],
            "augmented_q": states, "mapping": maps, "active": active,
            "augmented_v": velocity, "augmented_a": acceleration,
            "active_v": active_v, "active_a": active_a, "rcond": np.asarray(rc),
            "tool_target_pos": tool_poses[:, :3, 3], "tool_target_R": tool_poses[:, :3, :3],
            "closure_residual": np.asarray(closure), "tangent_residual": np.asarray(tangent),
            "acceleration_constraint_residual": np.asarray(acceleration_residual), "info": info}
