"""PACDM foot tasks and unilateral-contact manifold audits for Unitree Go2.

The physical robot is a floating-base tree. Foot-to-world edges here are
imposed stance/task constraints, not permanent structural closed chains.
Each edge is a pure-translation SE(3) subgroup: its three rotational rows
are identically zero and all three Cartesian rows remain in the acceptance
test. This adapter leaves the supplied PACDM implementation unchanged.

The six base coordinates are x, y, z, yaw, pitch, roll. The base chart is
local (pitch must stay away from +/-pi/2); it is not a quaternion chart.
Massless toe-target coordinates carry neither inertia nor actuator force.
"""
from __future__ import annotations

import numpy as np

from vendor.pacdm_original import PACDM, PointGraph, rank, skew


def _feet(cmg):
    """Normalize source toe-site metadata without inventing robot geometry."""
    records = cmg.get("feet", cmg.get("foot_sites"))
    if records is None:
        raise ValueError("CMG must supply feet with source body and point_m")
    if isinstance(records, dict):
        records = [dict(value, id=name) for name, value in records.items()]
    feet = []
    for record in records:
        feet.append({"id": record.get("id", record.get("name")),
                     "body": record.get("body", record.get("body_id")),
                     "point_m": np.asarray(record.get("point_m", record.get("position_m")), float),
                     "joint_ids": record.get("joint_ids")})
    if len(feet) != 4 or any(f["point_m"].shape != (3,) for f in feet):
        raise ValueError("Expected four source toe sites with Cartesian local coordinates")
    return feet


def _bounds(cmg):
    records = {joint["id"]: joint for joint in cmg["joints"]}
    return (np.asarray([records[name]["limits"]["lower"] for name in cmg["coordinate_ids"]]),
            np.asarray([records[name]["limits"]["upper"] for name in cmg["coordinate_ids"]]))


def _translation_residual(errors, point_jacobians, n, defects):
    """Exact subgroup logarithm and differential, including redundant rows."""
    nc = len(errors)
    residual = np.zeros((nc, 6))
    jacobian = np.zeros((nc, 6, n))
    delta = np.broadcast_to(np.eye(4), (nc, 4, 4)).copy()
    residual[:, 3:] = errors
    if defects is not None:
        defects = np.asarray(defects)
        if defects.shape != (nc, 4, 4) or not np.allclose(defects[:, :3, :3], np.eye(3), atol=1e-12):
            raise ValueError("Foot homotopy defects must be pure translations")
        residual[:, 3:] -= defects[:, :3, 3]
    jacobian[:, 3:] = point_jacobians
    delta[:, :3, 3] = residual[:, 3:]
    return residual.reshape(-1), jacobian.reshape(-1, n), delta


class FootKinematics:
    """Source-CMG world toe positions and analytic angular-first Jacobians."""

    def __init__(self, cmg, seed):
        self.cmg = cmg
        self.ids = list(cmg["coordinate_ids"])
        self.nt = len(self.ids)
        if self.nt != 18:
            raise ValueError("Go2 requires six base and twelve leg coordinates")
        self.feet = _feet(cmg)
        tree = dict(cmg, closures=[], independent_ids=self.ids)
        self.tree = PointGraph(tree, np.asarray(seed)[:self.nt])
        self.leg_indices = []
        for foot in self.feet:
            if foot["joint_ids"] is not None:
                indices = [self.ids.index(name) for name in foot["joint_ids"]]
            else:
                indices = []
                body = foot["body"]
                while body != cmg["root_body"]:
                    joint = next(j for j in cmg["joints"] if j["follower_body"] == body)
                    if joint["id"] in self.ids and self.ids.index(joint["id"]) >= 6:
                        indices.append(self.ids.index(joint["id"]))
                    body = joint["base_body"]
                indices.reverse()
            if len(indices) != 3:
                raise ValueError("Each toe must descend from three actuated leg joints")
            self.leg_indices.append(np.asarray(indices, int))

    def points_and_jacobians(self, q):
        poses, spatial = self.tree.poses(np.asarray(q)[:self.nt])
        points = []
        jacobians = []
        for foot in self.feet:
            pose = poses[foot["body"]]
            point = pose[:3, :3] @ foot["point_m"] + pose[:3, 3]
            e = spatial[foot["body"]]
            points.append(point)
            jacobians.append(e[3:] - skew(point) @ e[:3])
        return np.asarray(points), np.asarray(jacobians)

    def points(self, q):
        return self.points_and_jacobians(q)[0]


class ContactGraph:
    """Four Cartesian foot tasks solved by the unchanged PACDM core.

    Thirty coordinates = 18 physical + 12 prescribed toe coordinates.
    Eighteen independent coordinates = base6 + toe targets12.
    Twelve dependent coordinates = twelve leg joints.
    The 24-row residual has rank 12. Every rotational row is identically
    zero, representing a position-only task without an orientation lock.
    Swing targets move; stance targets stay fixed. Actual MuJoCo contact
    remains unilateral with friction and is never welded to these targets.
    """

    def __init__(self, cmg, seed):
        self.cmg = cmg
        self.kinematics = FootKinematics(cmg, seed)
        self.physical_ids = self.kinematics.ids
        self.nt = 18
        self.n = 30
        self.nc = 4
        self.feet = self.kinematics.feet
        self.ids = self.physical_ids + [f"target_{foot['id']}_{axis}"
                                       for foot in self.feet for axis in "xyz"]
        self.active = np.r_[np.arange(6), np.arange(18, 30)]
        self.passive = np.arange(6, 18)
        lower, upper = _bounds(cmg)
        self.lower = np.r_[lower, np.full(12, -np.inf)]
        self.upper = np.r_[upper, np.full(12, np.inf)]
        self.paths = [{"cut": "toe_task_" + foot["id"],
                       "cut_type": "position-only task; pure-translation SE3 subgroup",
                       "body": foot["body"], "point_m": foot["point_m"].tolist()}
                      for foot in self.feet]
        self.solver = PACDM(self)
        self._last_q = None
        self._last_N = None
        self._last_info = None

    def lift(self, q):
        q = np.asarray(q, float)[:self.nt]
        return np.r_[q, self.kinematics.points(q).reshape(-1)]

    augment = lift

    def residual(self, q, defects=None):
        q = np.asarray(q)
        points, physical_jacobian = self.kinematics.points_and_jacobians(q[:18])
        jacobian = np.pad(physical_jacobian, ((0, 0), (0, 0), (0, 12)))
        for foot in range(4):
            jacobian[foot, :, 18 + 3 * foot:21 + 3 * foot] = -np.eye(3)
        return _translation_residual(points - q[18:].reshape(4, 3), jacobian, self.n, defects)

    def solve_feet(self, base, feet, seed=None):
        """Return q18, its 18x18 tangent map, and exact PACDM diagnostics.

        Tangent-map columns are base6 followed by toe-target12. A saved
        solution supplies continuation unless an explicit seed is passed.
        An initial acquisition or failed predictor uses defect homotopy.
        """
        independent = np.r_[np.asarray(base, float).reshape(6), np.asarray(feet, float).reshape(12)]
        if seed is None and self._last_q is not None:
            predicted = self._last_q[self.passive] + self._last_N[self.passive] @ (
                independent - self._last_q[self.active])
            q, solve = self.solver.correct(independent, predicted, None,
                                           np.asarray(self._last_info["rows"]), maxiter=8)
            fallback = not solve["success"]
            if fallback:
                q, solve = self.solver.acquire(independent, self._last_q)
        else:
            if seed is None:
                raise ValueError("The initial solve requires a physical seed")
            q, solve = self.solver.acquire(independent, self.lift(seed))
            fallback = False
        if not solve["success"]:
            raise RuntimeError("PACDM foot assembly failed: " + str(solve))
        tangent, info = self.solver.mapping(q)
        if not info["success"]:
            raise RuntimeError("PACDM foot differential failed: " + str(info))
        self._last_q, self._last_N, self._last_info = q, tangent, info
        info = dict(info, correction=solve, continuation_fallback=fallback)
        return q[:18].copy(), tangent[:18].copy(), info


class StanceGraph:
    """Physical fixed-anchor stance manifold for a nonempty support set.

    For m nonsingular stance legs, rank is 3m and mobility is 18-3m.
    Base6 and the 3(4-m) swing-joint coordinates are independent. This is
    the ideal stationary toe-center approximation; finite-radius source
    feet can also roll, slip, or detach in simulation. This graph does not
    decide unilateral complementarity or replace the contact solver.
    """

    def __init__(self, cmg, seed, stance, anchors=None):
        self.cmg = cmg
        self.kinematics = FootKinematics(cmg, seed)
        self.ids = self.kinematics.ids
        self.nt = self.n = 18
        stance = np.asarray(stance)
        self.stance = np.flatnonzero(stance) if stance.dtype == bool else stance.astype(int)
        if not 1 <= len(self.stance) <= 4 or len(set(self.stance.tolist())) != len(self.stance):
            raise ValueError("Stance must contain one to four distinct foot indices")
        if np.any(self.stance < 0) or np.any(self.stance > 3):
            raise ValueError("Foot indices must lie in [0,3]")
        self.nc = len(self.stance)
        all_anchors = self.kinematics.points(seed) if anchors is None else np.asarray(anchors)
        self.anchors = all_anchors[self.stance].copy() if all_anchors.shape == (4, 3) else all_anchors.copy()
        if self.anchors.shape != (self.nc, 3):
            raise ValueError("Anchor array must contain all four feet or the selected stance feet")
        self.passive = np.concatenate([self.kinematics.leg_indices[i] for i in self.stance])
        self.active = np.setdiff1d(np.arange(18), self.passive)
        self.lower, self.upper = _bounds(cmg)
        self.paths = [{"cut": "stance_" + self.kinematics.feet[i]["id"],
                       "cut_type": "ideal stationary toe-center stance mode (position only)"}
                      for i in self.stance]

    def lift(self, q):
        return np.asarray(q, float)[:18].copy()

    augment = lift

    def residual(self, q, defects=None):
        points, jacobians = self.kinematics.points_and_jacobians(q)
        return _translation_residual(points[self.stance] - self.anchors,
                                     jacobians[self.stance], self.n, defects)


def solve_feet(cmg, base, feet, seed):
    """One-shot foot assembly; reuse ContactGraph for trajectory continuation."""
    return ContactGraph(cmg, seed).solve_feet(base, feet, seed)


def track_feet(cmg, bases, feet, seed):
    """Assemble a sampled gait and its base/toe differential maps with PACDM."""
    graph = ContactGraph(cmg, seed)
    bases, feet = np.asarray(bases), np.asarray(feet)
    if bases.shape != (len(feet), 6) or feet.shape[1:] != (4, 3):
        raise ValueError("Expected bases[N,6] and feet[N,4,3]")
    qs, maps, diagnostics = [], [], []
    for index, (base, targets) in enumerate(zip(bases, feet)):
        q, mapping, info = graph.solve_feet(base, targets, seed if index == 0 else None)
        qs.append(q)
        maps.append(mapping)
        diagnostics.append(info)
    report = {"samples": len(qs), "physical_coordinates": 18, "task_coordinates": 12,
              "constraint_rank": 12, "independent_coordinates": 18,
              "max_residual_inf": max(info["residual_inf"] for info in diagnostics),
              "max_tangent_residual": max(info["tangent_residual"] for info in diagnostics),
              "min_rcond": min(info["rcond"] for info in diagnostics),
              "fallback_count": sum(info["continuation_fallback"] for info in diagnostics)}
    return np.asarray(qs), np.asarray(maps), report


def support_audit(cmg, q, stance):
    """Report ideal support rank/mobility at current geometric anchors."""
    graph = StanceGraph(cmg, q, stance)
    residual, jacobian, _ = graph.residual(q)
    mapping, info = PACDM(graph).mapping(q)
    return dict(info, stance_feet=[graph.kinematics.feet[i]["id"] for i in graph.stance],
                rank=int(rank(jacobian)), mobility=int(18 - rank(jacobian)),
                residual_rows=len(residual), physical_coordinates=18,
                independent_coordinates=len(graph.active),
                map_shape=None if mapping is None else list(mapping.shape))
