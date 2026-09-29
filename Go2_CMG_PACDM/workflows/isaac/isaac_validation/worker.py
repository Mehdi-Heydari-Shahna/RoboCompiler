"""Newline JSON controller service, run with the separate Miniforge Python.

Protocol v1: one object ``{id, op, ...parameters}`` per UTF-8 line. Replies
are ``{id, ok: true, result: ...}`` or ``{id, ok: false, error, traceback}``.
Every command is synchronous; stdout is reserved exclusively for replies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

PROTOCOL_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]


class WorkerEngine:
    """Simulator-independent interface to the unchanged original controller."""

    def __init__(self, root=ROOT):
        import numpy as np
        from scipy.interpolate import BPoly
        from go2.model import load_model

        self.np = np
        self.root = Path(root).resolve()
        self.cmg = load_model(self.root / "data/go2_cmg.json")
        with np.load(self.root / "data/reference.npz", allow_pickle=False) as data:
            self.reference = {name: data[name].copy() for name in data.files}
        self.interp = BPoly.from_derivatives(
            self.reference["time"],
            np.stack([self.reference["q"], self.reference["v"], self.reference["a"]], axis=1),
        )
        self.controller = None
        self.duration = float(self.reference["time"][-1])

    def _vector(self, value, name):
        value = self.np.asarray(value, dtype=float)
        if value.shape != (18,) or not self.np.isfinite(value).all():
            raise ValueError(f"{name} must contain exactly 18 finite values in CMG order")
        return value

    def _require_init(self):
        if self.controller is None:
            raise RuntimeError("Send init before command or audit_state")

    def init(self, duration=26.0, feedforward=True, friction=0.5):
        from .controller import WholeBodyController

        np = self.np
        duration = float(duration)
        if not np.isfinite(duration) or not 0 < duration <= float(self.reference["time"][-1]):
            raise ValueError("duration must be positive and within the saved reference")
        friction = float(friction)
        if not np.isfinite(friction) or friction <= 0:
            raise ValueError("controller friction coefficient must be positive and finite")
        if not isinstance(feedforward, bool):
            raise ValueError("feedforward must be a JSON boolean")
        self.controller = WholeBodyController(self.cmg, friction=friction, feedforward=feedforward)
        self.duration = duration
        joints = {joint["id"]: joint for joint in self.cmg["joints"]}
        names = self.cmg["coordinate_ids"][6:]
        source = (Path(__file__).with_name("controller.py")).read_text(encoding="utf-8")
        source = source[source.index("class WholeBodyController:"):]
        q0 = self.reference["q"][0].copy()
        v0 = self.reference["v"][0].copy()
        return dict(
            protocol_version=PROTOCOL_VERSION, q=q0, v=v0, q0=q0, v0=v0,
            joint_names=names, coordinate_ids=self.cmg["coordinate_ids"],
            damping=self.cmg["damping"], frictionloss=self.cmg["frictionloss"],
            armature=self.cmg["armature"], limits=self.controller.limits,
            lower=[joints[name]["limits"]["lower"] for name in names],
            upper=[joints[name]["limits"]["upper"] for name in names],
            duration=duration, reference_duration=float(self.reference["time"][-1]),
            total_mass_kg=self.cmg["total_mass_kg"],
            controller_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            feedforward=feedforward, controller_friction=friction,
        )

    def command(self, t, q, v):
        from go2.task import target_trajectory

        self._require_init()
        np = self.np
        t = float(t)
        if not np.isfinite(t) or not -1e-10 <= t <= self.duration + 1e-10:
            raise ValueError("command time must be within the requested simulation duration")
        t = float(np.clip(t, 0.0, self.duration))
        q, v = self._vector(q, "q"), self._vector(v, "v")
        if abs(q[4]) >= 1.45:
            raise ValueError("Base pitch left the regular CMG Euler chart (|pitch| < 1.45 rad)")
        active, av, aa, stance = (value[0] for value in target_trajectory(np.asarray([t])))
        qr, vr, ar = self.interp(t), self.interp(t, nu=1), self.interp(t, nu=2)
        tau, forces, violation, iterations = self.controller.command(q, v, qr, vr, ar, active, av, aa, stance)
        feet = self.controller.kin.points(q)
        return dict(tau=tau, predicted_forces=forces, qp_violation=float(violation),
                    qp_iterations=int(iterations), q_ref=qr, v_ref=vr, a_ref=ar,
                    active=active, active_v=av, active_a=aa, stance=stance, feet=feet)

    def audit_state(self, q, v):
        from go2.model import velocity_map

        self._require_init()
        np = self.np
        q, v = self._vector(q, "q"), self._vector(v, "v")
        feet, jacobians = self.controller.kin.points_and_jacobians(q)
        poses = self.controller.backend.poses(q)
        transform = velocity_map(q)
        transform[3:6, 3:6] = poses["base"][:3, :3] @ transform[3:6, 3:6]
        mass = self.controller.backend.mass(q)
        return dict(mass=mass, mass_with_armature=mass + np.diag(self.controller.armature),
                    bias=self.controller.backend.bias(q, v), feet=feet, jacobians=jacobians,
                    body_poses=poses, chart_to_world_velocity=transform,
                    chart_velocity=v, world_base_twist=(transform @ v)[:6])

    def validate_reference(self):
        from go2.task_validation import validate_reference

        return validate_reference(self.root)

    def make_reference(self, dt=0.01):
        from go2.task import make_reference

        dt = float(dt)
        if not self.np.isfinite(dt) or not 0.001 <= dt <= 0.02:
            raise ValueError("reference dt must lie between 0.001 and 0.02 seconds")
        (self.root / "results").mkdir(exist_ok=True)
        result = make_reference(self.root, dt=dt)
        self.__init__(self.root)
        return result


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def serve(input_stream, output_stream, root=ROOT):
    engine = None
    for line in input_stream:
        request_id = None
        shutdown = False
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            request_id = request.pop("id", None)
            operation = request.pop("op", None)
            if operation == "ping":
                result = {"protocol_version": PROTOCOL_VERSION, "python": sys.executable,
                          "root": str(Path(root).resolve())}
            elif operation == "shutdown":
                result, shutdown = {"closed": True}, True
            else:
                if operation not in {"init", "command", "audit_state", "validate_reference", "make_reference"}:
                    raise ValueError(f"Unknown operation: {operation!r}")
                if engine is None:
                    engine = WorkerEngine(root)
                result = getattr(engine, operation)(**request)
            response = {"id": request_id, "ok": True, "result": result}
            encoded = json.dumps(response, default=_json_default, allow_nan=False)
        except Exception as error:
            encoded = json.dumps({"id": request_id, "ok": False,
                                  "error": f"{type(error).__name__}: {error}",
                                  "traceback": traceback.format_exc()})
        output_stream.write(encoded + "\n")
        output_stream.flush()
        if shutdown:
            return


def main():
    # A duplicated descriptor preserves the protocol pipe while redirecting
    # *both* Python and native library stdout to the diagnostic log (fd 2).
    # contextlib.redirect_stdout alone does not catch native OSQP output.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        serve(sys.stdin, protocol)
    finally:
        protocol.close()


if __name__ == "__main__":
    main()
