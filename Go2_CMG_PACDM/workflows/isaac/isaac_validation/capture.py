"""Live RGB capture at a fixed physics state, with bounded readiness checks.

The default uses capture-on-play with blocking World.render calls. Only the
explicit Replicator mode disables capture-on-play and schedules its own step.
Neither mode may advance physics. Imports of NVIDIA modules remain lazy.
"""
from __future__ import annotations
import numpy as np

from .lifecycle import cleanup_actions

_CAPTURE_ON_PLAY = "/omni/replicator/captureOnPlay"
_MULTI_TICK = "/rtx/hydra/supportMultiTickRate"


def configure_capture_camera(stage, camera_path):
    """Apply the installed 6.x sensor API without modifying camera optics/pose.

    RtxCamera is used only for USD authoring, not for a second render product
    or a separate simulation loop. Zero Hz means autotrigger on render ticks.
    Older installations lacking the API can retain their legacy USD camera
    only when multi-tick rendering is not enabled.
    """
    import carb.settings
    settings = carb.settings.get_settings()
    multi_tick = settings.get(_MULTI_TICK)
    prim = stage.GetPrimAtPath(camera_path)
    if not prim or prim.GetTypeName() != "Camera":
        raise RuntimeError(f"Capture camera is not a USD Camera: {camera_path}")
    try:
        from isaacsim.sensors.experimental.rtx import RtxCamera
    except ModuleNotFoundError as exc:
        # Do not hide a missing dependency inside an otherwise-installed API.
        parents = {"isaacsim", "isaacsim.sensors", "isaacsim.sensors.experimental",
                   "isaacsim.sensors.experimental.rtx"}
        if exc.name not in parents or multi_tick:
            raise RuntimeError("RTX camera authoring API is unavailable; cannot configure "
                               "the camera for this multi-tick renderer") from exc
        return dict(path=camera_path, authoring="legacy USD camera",
                    multi_tick_setting=multi_tick, sensor_schema_applied=False,
                    tick_rate_hz=None, applied_schemas=list(prim.GetAppliedSchemas()))
    RtxCamera(camera_path, tick_rate=0.0, reset_xform_op_properties=False)
    schemas = list(prim.GetAppliedSchemas())
    if "OmniSensorAPI" not in schemas:
        raise RuntimeError("RtxCamera did not apply OmniSensorAPI to the capture camera")
    # Schema namespaces differ across Kit releases. Inspect the installed
    # schema rather than creating an unrecognized custom tick-rate attribute.
    rates = {attr.GetName(): float(attr.Get()) for attr in prim.GetAttributes()
             if attr.GetName().startswith("omni:sensor:")
             and attr.GetName().endswith(":tickRate") and attr.Get() is not None}
    if not rates or any(not np.isfinite(value) or value != 0.0 for value in rates.values()):
        raise RuntimeError(f"Capture camera autotrigger could not be verified: {rates}")
    return dict(path=camera_path, authoring="isaacsim.sensors.experimental.rtx.RtxCamera",
                multi_tick_setting=multi_tick, sensor_schema_applied=True,
                tick_rate_hz=0.0, tick_rate_attributes=rates, applied_schemas=schemas)


class PhysicsFrameCapture:
    def __init__(self, world, robot, camera_path, *, mode="render", resolution=(1280, 720),
                 warmup_max_passes=64, frame_max_passes=16):
        if mode not in {"render", "replicator"}:
            raise ValueError(f"Unknown capture mode: {mode!r}")
        if (len(resolution) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                                        for x in resolution)):
            raise ValueError("Capture resolution must contain two positive integers")
        for name, value, minimum in (("warmup_max_passes", warmup_max_passes, 4),
                                     ("frame_max_passes", frame_max_passes, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.world, self.robot, self.camera_path = world, robot, camera_path
        self.mode, self.resolution = mode, tuple(resolution)
        self.warmup_max_passes, self.frame_max_passes = warmup_max_passes, frame_max_passes
        self.rep = self.product = self.rgb = self.settings = None
        self.previous_block = self.previous_capture_on_play = None
        self.capture_setting_changed = False
        self.warmup = self.last_attempt = None
        self.frames = []
        self.started = self.closed = False
        self.failed = False

    def start(self):
        if self.closed or self.started:
            raise RuntimeError("Capture cannot be started twice or after close")
        self.started = True
        self.failed = True
        import carb.settings
        import omni.replicator.core as rep
        self.rep = rep
        self.settings = carb.settings.get_settings()
        old_value = self.settings.get(_CAPTURE_ON_PLAY)
        # The documented default is True when the extension has no stored value.
        self.previous_capture_on_play = True if old_value is None else bool(old_value)
        self.capture_setting_changed = True
        rep.orchestrator.set_capture_on_play(self.mode == "render")
        self.product = rep.create.render_product(self.camera_path, self.resolution)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb.attach([self.product])
        if self.mode == "render":
            self.previous_block = self.world.get_block_on_render()
            self.world.set_block_on_render(True)
        # Readiness means BOTH unchanged physics AND a valid CPU image.
        # Four is only a minimum pipeline drain, never a readiness assumption.
        try:
            _, self.warmup = self._render_ready("warmup", 4, self.warmup_max_passes)
            self.failed = False
        except Exception:
            self.warmup = self.last_attempt
            self.failed = True
            raise

    def _render_ready(self, phase, minimum_passes, maximum_passes):
        q0, v0 = (np.asarray(x).copy() for x in self.robot.state())
        t0 = float(self.world.current_time)
        attempt = dict(phase=phase, passed=False, rgb_ready=False, physics_time_s=t0 if np.isfinite(t0) else None,
                       render_passes=0, max_render_passes=maximum_passes,
                       max_abs_dq=0.0, max_abs_dv=0.0, abs_dt_s=0.0,
                       samples=[], last_rgb_shape=None, last_rgb_dtype=None)
        self.last_attempt = attempt
        if not np.isfinite(q0).all() or not np.isfinite(v0).all() or not np.isfinite(t0):
            attempt["error"] = "Nonfinite physics state before rendering"
            raise RuntimeError(attempt["error"])

        def guard():
            q1, v1 = (np.asarray(x) for x in self.robot.state())
            if q1.shape != q0.shape or v1.shape != v0.shape:
                raise RuntimeError("Physics state shape changed during rendering")
            dq = float(np.max(np.abs(q1 - q0)))
            dv = float(np.max(np.abs(v1 - v0)))
            dt = abs(float(self.world.current_time) - t0)
            if not np.isfinite([dq, dv, dt]).all() or dq > 1e-7 or dv > 1e-7 or dt > 1e-9:
                raise RuntimeError(f"Rendering advanced/changed physics: "
                                   f"max|dq|={dq:g}, max|dv|={dv:g}, |dt|={dt:g}. "
                                   "The state was NOT restored or overwritten.")
            attempt["max_abs_dq"] = max(attempt["max_abs_dq"], dq)
            attempt["max_abs_dv"] = max(attempt["max_abs_dv"], dv)
            attempt["abs_dt_s"] = max(attempt["abs_dt_s"], dt)

        try:
            for number in range(1, maximum_passes + 1):
                attempt["render_passes"] = number
                if self.mode == "render":
                    self.world.render()
                else:
                    self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0,
                                               pause_timeline=False, wait_for_render=True)
                guard()
                # Fixed-state rendering can have pipeline latency. Only read
                # after the minimum drain; retry empty buffers, not bad images.
                if number < minimum_passes:
                    continue
                raw = self.rgb.get_data()
                if isinstance(raw, dict):
                    if "data" not in raw:
                        raise RuntimeError("Invalid live RGB mapping: missing 'data'")
                    raw = raw["data"]
                frame = np.asarray(raw) if raw is not None else np.empty(0, dtype=np.uint8)
                shape, dtype = list(frame.shape), str(frame.dtype)
                attempt.update(last_rgb_shape=shape, last_rgb_dtype=dtype)
                attempt["samples"].append(dict(render_pass=number, shape=shape, dtype=dtype,
                                               empty=bool(frame.size == 0)))
                guard()  # also reject a data-read API that advances physics
                if frame.size == 0:
                    continue
                width, height = self.resolution
                if (frame.dtype != np.uint8 or frame.ndim != 3
                        or frame.shape[:2] != (height, width) or frame.shape[2] not in (3, 4)):
                    raise RuntimeError(f"Invalid live RGB data: shape={frame.shape}, dtype={frame.dtype}")
                owned = np.ascontiguousarray(frame[:, :, :3]).copy()
                attempt.update(passed=True, rgb_ready=True)
                return owned, attempt
            raise RuntimeError(f"Live RGB not ready during {phase} after {maximum_passes} render passes; "
                               f"shape={attempt['last_rgb_shape']}, dtype={attempt['last_rgb_dtype']}, "
                               f"mode={self.mode}, capture_on_play={self.mode == 'render'}. "
                               "No synthetic frame was substituted. See capture_timing.json and the native log.")
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            self.failed = True
            raise

    def frame(self, time_s):
        if self.closed or self.failed or not self.warmup or not self.warmup.get("rgb_ready"):
            raise RuntimeError("Capture is not initialized or has failed")
        if not np.isfinite(time_s) or time_s < 0 or (self.frames and time_s <= self.frames[-1]["time_s"]):
            raise ValueError("Video timestamps must be finite, nonnegative and increasing")
        try:
            image, guard = self._render_ready("frame", 2 if self.mode == "render" else 1, self.frame_max_passes)
        except Exception:
            self.failed = True
            raise
        self.frames.append(dict(time_s=float(time_s), **guard))
        return image

    def audit(self):
        ready = bool(self.warmup and self.warmup.get("rgb_ready") is True)
        return dict(schema_version=2, mode=self.mode, camera_path=self.camera_path,
                    resolution=list(self.resolution), capture_on_play=self.mode == "render",
                    annotator_device="cpu", warmup=self.warmup, camera_ready=ready,
                    frame_count=len(self.frames), frames=self.frames, last_attempt=self.last_attempt,
                    failed=self.failed,
                    physics_unchanged=(ready and not self.failed and bool(self.frames)
                                       and all(item["passed"] and item["rgb_ready"] for item in self.frames)),
                    scope="Checks CPU image readiness and physics state/time during render/data-read calls; "
                          "not independent image timestamp attestation.")

    def close(self):
        if self.closed:
            return
        self.closed = True
        actions = []
        if self.rgb is not None:
            actions.append(("rgb.detach", self.rgb.detach))
        if self.product is not None:
            actions.append(("render_product.destroy", lambda: self.product.destroy()))
        if self.previous_block is not None:
            actions.append(("restore_block_on_render", lambda: self.world.set_block_on_render(self.previous_block)))
        if self.capture_setting_changed:
            actions.append(("restore_capture_on_play", lambda: self.rep.orchestrator.set_capture_on_play(
                self.previous_capture_on_play)))
        result = cleanup_actions(actions)
        self.rgb = self.product = self.rep = self.settings = None
        if not result["passed"]:
            raise RuntimeError("; ".join(result["errors"]))
