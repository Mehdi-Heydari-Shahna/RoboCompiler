"""Live RTX capture of the current PhysX scene; no replay or synthetic frames.

Instantiate only after SimulationApp and SimulationContext.initialize_physics().
Camera.get_rgba reads its attached RGB annotator directly (Isaac Sim 5.0 source),
whereas get_current_frame can lag because of its frequency-limited callback:
https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/source/extensions/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py
Camera axes/API: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.sensors.camera/docs/index.html
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


class VideoCaptureError(RuntimeError):
    """The video cannot be used as evidence for this simulation run."""


def _look_at_quaternion(eye, target):
    """Return wxyz USD camera orientation: local -Z forward, local +Y up."""
    back = np.asarray(eye, dtype=float) - np.asarray(target, dtype=float)
    back /= np.linalg.norm(back)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    rotation = np.column_stack((right, up, back))
    # Eigenvector form avoids a numerically delicate trace/180-degree branch.
    r = rotation
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3.0
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def _load_camera():
    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        try:
            from omni.isaac.sensor import Camera
        except ImportError as exc:
            raise VideoCaptureError(
                "Isaac camera extension is unavailable. Start this script through "
                "Isaac Sim's Python launcher and enable isaacsim.sensors.camera."
            ) from exc
    return Camera


def _load_encoding():
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg
        from PIL import Image, ImageDraw, ImageFont
        # Resolves a packaged or already installed executable; never downloads it.
        imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise VideoCaptureError(
            "Video requires Pillow, imageio and imageio-ffmpeg in the Isaac Python "
            "environment, including a local FFmpeg executable. Run the setup script."
        ) from exc
    return imageio, imageio_ffmpeg, Image, ImageDraw, ImageFont


class LiveVideo:
    """Capture a fixed camera after each caller-selected simulation sample.

    The caller advances PhysX and schedules captures at approximately ``fps``.
    This class never steps physics, assigns body poses, or invents extra frames.
    A full validation video must cover the 22-second experiment. ``passed`` in
    finish() refers only to video integrity and coverage, not controller quality.
    """

    def __init__(self, stage, simulation_context, output_path, fps=30,
                 resolution=(1280, 720), run_id=""):
        self.path = Path(output_path)
        self.fps = float(fps)
        self.resolution = tuple(map(int, resolution))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Video fps must be finite and positive.")
        if len(self.resolution) != 2 or any(n < 2 or n % 2 for n in self.resolution):
            raise ValueError("Video resolution must contain two positive even dimensions.")
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("LiveVideo output must have the .mp4 extension.")
        self.stage = stage
        self.sim = simulation_context
        self.run_id = str(run_id)
        self.frames = 0
        self.times = []
        self._writer = None
        self._closed = False
        self._failure = None
        self._metadata = None
        self._manifest = None
        self._imageio, self._ffmpeg, self._Image, self._Draw, self._Font = _load_encoding()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise VideoCaptureError(f"Refusing to overwrite an existing video: {self.path}")
        Camera = _load_camera()
        self.camera = Camera(
            prim_path="/World/ValidationCamera", name="validation_camera",
            frequency=None, resolution=self.resolution,
        )
        eye = np.array([1.6, -1.8, 1.25])
        self.camera.set_world_pose(
            position=eye,
            orientation=_look_at_quaternion(eye, [0.0, 0.0, 0.38]),
            camera_axes="usd",
        )
        # Isaac 5.0 scales both setter values by 10 into the USD attributes;
        # specifying both lengths explicitly makes FOV independent of defaults.
        # At 1280x720 this is 75.75 deg horizontal / 47.26 deg vertical, with
        # enough room for the complete 1.3-m base and probe plus the overlays.
        self.camera.set_focal_length(1.8)
        self.camera.set_horizontal_aperture(2.8)
        self.camera.set_vertical_aperture(2.8*self.resolution[1]/self.resolution[0])
        self.camera.set_clipping_range(near_distance=0.02, far_distance=100.0)
        self.camera.initialize()
        focal = float(self.camera.get_focal_length())
        horizontal = float(self.camera.get_horizontal_aperture())
        vertical = float(self.camera.get_vertical_aperture())
        if not np.allclose([horizontal/focal, vertical/focal],
                           [2.8/1.8, (2.8/1.8)*self.resolution[1]/self.resolution[0]],
                           rtol=1e-5, atol=1e-7):
            raise VideoCaptureError("Camera lens readback differs from the requested field of view.")
        distance = float(np.linalg.norm(eye - np.array([0.0, 0.0, 0.38])))
        self._camera_metadata = {
            "eye_m": eye.tolist(), "target_m": [0.0, 0.0, 0.38], "axes": "usd",
            "horizontal_fov_deg": math.degrees(2*math.atan(horizontal/(2*focal))),
            "vertical_fov_deg": math.degrees(2*math.atan(vertical/(2*focal))),
            "target_distance_m": distance,
            "view_width_at_target_m": distance*horizontal/focal,
            "view_height_at_target_m": distance*vertical/focal,
        }
        self._font = self._font_at_size(max(14, round(self.resolution[1] / 40)))
        self._title_font = self._font_at_size(max(19, round(self.resolution[1] / 25)))
        # RTX/annotator initialization may take multiple render frames. The writer
        # is deliberately unopened until a real, correctly sized RGBA frame exists.
        self._read_live_frame(warmup=True)

    def _font_at_size(self, size):
        for name in ("DejaVuSans.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                return self._Font.truetype(name, size=size)
            except OSError:
                continue
        try:
            return self._Font.load_default(size=size)
        except TypeError:
            return self._Font.load_default()

    def _render_without_physics(self):
        before = (float(self.sim.current_time), int(self.sim.current_time_step_index))
        self.sim.render()
        after = (float(self.sim.current_time), int(self.sim.current_time_step_index))
        if after[1] != before[1] or abs(after[0] - before[0]) > 1e-12:
            raise VideoCaptureError(
                "Rendering advanced physics/time. The capture would not match its "
                "logged simulation sample; stop and inspect the installed Isaac configuration."
            )

    def _read_live_frame(self, warmup=False):
        width, height = self.resolution
        attempts = 30 if warmup else 8
        for index in range(attempts):
            self._render_without_physics()
            # Three synchronous renders flush the RTX render pipeline at the same
            # physics state. Direct get_rgba bypasses Camera's throttled frame dict.
            if index < 2:
                continue
            rgba = self.camera.get_rgba()
            if rgba is None:
                continue
            rgba = np.asarray(rgba)
            if rgba.size == 0:
                continue
            if rgba.shape != (height, width, 4):
                raise VideoCaptureError(
                    f"Camera returned RGBA shape {rgba.shape}; expected {(height, width, 4)}."
                )
            if rgba.dtype != np.uint8:
                raise VideoCaptureError(f"Camera RGBA must be uint8, received {rgba.dtype}.")
            if not np.any(rgba[:, :, 3]) or not np.any(rgba[:, :, :3]):
                continue
            return np.ascontiguousarray(rgba[:, :, :3]).copy()
        raise VideoCaptureError(
            "RTX camera produced no nonempty, nonblack RGBA frame after render-only "
            "warmup. Check the NVIDIA GPU/driver, camera extension and scene lighting. "
            "No replacement video frames were generated."
        )

    def _overlay(self, rgb, t, position_error_m, orientation_error_deg, closure_error_m, phase):
        frame = self._Image.fromarray(rgb, "RGB")
        drawing = self._Draw.Draw(frame)
        width, height = self.resolution
        scale = height / 720.0
        padding = max(8, round(16 * scale))
        header_h = max(37, round(59 * scale))
        footer_h = max(84, round(90 * scale))
        drawing.rectangle((0, 0, width, header_h), fill=(13, 24, 39))
        drawing.text((padding, padding), "Stewart CMG / PACDM - Isaac Sim / PhysX",
                     font=self._title_font, fill=(245, 248, 252))
        drawing.rectangle((0, height-footer_h, width, height), fill=(13, 24, 39))
        lines = [
            f"t = {t:6.3f} s   |   {phase}   |   live PhysX state",
            f"Position: {1000*position_error_m:.3f} mm   Orientation: {orientation_error_deg:.3f} deg   Closure: {1000*closure_error_m:.3f} mm",
            f"Run: {self.run_id}   |   Validation decision: see validation.json",
        ]
        for index, line in enumerate(lines):
            drawing.text((padding, height-footer_h + padding + index * max(23, round(23*scale))),
                         line, font=self._font, fill=(225, 234, 244))
        return np.asarray(frame)

    def capture(self, t, position_error_m, orientation_error_deg, closure_error_m, phase):
        if self._closed:
            raise VideoCaptureError("Cannot capture after video finalization.")
        if self._failure:
            raise VideoCaptureError(f"Video capture already failed: {self._failure}")
        values = [float(v) for v in (t, position_error_m, orientation_error_deg, closure_error_m)]
        if not all(math.isfinite(v) for v in values) or any(v < 0 for v in values):
            raise ValueError("Video time and error magnitudes must be finite and nonnegative.")
        t, position_error_m, orientation_error_deg, closure_error_m = values
        if self.times and t <= self.times[-1]:
            raise ValueError("Each captured simulation time must increase strictly.")
        raw = None
        try:
            raw = self._read_live_frame()
            annotated = self._overlay(raw, *values, str(phase))
            if self._writer is None:
                self._writer = self._imageio.get_writer(
                    str(self.path), format="FFMPEG", mode="I", fps=self.fps,
                    codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                    ffmpeg_log_level="error", output_params=["-movflags", "+faststart"],
                )
                self._manifest = self.path.with_suffix(".frames.jsonl").open("x", encoding="utf-8")
            self._writer.append_data(annotated)
            record = {
                "frame": self.frames, "t_s": t, "run_id": self.run_id,
                "position_error_m": position_error_m,
                "orientation_error_deg": orientation_error_deg,
                "closure_error_m": closure_error_m, "phase": str(phase),
                "raw_rgb_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            }
            self._manifest.write(json.dumps(record, allow_nan=False) + "\n")
            self._manifest.flush()
            self.times.append(t)
            self.frames += 1
        except Exception as exc:
            self._failure = str(exc)
            if raw is not None:
                # Preserve the actual unannotated capture when encoding fails.
                self._Image.fromarray(raw, "RGB").save(self.path.with_suffix(".failed-frame.png"))
            raise VideoCaptureError(f"Live video capture failed: {exc}") from exc

    def finish(self):
        """Close, decode-check and report video coverage; safe to call twice."""
        if self._metadata is not None:
            return dict(self._metadata)
        self._closed = True
        try:
            if self._writer is not None:
                self._writer.close()
        except Exception as exc:
            self._failure = f"FFmpeg close failed: {exc}"
        finally:
            if self._manifest is not None:
                self._manifest.close()
        encoded_frames = 0
        encoded_duration = 0.0
        valid_file = False
        if self.path.is_file() and self.path.stat().st_size > 0:
            try:
                encoded_frames, encoded_duration = self._ffmpeg.count_frames_and_secs(str(self.path))
                reader = self._imageio.get_reader(str(self.path), format="FFMPEG")
                try:
                    first = reader.get_data(0)
                    last = reader.get_data(encoded_frames - 1)
                    expected_shape = (self.resolution[1], self.resolution[0], 3)
                    valid_file = first.shape == expected_shape and last.shape == expected_shape
                finally:
                    reader.close()
            except Exception as exc:
                self._failure = f"Encoded video integrity check failed: {exc}"
        start = self.times[0] if self.times else None
        end = self.times[-1] if self.times else None
        # Capture at the nearest 2-ms physics sample introduces <=2-ms jitter.
        sample_period = 1.0 / self.fps
        coverage = bool(self.times and start <= sample_period + 0.002
                        and end >= 22.0 - sample_period - 0.002
                        and end <= 22.0 + sample_period + 0.002
                        and self.frames >= math.floor(22.0 * self.fps)
                        and self.frames <= math.ceil(22.0 * self.fps) + 1)
        cadence = bool(len(self.times) > 1 and np.all(
            np.abs(np.diff(self.times) - sample_period) <= 0.004 + 1e-9))
        passed = bool(not self._failure and valid_file and coverage and cadence
                      and encoded_frames == self.frames)
        self._metadata = {
            "passed": passed, "path": self.path.name, "frames": self.frames,
            "fps": self.fps, "sim_start_s": start, "sim_end_s": end,
            "run_id": self.run_id, "source": "live PhysX physics",
            "encoded_frames": int(encoded_frames), "encoded_duration_s": float(encoded_duration),
            "valid_file": bool(valid_file), "full_22_second_coverage": coverage,
            "capture_cadence_ok": cadence,
            "frame_manifest": self.path.with_suffix(".frames.jsonl").name,
            "camera": dict(self._camera_metadata),
            "failure": self._failure,
        }
        return dict(self._metadata)
