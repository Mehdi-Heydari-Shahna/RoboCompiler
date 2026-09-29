#!/usr/bin/env python3
"""Optional zero-physics-step camera capture for an Isaac Sim 6.1 native run.

Import after SimulationApp construction and stage opening. The runner calls
`capture()` only after the matching `simulate()` / `fetch_results()` pair.
Replicator's `delta_time=0.0` renders the fetched state without advancing its
timeline; the native physics runner alone owns its integration clock.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import shutil


class VideoCapture:
    def __init__(self, output_dir, fps=20, resolution=(960, 540),
                 position=(7.0, -20.0, 12.0), look_at=(4.0, 1.0, 0.8)):
        self.output_dir = Path(output_dir).resolve()
        self.raw_dir = self.output_dir / "replicator_rgb"
        self.frames_dir = self.output_dir / "frames"
        self.video_path = self.output_dir / "isaac_native.mp4"
        if not isinstance(fps, int) or fps < 1 or fps > 120:
            raise ValueError("fps must be an integer in 1..120")
        if (len(resolution) != 2 or any(not isinstance(dim, int) or dim < 2 or dim % 2
                                         for dim in resolution)):
            raise ValueError("Video width and height must be positive even integers")
        self.fps = fps
        self.resolution = tuple(resolution)
        self.position = tuple(float(x) for x in position)
        self.look_at = tuple(float(x) for x in look_at)
        self._rep = None
        self._writer = None
        self._rp = None
        self._timeline = None
        self._captures = []
        self._finished = False

    def start(self):
        """Install a fixed camera, one light and a writer without simulation."""
        if self._rep is not None or self._finished:
            raise RuntimeError("Video capture has already been initialized")
        if self.raw_dir.exists() or self.frames_dir.exists() or self.video_path.exists():
            raise FileExistsError("Video output already exists; use a fresh result directory")
        import omni.replicator.core as rep
        import omni.timeline
        self._timeline = omni.timeline.get_timeline_interface()
        if self._timeline.is_playing():
            raise RuntimeError("Stop the timeline before setting up native video capture")
        self.raw_dir.mkdir(parents=True, exist_ok=False)
        self._rep = rep
        rep.orchestrator.set_capture_on_play(False)
        # These are presentation-only prims. No robot or physics prim is moved.
        cam = rep.functional.create.camera(position=self.position, look_at=self.look_at,
                                           parent="/World", name="NativeRunCamera")
        rep.functional.create.dome_light(intensity=500, parent="/World", name="NativeRunDomeLight")
        self._rp = rep.create.render_product(cam, self.resolution, name="NativeRunRGB")
        self._rp.hydra_texture.set_updates_enabled(False)
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(self.raw_dir))
        self._writer = rep.writers.get("BasicWriter")
        self._writer.initialize(backend=backend, rgb=True)
        self._writer.attach(self._rp)
        return {"video_capture_started": True, "fps": self.fps,
                "resolution": list(self.resolution), "camera_position_m": list(self.position),
                "camera_look_at_m": list(self.look_at), "raw_png_dir": str(self.raw_dir)}

    def capture(self, native_step, sim_time_s):
        """Render one previously fetched native state; no simulate() is called."""
        if self._rep is None or self._finished:
            raise RuntimeError("Call start() before capture() and not after finish()")
        if not isinstance(native_step, int) or native_step < 1:
            raise ValueError("Captured state must follow a completed native step")
        if not math.isfinite(sim_time_s) or sim_time_s <= 0:
            raise ValueError("Simulation timestamp must be positive and finite")
        if self._captures and (native_step <= self._captures[-1]["native_step"] or
                               sim_time_s <= self._captures[-1]["sim_time_s"]):
            raise ValueError("Video captures must have strictly increasing native steps and times")
        if self._timeline.is_playing():
            raise RuntimeError("Timeline played during native simulation capture")
        # Rendering is sparse while physics advances at 1 kHz. Clear temporal
        # reconstruction history so a previous frame does not ghost the new pose.
        import omni.usd
        omni.usd.get_context().reset_renderer_accumulation()
        # fetch_results() publishes the completed native step's transforms to
        # USD. Do not write poses/velocities explicitly during video capture:
        # USD change tracking could otherwise feed back to the live scene.
        before_time = float(self._timeline.get_current_time())
        self._rp.hydra_texture.set_updates_enabled(True)
        try:
            self._rep.orchestrator.step(delta_time=0.0, wait_for_render=True,
                                        pause_timeline=True)
        finally:
            self._rp.hydra_texture.set_updates_enabled(False)
        if self._timeline.is_playing() or not math.isclose(float(self._timeline.get_current_time()),
                                                          before_time, abs_tol=1e-9):
            raise RuntimeError("Replicator capture unexpectedly advanced the timeline")
        self._captures.append({"frame_index": len(self._captures), "native_step": native_step,
                               "sim_time_s": float(sim_time_s)})

    def finish(self):
        """Flush writer and verify PNGs before controller-side MP4 encoding.

        Returns explicit PNG evidence status. A capture failure raises and
        must be reflected in the parent runner's overall result.
        """
        if self._rep is None or self._finished:
            raise RuntimeError("Video capture is not active")
        self._finished = True
        try:
            self._rep.orchestrator.wait_until_complete()
        finally:
            self._writer.detach()
            self._rp.destroy()
        pngs = sorted((path for path in self.raw_dir.rglob("*.png")
                       if path.name.startswith("rgb")),
                      key=lambda path: _png_sort_key(path))
        if not self._captures or len(pngs) != len(self._captures):
            raise RuntimeError("BasicWriter RGB PNG count differs from requested captures: %d versus %d"
                               % (len(pngs), len(self._captures)))
        numeric_indexes = [_png_sort_key(png)[0] for png in pngs]
        if numeric_indexes != list(range(numeric_indexes[0], numeric_indexes[0] + len(pngs))):
            raise RuntimeError("BasicWriter RGB PNG sequence is not contiguous")
        for png in pngs:
            _validate_png(png, self.resolution)
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        for index, png in enumerate(pngs):
            shutil.move(str(png), str(self.frames_dir / ("frame_%06d.png" % index)))
        # Keep one authoritative image sequence in the results archive.
        shutil.rmtree(self.raw_dir)
        metadata_path = self.output_dir / "video_frames.json"
        metadata_path.write_text(json.dumps({"scope": "frames of native fetched PhysX states",
                                             "capture_timeline_delta_s": 0.0,
                                             "fps": self.fps, "resolution": list(self.resolution),
                                             "frame_count": len(self._captures),
                                             "frames": self._captures}, indent=2) + "\n",
                                 encoding="utf-8")
        return {"png_complete": True, "captured_frames": len(self._captures),
                "frame_manifest": str(metadata_path),
                "frame_directory": str(self.frames_dir), "fps": self.fps,
                "resolution": list(self.resolution),
                "mp4_created": False, "mp4_status": "DEFERRED_TO_CONTROLLER_LAUNCHER",
                "capture_timeline_delta_s": 0.0}


def _png_sort_key(path):
    import re
    matched = re.search(r"(\d+)$", path.stem)
    if matched is None:
        raise RuntimeError("Writer produced an RGB PNG without a numeric frame index: " + str(path))
    return int(matched.group(1)), str(path)


def _validate_png(path, resolution):
    with path.open("rb") as frame:
        hdr = frame.read(24)
    if len(hdr) != 24 or hdr[:8] != b"\x89PNG\r\n\x1a\n" or hdr[12:16] != b"IHDR":
        raise RuntimeError("Writer frame is not a valid PNG: " + str(path))
    size = struct.unpack(">II", hdr[16:24])
    if size != resolution:
        raise RuntimeError("Writer PNG dimensions differ from camera resolution: " + str(path))
