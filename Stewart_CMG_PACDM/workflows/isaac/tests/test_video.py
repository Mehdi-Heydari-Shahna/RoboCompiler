"""Host-side capture contract tests with mock Isaac/encoder dependencies.

These do not claim that Isaac or an RTX renderer ran on the test machine.
The installed Isaac run remains the required end-to-end video verification.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from isaac_validation import video


@pytest.fixture
def capture_env(monkeypatch, tmp_path):
    sim = SimpleNamespace(current_time=0.0, current_time_step_index=0, renders=0,
                          advance=False, empty=False, pixel_value=80)

    def render():
        sim.renders += 1
        if sim.advance:
            sim.current_time += 0.002
            sim.current_time_step_index += 1

    sim.render = render
    enc = SimpleNamespace(frames=[], created=0, fail=False, path=None)

    class Camera:
        def __init__(self, **kwargs):
            sim.camera_settings = kwargs
            self.resolution = kwargs["resolution"]
        def set_world_pose(self, **kwargs):
            sim.camera_pose = kwargs
        def set_focal_length(self, value):
            self.focal = value
        def set_horizontal_aperture(self, value):
            self.horizontal = value
        def set_vertical_aperture(self, value):
            self.vertical = value
        def get_focal_length(self):
            return self.focal
        def get_horizontal_aperture(self):
            return self.horizontal
        def get_vertical_aperture(self):
            return self.vertical
        def set_clipping_range(self, **kwargs):
            pass
        def initialize(self):
            sim.initialized = True
        def get_rgba(self):
            if sim.empty:
                return np.array([])
            w, h = self.resolution
            result = np.full((h, w, 4), sim.pixel_value, dtype=np.uint8)
            result[:, :, 3] = 255
            return result

    class Writer:
        def append_data(self, frame):
            if enc.fail:
                raise RuntimeError("deliberate encoder failure")
            enc.frames.append(frame.copy())
        def close(self):
            # Test-only marker; this fixture is not a real video encoder.
            enc.path.write_bytes(b"mock encoder output")

    class Reader:
        def get_data(self, index):
            return enc.frames[index]
        def close(self):
            pass

    def get_writer(path, **kwargs):
        enc.path = Path(path)
        enc.created += 1
        return Writer()

    io = SimpleNamespace(get_writer=get_writer, get_reader=lambda *a, **k: Reader())
    ffmpeg = SimpleNamespace(count_frames_and_secs=lambda path: (len(enc.frames), len(enc.frames) / 30))
    monkeypatch.setattr(video, "_load_camera", lambda: Camera)
    monkeypatch.setattr(video, "_load_encoding", lambda: (io, ffmpeg, Image, ImageDraw, ImageFont))
    return sim, enc, tmp_path / "validation.mp4"


def make_video(env):
    sim, enc, path = env
    return video.LiveVideo(None, sim, path, resolution=(320, 240), run_id="test-run")


def test_usd_camera_looks_at_target():
    eye = np.array([1.6, -1.8, 1.25])
    target = np.array([0.0, 0.0, 0.38])
    w, x, y, z = video._look_at_quaternion(eye, target)
    rotation = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])
    direction = (target-eye) / np.linalg.norm(target-eye)
    np.testing.assert_allclose(rotation @ [0, 0, -1], direction, atol=1e-14)
    assert (rotation @ [0, 1, 0])[2] > 0
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)


def test_camera_fov_contains_complete_robot_clear_of_overlays():
    # Conservative envelope encloses the 1.28-m base at z=-0.11 and the probe
    # above the platform. This checks geometry, not an unavailable RTX rendering.
    eye = np.array([1.6, -1.8, 1.25])
    target = np.array([0.0, 0.0, 0.38])
    back = (eye-target)/np.linalg.norm(eye-target)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    rotation = np.column_stack((right, up, back))
    angles = np.linspace(0, 2*np.pi, 360)
    points = np.array([[0.65*np.cos(a), 0.65*np.sin(a), z]
                       for a in angles for z in [-0.11, 1.05]])
    camera_points = (points-eye) @ rotation
    focal_px = 1280*1.8/2.8
    pixels = np.column_stack((640 + focal_px*camera_points[:, 0]/-camera_points[:, 2],
                              360 - focal_px*camera_points[:, 1]/-camera_points[:, 2]))
    assert np.all(camera_points[:, 2] < 0)
    assert pixels[:, 0].min() > 0 and pixels[:, 0].max() < 1280
    assert pixels[:, 1].min() > 59 and pixels[:, 1].max() < 720-90
    assert pixels[:, 0].max()-pixels[:, 0].min() > 450


def test_empty_camera_does_not_open_writer(capture_env):
    sim, enc, _ = capture_env
    sim.empty = True
    with pytest.raises(video.VideoCaptureError, match="no nonempty"):
        make_video(capture_env)
    assert enc.created == 0
    assert sim.current_time == 0
    assert sim.current_time_step_index == 0


def test_render_cannot_advance_physics(capture_env):
    sim, enc, _ = capture_env
    sim.advance = True
    with pytest.raises(video.VideoCaptureError, match="advanced physics/time"):
        make_video(capture_env)
    assert enc.created == 0


def test_current_frame_capture_and_short_run_fails_coverage(capture_env):
    sim, enc, path = capture_env
    live = make_video(capture_env)
    assert enc.created == 0
    assert sim.camera_pose["camera_axes"] == "usd"
    for index, t in enumerate([0.0, 0.034, 0.066]):
        sim.current_time = t
        sim.current_time_step_index = round(t/0.002)
        sim.pixel_value = 60 + index * 20
        live.capture(t, 0.001, 0.2, 0.0001, "tracking")
        # Unannotated centre stays equal to this simulation sample, not warmup.
        np.testing.assert_array_equal(enc.frames[-1][100, 160], [sim.pixel_value] * 3)
        assert sim.current_time == t
    result = live.finish()
    assert result["frames"] == 3
    assert result["source"] == "live PhysX physics"
    assert result["camera"]["horizontal_fov_deg"] == pytest.approx(75.7499673)
    assert result["valid_file"]
    assert not result["passed"]
    assert not result["full_22_second_coverage"]
    assert live.finish() == result
    records = [json.loads(line) for line in path.with_suffix(".frames.jsonl").read_text().splitlines()]
    assert [r["t_s"] for r in records] == [0, 0.034, 0.066]
    assert len({r["raw_rgb_sha256"] for r in records}) == 3


def test_full_22_second_sequence_and_decodable_frame_count(capture_env, monkeypatch):
    sim, enc, _ = capture_env
    live = make_video(capture_env)
    monkeypatch.setattr(live, "_overlay", lambda rgb, *args: rgb)
    for index in range(661):
        t = round(index / 30 / 0.002) * 0.002
        sim.current_time = t
        sim.current_time_step_index = round(t/0.002)
        live.capture(t, 0.001, 0.2, 0.0001, "tracking")
    result = live.finish()
    assert result["passed"]
    assert result["encoded_frames"] == 661
    assert result["sim_start_s"] == 0
    assert result["sim_end_s"] == 22


def test_encoder_failure_preserves_actual_raw_frame(capture_env):
    sim, enc, path = capture_env
    live = make_video(capture_env)
    enc.fail = True
    with pytest.raises(video.VideoCaptureError, match="deliberate encoder failure"):
        live.capture(0, 0, 0, 0, "hold")
    raw = np.asarray(Image.open(path.with_suffix(".failed-frame.png")))
    np.testing.assert_array_equal(raw[0, 0], [sim.pixel_value] * 3)
    assert not live.finish()["passed"]


def test_nonincreasing_or_nonfinite_samples_rejected(capture_env):
    live = make_video(capture_env)
    live.capture(0.034, 0, 0, 0, "hold")
    with pytest.raises(ValueError, match="increase strictly"):
        live.capture(0.034, 0, 0, 0, "hold")
    with pytest.raises(ValueError, match="finite"):
        live.capture(0.066, float("nan"), 0, 0, "hold")
    live.finish()
