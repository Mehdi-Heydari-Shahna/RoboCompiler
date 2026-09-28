"""Validated live RGB input, lazily encoded with FFmpeg."""
import hashlib
from pathlib import Path
import subprocess
import numpy as np


class VideoWriter:
    def __init__(self, path, ffmpeg, width=1280, height=720, fps=25):
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (width, height)):
            raise ValueError('Video width and height must be positive integers')
        if width % 2 or height % 2:
            raise ValueError('H.264 yuv420p video requires even width and height')
        if not isinstance(fps, (float, int)) or isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video fps must be positive and finite')
        self.path, self.ffmpeg = Path(path), str(ffmpeg)
        if self.path.exists():
            raise FileExistsError(f'Refusing to overwrite existing video: {self.path}')
        self.width, self.height, self.fps = width, height, fps
        self.frames, self.hashes = 0, set()
        self.proc = self.stderr = self.record = None
        self.closed = False

    def _start(self):
        # No encoder and no empty movie until the first valid live frame.
        self.stderr = self.path.with_suffix('.ffmpeg.log').open('wb')
        try:
            self.proc = subprocess.Popen([
                self.ffmpeg, '-hide_banner', '-loglevel', 'warning', '-n',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps), '-i', '-', '-an', '-c:v', 'libx264',
                '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.stderr)
        except Exception:
            self.stderr.close()
            raise

    def append(self, frame):
        if self.closed:
            raise RuntimeError('Cannot append after closing video')
        if isinstance(frame, dict):
            frame = frame['data']
        frame = np.asarray(frame)
        if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[:2] != (self.height, self.width)
                or frame.shape[2] not in (3, 4)):
            raise RuntimeError(f'Invalid Isaac RGB render product: {frame.shape}, {frame.dtype}')
        rgb = np.ascontiguousarray(frame[:, :, :3]).tobytes()
        if self.proc is None:
            self._start()
        try:
            written = self.proc.stdin.write(rgb)
            if written != len(rgb):
                raise RuntimeError(f'FFmpeg received a partial frame: {written}/{len(rgb)} bytes')
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('FFmpeg input failed; see encoder log') from exc
        self.hashes.add(hashlib.sha256(rgb).hexdigest())
        self.frames += 1

    def close(self):
        if self.record is not None:
            return self.record.copy()
        self.closed = True
        code, error = None, None
        if self.proc is not None:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    try:
                        self.proc.stdin.close()
                    except (BrokenPipeError, OSError) as exc:
                        error = f'FFmpeg input close failed: {exc}'
                try:
                    code = self.proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    code = self.proc.wait()
                    error = 'FFmpeg did not finish within 120 seconds; see encoder log'
            finally:
                if self.stderr is not None:
                    self.stderr.close()
        size = self.path.stat().st_size if self.path.is_file() else 0
        encoded = code == 0 and error is None and self.frames > 0 and size > 0
        valid = encoded and self.frames > 1 and len(self.hashes) > 1 and size > 1024
        self.record = dict(path=self.path.name, file=self.path.name, frames=self.frames,
                           unique_frames=len(self.hashes), fps=self.fps,
                           duration_s=self.frames/self.fps, valid=bool(valid), encoded=bool(encoded),
                           encoder_started=self.proc is not None, encoder_exit_code=code,
                           encoder_error=error, file_size_bytes=size,
                           source='live Isaac Sim / PhysX RGB render product',
                           sha256=hashlib.sha256(self.path.read_bytes()).hexdigest() if size else None)
        return self.record.copy()

def camera(stage):
    from pxr import Gf, UsdGeom, UsdLux
    cam = UsdGeom.Camera.Define(stage, '/World/ValidationCamera')
    # Full course visible throughout: rail crossing, crouch, turn and docking.
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(3.3, -3.8, 2.0),
                                 Gf.Vec3d(1.05, 0.05, .22), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())
    cam.CreateFocalLengthAttr(35.)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    dome = UsdLux.DomeLight.Define(stage, '/World/SkyLight')
    dome.CreateIntensityAttr(900.)
    light = UsdLux.DistantLight.Define(stage, '/World/KeyLight')
    light.CreateIntensityAttr(1800.)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(25, -30, -35))
    return str(cam.GetPath())
