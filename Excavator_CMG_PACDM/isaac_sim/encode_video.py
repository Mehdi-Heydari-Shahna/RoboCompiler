#!/usr/bin/env python3
"""Encode an ordered sequence of measured Isaac RGB PNGs, if FFmpeg is present.

An MP4 is optional evidence; failed encoding leaves all source PNGs untouched and
is reported as such. This module does not load Isaac Sim or step physics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def _find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return str(executable)
    try:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if executable and Path(executable).is_file():
            return str(executable)
    except (ImportError, OSError, RuntimeError):
        pass
    return None


def encode_png_sequence(frames_dir, output_mp4, fps=20, expected_count=None):
    """Encode frame_000000.png .. frame_N.png, preserving every source PNG.

    Returns a JSON-serializable result. Callers must check `mp4_created` before
    presenting the movie as evidence. Incomplete or non-contiguous PNGs fail.
    """
    frames_dir = Path(frames_dir).resolve()
    output_mp4 = Path(output_mp4).resolve()
    result = {"mp4_created": False, "mp4_path": None, "fps": int(fps),
              "png_directory": str(frames_dir), "png_count": 0}
    started_encoding = False
    try:
        if not isinstance(fps, int) or not 1 <= fps <= 120:
            raise ValueError("FPS must be an integer in 1..120")
        frames = sorted(frames_dir.glob("frame_*.png"))
        result["png_count"] = len(frames)
        if not frames:
            raise ValueError("No captured PNGs")
        if expected_count is not None and len(frames) != expected_count:
            raise ValueError("PNG count differs from the requested captures")
        expected_names = ["frame_%06d.png" % index for index in range(len(frames))]
        if [p.name for p in frames] != expected_names:
            raise ValueError("Frame PNG sequence has a gap or an unexpected name")
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("FFmpeg executable unavailable; PNG sequence preserved")
        if output_mp4.exists():
            raise FileExistsError("Refusing to overwrite existing MP4: " + str(output_mp4))
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-framerate", str(fps), "-start_number", "0", "-i",
               str(frames_dir / "frame_%06d.png"), "-frames:v", str(len(frames)),
               "-an", "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output_mp4)]
        started_encoding = True
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=max(120, len(frames) * 4))
        if completed.returncode != 0:
            raise RuntimeError("FFmpeg failed: " + completed.stderr[-1600:])
        if not output_mp4.is_file() or output_mp4.stat().st_size < 1024:
            raise RuntimeError("FFmpeg did not produce a nonempty MP4")
        with output_mp4.open("rb") as movie:
            header = movie.read(32)
        if b"ftyp" not in header:
            raise RuntimeError("Produced file has no MP4 ftyp header")
        result.update({"mp4_created": True, "mp4_path": str(output_mp4),
                       "mp4_bytes": output_mp4.stat().st_size,
                       "ffmpeg_executable": ffmpeg})
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result["mp4_error"] = "%s: %s" % (type(exc).__name__, exc)
        # A partial movie must never be mistaken for complete video evidence.
        if started_encoding and output_mp4.exists() and not result["mp4_created"]:
            output_mp4.unlink()
    return result


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir")
    parser.add_argument("output_mp4")
    parser.add_argument("--fps", type=int, default=20)
    arguments = parser.parse_args()
    outcome = encode_png_sequence(arguments.frames_dir, arguments.output_mp4, fps=arguments.fps)
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["mp4_created"] else 1)
