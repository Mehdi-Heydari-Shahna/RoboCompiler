"""Encoder unit tests. Synthetic pixels are NOT Isaac validation evidence."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from isaac_validation.video import VideoWriter


def test_no_frames_means_no_encoder_no_movie_and_not_encoded(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Encoder must not start before receiving a valid RGB frame')
    monkeypatch.setattr('isaac_validation.video.subprocess.Popen', forbidden)
    writer = VideoWriter(tmp_path/'empty.mp4', 'unused-ffmpeg')
    result = writer.close()
    assert result['encoded'] is result['valid'] is False
    assert result['encoder_started'] is False
    assert result['encoder_exit_code'] is None
    assert result['frames'] == result['file_size_bytes'] == 0
    assert not (tmp_path/'empty.mp4').exists()
    assert not (tmp_path/'empty.ffmpeg.log').exists()
    assert writer.close() == result


@pytest.mark.parametrize('shape,dtype', [((0,),np.uint8), ((2,4),np.uint8), ((2,4,2),np.uint8),
                                       ((2,4,5),np.uint8), ((2,4,3),float), ((4,2,3),np.uint8)])
def test_invalid_frame_never_starts_encoder(tmp_path, monkeypatch, shape, dtype):
    monkeypatch.setattr('isaac_validation.video.subprocess.Popen', lambda *a, **kw: pytest.fail('No encoder expected'))
    writer = VideoWriter(tmp_path/'invalid.mp4', 'ffmpeg', width=4, height=2)
    with pytest.raises(RuntimeError, match='Invalid Isaac RGB'):
        writer.append(np.zeros(shape, dtype=dtype))
    assert writer.frames == 0
    assert not writer.close()['encoded']


@pytest.mark.parametrize('kwargs', [{'width': 0}, {'width': 3}, {'height': -2}, {'height': 3},
                                   {'fps': 0}, {'fps': float('nan')}, {'fps': True}])
def test_invalid_encoder_parameters_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        VideoWriter(tmp_path/'invalid.mp4', 'ffmpeg', **kwargs)


def test_existing_video_cannot_be_reused_or_overwritten(tmp_path):
    movie = tmp_path/'old.mp4'
    movie.write_bytes(b'old run evidence')
    with pytest.raises(FileExistsError):
        VideoWriter(movie, 'ffmpeg')
    assert movie.read_bytes() == b'old run evidence'


def test_closed_writer_rejects_append_without_starting_encoder(tmp_path):
    writer = VideoWriter(tmp_path/'closed.mp4', 'ffmpeg', width=4, height=2)
    writer.close()
    with pytest.raises(RuntimeError, match='after closing'):
        writer.append(np.zeros((2,4,3), dtype=np.uint8))


def test_encoder_start_failure_closes_log_and_preserves_failure(tmp_path):
    writer = VideoWriter(tmp_path/'failed.mp4', tmp_path/'missing-encoder', width=4, height=2)
    with pytest.raises(OSError):
        writer.append(np.zeros((2,4,3), dtype=np.uint8))
    assert writer.stderr.closed
    assert writer.frames == 0 and not writer.close()['encoded']


def test_native_encoder_failure_is_recorded(tmp_path, monkeypatch):
    import io
    proc = SimpleNamespace(stdin=io.BytesIO(), wait=lambda **kw: 7)
    monkeypatch.setattr('isaac_validation.video.subprocess.Popen', lambda *a, **kw: proc)
    writer = VideoWriter(tmp_path/'failed.mp4', 'fake', width=4, height=2)
    writer.append(np.zeros((2,4,3), dtype=np.uint8))
    result = writer.close()
    assert result['encoder_exit_code'] == 7 and not result['encoded']
    assert result['encoder_started'] and writer.stderr.closed


def test_real_ffmpeg_roundtrip_of_synthetic_frames(tmp_path):
    # This exercises the local encoder/decoder only, not any NVIDIA runtime.
    ffmpeg = pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    cv2 = pytest.importorskip('cv2')
    writer = VideoWriter(tmp_path/'synthetic-not-isaac.mp4', ffmpeg, width=64, height=48, fps=25)
    rng = np.random.default_rng(20260924)
    for _ in range(6):
        writer.append(rng.integers(0,256,size=(48,64,4),dtype=np.uint8))
    record = writer.close()
    assert record['encoded'] and record['valid'] and record['encoder_exit_code'] == 0
    cap = cv2.VideoCapture(str(tmp_path/'synthetic-not-isaac.mp4'))
    decoded = 0
    try:
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FPS) == 25
        while True:
            ok, frame = cap.read()
            if not ok: break
            assert frame.shape == (48,64,3)
            decoded += 1
    finally:
        cap.release()
    assert decoded == record['frames'] == record['unique_frames'] == 6
    assert record['duration_s'] == 0.24
    assert writer.close() == record



def test_partial_input_write_does_not_count_as_a_frame(tmp_path, monkeypatch):
    proc = SimpleNamespace(stdin=SimpleNamespace(write=lambda data: len(data)-1, closed=True), wait=lambda **kw: 0)
    monkeypatch.setattr('isaac_validation.video.subprocess.Popen', lambda *a, **kw: proc)
    writer = VideoWriter(tmp_path/'partial.mp4', 'fake', width=4, height=2)
    with pytest.raises(RuntimeError, match='partial frame'):
        writer.append(np.zeros((2,4,3), dtype=np.uint8))
    assert writer.frames == 0 and not writer.close()['encoded']


def test_encoder_timeout_is_killed_and_never_marked_encoded(tmp_path, monkeypatch):
    import io
    import subprocess
    calls = []
    def wait(**kwargs):
        if kwargs:
            raise subprocess.TimeoutExpired('synthetic-ffmpeg', 120)
        return -9
    proc = SimpleNamespace(stdin=io.BytesIO(), wait=wait, kill=lambda: calls.append('kill'))
    monkeypatch.setattr('isaac_validation.video.subprocess.Popen', lambda *a, **kw: proc)
    writer = VideoWriter(tmp_path/'timeout.mp4', 'fake', width=4, height=2)
    writer.append(np.zeros((2,4,3), dtype=np.uint8))
    result = writer.close()
    assert calls == ['kill'] and not result['encoded']
    assert '120 seconds' in result['encoder_error']
    assert writer.stderr.closed
