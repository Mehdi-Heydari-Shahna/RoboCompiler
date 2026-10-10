"""Cross-process and original-controller equivalence tests (no Isaac/MuJoCo)."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation

from isaac_validation.client import WorkerClient, WorkerError

ROOT = Path(__file__).resolve().parents[1]


def original_controller_source():
    text = (ROOT / "go2/simulation.py").read_text(encoding="utf-8")
    return text[text.index("class WholeBodyController:"):text.index("\ndef run_case(")]


def test_controller_class_is_source_identical():
    text = (ROOT / "isaac_validation/controller.py").read_text(encoding="utf-8")
    assert text[text.index("class WholeBodyController:"):] == original_controller_source()


@pytest.fixture
def worker(tmp_path):
    if importlib.util.find_spec("pinocchio") is None or importlib.util.find_spec("osqp") is None:
        pytest.skip("Use the separate controller environment with Pinocchio and OSQP installed")
    with WorkerClient(sys.executable, ROOT, tmp_path / "controller.log") as result:
        yield result


def test_subprocess_matches_original_controller_across_gait(worker):
    import osqp
    from go2.contact import FootKinematics
    from go2.model import load_model
    from go2.pin_backend import PinBackend

    namespace = dict(np=np, sparse=sparse, osqp=osqp, PinBackend=PinBackend, FootKinematics=FootKinematics)
    exec(compile(original_controller_source(), "supplied_go2_controller", "exec"), namespace)
    original = namespace["WholeBodyController"](load_model())
    initial = worker.init()
    assert initial["controller_sha256"] == hashlib.sha256(original_controller_source().encode()).hexdigest()
    assert len(initial["joint_names"]) == 12
    assert len(initial["q"]) == 18
    assert np.allclose(initial["v"], 0)
    from scipy.interpolate import BPoly
    with np.load(ROOT / "data/reference.npz", allow_pickle=False) as data:
        reference = BPoly.from_derivatives(data["time"], np.stack([data["q"], data["v"], data["a"]], axis=1))
    for index, t in enumerate((0.0, 2.12, 8.04, 15.4, 24.35, 26.0)):
        q, v = reference(t), reference(t, nu=1)
        q[6:] += 1e-4 * np.sin(np.arange(12) + index)
        v[6:] += 1e-4 * np.cos(np.arange(12) + index)
        result = worker.command(t, q, v)
        from go2.task import target_trajectory
        active, av, aa, stance = (array[0] for array in target_trajectory(np.asarray([t])))
        expected = original.command(q, v, reference(t), reference(t, nu=1), reference(t, nu=2), active, av, aa, stance)
        np.testing.assert_allclose(result["tau"], expected[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(result["predicted_forces"], expected[1], rtol=1e-9, atol=1e-9)
        assert result["qp_violation"] == pytest.approx(expected[2], abs=1e-9)
        assert result["qp_iterations"] == expected[3]
        assert np.max(np.abs(result["tau"]) / initial["limits"]) <= 1.0 + 1e-12


def test_audit_physical_pose_and_world_velocity_conventions(worker):
    from go2.model import load_model

    initial = worker.init()
    q = np.asarray(initial["q"])
    q[:6] = [0.1, -0.1, 0.4, 0.35, -0.12, 0.08]
    v = np.linspace(-0.1, 0.1, 18)
    result = worker.audit_state(q, v)
    mass = np.asarray(result["mass"])
    np.testing.assert_allclose(mass, mass.T, atol=1e-13)
    assert np.linalg.eigvalsh(mass).min() > 0
    np.testing.assert_allclose(np.asarray(result["mass_with_armature"]) - mass,
                               np.diag(initial["armature"]), atol=1e-14)
    cmg = load_model()
    for index, foot in enumerate(cmg["feet"]):
        pose = np.asarray(result["body_poses"][foot["body"]])
        np.testing.assert_allclose(result["feet"][index], pose[:3, :3] @ foot["point_m"] + pose[:3, 3], atol=1e-12)
    step = 1e-6
    rplus = Rotation.from_euler("ZYX", (q + step * v)[3:6]).as_matrix()
    rminus = Rotation.from_euler("ZYX", (q - step * v)[3:6]).as_matrix()
    rotation = Rotation.from_euler("ZYX", q[3:6]).as_matrix()
    skew = ((rplus - rminus) / (2 * step)) @ rotation.T
    omega_world = np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]])
    actual = np.asarray(result["chart_to_world_velocity"]) @ v
    np.testing.assert_allclose(actual[:3], v[:3], atol=1e-13)
    np.testing.assert_allclose(actual[3:6], omega_world, atol=2e-10)


def test_invalid_state_is_diagnostic_and_does_not_desynchronize(worker):
    initial = worker.init()
    with pytest.raises(WorkerError, match="exactly 18 finite"):
        worker.command(0.0, [0.0] * 17, initial["v"])
    assert len(worker.command(0.0, initial["q"], initial["v"])["tau"]) == 12
    with pytest.raises(WorkerError, match="within the requested"):
        worker.command(27.0, initial["q"], initial["v"])
    with pytest.raises(WorkerError, match="Unknown operation"):
        worker.request("not_a_command")
    assert worker.request("ping")["protocol_version"] == 1


def test_timeout_terminates_the_worker_instead_of_reusing_late_reply(tmp_path):
    root = tmp_path / "fake_package"
    package = root / "isaac_validation"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    with WorkerClient(sys.executable, root, tmp_path / "stall.log", timeout=0.2) as client:
        with pytest.raises(WorkerError, match="timed out"):
            client.request("ping")
        assert client.process.poll() is not None
        with pytest.raises(WorkerError, match="closed"):
            client.request("ping")


def test_malformed_protocol_is_diagnostic_and_terminates(tmp_path):
    root = tmp_path / "fake_package"
    package = root / "isaac_validation"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text(
        "import sys\nfor line in sys.stdin:\n print('unexpected native output', flush=True)\n", encoding="utf-8")
    with WorkerClient(sys.executable, root, tmp_path / "invalid.log") as client:
        with pytest.raises(WorkerError, match="Invalid JSON"):
            client.request("ping")
        assert client.process.poll() is not None


def test_unchanged_reference_validation_passes(worker):
    result = worker.validate_reference()
    assert result["passed"], result["checks"]
    assert result["samples"] == 2601
    assert result["duration_s"] == pytest.approx(26.0)
