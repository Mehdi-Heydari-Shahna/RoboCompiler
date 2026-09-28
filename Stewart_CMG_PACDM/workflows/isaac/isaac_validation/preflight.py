"""Recompute PACDM evidence before allowing the Isaac Sim mission.

The default path needs only NumPy, SciPy and the unchanged PACDM source. It
does not import Isaac Sim, Pinocchio or MuJoCo. Checks are fresh computations;
the supplied MuJoCo results remain historical comparison data. A passed
preflight is not an Isaac Sim validation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from stewart.model import inverse_seed, make_cmg
from vendor.pacdm_original import PACDM, PointGraph


PACDM_SHA256 = "492209e3a33281684751990ce97e02459e18a5529c2b7c4e8bae124eadc310ca"
MANIFEST_SHA256 = "38ade3636ff7531f753468f799d352b73a7adab5c4daa3209e9807a1ca469d48"
CMG_SHA256 = "401d7c2af4f9801e39040d33ab627a9fa1598831f674b48e830507629a6fcda7"
REFERENCE_SHA256 = "f4d78ee06cd990185a33199e8376e4a608d120d321f2cf47fa9c8e998a53c394"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _equivalent(left, right):
    """Ignore harmless floating point roundoff in regenerated CMG values."""
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_equivalent(left[k], right[k]) for k in left))
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right)
                and all(_equivalent(a, b) for a, b in zip(left, right)))
    if isinstance(left, (float, int)) and not isinstance(left, bool):
        return bool(isinstance(right, (float, int))
                    and np.isclose(left, right, rtol=0, atol=1e-12))
    return left == right


def _mujoco_feedforward(cmg, reference, indices):
    """Independent unconstrained tree inverse dynamics, projected by PACDM.

    Equality constraints are disabled here because their reaction forces
    must not be included in M(q)qdd + b(q,qd) before projection.
    """
    import mujoco
    from stewart.model import compile_mujoco

    with tempfile.TemporaryDirectory(prefix="stewart_preflight_") as tmp:
        xml = compile_mujoco(cmg, Path(tmp) / "tree.xml")
        model = mujoco.MjModel.from_xml_path(str(xml))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        data = mujoco.MjData(model)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
                     for key in cmg["coordinate_ids"]]
        if any(j < 0 for j in joint_ids):
            raise ValueError("MuJoCo tree is missing a CMG coordinate")
        qi = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
        vi = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
        errors = []
        for k in indices:
            data.qpos[qi] = reference["q"][k]
            data.qvel[vi] = reference["velocity"][k]
            data.qacc[vi] = reference["acceleration"][k]
            mujoco.mj_inverse(model, data)
            measured = reference["mapping"][k].T @ data.qfrc_inverse[vi]
            errors.append(np.max(np.abs(measured - reference["feedforward_force"][k])))
        return float(max(errors)), mujoco.__version__


def run_preflight(root, *, samples=25, check_mujoco=False):
    """Return a JSON-compatible report; every required failure fails closed.

    ``root`` is the extracted package directory. Sampling uses deterministic
    equally spaced reference rows. Full-array structural, time, finiteness,
    inverse geometry and joint-limit checks do not use sampling.
    """
    started = time.perf_counter()
    root = Path(root).resolve()
    checks = {}
    details = {
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "requested_samples": int(samples), "sample_failures": [],
        "method": "Fresh perturbed-seed PACDM acquisition and differential mapping; "
                  "immutable supplied reference and model content checks.",
        "scope": "CPU reference/model verification only; no Isaac Sim execution.",
        "runtime_role": "PACDM generates the reference and differential map; "
                        "cached Pinocchio-projected force feedforward drives Isaac physics. "
                        "PACDM is not substituted for the PhysX physics integrator.",
    }
    report = {"schema": "stewart.isaac.preflight/1", "passed": False,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "details": details}

    def check(name, value, limit, relation="<=", unit="dimensionless"):
        finite = bool(np.isfinite(value))
        passed = finite and {"<=": lambda: value <= limit,
                             ">=": lambda: value >= limit,
                             "==": lambda: value == limit}[relation]()
        checks[name] = {"passed": bool(passed), "value": float(value) if finite else None,
                        "limit": limit, "relation": relation, "unit": unit}

    def finish():
        report["passed"] = bool(checks) and all(c["passed"] for c in checks.values())
        report["elapsed_s"] = time.perf_counter() - started
        return report

    try:
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 25:
            raise ValueError("At least 25 deterministic PACDM samples are required")
        anchored = {"vendor/pacdm_original.py": PACDM_SHA256,
                    "baseline/original_SHA256SUMS.json": MANIFEST_SHA256,
                    "data/stewart.cmg.json": CMG_SHA256,
                    "baseline/reference.npz": REFERENCE_SHA256}
        details["sha256"] = {p: sha256(root / p) for p in anchored}
        for path, expected in anchored.items():
            check("hash." + path, int(details["sha256"][path] == expected), 1, "==")
        manifest = json.loads((root / "baseline/original_SHA256SUMS.json").read_text())
        # Also protect cached historical comparator traces and the original
        # source used by the optional reference regeneration command.
        checksums = {p: manifest[p] for p in ("stewart/model.py", "stewart/reference.py",
                                             "stewart/pin_backend.py")}
        for path in sorted((root / "baseline").iterdir()):
            key = "results/" + path.name
            if path.suffix in (".npz", ".json") and key in manifest:
                checksums["baseline/" + path.name] = manifest[key]
        for path, expected in checksums.items():
            actual = sha256(root / path)
            details["sha256"][path] = actual
            check("hash." + path, int(actual == expected), 1, "==")
        if not all(c["passed"] for c in checks.values()):
            details["initialization_error"] = "A baseline/model/source hash differs from the supplied release"
            return finish()

        cmg = json.loads((root / "data/stewart.cmg.json").read_text())
        check("cmg.matches_declared_benchmark", int(_equivalent(cmg, make_cmg())), 1, "==")
        with np.load(root / "baseline/reference.npz", allow_pickle=False) as archive:
            ref = {key: archive[key].copy() for key in archive.files}
        shapes = {"time": (1101,), "q": (1101, 24), "q_augmented": (1101, 42),
                  "velocity": (1101, 24), "acceleration": (1101, 24),
                  "lengths": (1101, 6), "length_velocity": (1101, 6),
                  "length_acceleration": (1101, 6), "feedforward_force": (1101, 6),
                  "mapping": (1101, 24, 6), "target_pose": (1101, 6),
                  "coordinate_ids": (24,), "closure_residual": (1101,),
                  "tangent_residual": (1101,), "acceleration_residual": (1101,),
                  "selected_block_condition": (1101,), "reduced_mass_min_eigenvalue": (1101,)}
        schema_ok = set(ref) == set(shapes) and all(ref[k].shape == s for k, s in shapes.items())
        check("reference.array_schema", int(schema_ok), 1, "==")
        if not schema_ok:
            details["initialization_error"] = "Reference array names or shapes are invalid"
            return finish()
        finite = all(np.isfinite(v).all() for k, v in ref.items() if k != "coordinate_ids")
        check("reference.all_values_finite", int(finite), 1, "==")
        check("reference.coordinate_ids", int(ref["coordinate_ids"].tolist() == cmg["coordinate_ids"]), 1, "==")
        t = ref["time"]
        check("reference.time_start", abs(t[0]), 1e-12, unit="s")
        check("reference.duration", abs(t[-1] - 22.), 1e-12, unit="s")
        check("reference.strictly_increasing_time", int(np.all(np.diff(t) > 0)), 1, "==")
        check("reference.uniform_interval", np.max(abs(np.diff(t) - .02)), 1e-12, unit="s")
        if not all(c["passed"] for c in checks.values()):
            return finish()

        graph = PointGraph(cmg, ref["q"][0])
        solver = PACDM(graph)
        geometric = np.asarray([inverse_seed(cmg, p) for p in ref["target_pose"]])
        check("reference.inverse_geometry_q", np.max(abs(ref["q"] - geometric)), 2e-8, unit="mixed SI")
        check("reference.inverse_geometry_lengths", np.max(abs(ref["lengths"] - geometric[:, graph.active])), 1e-12, unit="m")
        check("reference.augmented_physical_coordinates", np.max(abs(ref["q_augmented"][:, :24] - ref["q"])), 1e-12, unit="mixed SI")
        check("reference.active_coordinates", np.max(abs(ref["q"][:, graph.active] - ref["lengths"])), 1e-12, unit="m")
        check("reference.active_velocity", np.max(abs(ref["velocity"][:, graph.active] - ref["length_velocity"])), 1e-12, unit="m/s")
        check("reference.active_acceleration", np.max(abs(ref["acceleration"][:, graph.active] - ref["length_acceleration"])), 1e-12, unit="m/s^2")
        limit_margin = min(np.min(ref["q_augmented"] - graph.lower), np.min(graph.upper - ref["q_augmented"]))
        check("reference.joint_limit_margin", limit_margin, 0, ">=", "mixed SI")
        check("reference.feedforward_within_force_limit", np.max(abs(ref["feedforward_force"])), cmg["actuation"]["force_limit_N"], unit="N")
        check("reference.cached_reduced_mass_positive", np.min(ref["reduced_mass_min_eigenvalue"]), 1e-8, ">=", "kg")

        indices = np.unique(np.linspace(0, len(t) - 1, min(samples, len(t)), dtype=int))
        details["sample_indices"] = indices.tolist()
        details["sample_times_s"] = t[indices].tolist()
        details["perturbation_seed"] = 20260924
        rng = np.random.default_rng(details["perturbation_seed"])
        maxima = {k: 0. for k in ("reconstructed_pose", "closure", "point_closure", "tangent",
                                  "mapping_cache", "velocity_cache", "acceleration_cache",
                                  "acceleration_constraint")}
        min_rank = graph.n
        completed = 0
        for index in indices:
            try:
                stored = ref["q_augmented"][index]
                N, mapping = solver.mapping(stored)
                if N is None or not mapping["success"]:
                    raise ValueError("Stored state fails PACDM rank/closure gate: " + str(mapping))
                min_rank = min(min_rank, mapping["rank_full"], mapping["rank_passive"])
                maxima["closure"] = max(maxima["closure"], mapping["residual_inf"])
                maxima["tangent"] = max(maxima["tangent"], mapping["tangent_residual"])
                maxima["mapping_cache"] = max(maxima["mapping_cache"], float(np.max(abs(N[:24] - ref["mapping"][index]))))
                r, J, D = graph.residual(stored)
                maxima["point_closure"] = max(maxima["point_closure"], float(np.max(np.linalg.norm(D[:, :3, 3], axis=1))))
                v = N @ ref["length_velocity"][index]
                h = 1e-5 / max(1., np.linalg.norm(v))
                Jdot = (graph.residual(stored + h * v)[1] - graph.residual(stored - h * v)[1]) / (2 * h)
                rows = np.asarray(mapping["rows"])
                curvature = np.zeros(graph.n)
                curvature[graph.passive] = -np.linalg.solve(J[np.ix_(rows, graph.passive)], (Jdot @ v)[rows])
                a = N @ ref["length_acceleration"][index] + curvature
                maxima["velocity_cache"] = max(maxima["velocity_cache"], float(np.max(abs(v[:24] - ref["velocity"][index]))))
                maxima["acceleration_cache"] = max(maxima["acceleration_cache"], float(np.max(abs(a[:24] - ref["acceleration"][index]))))
                maxima["acceleration_constraint"] = max(maxima["acceleration_constraint"], float(np.max(abs(J @ a + Jdot @ v))))
                seed = geometric[index] + rng.uniform(-.002, .002, graph.nt)
                seed[graph.active] = ref["lengths"][index]
                reconstructed, info = solver.acquire(ref["lengths"][index], graph.lift(seed))
                if not info["success"]:
                    raise ValueError("Fresh perturbed-seed acquisition failed: " + str(info))
                maxima["reconstructed_pose"] = max(maxima["reconstructed_pose"], float(np.max(abs(reconstructed[:6] - ref["target_pose"][index]))))
                completed += 1
            except Exception as error:
                details["sample_failures"].append({"index": int(index), "error": f"{type(error).__name__}: {error}"})
        check("pacdm.samples_completed", completed, len(indices), "==", "count")
        check("pacdm.physical_and_passive_rank", min_rank, len(graph.passive), "==", "rank")
        for name, limit in (("reconstructed_pose", 2e-8), ("closure", 1e-8),
                            ("point_closure", 1e-8), ("tangent", 1e-10),
                            ("mapping_cache", 1e-9), ("velocity_cache", 1e-9),
                            ("acceleration_cache", 1e-7), ("acceleration_constraint", 1e-8)):
            check("pacdm." + name, maxima[name], limit, unit="mixed SI")
        if check_mujoco:
            difference, version = _mujoco_feedforward(cmg, ref, indices)
            details["versions"]["mujoco"] = version
            details["optional_mujoco_check"] = "Executed unconstrained tree inverse dynamics and PACDM projection"
            check("reference.mujoco_projected_feedforward_difference", difference, 1e-7, unit="N")
        else:
            details["optional_mujoco_check"] = "Not requested; no MuJoCo imported"
    except Exception as error:
        details["initialization_error"] = f"{type(error).__name__}: {error}"
        check("preflight.exception_free", 0, 1, "==")
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--check-mujoco", action="store_true", help="Also verify cached force with an independently lowered MuJoCo tree")
    args = parser.parse_args()
    report = run_preflight(args.root, samples=args.samples, check_mujoco=args.check_mujoco)
    output = args.output or args.root / "local_checks/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    failures = [name for name, value in report["checks"].items() if not value["passed"]]
    print(json.dumps({"passed": report["passed"], "failed_checks": failures, "report": str(output)}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
