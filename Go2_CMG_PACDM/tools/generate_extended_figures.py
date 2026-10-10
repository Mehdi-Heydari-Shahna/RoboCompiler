#!/usr/bin/env python3
"""Generate the Pinocchio execution and generated-dynamics manuscript figures.

Run from any directory with ``python tools/generate_extended_figures.py``.
Only NumPy and Matplotlib are required; recorded trajectories and witness CSVs
are read directly, without importing a simulator or rerunning dynamics.

Default inputs, relative to --root:
  workflows/pinocchio/results_pinocchio/{nominal,payload,fine,finer}.npz
  workflows/pinocchio/data/go2_cmg.json
  workflows/generated/results_native/{dynamics_raw,curvature_ablation}.csv

Tracking RMS uses t >= 2 s. Refinement RMS uses the complete common timeline.
The original Pinocchio summary JSON uses all-course tracking RMS; it is not
used as a substitute for the manuscript's common-interval metric.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LEGS = ("FL", "FR", "RL", "RR")
COLORS = ("#007F7C", "#214A73", "#D47B22", "#76579C")
REFERENCE = "#272727"
PIN_FIELDS = {"time": (), "q": (18,), "q_ref": (18,),
              "torque": (12,), "contact_force": (4, 3)}


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Read the numeric fields required by the execution figure."""
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(PIN_FIELDS) - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing trajectory fields {missing}")
        arrays = {key: np.asarray(archive[key], dtype=float) for key in PIN_FIELDS}
    n = len(arrays["time"])
    for key, trailing_shape in PIN_FIELDS.items():
        values = arrays[key]
        if values.shape != (n, *trailing_shape) or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid shape or nonfinite values in {key}")
    t = arrays["time"]
    if n < 2 or np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: at least two increasing timestamps are required")
    expected_time = np.linspace(0, 26, 2601)
    if t.shape != expected_time.shape or not np.allclose(t, expected_time, rtol=0, atol=1e-10):
        raise ValueError(f"{path}: expected the complete 26-s course at 10-ms recording intervals")
    return arrays


def position_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    """Euclidean position errors; all RMS intervals are explicit."""
    error = np.linalg.norm(log["q"][:, :3] - log["q_ref"][:, :3], axis=1)
    course = log["time"] >= 2.0
    if not course.any():
        raise ValueError("Tracking RMS requires at least one sample at t >= 2 s")
    return {"rms_t_ge_2_s_mm": float(1000 * np.sqrt(np.mean(error[course] ** 2))),
            "rms_all_samples_mm": float(1000 * np.sqrt(np.mean(error ** 2))),
            "final_mm": float(1000 * error[-1]),
            "peak_mm": float(1000 * error.max()),
            "rms_t_ge_2_s_sample_count": int(course.sum())}


def path_difference(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> np.ndarray:
    """Euclidean base-path difference in mm at common recorded timestamps."""
    if (a["time"].shape != b["time"].shape or
            not np.allclose(a["time"], b["time"], rtol=0, atol=1e-10)):
        raise ValueError("Refinement trajectories do not have aligned timestamps")
    return 1000 * np.linalg.norm(a["q"][:, :3] - b["q"][:, :3], axis=1)


def motor_limits(cmg: dict) -> np.ndarray:
    """Recover positive joint-torque limits in the recorded coordinate order."""
    coordinates = cmg["coordinate_ids"][6:]
    actuators = {entry["joint"]: entry for entry in cmg["actuation"]["actuators"]}
    limits = []
    for joint in coordinates:
        actuator = actuators[joint]
        lo, hi = np.asarray(actuator["ctrl_range"], dtype=float) * float(actuator["gear"])
        if not np.isfinite([lo, hi]).all() or not np.isclose(lo, -hi) or hi <= 0:
            raise ValueError(f"Symmetric positive torque bounds required for {joint}")
        limits.append(hi)
    if len(limits) != 12:
        raise ValueError("The Go2 trajectory requires twelve motor torque limits")
    return np.asarray(limits)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: CSV contains no witnesses")
    return rows


def column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[key]) for row in rows])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or nonnumeric witness column: {key}") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite witness column: {key}")
    return values


def generated_metrics(dynamics: list[dict[str, str]],
                      curvature: list[dict[str, str]]) -> dict:
    """Aggregate successful raw native witnesses without using summary CSVs."""
    if any(row.get("success", "").lower() != "true" for row in dynamics + curvature):
        raise ValueError("Generated figure requires successful native witness records")
    support = column(dynamics, "support_count")
    native = column(dynamics, "native_acceleration_relative")
    if np.any(native < 0) or set(support) != {2, 3, 4}:
        raise ValueError("Expected nonnegative native discrepancies for 2/3/4-foot support")
    scales = column(curvature, "speed_scale")
    full = column(curvature, "full_residual_m_s2")
    omitted = column(curvature, "omitted_residual_m_s2")
    if np.any(full < 0) or np.any(omitted < 0) or np.any(scales <= 0):
        raise ValueError("Curvature scales must be positive and residuals nonnegative")
    grouped = []
    for scale in sorted(set(scales)):
        use = scales == scale
        grouped.append({"speed_scale": float(scale), "witness_count": int(use.sum()),
                        "full_max_m_s2": float(full[use].max()),
                        "omitted_max_m_s2": float(omitted[use].max())})
    return {"native_witness_count": len(dynamics),
            "native_witness_count_by_support": {
                str(int(k)): int(np.count_nonzero(support == k)) for k in sorted(set(support))},
            "native_acceleration_max_normalized_component_error": float(native.max()),
            "normalization": "max(abs(a-b)) / max(1, max(abs(b))); b = native acceleration",
            "curvature_residual": "maximum absolute Cartesian point-acceleration component",
            "curvature": grouped}


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                        "mathtext.fontset": "dejavuserif", "font.size": 9,
                        "axes.titlesize": 9.2, "axes.labelsize": 9,
                        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
                        "legend.fontsize": 7.8, "axes.linewidth": .65,
                        "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
                        "savefig.facecolor": "white", "axes.axisbelow": True})


def prepare_axes(axes) -> None:
    for ax in np.asarray(axes).flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#d9dfe3", lw=.45, alpha=.75)
        ax.tick_params(direction="out", length=3, width=.6)
        ax.yaxis.set_major_locator(MaxNLocator(5))


def title(ax, letter: str, text: str) -> None:
    ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold", pad=7)


def time_axis(ax, t: np.ndarray, shade_pushes: bool = True) -> None:
    ax.set(xlabel="Time (s)", xlim=(float(t[0]), float(t[-1])))
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    if shade_pushes:
        for start, end in ((1.2, 1.35), (24.3, 24.5)):
            ax.axvspan(start, end, color="#717c82", alpha=.19, lw=0, zorder=0)


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", metadata={"CreationDate": None,
                "Title": stem.replace("_", " "), "Creator": "CMG-PACDM figure generation"})
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_pinocchio(logs: dict[str, dict[str, np.ndarray]], limits: np.ndarray,
                   output: Path) -> dict:
    n = logs["nominal"]
    t, q, reference = n["time"], n["q"], n["q_ref"]
    metrics = {case: position_metrics(log) for case, log in logs.items()}
    differences = (path_difference(n, logs["fine"]),
                   path_difference(logs["fine"], logs["finer"]))
    errors = {case: 1000 * np.linalg.norm(log["q"][:, :3] - log["q_ref"][:, :3], axis=1)
              for case, log in logs.items()}
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 6.85))
    fig.subplots_adjust(left=.095, right=.985, top=.963, bottom=.075, wspace=.27, hspace=.46)
    prepare_axes(axes)
    ax = axes[0, 0]
    title(ax, "a", "Horizontal base path")
    ax.plot(reference[:, 0], reference[:, 1], color=REFERENCE, lw=1.5, ls="--", label="PACDM reference")
    ax.plot(q[:, 0], q[:, 1], color=COLORS[0], lw=.85, label="Pinocchio plant")
    ax.set(xlabel="World x (m)", ylabel="World y (m)")
    ax.legend(loc="upper left", frameon=False)
    ax = axes[0, 1]
    title(ax, "b", "Base height")
    ax.plot(t, reference[:, 2], color=REFERENCE, lw=1.5, ls="--", label="PACDM reference")
    ax.plot(t, q[:, 2], color=COLORS[0], lw=.85, label="Pinocchio plant")
    ax.set_ylabel("Height (m)")
    ax.legend(loc="lower left", frameon=False)
    time_axis(ax, t)
    ax = axes[1, 0]
    title(ax, "c", "Base-position error")
    for case, label, color in (("nominal", "Nominal", COLORS[0]),
                               ("payload", "1.5-kg payload", COLORS[2])):
        ax.plot(logs[case]["time"], errors[case], color=color,
                label=f"{label}: {metrics[case]['rms_t_ge_2_s_mm']:.3f} mm", lw=.9)
    ax.set(ylabel="Euclidean error (mm)", ylim=(0, 1.55 * max(errors["nominal"].max(), errors["payload"].max())))
    ax.legend(loc="upper left", frameon=False, title=r"RMS over $t\geq2$ s", title_fontsize=7.8)
    time_axis(ax, t)
    ax = axes[1, 1]
    title(ax, "d", "World-z foot reactions")
    use = (t >= 6) & (t <= 7.6)
    if not use.any():
        raise ValueError("Pinocchio trajectory does not cover the 6-7.6 s contact window")
    forces = n["contact_force"][:, :, 2]
    for leg, label in enumerate(LEGS):
        ax.plot(t[use], forces[use, leg], color=COLORS[leg], lw=.85, label=label)
    ax.set(xlabel="Time (s)", ylabel="World-z reaction (N)", xlim=(6, 7.6),
           ylim=(min(0, float(forces[use].min()) * 1.1), float(forces[use].max()) * 1.27))
    ax.set_xticks([6, 6.4, 6.8, 7.2, 7.6])
    ax.legend(ncol=4, loc="upper right", frameon=False, columnspacing=.6, handlelength=1.2)
    ax = axes[2, 0]
    title(ax, "e", "Normalized motor effort")
    effort = 100 * (np.abs(n["torque"]) / limits).reshape(len(t), 4, 3).max(axis=2)
    for leg, label in enumerate(LEGS):
        ax.plot(t, effort[:, leg], color=COLORS[leg], lw=.75, label=label)
    ax.axhline(100, color=REFERENCE, ls="--", lw=.8)
    ax.set(ylabel="Maximum per leg (%)", ylim=(0, max(120, float(effort.max()) * 1.15)))
    ax.legend(ncol=4, loc="upper left", frameon=False, columnspacing=.6, handlelength=1.2)
    time_axis(ax, t)
    ax = axes[2, 1]
    title(ax, "f", "Adjacent-timestep refinement")
    refinement = {}
    for difference, label, key, color in zip(differences,
            ("1 / 0.5 ms", "0.5 / 0.25 ms"), ("1_to_0.5_ms", "0.5_to_0.25_ms"),
            (COLORS[1], COLORS[3])):
        rms = float(np.sqrt(np.mean(difference ** 2)))
        ax.plot(t, difference, color=color, label=f"{label}: {rms:.3f} mm", lw=.85)
        refinement[key] = {"max_mm": float(difference.max()), "rms_all_samples_mm": rms}
    ax.set(ylabel="Base-path difference (mm)", ylim=(0, 1.6 * max(x.max() for x in differences)))
    ax.legend(loc="upper left", frameon=False, title="RMS over full course", title_fontsize=7.8)
    time_axis(ax, t)
    save_figure(fig, output, "Go2_Pinocchio_Execution")
    return {"tracking": metrics, "refinement": refinement,
            "sample_count": len(t), "time_start_s": float(t[0]), "time_end_s": float(t[-1]),
            "force_definition": "world-z component of physical contact impulses / integration step",
            "motor_effort_definition": "maximum absolute motor torque / authored limit within each leg; logged samples",
            "motor_limits_Nm": limits.tolist()}


def plot_generated(dynamics: list[dict[str, str]], curvature: list[dict[str, str]],
                   output: Path) -> dict:
    metrics = generated_metrics(dynamics, curvature)
    native = column(dynamics, "native_acceleration_relative")
    support = column(dynamics, "support_count")
    if np.any(native <= 0):
        raise ValueError("Native discrepancies must be positive for the logarithmic witness plot")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15))
    fig.subplots_adjust(left=.10, right=.985, top=.89, bottom=.23, wspace=.39)
    prepare_axes(axes)
    ax = axes[0]
    title(ax, "a", "Native acceleration consistency")
    for k, color in zip((2, 3, 4), COLORS):
        values = native[support == k]
        # Offsets separate recorded witnesses; they do not encode another variable.
        ax.scatter(k + np.linspace(-.14, .14, len(values)), values,
                   color=color, s=16, alpha=.8, linewidths=0,
                   label=f"{k} feet ({len(values)})")
    ax.set_yscale("log")
    ax.set(xlabel="Supporting-foot count", ylabel=r"Normalized component error $\delta_{\max}$",
           xlim=(1.6, 4.4), xticks=[2, 3, 4], ylim=(native.min() * .5, native.max() * 5))
    ax.text(.025, .96, f"Maximum {native.max():.3e}", transform=ax.transAxes, va="top", fontsize=8)
    ax = axes[1]
    title(ax, "b", "Curvature contribution")
    points = metrics["curvature"]
    scales = np.asarray([row["speed_scale"] for row in points])
    full = np.asarray([row["full_max_m_s2"] for row in points])
    omitted = np.asarray([row["omitted_max_m_s2"] for row in points])
    if np.any(full <= 0) or np.any(omitted <= 0):
        raise ValueError("Curvature residuals must be positive for the logarithmic plot")
    ax.semilogy(scales, full, color=COLORS[0], marker="o", ms=4, label="Complete mapping")
    ax.semilogy(scales, omitted, color=COLORS[2], marker="s", ms=4, label="Curvature omitted")
    ax.set(xlabel="Prescribed velocity scale", ylabel=r"Maximum point residual (m s$^{-2}$)",
           xticks=scales, ylim=(full.min() * .3, omitted.max() * 12))
    ax.legend(loc="center right", frameon=False)
    save_figure(fig, output, "Go2_Generated_Dynamics")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Repository root")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_figures"),
                        help="Output directory; relative paths are resolved under --root")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    pin = root / "workflows/pinocchio"
    native = root / "workflows/generated/results_native"
    paths = [pin / "results_pinocchio" / f"{name}.npz" for name in ("nominal", "payload", "fine", "finer")]
    cmg_path = pin / "data/go2_cmg.json"
    dynamics_path, curvature_path = native / "dynamics_raw.csv", native / "curvature_ablation.csv"
    paths.extend([cmg_path, dynamics_path, curvature_path])
    logs = {name: load_trajectory(pin / "results_pinocchio" / f"{name}.npz")
            for name in ("nominal", "payload", "fine", "finer")}
    limits = motor_limits(json.loads(cmg_path.read_text(encoding="utf-8")))
    dynamics, curvature = read_rows(dynamics_path), read_rows(curvature_path)
    # Validate the native CSVs before writing any figures; blank Linux native
    # columns must not be substituted for the executed Windows native run.
    witness_metrics = generated_metrics(dynamics, curvature)
    if witness_metrics["native_witness_count_by_support"] != {"2": 16, "3": 16, "4": 16}:
        raise ValueError("The manuscript figure requires sixteen native witnesses per support count")
    scale_counts = {row["speed_scale"]: row["witness_count"] for row in witness_metrics["curvature"]}
    if scale_counts != {.5: 12, 1.: 12, 2.: 12, 4.: 12}:
        raise ValueError("The manuscript figure requires twelve curvature witnesses at each of 0.5/1/2/4 speed")
    output.mkdir(parents=True, exist_ok=True)
    style()
    report = {"schema_version": 1,
              "source_sha256": {str(path.relative_to(root)).replace("\\", "/"):
                  hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              "pinocchio": plot_pinocchio(logs, limits, output),
              "generated_native": plot_generated(dynamics, curvature, output)}
    (output / "extended_figure_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Go2_Pinocchio_Execution and Go2_Generated_Dynamics (PDF/PNG) to {output}")
    print(f"Nominal tracking RMS, t >= 2 s: {report['pinocchio']['tracking']['nominal']['rms_t_ge_2_s_mm']:.6f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
