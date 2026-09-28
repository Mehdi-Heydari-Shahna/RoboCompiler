#!/usr/bin/env python3
"""Generate supplementary figures from recorded Panda CMG/PACDM NPZ files.

Usage:
    python tools/plot_panda_supplement.py --root . --out figures

The plotting script reads saved reference/simulation states. It does not rerun
PACDM or MuJoCo and does not alter the source benchmark or the saved logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np


INK = "#19384A"
TEAL = "#087F8C"
AMBER = "#D88620"
PLUM = "#8A5C9E"
BLUE = "#446A9E"
GRID = "#DCE4E9"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.titleweight": "semibold", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#A4B7C2", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.55,
        "lines.solid_capstyle": "round", "legend.frameon": False,
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "figure.dpi": 160, "savefig.dpi": 240,
    })


def save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def reference_figure(ref: dict, nominal: dict, out: Path) -> None:
    t = ref["time"]
    assert len(t) == 2201 and nominal["q"].shape[0] == len(t)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.1), layout="constrained")
    ax = axes[0]
    ax.plot(t, ref["q"][:, 2], color=TEAL, lw=1.85, label="PACDM reference")
    ax.plot(nominal["time"], nominal["q"][:, 2], color=AMBER,
            lw=1.2, ls=(0, (5, 2.5)), label="MuJoCo measured")
    ax.scatter([13.5, 15.], [.25, -.20], s=19, facecolors="white",
               edgecolors=TEAL, zorder=5)
    ax.set(title="(a) Prescribed arm redundancy", xlabel="Time [s]",
           ylabel="Joint 3 [rad]", xlim=(0, 22), ylim=(-.235, .285))
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    for key, label, color, width in [
        ("closure_residual", "Closure", TEAL, 1.45),
        ("tangent_residual", "Tangent", AMBER, 1.10),
        ("acceleration_constraint_residual", "Acceleration", PLUM, 1.05),
    ]:
        # A finite plotting floor keeps exact zeros visible on logarithmic axes.
        ax.plot(t, np.maximum(ref[key], 1e-18), color=color,
                lw=width, label=label)
    ax.set(title="(b) Ideal reference residuals", xlabel="Time [s]",
           ylabel="Max. component [declared chart]", xlim=(0, 22),
           ylim=(1e-18, 4e-9), yscale="log")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[2]
    ax.plot(t, ref["rcond"], color=BLUE, lw=1.5)
    ix = int(np.argmin(ref["rcond"]))
    ax.scatter(t[ix], ref["rcond"][ix], color=BLUE, s=25, zorder=5)
    ax.annotate(f"minimum = {ref['rcond'][ix]:.5f}",
                xy=(t[ix], ref["rcond"][ix]), xytext=(6.0, .0523),
                arrowprops=dict(arrowstyle="-", color=INK, lw=.8),
                fontsize=8, color=INK)
    ax.set(title="(c) Passive solve conditioning", xlabel="Time [s]",
           ylabel="Reciprocal condition estimate", xlim=(0, 22),
           ylim=(.047, .064))
    for a in axes:
        a.xaxis.set_major_locator(MultipleLocator(5))
        a.tick_params(length=3, width=.6)
    save(fig, out, "Panda_PACDM_Reference")


def rollout_figure(nominal: dict, summary: dict, out: Path) -> None:
    t = nominal["time"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.55), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    ax = axes[0]
    # The benchmark barrier occupies y = [-0.018, +0.018] m and z = [0, .200] m.
    ax.add_patch(Rectangle((-18, 0), 36, 200, facecolor="#DFE6EB",
                           edgecolor="#9EADB7", lw=.8))
    y = nominal["object_pos"][:, 1] * 1000
    z = nominal["object_pos"][:, 2] * 1000
    ax.plot(y, z, color=TEAL, lw=2.1, zorder=4)
    over = (t >= 8) & (t <= 12)
    ax.plot(y[over], 200 + 1000 * nominal["barrier_clearance"][over],
            color=BLUE, lw=1.3, ls=(0, (3.0, 2.0)), zorder=4)
    ax.scatter([y[0], y[-1]], [z[0], z[-1]], s=29,
               c=[INK, AMBER], zorder=5)
    ax.annotate("Pickup", xy=(y[0], z[0]), xytext=(-175, 116),
                arrowprops=dict(arrowstyle="-", lw=.6, color=INK), fontsize=8)
    ax.annotate("Socket", xy=(y[-1], z[-1]), xytext=(110, 117),
                arrowprops=dict(arrowstyle="-", lw=.6, color=INK), fontsize=8)
    ax.text(-215, 373,
            f"Min. corner clearance (8–12 s):\n"
            f"{summary['minimum_payload_barrier_clearance_m'] * 1000:.1f} mm",
            va="top", fontsize=8, color=INK,
            bbox={"boxstyle": "round,pad=.28", "fc": "white", "ec": "none", "alpha": .94})
    ax.set(title="(a) Free-payload trajectory", xlabel="World y [mm]",
           ylabel="World height [mm]", xlim=(-230, 244), ylim=(0, 400))
    ax.text(-95, 335, "Payload centre", fontsize=8, color=TEAL, va="bottom")
    ax.text(36, 284, "Lowest corner", fontsize=8, color=BLUE, va="top")
    ax.text(0, 143, "200-mm\nbarrier", fontsize=7.5, color=INK,
            rotation=90, va="center", ha="center")

    ax = axes[1]
    ax.plot(t, nominal["pose_error"] * 1000, color=TEAL,
            lw=1.5, label="Position")
    ax.set(title="(b) Tool-target tracking", xlabel="Time [s]",
           ylabel="Position error [mm]", xlim=(0, 22), ylim=(0, .315))
    ax2 = ax.twinx()
    ax2.plot(t, np.rad2deg(nominal["angle_error"]), color=AMBER,
             lw=1.2, label="Orientation")
    ax2.set(ylabel="Orientation error [deg]", ylim=(0, .035))
    ax2.tick_params(axis="y", colors=AMBER, labelsize=8)
    ax2.yaxis.label.set_color(AMBER)
    ax2.grid(False)
    for a, b in [(13.0, 13.16), (14.5, 14.62)]:
        ax.axvspan(a, b, color="#F5C976", alpha=.32, lw=0)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8)

    ax = axes[2]
    mask = (t >= 8) & (t <= 16)
    ax.plot(t[mask], nominal["left_normal_force"][mask],
            color=BLUE, lw=1.55, label="Left finger")
    ax.plot(t[mask], nominal["right_normal_force"][mask],
            color=PLUM, lw=1.55, label="Right finger")
    ax.axvspan(13.0, 13.16, color="#F5C976", alpha=.45, lw=0)
    ax.axvspan(14.5, 14.62, color="#E7B18B", alpha=.45, lw=0)
    ax.text(13.00, 10.55, "F", ha="center", fontsize=8, color="#9B5B0A")
    ax.text(14.57, 10.55, "T", ha="center", fontsize=8, color="#A25D39")
    ax.set(title="(c) Bilateral payload contact", xlabel="Time [s]",
           ylabel="Finger normal force [N]", xlim=(8, 16), ylim=(6.9, 11.0))
    ax.legend(loc="lower left", fontsize=8)
    axes[1].xaxis.set_major_locator(MultipleLocator(5))
    axes[2].xaxis.set_major_locator(MultipleLocator(2))
    for a in axes:
        a.tick_params(length=3, width=.6)
    save(fig, out, "Panda_Contact_Rollout")


def ablation_figure(nominal: dict, ablation: dict,
                    nominal_summary: dict, ablation_summary: dict,
                    out: Path) -> None:
    t = nominal["time"]
    assert np.allclose(t, ablation["time"])
    fig, ax = plt.subplots(figsize=(5.0, 2.9), layout="constrained")
    ax.axvspan(8, 16, facecolor="#EAF3F3", lw=0)
    ax.plot(t, ablation["pose_error"] * 1000, color=AMBER, lw=1.5,
            label=f"Without feedforward: RMS {ablation_summary['rms_tool_position_error_m'] * 1000:.3f} mm")
    ax.plot(t, nominal["pose_error"] * 1000, color=TEAL, lw=1.75,
            label=f"With Pinocchio feedforward: RMS {nominal_summary['rms_tool_position_error_m'] * 1000:.3f} mm")
    ax.set(xlabel="Time [s]", ylabel="Tool-position error [mm]",
           xlim=(0, 22), ylim=(0, 7.4))
    ax.legend(loc="center", bbox_to_anchor=(.53, .42), fontsize=7.8)
    ax.text(12, 1.0, "Transfer (8–16 s)", ha="center", fontsize=8,
            color="#527F83")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(length=3, width=.6)
    save(fig, out, "Panda_Feedforward_Comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Benchmark repository root")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output folder for PNG and vector PDF figures")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = load_npz(root / "data/reference.npz")
    nominal = load_npz(root / "results/nominal.npz")
    ablation = load_npz(root / "results/no_feedforward.npz")
    normal_summary = json.loads((root / "results/nominal.json").read_text())
    ablation_summary = json.loads((root / "results/no_feedforward.json").read_text())
    setup_style()
    reference_figure(ref, nominal, out)
    rollout_figure(nominal, normal_summary, out)
    ablation_figure(nominal, ablation, normal_summary, ablation_summary, out)
    for name in ("Panda_PACDM_Reference", "Panda_Contact_Rollout",
                 "Panda_Feedforward_Comparison"):
        print(out / f"{name}.pdf")


if __name__ == "__main__":
    main()
