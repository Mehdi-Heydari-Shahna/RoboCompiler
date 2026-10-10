"""Build measured reports from one completed, hash-consistent experiment.

The acceptance thresholds are read from run_validation.py's saved checks.
This module never changes a result, threshold, trajectory or status.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np


LABELS = {
    "nominal": "Nominal", "half_step": "Half step", "quarter_step": "Quarter step",
    "heavy_load": "Higher tool force", "initial_offset": "Initial offset",
    "no_feedforward": "No arm feedforward",
}
COLORS = {
    "nominal": "#087f8c", "half_step": "#536ab9", "quarter_step": "#50a5c2",
    "heavy_load": "#9656a6", "initial_offset": "#bd861a", "no_feedforward": "#c34742",
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _num(value, precision=4):
    if value is None:
        return "not finite"
    if not np.isfinite(float(value)):
        raise ValueError("Reports require finite measured values")
    return f"{float(value):.{precision}g}"


def _escape(value):
    return html.escape(str(value), quote=True)


def _table(headers, rows):
    return ("<div class='table-scroll'><table><thead><tr>"+
            "".join("<th>"+_escape(v)+"</th>" for v in headers)+
            "</tr></thead><tbody>"+
            "".join("<tr>"+"".join("<td>"+_escape(v)+"</td>" for v in row)+"</tr>" for row in rows)+
            "</tbody></table></div>")


def _markdown_table(headers, rows):
    def row(values):
        return "| "+" | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values)+" |"
    return "\n".join([row(headers), row(["---"]*len(headers))]+[row(v) for v in rows])


def _plot(results, summary, cases, logs, effort_limits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#233748", "text.color": "#233748",
                         "xtick.color": "#40566a", "ytick.color": "#40566a",
                         "savefig.facecolor": "white"})
    figure, axes = plt.subplots(3, 2, figsize=(15.6, 13.8), constrained_layout=True)
    axes = axes.ravel()
    for name, item in logs.items():
        color = COLORS.get(name, "#546a7b")
        style = "--" if name in ("half_step", "quarter_step", "no_feedforward") else "-"
        width = 1.8 if name == "nominal" else 1.15
        label = f"{LABELS.get(name, name)} ({cases[name]['dt_s']*1e3:g} ms)"
        axes[0].plot(item["time"], 1e3*item["pose_error"], style, color=color,
                     linewidth=width, label=label)
        axes[1].plot(item["time"], np.rad2deg(item["angle_error"]), style,
                     color=color, linewidth=width, label=label)
        axes[3].plot(item["time"], item["energy_balance"], style, color=color,
                     linewidth=width, label=LABELS.get(name, name))
    position_gate = summary["checks"]["nominal.tool_position"]["limit"]*1e3
    orientation_gate = summary["checks"]["nominal.tool_rotation"]["limit"]
    axes[0].axhline(position_gate, color="#6c7881", linestyle=":", linewidth=1.3,
                    label=f"Positive-case gate ({position_gate:g} mm)")
    axes[1].axhline(orientation_gate, color="#6c7881", linestyle=":", linewidth=1.3)
    axes[0].set(title=f"A  Tool-position tracking — all {len(cases)} cases", ylabel="Position error [mm]")
    axes[1].set(title=f"B  Tool-orientation tracking — all {len(cases)} cases", ylabel="Rotation error [deg]")
    # Symmetric log has a linear region containing zero: no zero samples are
    # removed or replaced by artificial positive values.
    axes[0].set_yscale("symlog", linthresh=.01)
    axes[1].set_yscale("symlog", linthresh=.001)
    axes[0].set_ylim(bottom=0.)
    axes[1].set_ylim(bottom=0.)
    axes[0].legend(loc="upper left", fontsize=8.1, ncol=2, framealpha=.93)
    nominal = logs["nominal"]
    axes[2].plot(nominal["time"], nominal["energy"]-nominal["energy"][0],
                 color=COLORS["nominal"], linewidth=2.6, label="Mechanical energy change")
    axes[2].plot(nominal["time"], nominal["work"], color="#e09532", linestyle="--",
                 linewidth=1.5, label="Integrated net work")
    axes[2].set(title="C  Independent energy and work — nominal", ylabel="Energy change / work [J]")
    axes[2].legend(fontsize=9)
    axes[3].set(title=f"D  Energy–work balance — all {len(cases)} cases", ylabel="E(t) − E(0) − W(t) [J]")
    axes[3].set_yscale("symlog", linthresh=1e-12)
    largest_refinement = max(pair["max_position_difference_m"] for pair in summary["refinement"])
    refinement_unit, refinement_scale = (("pm", 1e12) if largest_refinement < 1e-9 else
                                         ("nm", 1e9) if largest_refinement < 1e-6 else ("µm", 1e6))
    for pair in summary["refinement"]:
        a, b = logs[pair["coarse"]], logs[pair["fine"]]
        position = np.linalg.norm(a["tool_pos"]-b["tool_pos"], axis=1)
        axes[4].plot(a["time"], position*refinement_scale, linewidth=1.4,
                     label=f"{cases[pair['coarse']]['dt_s']*1e3:g} vs {cases[pair['fine']]['dt_s']*1e3:g} ms")
    axes[4].set(title="E  Full-cycle timestep refinement", ylabel=f"Tool-position difference [{refinement_unit}]")
    axes[4].set_ylim(bottom=0.)
    axes[4].legend(fontsize=9)
    case_names = list(cases)
    ratios = np.asarray([cases[name]["peak_actuator_effort"] for name in case_names])/effort_limits
    maximum = max(1., float(ratios.max()))
    im = axes[5].imshow(ratios, cmap="YlGnBu", aspect="auto", vmin=0., vmax=maximum)
    axes[5].set(title="F  Peak actuator effort / source limit", xlabel="Actuator")
    axes[5].set_xticks(np.arange(8), ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Tendon"])
    axes[5].set_yticks(np.arange(len(case_names)), [LABELS.get(name, name) for name in case_names])
    for i in range(len(case_names)):
        for j in range(8):
            axes[5].text(j, i, f"{ratios[i,j]:.3f}", ha="center", va="center", fontsize=8.8,
                         color="white" if ratios[i, j] > .6*maximum else "#203749")
    figure.colorbar(im, ax=axes[5], fraction=.04, pad=.025, label="Ratio (limit = 1)")
    for ax in axes[:5]:
        ax.set_xlabel("Time [s]")
        ax.set_xlim(nominal["time"][0], nominal["time"][-1])
        ax.grid(True, color="#dfe6eb", linewidth=.55, alpha=.85)
        ax.set_axisbelow(True)
    status = "PASS" if summary["passed"] else "FAIL"
    figure.suptitle(f"Panda · PACDM / Pinocchio numerical validation\n"
                    f"{status} · {summary['passed_count']}/{summary['check_count']} acceptance checks",
                    fontsize=17, fontweight="bold")
    figure.savefig(results/"performance.png", dpi=165)
    plt.close(figure)


def build_report(root):
    """Write HTML, PNG and Markdown only for completed, consistent evidence.

    A failed completed experiment is reported as FAIL. An unfinished experiment
    raises instead of creating a misleading finished report. Video is embedded
    only if its recorded source hash matches the current nominal trajectory.
    """
    root = Path(root).resolve()
    results = root/"results"
    summary = _load(results/"validation.json")
    if "status" in summary or not all(k in summary for k in ("checks", "cases", "refinement", "trajectory_SHA256")):
        raise RuntimeError("Validation is incomplete; report generation waits for the completed aggregate")
    mechanics = _load(results/"mechanics.json")
    task = _load(results/"task_validation.json")
    cmg = _load(root/"data/panda_cmg.json")
    cases = summary["cases"]
    checks = summary["checks"]
    passed_count = sum(bool(item["passed"]) for item in checks.values())
    if passed_count != summary["passed_count"] or len(checks) != summary["check_count"]:
        raise ValueError("Saved check totals do not match the actual checks")
    if summary["passed"] != all(item["passed"] for item in checks.values()):
        raise ValueError("Saved overall status does not match actual checks")
    for prefix, record in (("mechanics", mechanics), ("task", task)):
        for name, item in record["checks"].items():
            if checks.get(prefix+"."+name) != item:
                raise ValueError(f"{prefix} evidence does not match completed aggregate: {name}")
    logs = {}
    for name, record in cases.items():
        if _load(results/f"{name}.json") != record:
            raise ValueError(f"Case summary changed after aggregation: {name}")
        path = results/f"{name}.npz"
        if _sha(path) != summary["trajectory_SHA256"][name]:
            raise ValueError(f"Trajectory changed after aggregation: {name}")
        with np.load(path, allow_pickle=False) as archive:
            logs[name] = {key: archive[key] for key in archive.files}
        if any(not np.all(np.isfinite(value)) for value in logs[name].values()):
            raise ValueError(f"Nonfinite plotted data in {name}")
    if _sha(root/"data/reference.npz") != summary["reference_SHA256"]:
        raise ValueError("Reference changed after aggregation")
    actual_core_sha = _sha(root/"vendor/pacdm_original.py")
    expected_core_sha = summary["original_PACDM_SHA256"]
    core_unchanged = actual_core_sha == expected_core_sha
    effort_limits = np.max(abs(np.asarray([item["force_range"]
        for item in cmg["actuation"]["actuators"]], dtype=float)), axis=1)
    if effort_limits.shape != (8,) or np.any(effort_limits <= 0):
        raise ValueError("Invalid source effort limits")
    _plot(results, summary, cases, logs, effort_limits)

    case_headers = ["Case", "Step (ms)", "Equivalent force load (kg)", "Max position (mm)",
                    "RMS position (mm)", "Max rotation (deg)", "Max joint error (deg)",
                    "Final position (mm)", "Energy/work error (J)", "Saturated RHS evaluations"]
    case_rows = [[LABELS.get(name, name), _num(record["dt_s"]*1e3), _num(record["load_kg_equivalent"]),
                  _num(record["max_tool_position_error_m"]*1e3),
                  _num(record["rms_tool_position_error_m"]*1e3),
                  _num(record["max_tool_orientation_error_deg"]),
                  _num(record["max_arm_joint_error_deg"]),
                  _num(record["final_tool_error_m"]*1e3),
                  _num(record["max_energy_work_balance_error_J"]),
                  record["saturation_rhs_evaluations"]] for name, record in cases.items()]
    refinement_headers = ["Steps compared (ms)", "Max tool difference (µm)",
                          "Max rotation difference (deg)", "Max joint difference (rad)", "Samples"]
    refinement_rows = [[f"{_num(cases[p['coarse']]['dt_s']*1e3)} / {_num(cases[p['fine']]['dt_s']*1e3)}",
                        _num(p["max_position_difference_m"]*1e6), _num(p["max_rotation_difference_deg"]),
                        _num(p["max_joint_difference_rad"]), p["compared_samples"]] for p in summary["refinement"]]
    failed = {name: item for name, item in checks.items() if not item["passed"]}
    check_headers = ["Check", "Result", "Measured value", "Required", "Unit"]
    check_rows = [[name, "PASS" if item["passed"] else "FAIL", _num(item["value"], 7),
                   item["relation"]+" "+_num(item["limit"], 7), item.get("unit", "")]
                  for name, item in checks.items()]
    failed_rows = [row for row in check_rows if row[1] == "FAIL"]
    sections = [("Task / route", task["checks"]), ("Independent mechanics", mechanics["checks"]),
                ("Integrated cases and experiment gates", {name: item for name, item in checks.items()
                    if not name.startswith(("task.", "mechanics."))})]
    count_rows = [[name, sum(v["passed"] for v in items.values()), len(items),
                   sum(not v["passed"] for v in items.values())] for name, items in sections]
    nominal = cases["nominal"]
    duration = _num(nominal["duration_s"])
    status = "PASS" if summary["passed"] else "FAIL"
    scope = list(summary["scope"])
    methods = [
        ("CMG and original PACDM", "panda/model.py; vendor/pacdm_original.py; data/panda_cmg.json",
         "Authored model records compile directly into CMG. The original PACDM core assembles the 15-coordinate task graph and its 15×8 differential map. Nine coordinates are physical; six are virtual target coordinates. One finger equality is physical."),
        ("Reference and acceleration curvature", "panda/task.py; panda/task_validation.py; data/reference.npz",
         "Piecewise quintic targets prescribe tool pose, joint3 redundancy and finger opening. PACDM computes q, v=N v_active and a=N a_active+curvature with J a+Jdot v=0. The reference is interpolated using position, velocity and acceleration at every knot."),
        ("Independent kinematics and mechanics", "pin_validation/mechanics.py; results/mechanics.json; results/task_validation.json",
         "Native Pinocchio FK, frame Jacobians, velocity and classical acceleration are checked against graph kinematics and explicit XYZ target derivatives. Tests include finite differences, rank, RNEA/CRBA/bias consistency, positive mass matrices, energy, virtual work, physical-finger KKT/reduction equivalence, and native ABA/RNEA round trips."),
        ("Integrated plant and controller", "pin_validation/simulation.py; results/*.npz; results/*.json",
         "Pinocchio supplies mass, nonlinear effects and inverse dynamics. A custom RK4 integrator advances eight physical coordinates, velocities and accumulated work. Source armature/damping, actuator transmission, arm servo coefficients and effort/setpoint limits are applied. A documented 1000 N/m, 30 N s/m finger servo is a benchmark choice. Feedforward uses the PACDM acceleration reference and Pinocchio inverse dynamics."),
        ("Loads and experiments", "pin_validation/simulation.py; results/validation.json",
         "A smooth downward tool wrench and brief force/torque disturbances test tracking. The higher-load case doubles the equivalent downward force. Initial joint offsets test recovery. The ablation disables arm inverse-dynamics and modeled-damping feedforward only; the same finger feedforward is retained to isolate arm compensation. It does not compare PACDM with a different inverse-kinematics method. Three timesteps cover the full cycle."),
        ("Negative controls and acceptance", "pin_validation/mechanics.py; run_validation.py; results/validation.json",
         "Wrong tool offset, wrong joint axis, omitted acceleration curvature and omitted armature must be detectable. All acceptance values in this report come from the saved aggregate checks defined by run_validation.py; failed completed runs remain marked FAIL."),
        ("Recorded-state video", "pin_validation/render.py; results/render_metadata.json; results/pinocchio_validation.mp4",
         "VTK draws source meshes using Pinocchio FK of saved integrated q. Robot meshes never use reference q. The path and error overlays use the saved log; a rendered scene does not add contact or object dynamics."),
    ]
    method_html = "".join(f"<article class='method'><h3>{_escape(title)}</h3><p>{_escape(text)}</p>"
                          f"<p class='code'>{_escape(evidence)}</p></article>" for title, evidence, text in methods)

    video = None
    video_issues = []
    metadata_path = results/"render_metadata.json"
    if metadata_path.exists():
        candidate = _load(metadata_path)
        video_path = results/candidate.get("video", "missing")
        if candidate.get("source_case") != "nominal":
            video_issues.append("render provenance is not for the nominal case")
        if candidate.get("source_trajectory_sha256") != summary["trajectory_SHA256"]["nominal"]:
            video_issues.append("source trajectory hash does not match the current nominal run")
        if candidate.get("cmg_sha256") != _sha(root/"data/panda_cmg.json"):
            video_issues.append("rendered CMG hash does not match the current model")
        if not video_path.is_file():
            video_issues.append("recorded video file is missing")
        elif candidate.get("video_sha256") != _sha(video_path):
            video_issues.append("video byte hash is missing or does not match recorded provenance")
        if not video_issues:
            video = candidate
    else:
        video_issues.append("render provenance is missing")
    if video:
        poster = f" poster='{_escape(video['poster'])}'" if (results/video["poster"]).is_file() else ""
        video_html = (f"<video controls preload='metadata'{poster}><source src='{_escape(video['video'])}' type='video/mp4'></video>"
                      f"<p class='caption'>{video['frames']} frames at {video['fps']} fps · "
                      f"{_escape(' × '.join(map(str, video['resolution'])))} pixels. "
                      "The displayed motion is the recorded nominal integrated trajectory. "
                      f"<a href='{_escape(video['video'])}'>Open MP4</a> · <a href='render_metadata.json'>Render provenance</a>.</p>")
        video_md = f"[Nominal video]({video['video']}) · [render provenance](render_metadata.json). The source trajectory, CMG and video byte hashes match the current evidence."
    else:
        reason = "; ".join(video_issues)
        video_html = ("<p class='notice'>Video omitted: "+_escape(reason)+". Numerical acceptance and video validity are separate. "
                      "Run <code>python run_validation.py --render-only</code>, then <code>python -m pin_validation.report</code>.</p>")
        video_md = "Video omitted: "+reason+". Numerical acceptance and video validity are separate."
    positive_names = [name for name in cases if name != "no_feedforward"]
    positive_max = max(cases[name]["max_tool_position_error_m"] for name in positive_names)
    abi = summary["checks"]["ablation.feedforward_RMS_improvement"]["value"]
    environment_rows = [[k, v] for k, v in summary["environment"].items()]
    gate_names = ["nominal.tool_position", "nominal.tool_rotation", "nominal.joint_tracking",
                  "nominal.final_position", "nominal.energy_work_balance", "nominal.no_saturation",
                  "refinement.nominal.tool_position", "refinement.nominal.tool_rotation",
                  "refinement.nominal.joint_position", "ablation.feedforward_RMS_improvement"]
    gate_rows = [[name, checks[name]["relation"]+" "+_num(checks[name]["limit"]), checks[name].get("unit", "")]
                 for name in gate_names if name in checks]
    links = [("validation.json", "Complete acceptance checks"), ("mechanics.json", "Independent mechanics"),
             ("task_validation.json", "Route and interpolation validation"), ("performance.png", "Numerical figure"),
             ("VALIDATION_REPORT.md", "Concise report"), ("../data/reference.npz", "PACDM reference"),
             ("../data/panda_cmg.json", "CMG model")]
    evidence_html = " · ".join(f"<a href='{path}'>{_escape(label)}</a>" for path, label in links)
    failure_html = (_table(check_headers, failed_rows) if failed_rows else
                    "<p>No failed acceptance checks in this completed run.</p>")
    css = """
    :root{--ink:#203748;--muted:#617585;--line:#d9e3e9;--accent:#087f8c;--bg:#eff4f7}
    *{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
    main{max-width:1250px;margin:0 auto;padding:32px 24px 70px}header{background:#132b3c;color:#f4f8fa;border-radius:16px;padding:35px 38px}
    .eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:12px;color:#9cd4d9;font-weight:700}h1{font-size:38px;line-height:1.18;margin:12px 0}h2{font-size:24px;margin:0 0 18px}h3{font-size:17px;margin:0 0 8px}
    header p{color:#c2d3df;max-width:940px;margin-bottom:0}.badge{display:inline-block;border-radius:30px;padding:5px 13px;font-size:14px;font-weight:800;background:#087f68;color:white}.badge.fail{background:#ab3734}
    section{background:white;border:1px solid var(--line);border-radius:13px;padding:27px 30px;margin-top:22px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:22px}.metric{background:white;border:1px solid var(--line);border-radius:12px;padding:19px}.metric strong{display:block;font-size:25px;color:var(--accent);line-height:1.25}.metric span{font-size:13px;color:var(--muted)}
    p{margin:10px 0}a{color:#076f82;text-underline-offset:3px}.caption,.small{font-size:13px;color:var(--muted)}.notice{background:#fff5da;border-left:4px solid #d49926;padding:13px 17px}.scope{padding-left:22px}.scope li{margin:6px 0}
    video,img{width:100%;max-width:100%;border-radius:8px;display:block}video{background:#0b1521}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;background:#eaf1f5;font-weight:700;white-space:nowrap}td,th{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}tr:nth-child(even) td{background:#f8fafb}td:first-child{font-weight:600}
    .methods{display:grid;grid-template-columns:1fr 1fr;gap:20px}.method{padding:17px;border:1px solid var(--line);border-radius:9px}.method p{font-size:14px}.code,code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.code{color:#527084}details{margin-top:20px}summary{cursor:pointer;font-weight:700}pre{overflow-x:auto;background:#142e40;color:#eaf4f8;padding:18px;border-radius:8px;font-size:13px}.hash{word-break:break-all;font-family:monospace;font-size:13px}footer{margin-top:24px;color:var(--muted);font-size:12px;text-align:center}
    @media(max-width:800px){main{padding:14px 10px 35px}header,section{padding:22px 18px}h1{font-size:29px}.metrics{grid-template-columns:1fr 1fr}.methods{grid-template-columns:1fr}.metric strong{font-size:21px}}
    @media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}.table-scroll{overflow:visible}details{display:block}video{max-height:350px}a{color:inherit}}
    """
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panda PACDM / Pinocchio — measured validation</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">Reproducible numerical experiment · Panda</div><h1>PACDM validation in Pinocchio</h1>
<span class="badge {'fail' if not summary['passed'] else ''}">{status} · {passed_count}/{len(checks)} checks</span>
<p>The unchanged PACDM reference drives an independently integrated Pinocchio robot through the complete {duration}-second route. This report links every conclusion to saved numerical evidence.</p></header>
<div class="metrics"><div class="metric"><strong>{_num(positive_max*1e3)} mm</strong><span>Largest tool-position error across positive cases</span></div>
<div class="metric"><strong>{_num(nominal['max_tool_orientation_error_deg'])}°</strong><span>Nominal maximum orientation error</span></div>
<div class="metric"><strong>{_num(abi)}×</strong><span>Nominal RMS improvement with arm feedforward</span></div>
<div class="metric"><strong>{len(cases)} × {duration} s</strong><span>Completed full-cycle experiments</span></div></div>
<section><h2>What this validates</h2><ul class="scope">{''.join('<li>'+_escape(s)+'</li>' for s in scope)}</ul>
<p class="notice">The Panda arm is a serial chain. PACDM's tool–target closure is an imposed task; only the finger equality is a permanent physical constraint. Equivalent load in kilograms specifies a downward tool force, not a simulated payload body.</p></section>
<section><h2>Video of integrated states</h2>{video_html}</section>
<section><h2>Measured performance</h2><img src="performance.png" alt="Tracking, energy–work balance, timestep refinement and source actuator effort ratios">
<p class="caption">Tracking and energy-balance plots include every case, including the compensation ablation. Refined curves may overlap. Symmetric logarithmic axes preserve zero values with a linear region around zero. Timestep differences compare numerical trajectories and do not establish physical positioning precision. Plot samples are stored every {_num(nominal['stored_spacing_s']*1e3)} ms; acceptance maxima are measured at every integration step. Saturation and setpoint counts include evaluated RK4 stages.</p></section>
<section><h2>Full-cycle experiments</h2>{_table(case_headers,case_rows)}
<p class="small">Positive cases are nominal, both refinements, higher tool force and initial offset. The no-arm-feedforward case retains finger compensation; it is an ablation and must exceed the positive position gate while meeting its separately stated dynamics, limits and energy checks. Source effort magnitudes are {_escape(', '.join(_num(x) for x in effort_limits))}; the first seven are N m and the tendon is N.</p>
<details><summary>Selected acceptance gates</summary>{_table(['Gate','Required','Unit'],gate_rows)}</details></section>
<section><h2>Timestep refinement</h2>{_table(refinement_headers,refinement_rows)}
<p>The second-to-first peak position-difference ratio is <strong>{_num(summary['refinement_position_ratio'])}</strong>. Each comparison uses identical stored physical times throughout the complete cycle. This is measured convergence for this route and integrator.</p></section>
<section><h2>Algorithms and independent evidence</h2><div class="methods">{method_html}</div></section>
<section><h2>Acceptance record</h2>{_table(['Group','Passed','Total','Failed'],count_rows)}<h3 style="margin-top:22px">Failed checks</h3>{failure_html}
<details><summary>All {len(checks)} measured acceptance checks</summary>{_table(check_headers,check_rows)}</details></section>
<section><h2>Reproduce and inspect</h2><p>{evidence_html}</p><pre>python run_validation.py --render --workers 2
python run_validation.py --audit-existing</pre><p class="small">Use the pinned environment. The recorded execution environment is listed below; operating systems not listed have not been verified by this run.</p>{_table(['Component','Recorded version'],environment_rows)}
<h3 style="margin-top:22px">Original PACDM core</h3><p>{'SHA-256 matches the unchanged original core.' if core_unchanged else 'WARNING: the current core SHA-256 does not match the validated original.'}</p>
<p class="hash">{_escape(actual_core_sha)}</p><p class="small">Expected original SHA-256: <span class="hash">{_escape(expected_core_sha)}</span></p>
<p class="small">Reference and every case trajectory were checked against the completed aggregate's hashes before this report was built. Data tables come from saved JSON; figure lines come from the corresponding NPZ arrays.</p></section>
<footer>Finite numerical evidence · report generated from completed local evidence</footer>
</main></body></html>"""
    (results/"report.html").write_text(document, encoding="utf-8")
    md = ["# Panda PACDM / Pinocchio numerical validation", "",
          f"**{status}: {passed_count}/{len(checks)} acceptance checks.** Recorded rigid-body tool-wrench experiment.", "",
          f"{len(cases)} full-cycle experiments, {duration} s each. Nominal peak tool error: **{_num(nominal['max_tool_position_error_m']*1e3)} mm**; orientation error: **{_num(nominal['max_tool_orientation_error_deg'])}°**. Largest positive-case position error: **{_num(positive_max*1e3)} mm**.", "",
          video_md, "", "[Interactive HTML report](report.html) · [Performance figure](performance.png) · [Complete checks](validation.json)", "",
          "## Scope", ""]+["- "+s for s in scope]+["", "Equivalent load specifies a tool force. No payload mass/inertia, collision, frictional grasp, socket interaction or hardware behavior is validated.", "",
          "## Measured cases", "", _markdown_table(case_headers, case_rows), "",
          f"Maxima are measured at integration steps; trajectory files retain {_num(nominal['stored_spacing_s']*1e3)} ms samples. Saturation counts include evaluated RK4 stages. The no-arm-feedforward experiment retains the same finger compensation and is an ablation, not a positive tracking case.", "",
          "## Timestep refinement", "", _markdown_table(refinement_headers, refinement_rows), "",
          f"Peak position-difference contraction ratio: {_num(summary['refinement_position_ratio'])}. Arm feedforward improves nominal position RMS by {_num(abi)}× relative to its ablation. This ablation removes arm inverse-dynamics and modeled-damping compensation while retaining finger compensation; it does not compare PACDM against another IK algorithm.", "",
          "## Acceptance checks", "", _markdown_table(["Group", "Passed", "Total", "Failed"], count_rows), "",
          "Failed checks: "+("none." if not failed else "see table below.")]
    if failed:
        md += ["", _markdown_table(check_headers, failed_rows)]
    md += ["", "## Methods and evidence", ""]
    for title, evidence, description in methods:
        md += [f"- **{title}:** {description} Evidence: `{evidence}`."]
    md += ["", "## Provenance and reproduction", "",
           "Original PACDM SHA-256 (actual):", "", f"`{actual_core_sha}`", "",
           "Matches the pinned original: **"+("yes" if core_unchanged else "NO")+"**.", "",
           f"Reference SHA-256: `{summary['reference_SHA256']}`.", "",
           "Recorded environment: "+"; ".join(f"{k} {v}" for k,v in summary["environment"].items())+".", "",
           "```bash", "python run_validation.py --render --workers 2", "python run_validation.py --audit-existing", "```", "",
           "Thresholds are saved in `results/validation.json` and defined by `run_validation.py`. Every trajectory hash was checked against the completed aggregate before report generation. The current report does not replace independent hardware measurement or a contact simulation.", ""]
    (results/"VALIDATION_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    return {"html": results/"report.html", "figure": results/"performance.png",
            "markdown": results/"VALIDATION_REPORT.md", "passed": summary["passed"],
            "matching_video_embedded": video is not None}


if __name__ == "__main__":
    paths = build_report(Path(__file__).resolve().parents[1])
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in paths.items()}, indent=2))
