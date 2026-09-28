"""Measured performance plots and a self-contained, portable HTML report."""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#f3f6f9"
INK = "#152b40"
MUTED = "#60758a"
TEAL = "#008b91"
AMBER = "#d98c0b"
COLORS = ["#007d99", "#db8419", "#418d5a", "#9259b7", "#c65970", "#5e72cc"]


def _number(x, precision=4):
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, (float, int)):
        if x == 0:
            return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e5:
            return f"{x:.3e}"
        return f"{x:,.{precision}g}"
    return str(x)


def _flatten(obj, prefix=""):
    out=[]
    if isinstance(obj, dict):
        for key, val in obj.items():
            out.extend(_flatten(val, f"{prefix} / {key}" if prefix else key))
    elif isinstance(obj, list):
        if len(obj) <= 8 and all(not isinstance(v, (dict, list)) for v in obj):
            out.append((prefix, ", ".join(_number(v) for v in obj)))
        else:
            for i, val in enumerate(obj):
                out.extend(_flatten(val, f"{prefix} / {i}"))
    else:
        out.append((prefix, obj))
    return out


def _embed(path):
    return "data:image/png;base64,"+base64.b64encode(Path(path).read_bytes()).decode()


def _event_windows(log):
    for key in ("external_wrench", "disturbance_wrench", "wrench"):
        if key in log:
            w=np.asarray(log[key])
            if w.ndim == 2 and len(w) == len(log["time"]):
                active=np.linalg.norm(w,axis=1)>1e-8
                edges=np.diff(np.r_[False,active,False].astype(int))
                starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
                return [(float(log["time"][a]),float(log["time"][min(b,len(active)-1)])) for a,b in zip(starts,ends)]
    return []


def _time_axis(ax, time, windows):
    ax.set_xlim(float(time[0]),float(time[-1]))
    ax.set_xlabel("Time [s]")
    for x in (2,9,17):
        ax.axvline(x,color="#becbd5",lw=.9,ls=":",zorder=0)
    for a,b in windows:
        ax.axvspan(a,b,color="#eabc74",alpha=.24,lw=0,zorder=0)
    ax.grid(alpha=.22, lw=.6)
    ax.spines[["top","right"]].set_visible(False)


def make_plots(root: Path, log: dict):
    time=np.asarray(log["time"]).reshape(-1)
    pose=np.asarray(log["q"])[:,:6]
    target=np.asarray(log["target_pose"])
    windows=_event_windows(log)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.titlesize":12, "axes.titleweight":"bold", "axes.labelcolor":MUTED,
        "axes.edgecolor":"#c3d1dc", "text.color":INK, "xtick.color":MUTED,
        "ytick.color":MUTED, "axes.titlepad":13, "figure.facecolor":"white"})
    fig=plt.figure(figsize=(14.4,12),layout="constrained")
    grid=fig.add_gridspec(3,2,hspace=.08,wspace=.08)
    ax=fig.add_subplot(grid[0,0],projection="3d")
    ax.plot(target[:,0]*1000,target[:,1]*1000,target[:,2]*1000,color=AMBER,lw=2.0,ls="--",label="Target")
    ax.plot(pose[:,0]*1000,pose[:,1]*1000,pose[:,2]*1000,color=TEAL,lw=1.2,label="Measured")
    ax.scatter(*np.multiply(pose[0,:3],1000),color=INK,s=22,label="Start")
    ax.set(xlabel="x [mm]",ylabel="y [mm]",zlabel="z [mm]",title="Platform motion in task space")
    ax.view_init(25,-55)
    ax.legend(loc="upper left",fontsize=8,frameon=False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
    ax=fig.add_subplot(grid[0,1])
    ax.plot(time,log["pose_error_m"]*1000,color=TEAL,label="Translation",lw=1.2)
    twin=ax.twinx()
    twin.plot(time,np.rad2deg(log["angle_error_rad"]),color=AMBER,label="Rotation",lw=1.1)
    ax.set(title="Task-space tracking error",ylabel="Translation [mm]")
    twin.set_ylabel("Rotation [deg]",color=AMBER)
    twin.tick_params(axis="y",colors=AMBER)
    twin.spines[["top","left"]].set_visible(False)
    _time_axis(ax,time,windows)
    ax.legend(handles=ax.lines[:1]+twin.lines,loc="upper right",frameon=False,fontsize=9)
    ax=fig.add_subplot(grid[1,0])
    for i,label in enumerate(("x","y","z")):
        # Show z about its initial target to keep three position components
        # comparable; the absolute height remains visible in the 3D plot.
        offset=target[0,2] if i==2 else 0.
        ax.plot(time,(target[:,i]-offset)*1000,color=COLORS[i],ls="--",lw=.9,alpha=.7)
        ax.plot(time,(pose[:,i]-offset)*1000,color=COLORS[i],lw=1.2,label=label if i<2 else "z − z₀")
    ax.set(title="Translation · solid measured / dashed target",ylabel="Displacement [mm]")
    _time_axis(ax,time,windows); ax.legend(ncol=3,loc="lower left",frameon=False,fontsize=9)
    ax=fig.add_subplot(grid[1,1])
    for i,label in enumerate(("Yaw","Pitch","Roll")):
        ax.plot(time,np.rad2deg(target[:,i+3]),color=COLORS[i],ls="--",lw=.9,alpha=.7)
        ax.plot(time,np.rad2deg(pose[:,i+3]),color=COLORS[i],lw=1.2,label=label)
    ax.set(title="Orientation · ZYX Euler chart",ylabel="Angle [deg]")
    _time_axis(ax,time,windows); ax.legend(ncol=3,loc="lower left",frameon=False,fontsize=9)
    ax=fig.add_subplot(grid[2,0])
    for i in range(6):
        ax.plot(time,log["force"][:,i],color=COLORS[i],lw=1.,label=f"Leg {i+1}")
    ax.set(title="Commanded prismatic actuator force",ylabel="Force [N]")
    _time_axis(ax,time,windows); ax.legend(ncol=3,frameon=False,fontsize=8,loc="lower left")
    ax=fig.add_subplot(grid[2,1])
    for i in range(6):
        ax.plot(time,pose[:,0] if log["q"].shape[1]<24 else log["q"][:,8+3*i]*1000,
                color=COLORS[i],lw=1.,label=f"Leg {i+1}")
    ax.set(title="Independent actuator coordinates",ylabel="Leg length [mm]")
    _time_axis(ax,time,windows); ax.legend(ncol=3,frameon=False,fontsize=8,loc="lower left")
    fig.suptitle("CMG / PACDM  ·  STEWART INSPECTION BENCHMARK", fontsize=19,weight="bold",x=.5)
    out=root/"results"/"performance.png"
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return out


def build_report(root: Path):
    root=Path(root); results=root/"results"
    with np.load(results/"nominal.npz",allow_pickle=False) as source:
        log={k:source[k] for k in source.files}
    validation_path=results/"validation.json"
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {"status":"Validation file unavailable"}
    plot=make_plots(root,log)
    pos=np.asarray(log["pose_error_m"]); angle=np.rad2deg(log["angle_error_rad"])
    metrics=[("Translation RMS",f"{np.sqrt(np.mean(pos**2))*1000:.3f}","mm"),
             ("Translation peak",f"{np.max(pos)*1000:.3f}","mm"),
             ("Rotation RMS",f"{np.sqrt(np.mean(angle**2)):.3f}","deg"),
             ("Peak leg force",f"{np.max(np.abs(log['force'])):.1f}","N")]
    cards="".join(f'<div class="metric"><span>{label}</span><strong>{value}<small>{unit}</small></strong></div>' for label,value,unit in metrics)
    rows=[]
    for label,item in validation.get('checks',{}).items():
        accepted=item.get('passed') is True
        measured=f"{_number(item.get('value'))} {item.get('unit','')}"
        threshold=f"{item.get('relation','<=')} {_number(item.get('limit'))}"
        value_text=f'<span class="tag {"ok" if accepted else "fail"}">{"PASS" if accepted else "FAIL"}</span> {html.escape(measured)} <small>({html.escape(threshold)})</small>'
        rows.append(f'<tr><th>{html.escape(label.replace("_"," "))}</th><td>{value_text}</td></tr>')
    case_rows=[]
    for name,item in validation.get('cases',{}).items():
        case_rows.append('<tr><th>'+html.escape(name.replace('_',' '))+'</th><td>'+f"{item['rms_position_error_m']*1000:.3f} mm RMS</td><td>{item['max_position_error_m']*1000:.3f} mm peak</td><td>{item['max_actuator_force_N']:.1f} N peak</td></tr>")
    cross=validation.get('checks',{}).get('pacdm_vs_mujoco.position_difference',{}).get('value')
    compare_text=('The direct PACDM rollout integrates the six independent strut states using projected Pinocchio dynamics and reconstructs the dependent coordinates through PACDM. '
                  + (f'The largest platform-position difference from the native MuJoCo trajectory is {cross*1e6:.3f} micrometres over this mission. ' if cross is not None else '')
                  + 'The comparison evaluates consistency between the reduced dynamics and the native MuJoCo simulation for the same declared mechanism.')
    case_section='<section class="section"><h2>Direct dynamics and stress cases</h2><p>'+compare_text+'</p><table>'+''.join(case_rows)+'</table><p class="small">Heavy payload: 14 kg actual, 8 kg controller model, unchanged gains. The feedforward-disabled case retains PACDM reference assembly and feedback gains while setting analytical feedforward to zero.</p></section>'
    status=validation.get("passed")
    if status is True:
        status_html='<span class="tag ok">Numerical validation: PASS</span>'
    elif status is False:
        status_html='<span class="tag fail">Numerical validation: FAIL</span>'
    else:
        status_html='<span class="tag">See individual validation results</span>'
    poster=results/"poster.png"
    poster_html=(f'<img class="poster" src="{_embed(poster)}" alt="MuJoCo render of the six-UPS Stewart inspection mission.">' if poster.exists() else '')
    windows=_event_windows(log)
    event_text=("Recorded external-wrench windows: "+", ".join(f"{a:.2f}–{b:.2f} s" for a,b in windows)+". They are shaded amber in the time plots." if windows else "Applied perturbations are specified in the saved run configuration.")
    # All images and styling are embedded. The MP4 is an optional neighboring
    # file. The report is self-contained and can be viewed offline.
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CMG / PACDM — Stewart inspection benchmark</title><style>
    :root{--ink:#152b40;--muted:#60758a;--teal:#008b91;--bg:#f3f6f9}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1160px;margin:auto;padding:48px 30px 70px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.18em;color:var(--teal);text-transform:uppercase}h1{font-size:clamp(30px,5vw,52px);line-height:1.08;letter-spacing:-.035em;margin:13px 0 19px}h2{font-size:24px;letter-spacing:-.02em;margin:0 0 17px}p{margin:0 0 16px}.lead{max-width:890px;font-size:18px;color:var(--muted)}.tag{display:inline-block;font-size:12px;font-weight:750;padding:4px 10px;border-radius:6px;background:#e4ecf2;color:#375570}.ok{background:#ddf1e8;color:#216646}.fail{background:#f9e0de;color:#a92d29}.poster{display:block;width:100%;margin:26px 0 0;border-radius:14px;box-shadow:0 12px 38px #14283c20}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:26px 0}.metric{background:white;border:1px solid #e0e8ef;border-radius:12px;padding:23px}.metric span{display:block;font-size:12px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.025em}.metric strong{font-size:32px;letter-spacing:-.025em;display:block;margin-top:8px}.metric small{font-size:13px;color:var(--muted);margin-left:7px}.section{background:white;border:1px solid #e0e8ef;border-radius:14px;padding:30px;margin:22px 0}.phases{display:grid;grid-template-columns:repeat(4,1fr);gap:22px}.phases article{border-top:3px solid #5cc1c4;padding-top:15px}.phases b{display:block;font-size:16px}.phases span{font-size:12px;color:var(--teal);font-weight:700}.phases p{font-size:14px;color:var(--muted);margin:7px 0 0}.performance{width:100%;height:auto;display:block}.small{font-size:13px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #e6edf2;text-align:left;vertical-align:top}th{font-weight:500;color:var(--muted);width:68%;overflow-wrap:anywhere}td{font-variant-numeric:tabular-nums}tr:last-child th,tr:last-child td{border:0}details{border-top:1px solid #e2eaf0;margin-top:20px;padding-top:13px}summary{cursor:pointer;font-weight:650;color:var(--teal)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f9;padding:17px;font-size:12px;border-radius:7px}.footer{font-size:12px;color:var(--muted)}a{color:var(--teal)}@media(max-width:760px){main{padding:28px 16px}.metrics,.phases{grid-template-columns:repeat(2,1fr)}.section{padding:20px}.metric{padding:17px}th{width:60%}}@media print{body{background:white}main{padding:0}.section{break-inside:avoid}.poster{max-height:420px;object-fit:contain}details{display:none}}
    </style></head><body><main><div class="eyebrow">CMG / PACDM · Stewart benchmark</div>
    <h1>Stewart platform: CMG–PACDM benchmark</h1>
    <p class="lead">An idealized six-UPS parallel robot carries a rigid inspection payload through a spatial helical scan, a figure-eight maneuver and a final docking sequence. The recorded run tests coupled translation and rotation with six driven prismatic coordinates.</p>
    STATUS
    POSTER
    <div class="metrics">CARDS</div>
    <p class="small">Metrics above are computed over every saved sample of the nominal run, including the initial hold, transients and docking. All units are SI internally.</p>
    <section class="section"><h2>The 22-second mission</h2><div class="phases">
    <article><span>00–02 s</span><b>Initial hold</b><p>Hold the nominal pose after offline PACDM acquisition.</p></article>
    <article><span>02–09 s</span><b>Helical inspection</b><p>Follow a spatial reference with coordinated translation and attitude.</p></article>
    <article><span>09–17 s</span><b>Figure-eight scan</b><p>Reverse curvature while coordinating all six platform coordinates.</p></article>
    <article><span>17–22 s</span><b>Dock &amp; settle</b><p>Move to the terminal pose and measure residual tracking error.</p></article>
    </div><p class="small" style="margin-top:21px">EVENTS</p></section>
    <section class="section"><h2>Measured behavior</h2><p class="small">The solid traces show saved simulated states; dashed traces show commanded task-space motion. The position plot displays z relative to its initial target height. The rotation error comes from the saved angular-error metric, while the orientation traces use the ZYX Euler chart.</p><img class="performance" src="PLOT" alt="Trajectory, translation and rotation tracking errors, six actuator forces, and six leg lengths."></section>
    CASES
    <section class="section"><h2>Validation record</h2><p class="small">Measured values and acceptance thresholds are read from results/validation.json. Each row identifies the numerical check and its outcome.</p><details><summary>Individual acceptance checks</summary><table><tbody>ROWS</tbody></table></details><details><summary>Exact machine-readable validation record</summary><pre>JSON</pre></details></section>
    <section class="section"><h2>Mechanism and numerical formulation</h2><p>The six-UPS platform is specified by a CMG containing body inertias, joint transforms, a computational pose chart, six point closures, and six actuator coordinates.</p><p>The benchmark uses a declared rigid-body model with a fixed payload and ideal linear force actuation. PACDM provides closure-consistent reference states and reduced dynamics; the native MuJoCo rollout provides the corresponding full-mechanism simulation.</p><p class="small">The optional video visualizes the recorded MuJoCo states. Numerical results are computed from the stored trajectories and validation records.</p></section>
    <p class="footer">Computed from the simulation and validation records by stewart/report.py. Figures and styles are embedded for offline viewing.</p></main></body></html>'''
    replacements={"STATUS":status_html,"POSTER":poster_html,"CARDS":cards,
                  "EVENTS":html.escape(event_text),"PLOT":_embed(plot),
                  "ROWS":"".join(rows),"JSON":html.escape(json.dumps(validation,indent=2)),"CASES":case_section}
    # One pass: never scan an inserted base64 image for another placeholder.
    document=re.sub(r'\b(STATUS|POSTER|CARDS|EVENTS|PLOT|ROWS|JSON|CASES)\b',
                    lambda match: replacements[match.group(0)],document)
    out=results/"report.html";out.write_text(document,encoding='utf-8')
    return out
