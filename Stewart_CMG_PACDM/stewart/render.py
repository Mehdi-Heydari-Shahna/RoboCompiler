"""Render recorded benchmark states with MuJoCo; this module does not simulate.

All HUD quantities are sampled from the saved run. The two trails show platform
origins unless the model exposes an ``inspection_tip`` or ``probe`` site, in which case they
show that site and its target counterpart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Headless rendering works with Mesa EGL on the reference Linux environment.
# Set MUJOCO_GL before importing this module to select another supported backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = (230, 239, 247, 255)
MUTED = (151, 174, 193, 255)
CYAN = (61, 218, 225, 255)
GOLD = (255, 188, 75, 255)
NAVY = (12, 23, 38, 223)
PHASES = ((0., 2., "01", "Initial hold"), (2., 9., "02", "Helical inspection"),
          (9., 17., "03", "Figure-eight scan"), (17., 22., "04", "Precision docking"))


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        import matplotlib
        bundled=Path(matplotlib.get_data_path())/'fonts'/'ttf'/name
        try:
            return ImageFont.truetype(str(bundled),size)
        except OSError:
            return ImageFont.load_default(size=size)


def _phase(t: float):
    return next((p for p in PHASES if p[0] <= t < p[1]), PHASES[-1])


def _rot(angles):
    """ZYX rotation for angles ordered yaw, pitch, roll."""
    y, p, r = angles
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def _joint_addresses(model, log, root: Path, width: int):
    ids = log.get("coordinate_ids")
    if ids is None:
        cmg = json.loads((root / "data" / "stewart.cmg.json").read_text())
        ids = cmg.get("coordinate_ids")
        if ids is None and "coordinates" in cmg:
            coords = cmg["coordinates"]
            if isinstance(coords, list) and len(coords) == width:
                ids = [c if isinstance(c, str) else c.get("id", c.get("name")) for c in coords]
    if ids is not None:
        addresses = []
        for name in ids:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
            if jid < 0:
                raise ValueError(f"Recorded coordinate {name!r} is not an MJCF joint")
            if model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"Coordinate {name!r} is not a scalar joint")
            addresses.append(int(model.jnt_qposadr[jid]))
        if len(addresses) != width:
            raise ValueError("Coordinate metadata does not match saved q columns")
        return np.asarray(addresses)
    raise ValueError("Cannot map saved q to MJCF: supply coordinate_ids in the NPZ or CMG")


def _connector(scene, a, b, color, radius=.0025):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(b)-a) < 1e-8:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                       np.array([radius, radius, radius]), np.zeros(3), np.eye(3).ravel(),
                       np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a), np.asarray(b))
    scene.ngeom += 1


def _sphere(scene, point, color, radius=.011):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.full(3, radius), np.asarray(point), np.eye(3).ravel(),
                       np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _panel(draw, box):
    draw.rounded_rectangle(box, radius=17, fill=NAVY, outline=(86, 116, 139, 105), width=1)


def _hud(frame, t, index, log, trail_kind, width, height):
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, width, 116), fill=(9, 18, 31, 236))
    draw.rectangle((0, 114, width, 116), fill=(61, 218, 225, 105))
    draw.text((34, 19), "CMG / PACDM", font=_font(16, True), fill=CYAN)
    draw.text((33, 42), "STEWART  /  SIX-AXIS INSPECTION", font=_font(28, True), fill=INK)
    draw.text((35, 82), "6-UPS parallel robot   •   closed-chain dynamics   •   recorded simulation", font=_font(15), fill=MUTED)
    draw.text((width-161, 26), f"{t:05.2f} s", font=_font(27, True), fill=INK)
    draw.text((width-160, 65), "MISSION TIME", font=_font(11, True), fill=MUTED)
    if 'wrench' in log:
        w=np.asarray(log['wrench'][index]); fn=np.linalg.norm(w[:3]); tn=np.linalg.norm(w[3:])
        if fn>.1 or tn>.1:
            _panel(draw,(29,143,427,207))
            draw.text((47,154),'EXTERNAL WRENCH PULSE',font=_font(11,True),fill=GOLD)
            draw.text((47,176),f'{fn:.1f} N force   /   {tn:.1f} N m torque',font=_font(15,True),fill=INK)

    x = width-298
    _panel(draw, (x, 144, width-26, 430))
    phase = _phase(t)
    draw.text((x+20, 162), "PHASE " + phase[2], font=_font(11, True), fill=CYAN)
    draw.text((x+20, 186), phase[3], font=_font(17, True), fill=INK)
    pe = float(log["pose_error_m"][index])*1000
    ae = float(log["angle_error_rad"][index])*180/np.pi
    peak = float(np.max(np.abs(log["force"][index])))
    for yy, label, val, unit in ((229, "TRANSLATION ERROR", pe, "mm"),
                                (293, "ROTATION ERROR", ae, "deg"),
                                (357, "PEAK LEG FORCE", peak, "N")):
        draw.text((x+20, yy), label, font=_font(10, True), fill=MUTED)
        s = f"{val:.3f}" if unit != "N" else f"{val:,.1f}"
        draw.text((x+20, yy+19), s, font=_font(24, True), fill=INK)
        draw.text((width-74, yy+29), unit, font=_font(13), fill=MUTED)

    _panel(draw, (x, 444, width-26, 605))
    draw.text((x+20, 461), "ACTUATOR FORCE", font=_font(10, True), fill=MUTED)
    forces = np.asarray(log["force"][index])
    all_peak = max(1., float(np.nanmax(np.abs(log["force"]))))
    barleft, barwidth = x+35, 194
    for k, f in enumerate(forces):
        yy = 487+k*17
        draw.text((x+16, yy-2), str(k+1), font=_font(10), fill=MUTED)
        draw.rounded_rectangle((barleft, yy, barleft+barwidth, yy+6), radius=3, fill=(62, 80, 98, 170))
        frac = min(abs(float(f))/all_peak, 1.)
        if frac > .002:
            draw.rounded_rectangle((barleft, yy, barleft+barwidth*frac, yy+6), radius=3, fill=CYAN)

    _panel(draw, (28, height-168, 502, height-107))
    draw.line((47, height-143, 77, height-143), fill=CYAN, width=4)
    draw.text((88, height-152), "actual", font=_font(12), fill=INK)
    draw.line((170, height-143, 200, height-143), fill=GOLD, width=3)
    draw.text((211, height-152), "target", font=_font(12), fill=INK)
    draw.text((298, height-152), trail_kind, font=_font(11), fill=MUTED)

    # Timings are taken from the reference mission definition; measured values
    # above are always read from the saved simulation, including perturbations.
    draw.rectangle((0, height-90, width, height), fill=(9, 18, 31, 243))
    usable = width-68
    for start, end, number, label in PHASES:
        left = 34 + usable*start/22
        right = 34 + usable*end/22
        fill = CYAN if start <= t < end or t >= 22 and end == 22 else (62, 82, 101, 255)
        draw.rounded_rectangle((left, height-70, right-6, height-65), radius=2, fill=fill)
        short = {"Initial hold":"Hold", "Helical inspection":"Helical scan", "Figure-eight scan":"Figure-eight scan", "Precision docking":"Dock & settle"}[label]
        draw.text((left, height-51), short, font=_font(12, start <= t < end), fill=INK if start <= t < end else MUTED)
    px = 34+usable*np.clip(t/22, 0, 1)
    draw.ellipse((px-4, height-73, px+4, height-65), fill=INK)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def render_video(root: Path, case: str = "nominal", *, fps: int = 30,
                 width: int = 1280, height: int = 800, duration: float | None = None):
    """Make the MP4/poster from the saved states and their exact scene.

    ``duration`` is an optional quick-preview limit; omit it for the entire run.
    It does not change the simulated states or playback speed.
    """
    root = Path(root)
    results = root / "results"
    with np.load(results / f"{case}.npz", allow_pickle=False) as raw:
        log = {k: raw[k] for k in raw.files}
    required = ("time", "q", "target_pose", "force", "pose_error_m", "angle_error_rad")
    for key in required:
        if key not in log:
            raise ValueError(f"Saved run has no {key!r}; render after running the benchmark")
    times = np.asarray(log["time"]).reshape(-1)
    if len(times) < 2 or not np.all(np.diff(times)>0):
        raise ValueError("Recorded simulation times must increase strictly")
    model = mujoco.MjModel.from_xml_path(str(results / f"{case}.xml"))
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    addresses = _joint_addresses(model, log, root, log["q"].shape[1])
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=5000)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.65
    camera.lookat[:] = [0., 0., .50]
    camera.elevation = -24.
    camera.azimuth = 132.
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip")
    if site < 0:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "probe")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "platform")
    tip_offset = None
    if site >= 0 and body >= 0:
        data.qpos[addresses] = log["q"][0]
        mujoco.mj_forward(model, data)
        tip_offset = data.xmat[body].reshape(3, 3).T @ (data.site_xpos[site]-data.xpos[body])
    actual = np.asarray(log["q"][:, :3]).copy()
    target = np.asarray(log["target_pose"][:, :3]).copy()
    trail_kind = "platform origin"
    if tip_offset is not None:
        actual += np.array([_rot(a) @ tip_offset for a in log["q"][:, 3:6]])
        target += np.array([_rot(a) @ tip_offset for a in log["target_pose"][:, 3:6]])
        trail_kind = "inspection tip"
    stop = min(float(times[-1]), duration) if duration is not None else float(times[-1])
    frame_times = np.arange(float(times[0]), stop+1e-8, 1/fps)
    output = results / ("Stewart_CMG_PACDM_Demo.mp4" if case == "nominal" else f"Stewart_{case}.mp4")
    poster = results / ("poster.png" if case == "nominal" else f"poster_{case}.png")
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8,
                                macro_block_size=None, ffmpeg_params=["-movflags", "+faststart"])
    poster_frame = int(np.argmin(np.abs(frame_times-min(stop*.58, 12.8))))
    try:
        for frame_number, t in enumerate(frame_times):
            right = min(int(np.searchsorted(times, t)), len(times)-1)
            left = max(0, right-1)
            blend = (t-times[left])/(times[right]-times[left]) if right != left else 0.
            q = log["q"][left]*(1-blend)+log["q"][right]*blend
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            camera.azimuth = 132.+12.*np.sin(2*np.pi*t/45.)
            renderer.update_scene(data, camera=camera, scene_option=option)
            # A complete thin reference trace plus a bright four-second actual
            # trace exposes tracking without accumulating an opaque solid knot.
            reference_indices = np.linspace(0, len(times)-1, min(200, len(times)), dtype=int)
            for a, b in zip(reference_indices[:-1], reference_indices[1:]):
                _connector(renderer.scene, target[a], target[b], (1., .68, .19, .37), .0013)
            start = int(np.searchsorted(times, max(times[0], t-4.)))
            trace_indices = np.linspace(start, right, min(110, max(2, right-start+1)), dtype=int)
            for a, b in zip(trace_indices[:-1], trace_indices[1:]):
                _connector(renderer.scene, actual[a], actual[b], (.16, .93, .92, .96), .0025)
            _sphere(renderer.scene, target[right], (1., .7, .24, .98), .009)
            if 'wrench' in log and np.linalg.norm(log['wrench'][right,:3])>.1:
                force=log['wrench'][right,:3];p=data.xpos[body]
                _connector(renderer.scene,p-.20*force/np.linalg.norm(force),p,(1.,.28,.08,1.),.007)
            frame = renderer.render()
            composed = _hud(frame, float(t), right, log, trail_kind, width, height)
            writer.append_data(composed)
            if frame_number == poster_frame:
                Image.fromarray(composed).save(poster)
            if frame_number % (fps*5) == 0:
                print(f"Rendered {frame_number}/{len(frame_times)} frames", flush=True)
    finally:
        writer.close()
        renderer.close()
    (results / "render_metadata.json").write_text(json.dumps({
        "source_case": case, "video": output.name, "poster": poster.name,
        "fps": fps, "frames": len(frame_times), "resolution": [width, height],
        "render_backend": os.environ.get("MUJOCO_GL"),
        "source": "MuJoCo rendering of saved simulation coordinates",
        "trail_point": trail_kind,
        "hud": "Measured simulation values; nearest saved sample at each video time",
        "q_interpolation": "Linear interpolation of saved scalar joint coordinates",
    }, indent=2)+"\n")
    return output
