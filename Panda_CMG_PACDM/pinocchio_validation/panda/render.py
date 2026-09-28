"""Render the saved Panda states; rendering never advances the simulation.

The two views show the same physical state. Free-joint interpolation uses
MuJoCo's tangent-space difference/integration rather than quaternion lerp.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' or sys.platform == 'darwin' else 'egl')

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = '#e6eff6'
MUTED = '#8ca8bd'
CYAN = '#36d4d5'
AMBER = '#ffbb4e'
BG = '#0b1521'
PANEL = '#122232'


def _font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(str(Path('/usr/share/fonts/truetype/dejavu') / name), size)
    except OSError:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf' / name), size)


def phases(log):
    """Use the task's published timeline, with a recorded-label fallback."""
    try:
        from .simulation import PHASES
        return [(float(a), float(b), str(label)) for a, b, label in PHASES]
    except (ImportError, AttributeError):
        t = np.asarray(log['time'])
        values = np.asarray(log.get('phase', np.full(len(t), 'Transfer')))
        cuts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1, len(t)]
        return [(float(t[a]), float(t[b]) if b < len(t) else float(t[-1]), str(values[a]))
                for a, b in zip(cuts[:-1], cuts[1:])]


def _phase(t, timeline):
    for k, (start, end, label) in enumerate(timeline):
        if start <= t < end:
            return k, label
    return len(timeline)-1, timeline[-1][2]


def _connector(scene, a, b, rgba, radius=.0015):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(a)-b) < 1e-7:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                      np.full(3, radius), np.zeros(3), np.eye(3).ravel(), np.array(rgba, np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a), np.asarray(b))
    scene.ngeom += 1


def _load(root, case):
    with np.load(root/'results'/f'{case}.npz', allow_pickle=False) as saved:
        log = {key: saved[key] for key in saved.files}
    for key in ('time', 'qpos', 'tool_pos', 'object_pos', 'normal_force'):
        if key not in log:
            raise ValueError(f'Missing replay quantity {key!r}')
    if len(log['time']) < 2 or not np.all(np.diff(log['time']) > 0):
        raise ValueError('Recorded timestamps must increase strictly')
    return log


def _position_error(log):
    error = np.asarray(log.get('pose_error', log['tool_pos']-log['target_pos']))
    return error if error.ndim == 1 else np.linalg.norm(error[:, :3], axis=1)


def _compose(main, closeup, t, index, log, timeline, width=1280, height=800):
    canvas = Image.new('RGB', (1280, 800), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((29, 19), 'CMG / PACDM  ·  ROBOT GENERALITY', font=_font(12, True), fill=CYAN)
    draw.text((28, 40), 'FRANKA PANDA', font=_font(30, True), fill=INK)
    draw.text((361, 51), 'DEXTEROUS CARTRIDGE TRANSFER', font=_font(16, True), fill=MUTED)
    draw.text((1118, 35), f'{t:05.2f} s', font=_font(26, True), fill=INK)
    draw.line((28, 91, 1252, 91), fill='#294657', width=1)

    canvas.paste(Image.fromarray(main).resize((950, 607), Image.Resampling.LANCZOS), (28, 112))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((28, 112, 978, 719), radius=9, outline='#284256', width=1)
    phase, label = _phase(t, timeline)
    label = label.upper()
    phase_width = min(620, max(280, int(draw.textlength(label, font=_font(13, True)))+75))
    draw.rounded_rectangle((45, 128, 45+phase_width, 175), radius=8, fill=BG)
    draw.text((61, 141), f'{phase+1:02d}  /  {label}', font=_font(13, True), fill=INK)
    draw.rounded_rectangle((44, 684, 516, 709), radius=5, fill=BG)
    draw.text((55, 690), 'MuJoCo replay  ·  arm servos + inverse-dynamics feedforward', font=_font(10), fill=MUTED)

    draw.rounded_rectangle((996, 112, 1252, 389), radius=10, fill=PANEL)
    draw.text((1015, 128), 'MEASURED IN SIMULATION', font=_font(10, True), fill=MUTED)
    contact = int(log.get('pad_contacts', np.zeros(len(log['time'])))[index])
    values = [('PAD NORMAL FORCE', f'{float(log["normal_force"][index]):.1f}', 'N'),
              ('TOOL POSITION ERROR', f'{float(_position_error(log)[index])*1000:.3f}', 'mm'),
              ('PAYLOAD HEIGHT', f'{float(log["object_pos"][index, 2])*1000:.1f}', 'mm')]
    for row, (name, value, unit) in enumerate(values):
        y = 155 + row*70
        draw.text((1015, y), name, font=_font(10, True), fill=MUTED)
        draw.text((1015, y+16), value, font=_font(26, True), fill=INK)
        draw.text((1209, y+29), unit, font=_font(12), fill=MUTED)
    color = CYAN if contact else MUTED
    draw.ellipse((1015, 364, 1021, 370), fill=color)
    draw.text((1031, 358), f'{contact} finger–payload contacts', font=_font(11), fill=color)

    draw.text((999, 409), 'CONTACT CLOSE-UP', font=_font(11, True), fill=CYAN)
    canvas.paste(Image.fromarray(closeup).resize((256, 230), Image.Resampling.LANCZOS), (996, 432))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((996, 432, 1252, 662), radius=8, outline='#284256', width=1)
    draw.text((1000, 677), '7 arm joints + coupled fingers', font=_font(11), fill=MUTED)
    draw.text((1000, 697), 'PACDM task manifold + redundancy', font=_font(10, True), fill=INK)

    pulse = np.asarray(log.get('wrench', np.zeros((len(log['time']), 6)))[index])
    if np.linalg.norm(pulse) > 1e-8:
        draw.rounded_rectangle((650, 183, 960, 238), radius=7, fill='#352716')
        draw.text((666, 192), 'APPLIED PAYLOAD DISTURBANCE', font=_font(10, True), fill=AMBER)
        draw.text((666, 210), f'{np.linalg.norm(pulse[:3]):.2f} N  /  {np.linalg.norm(pulse[3:]):.3f} N m',
                  font=_font(13, True), fill=INK)

    draw.line((28, 741, 1252, 741), fill='#294657', width=1)
    # Equal-width labels keep short release phases legible. The white progress
    # marker remains proportional to physical time along the lower rule.
    duration = timeline[-1][1]
    for k, (_, _, label) in enumerate(timeline):
        left = 28 + 1224*k/len(timeline)
        right = 28 + 1224*(k+1)/len(timeline)
        draw.rounded_rectangle((left, 753, right-6, 757), radius=2,
                               fill=CYAN if k == phase else '#2c4355')
        label = label.replace(' / ', '/').replace(' and ', ' + ')
        max_width = right-left-10
        while draw.textlength(label, font=_font(10, k == phase)) > max_width and len(label) > 8:
            label = label[:-2].rstrip()+'…'
        draw.text((left, 765), label, font=_font(10, k == phase), fill=INK if k == phase else MUTED)
    progress = 28 + 1224*np.clip(t/duration, 0, 1)
    draw.line((28, 792, 1252, 792), fill='#294657', width=1)
    draw.ellipse((progress-3, 789, progress+3, 795), fill=INK)
    if (width, height) != (1280, 800):
        canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(canvas)


def render_video(root: Path, case='nominal', *, fps=30, width=1280, height=800, duration=None):
    """Write results/demo.mp4 and poster.png from native simulation states.

    ``duration`` produces a time-limited preview without changing replay speed.
    Non-nominal cases use separate filenames to preserve the principal result.
    """
    root = Path(root)
    results = root/'results'
    log = _load(root, case)
    timeline = phases(log)
    times = np.asarray(log['time'])
    model = mujoco.MjModel.from_xml_path(str((results/f'{case}.xml').resolve()))
    model.vis.global_.offwidth = max(950, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(607, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=607, width=950, max_geom=5000)
    detail = mujoco.Renderer(model, height=460, width=512, max_geom=3000)
    option = mujoco.MjvOption()
    for flag in (mujoco.mjtVisFlag.mjVIS_CONSTRAINT, mujoco.mjtVisFlag.mjVIS_JOINT,
                 mujoco.mjtVisFlag.mjVIS_ACTUATOR):
        option.flags[flag] = False
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.25, 0., .34]
    camera.distance = 1.62
    camera.azimuth = 155
    camera.elevation = -27
    close = mujoco.MjvCamera()
    close.type = mujoco.mjtCamera.mjCAMERA_FREE
    close.distance = .36
    close.elevation = -17
    close.azimuth = 145
    suffix = '' if case == 'nominal' else '_'+case
    output = results/f'demo{suffix}.mp4'
    poster = results/f'poster{suffix}.png'
    stop = min(float(times[-1]), duration) if duration is not None else float(times[-1])
    frame_times = np.arange(float(times[0]), stop+1e-8, 1/fps)
    # Choose an airborne state rather than a stationary setup frame.
    carried = np.flatnonzero(log['object_pos'][:, 2] > log['object_pos'][0, 2]+.12)
    desired = float(times[carried[len(carried)//2]]) if len(carried) else stop*.5
    poster_index = int(np.argmin(np.abs(frame_times-min(stop, desired))))
    tangent = np.zeros(model.nv)
    writer = imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8,
                               macro_block_size=None, ffmpeg_params=['-movflags', '+faststart'])
    try:
        for frame_index, t in enumerate(frame_times):
            right = min(int(np.searchsorted(times, t)), len(times)-1)
            left = max(0, right-1)
            blend = (t-times[left])/(times[right]-times[left]) if right != left else 0.
            data.qpos[:] = log['qpos'][left]
            mujoco.mj_differentiatePos(model, tangent, 1., log['qpos'][left], log['qpos'][right])
            mujoco.mj_integratePos(model, data.qpos, tangent, blend)
            if 'qvel' in log:
                data.qvel[:] = (1-blend)*log['qvel'][left]+blend*log['qvel'][right]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            if t >= 4:
                start = np.searchsorted(times, max(4., t-4.))
                indices = np.linspace(start, right, min(100, max(2, right-start+1)), dtype=int)
                for a, b in zip(indices[:-1], indices[1:]):
                    _connector(renderer.scene, log['object_pos'][a], log['object_pos'][b],
                               (.15, .82, .85, .55), .0012)
            main = renderer.render().copy()
            # Tool site positions are logged in world coordinates. The close-up
            # follows the physical gripper, including tracking error and slip.
            close.lookat[:] = ((1-blend)*log['tool_pos'][left]+blend*log['tool_pos'][right])
            close.lookat[2] += .025
            if 'tool_R' in log:
                # Maintain a comparable finger view while the payload turns.
                # This rotates the camera only; both views retain one state.
                tool_R = log['tool_R'][right]
                close.azimuth = 145 + np.rad2deg(np.arctan2(tool_R[1, 0], tool_R[0, 0]))
            detail.update_scene(data, camera=close, scene_option=option)
            frame = _compose(main, detail.render().copy(), float(t), right, log, timeline, width, height)
            writer.append_data(frame)
            if frame_index == poster_index:
                Image.fromarray(frame).save(poster)
            if frame_index % (fps*5) == 0:
                print(f'Rendered {frame_index}/{len(frame_times)} frames', flush=True)
    finally:
        writer.close()
        renderer.close()
        detail.close()
    (results/f'render_metadata{suffix}.json').write_text(json.dumps(dict(
        source_case=case, video=output.name, poster=poster.name, fps=fps,
        frames=len(frame_times), resolution=[width, height], render_backend=os.environ.get('MUJOCO_GL'),
        source='MuJoCo replay of saved native states; no simulation steps during rendering',
        interpolation='MuJoCo manifold difference/integration, including free-joint quaternions',
        trail='Recorded payload origin over the latest four seconds after t = 4 s',
        hud='Nearest saved sample at each video timestamp',
        closeup='The same saved state as the main viewport, with a camera following the measured tool site'),
        indent=2)+'\n', encoding='utf-8')
    return output
