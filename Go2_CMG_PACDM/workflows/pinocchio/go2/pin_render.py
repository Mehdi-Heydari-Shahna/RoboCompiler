"""Mesh video of saved, integrated Pinocchio states; no simulator is imported.

The authored XML is read *only* for visual mesh/material/local-placement data.
Every robot body transform comes from PinBackend.poses(q).  Rendered frames do
not advance physics and never replace the recorded state with a reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Linux headless; Windows and macOS use the native OpenGL context.
if sys.platform.startswith('linux'):
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# pyrender pins PyOpenGL 3.1.0; ignore an incompatible global accelerator.
import OpenGL
OpenGL.USE_ACCELERATE = False
import pyrender
from scipy.spatial.transform import Rotation
import trimesh

from .pin_backend import PinBackend

PHASES = [(0, 2, 'Balance + push'), (2, 8, 'Precision trot'),
          (8, 14, 'Clear the rails'), (14, 21, 'Turn + crouch'),
          (21, 24, 'Exit'), (24, 26, 'Recover + dock')]
INK = '#edf3f8'
MUTED = '#9cb2c6'
CYAN = '#42d9ce'
AMBER = '#f6bd62'
BG = '#0b1521'
PANEL = '#132434'
LINE = '#2b4255'
LEG_COLORS = ['#42d9ce', '#f6bd62', '#a5b9fa', '#e8a2c8']
LEG_NAMES = ['FL', 'FR', 'RL', 'RR']
LIMITS = np.tile([23.7, 23.7, 45.43], 4)
_FONT_CACHE = {}


def _font(size, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        try:
            _FONT_CACHE[key] = ImageFont.truetype(name, size)
        except OSError:
            import matplotlib
            _FONT_CACHE[key] = ImageFont.truetype(
                str(Path(matplotlib.get_data_path()) / 'fonts/ttf' / name), size)
    return _FONT_CACHE[key]


def _phase(t):
    for i, (a, b, label) in enumerate(PHASES):
        if a <= t < b:
            return i, label
    return len(PHASES)-1, PHASES[-1][2]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(root, case):
    path = root / 'results_pinocchio' / f'{case}.npz'
    with np.load(path, allow_pickle=False) as saved:
        log = {k: saved[k] for k in saved.files}
    for key in ('time', 'q', 'q_ref', 'torque', 'stance', 'support_force'):
        if key not in log:
            raise ValueError(f'Missing saved replay quantity: {key}')
    if len(log['time']) < 2 or not np.all(np.diff(log['time']) > 0):
        raise ValueError('Replay timestamps must increase strictly')
    if not np.all(np.isfinite(log['q'])) or log['q'].shape[1] != 18:
        raise ValueError('Expected finite scalar CMG coordinates of shape (samples, 18)')
    for key in ('q', 'q_ref', 'torque', 'stance', 'support_force'):
        if len(log[key]) != len(log['time']):
            raise ValueError(f'Replay sample count mismatch: {key}')
    return log, path


def _geom_defaults(xml):
    """Resolve inherited authored geom defaults without importing a simulator."""
    table = {'': {}}
    def walk(node, parent):
        attrs = dict(parent)
        geom = node.find('geom')
        if geom is not None:
            attrs.update(geom.attrib)
        table[node.get('class', '')] = attrs
        for child in node.findall('default'):
            walk(child, attrs)
    for node in xml.findall('default'):
        walk(node, {})
    return table


def _local_pose(attrs, degree=False):
    pose = np.eye(4)
    pose[:3, 3] = np.fromstring(attrs.get('pos', '0 0 0'), sep=' ')
    if 'quat' in attrs:
        quat = np.fromstring(attrs['quat'], sep=' ')
        pose[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    elif 'euler' in attrs:
        pose[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(attrs['euler'], sep=' '), degrees=degree).as_matrix()
    return pose


def _material(rgba, metallic=0.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.asarray(rgba, dtype=float), metallicFactor=metallic,
        roughnessFactor=0.55, doubleSided=False)


def _add_box(scene, size, pos, color, yaw=0.):
    pose = np.eye(4)
    pose[:3, 3] = pos
    pose[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.box(size),
              material=_material(color), smooth=False), pose=pose)


def _look_at(eye, target):
    backward = np.asarray(eye)-np.asarray(target)
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, up, backward])
    pose[:3, 3] = eye
    return pose


def _scene(root):
    """Return static scene and (body name, visual node, local transform) records."""
    xml_path = root / 'upstream/unitree_go2/go2.xml'
    xml = ET.parse(xml_path).getroot()
    compiler = xml.find('compiler')
    meshdir = xml_path.parent / compiler.get('meshdir', '')
    degree = compiler.get('angle', 'degree') == 'degree'
    defaults = _geom_defaults(xml)
    materials = {m.get('name'): np.fromstring(m.get('rgba', '.7 .7 .7 1'), sep=' ')
                 for m in xml.findall('./asset/material')}
    asset_meshes = {}
    mesh_paths = []
    for item in xml.findall('./asset/mesh'):
        path = meshdir / item.get('file')
        mesh = trimesh.load_mesh(str(path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.apply_scale(np.fromstring(item.get('scale', '1 1 1'), sep=' '))
        asset_meshes[item.get('name', path.stem)] = mesh
        mesh_paths.append(path)
    scene = pyrender.Scene(bg_color=[0.043, .077, .118, 1.],
                           ambient_light=[.22, .25, .30])
    visuals = []
    def visit(body, inherited_class=''):
        childclass = body.get('childclass', inherited_class)
        for geom in body.findall('geom'):
            attrs = dict(defaults.get(geom.get('class', childclass), {}))
            attrs.update(geom.attrib)
            if attrs.get('type') != 'mesh' or 'mesh' not in attrs:
                continue
            rgba = np.fromstring(attrs['rgba'], sep=' ') if 'rgba' in attrs else materials.get(attrs.get('material'), [.7, .7, .7, 1.])
            material = _material(rgba, 0.45 if attrs.get('material') == 'metal' else .1)
            node = scene.add(pyrender.Mesh.from_trimesh(
                asset_meshes[attrs['mesh']], material=material, smooth=True))
            visuals.append((body.get('name'), node, _local_pose(attrs, degree)))
        for child in body.findall('body'):
            visit(child, childclass)
    for body in xml.findall('./worldbody/body'):
        visit(body)
    # Ground/contact geometry is identical to the benchmark dimensions.
    _add_box(scene, [50., 50., .04], [1., 0., -.02], [.025, .040, .060, 1.])
    for x in np.arange(-3., 6., .25):
        _add_box(scene, [.0012, 7.5, .0001], [x, 0., .0001], [.065, .085, .105, 1.])
    for y in np.arange(-3., 3.01, .25):
        _add_box(scene, [9., .0012, .0001], [1., y, .0001], [.065, .085, .105, 1.])
    for x, h in [(.48, .025), (.86, .035)]:
        _add_box(scene, [.05, 1.2, h], [x, 0., h/2], [.95, .56, .10, 1.])
    _add_box(scene, [.05, 1.15, .04], [1.65, 0., .38], [.20, .62, .70, 1.])
    for y in [-.55, .55]:
        _add_box(scene, [.05, .05, .36], [1.65, y, .18], [.20, .62, .70, 1.])
    _add_box(scene, [.68, .56, .0004], [2.20, .30, .0003], [.08, .35, .38, 1.], yaw=.35)
    for x in np.arange(-.2, 2.6, .15):
        _add_box(scene, [.08, .014, .0002], [x, -.64, .0005], [.25, .60, .70, 1.])
    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(43.), znear=.02, zfar=50.))
    keylight = scene.add(pyrender.DirectionalLight(color=[1., .95, .90], intensity=1.8),
                         pose=_look_at([1., -2., 3.], [0., 0., 0.]))
    scene.add(pyrender.DirectionalLight(color=[.65, .82, 1.], intensity=1.0),
              pose=_look_at([-1., 2., 2.], [0., 0., 0.]))
    return scene, visuals, camera, keylight, mesh_paths, xml_path


def _trace(draw, times, q, qr, t, rect):
    x0, y0, x1, y1 = rect
    start = max(0., t-5.)
    stop = max(5., t)
    ids = np.flatnonzero((times >= start) & (times <= t))
    for height in (.26, .32):
        y = y1-(height-.23)/.12*(y1-y0)
        draw.line((x0, y, x1, y), fill=LINE)
        draw.text((x0+1, y-12), f'{height:.2f}', font=_font(9), fill=MUTED)
    for values, color, width in [(qr, MUTED, 1), (q, CYAN, 2)]:
        points = [(x0+(times[i]-start)/(stop-start)*(x1-x0),
                   y1-np.clip((values[i, 2]-.23)/.12, 0, 1)*(y1-y0)) for i in ids]
        if len(points) > 1:
            draw.line(points, fill=color, width=width)


def _compose(main, t, index, log, width=1280, height=800):
    canvas = Image.new('RGB', (1280, 800), BG)
    d = ImageDraw.Draw(canvas)
    d.text((28, 17), 'CMG–PACDM  /  INDEPENDENT DYNAMICS VALIDATION', font=_font(12, True), fill=CYAN)
    d.text((25, 38), 'PINOCCHIO', font=_font(34, True), fill=INK)
    d.text((296, 54), 'UNITREE GO2  ·  FLOATING BASE + CHANGING CONTACT', font=_font(14, True), fill=MUTED)
    d.text((1121, 35), f'{t:05.2f} s', font=_font(25, True), fill=INK)
    d.line((28, 91, 1252, 91), fill=LINE)
    canvas.paste(Image.fromarray(main), (28, 112))
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((28, 112, 958, 719), radius=10, outline=LINE)
    phase, label = _phase(t)
    pw = int(d.textlength(label.upper(), font=_font(13, True)))+89
    d.rounded_rectangle((45, 128, 45+pw, 175), radius=7, fill=BG)
    d.text((61, 142), f'{phase+1:02d}  /  {label.upper()}', font=_font(13, True), fill=INK)
    d.rounded_rectangle((44, 659, 647, 708), radius=5, fill=BG)
    d.text((57, 670), 'Pinocchio dynamics + independent contact solver', font=_font(12, True), fill=INK)
    d.text((57, 689), 'Saved integrated states  ·  Pinocchio forward kinematics  ·  no MuJoCo physics', font=_font(10), fill=MUTED)
    push = np.asarray(log.get('push', np.zeros((len(log['time']), 6)))[index, :3])
    magnitude = np.linalg.norm(push)
    if magnitude > 1e-8:
        d.rounded_rectangle((45, 185, 300, 225), radius=7, fill='#543e28')
        d.text((60, 197), f'EXTERNAL PUSH  {magnitude:.0f} N', font=_font(13, True), fill=AMBER)
    x = 979
    # The measured support values come only from the integrated simulation log.
    d.rounded_rectangle((x, 112, 1252, 349), radius=10, fill=PANEL)
    d.text((x+17, 131), 'FOOT SUPPORT', font=_font(12, True), fill=INK)
    d.text((x+17, 154), 'Vertical reaction  ·  contact > 2 N', font=_font(10), fill=MUTED)
    for k in range(4):
        yy = 187+k*36
        force = float(log['support_force'][index, k])
        on = force > 2.
        d.ellipse((x+17, yy, x+25, yy+8), fill=LEG_COLORS[k] if on else LINE)
        d.text((x+33, yy-4), LEG_NAMES[k], font=_font(10, True), fill=LEG_COLORS[k])
        d.rounded_rectangle((x+65, yy, x+188, yy+7), radius=3, fill=BG)
        length = 123*np.clip(force/150., 0, 1)
        if length > 0:
            d.rounded_rectangle((x+65, yy, x+65+length, yy+7), radius=3, fill=LEG_COLORS[k])
        d.text((x+198, yy-5), f'{force:3.0f} N', font=_font(11), fill=INK)
        d.text((x+65, yy+13), 'planned stance' if log['stance'][index, k] else 'planned swing', font=_font(8), fill=MUTED)
    d.text((x+17, 329), 'FL / FR / RL / RR  ·  force scale 150 N', font=_font(9), fill=MUTED)
    d.rounded_rectangle((x, 362, 1252, 546), radius=10, fill=PANEL)
    d.text((x+17, 379), 'BODY TRACKING', font=_font(12, True), fill=INK)
    error_mm = np.linalg.norm(log['q'][index, :3]-log['q_ref'][index, :3])*1000.
    d.text((x+17, 404), f'{error_mm:5.2f}', font=_font(25, True), fill=CYAN)
    d.text((x+106, 419), 'mm position error', font=_font(10), fill=MUTED)
    _trace(d, log['time'], log['q'], log['q_ref'], t, (x+17, 455, x+251, 513))
    d.text((x+17, 527), 'Height: integrated / reference (grey)', font=_font(9), fill=MUTED)
    d.rounded_rectangle((x, 559, 1252, 719), radius=10, fill=PANEL)
    d.text((x+17, 575), 'MOTOR LIMIT USAGE', font=_font(12, True), fill=INK)
    ratios = np.max(np.abs(log['torque'][index].reshape(4, 3))/LIMITS.reshape(4, 3), axis=1)
    for k, value in enumerate(ratios):
        yy = 605+20*k
        d.text((x+17, yy-4), LEG_NAMES[k], font=_font(10, True), fill=LEG_COLORS[k])
        d.rounded_rectangle((x+48, yy, x+202, yy+7), radius=3, fill=BG)
        length = 154*np.clip(value, 0, 1)
        if length > 0:
            d.rounded_rectangle((x+48, yy, x+48+length, yy+7), radius=3,
                                fill=AMBER if value > .9 else LEG_COLORS[k])
        d.text((x+211, yy-4), f'{100*value:3.0f}%', font=_font(10), fill=INK)
    d.text((x+17, 697), 'Largest |joint torque| / source limit', font=_font(9), fill=MUTED)
    phasewidth = 1224/len(PHASES)
    for k, (_, _, name) in enumerate(PHASES):
        xx = 28+k*phasewidth
        d.rectangle((xx, 741, xx+phasewidth-6, 745), fill=CYAN if k == phase else LINE)
        d.text((xx, 756), name, font=_font(11, k == phase), fill=INK if k == phase else MUTED)
    progress = 28+1224*np.clip(t/float(log['time'][-1]), 0, 1)
    d.line((28, 790, 1252, 790), fill=LINE)
    d.ellipse((progress-3, 787, progress+3, 793), fill=INK)
    if (width, height) != (1280, 800):
        canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(canvas)


def render(root, case='nominal', *, fps=30, width=1280, height=800, duration=None):
    """Write Go2_Pinocchio.mp4, poster.png and hashed replay provenance.

    Rendering duration is simulation duration (real-time playback). Duration-
    limited previews receive a separate suffix and never replace the main video.
    """
    root = Path(root).resolve()
    if fps <= 0 or (duration is not None and duration <= 0):
        raise ValueError('fps and optional duration must be positive')
    log, source_path = _load(root, case)
    cmg_path = root / 'data/go2_cmg.json'
    backend = PinBackend(json.loads(cmg_path.read_text(encoding='utf-8')))
    scene, visuals, camera, keylight, mesh_paths, xml_path = _scene(root)
    renderer = pyrender.OffscreenRenderer(930, 607)
    suffix = '' if case == 'nominal' else '_'+case
    if duration is not None:
        suffix += '_preview'
    results = root / 'results_pinocchio'
    output = results / f'Go2_Pinocchio{suffix}.mp4'
    poster = results / f'poster{suffix}.png'
    times = log['time']
    stop = min(float(times[-1]), duration) if duration is not None else float(times[-1])
    # Exactly 780 frames for a 26-second, 30-fps recording.
    frame_times = np.arange(float(times[0]), stop-1e-10, 1/fps)
    if not len(frame_times):
        raise ValueError('No frames requested')
    desired = float(times[np.argmin(abs(log['q'][:, 0]-1.65))])
    poster_index = int(np.argmin(abs(frame_times-min(stop, desired))))
    writer = imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8,
                                macro_block_size=None, ffmpeg_params=['-movflags', '+faststart'])
    flags = pyrender.RenderFlags.NONE
    try:
        for frame_index, t in enumerate(frame_times):
            right = min(int(np.searchsorted(times, t)), len(times)-1)
            left = max(0, right-1)
            blend = (t-times[left])/(times[right]-times[left]) if right != left else 0.
            # All 18 coordinates are scalar (XYZ, ZYX Euler, leg hinges), in
            # the regular local chart used for dynamics. No quaternion state.
            q = (1-blend)*log['q'][left]+blend*log['q'][right]
            poses = backend.poses(q)
            for body, node, local in visuals:
                scene.set_pose(node, poses[body] @ local)
            base = poses['base'][:3, 3]
            target = base+np.array([.05, 0., -.075])
            # Constant world orientation keeps rail/gate motion understandable.
            eye = target+np.array([.86, -1.06, .70])
            scene.set_pose(camera, _look_at(eye, target))
            # Keep the directional-light anchor near the robot as the camera follows.
            scene.set_pose(keylight, _look_at(base+[1., -2., 3.], base))
            nearest = left if blend < .5 else right
            main, _ = renderer.render(scene, flags=flags)
            frame = _compose(main, float(t), nearest, log, width, height)
            writer.append_data(frame)
            if frame_index == poster_index:
                Image.fromarray(frame).save(poster)
            if frame_index % (5*fps) == 0:
                print(f'Pinocchio mesh render: {frame_index}/{len(frame_times)} frames', flush=True)
    finally:
        writer.close()
        renderer.delete()
    metadata = dict(
        source_case=case, video=output.name, poster=poster.name,
        source_trajectory=str(source_path.relative_to(root)),
        trajectory_sha256=_sha256(source_path), cmg_sha256=_sha256(cmg_path),
        visual_xml_sha256=_sha256(xml_path),
        visual_mesh_sha256={p.name: _sha256(p) for p in mesh_paths},
        video_sha256=_sha256(output), fps=fps, frames=len(frame_times),
        duration_s=len(frame_times)/fps, resolution=[width, height],
        dynamics_source='Saved states from independently integrated Pinocchio rigid-body dynamics and contact solver',
        pose_source='PinBackend.poses(q): Pinocchio forward kinematics of saved scalar CMG states',
        render_backend=f'pyrender {pyrender.__version__} / OpenGL ({os.environ.get("PYOPENGL_PLATFORM", "native")})',
        no_mujoco_import=True, no_physics_steps_during_render=True,
        visual_asset_source='Original upstream Unitree Go2 OBJ meshes; authored local geom placements and colors parsed directly from XML',
        interpolation='Linear interpolation of the 18 scalar CMG coordinates in their regular Euler/hinge chart; HUD uses nearest saved sample',
        camera='Follows the measured source base body; no robot coordinates changed for framing',
        hud='Only saved integrated-state/force/torque values. Foot-support bars capped visually at 150 N; support indicator threshold 2 N.',
        environment='Rails x=0.48/0.86 m, heights=0.025/0.035 m; gate x=1.65 m, underside=0.36 m; grid and dock are visual annotations')
    (results / f'render_metadata{suffix}.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--case', default='nominal')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--duration', type=float)
    args = parser.parse_args()
    print(render(args.root, args.case, fps=args.fps, duration=args.duration))
