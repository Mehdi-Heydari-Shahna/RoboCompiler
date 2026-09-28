"""Evidence video of the saved Kangaroo Pinocchio trajectory, rendered with VTK.

MuJoCo is neither imported nor run.  The upstream XML files are read as XML
only, exactly as the v22 exporter reads them, for the visual STL file of each
body, its mesh scale and its colour.  Every moving mesh is placed by
``FloatingPinBackend.poses`` of a saved assembled state (pelvis pose and all
140 PACDM graph coordinates stored at that step).  Frame times are exact
multiples of the 5 ms storage period, so no state is interpolated or taken
from the reference.  Contact arrows show the saved rigid-contact impulses
divided by the step.

Run from the package root::

    python -m kangaroo_pin.pin_render
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW', 'vtkEGLRenderWindow')

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kangaroo_pin.pin_backend import FloatingPinBackend  # noqa: E402

W, H = 1600, 1000
VIEW_W, VIEW_H = 960, 790
BG, PANEL, INK, MUTED, RULE = '#0b1521', '#122232', '#edf4f8', '#95aec1', '#2b4355'
CYAN, AMBER, BLUE, RED, GREEN = '#37d4d8', '#ffb760', '#62caff', '#ff5a5a', '#56e39f'
# Replay segments (start, end, playback speed).  Every frame spacing
# speed / fps is a multiple of the 5 ms storage period.
SEGMENTS = [(0., .4, .15), (.4, 1.4, .6), (1.4, 6.4, .75), (6.4, 7.6, .45), (7.6, 10., .75)]
STAGES = [(0., .101, '01  RELEASE & FALL', 'Both feet start 5 cm above the floor; the pelvis is free.'),
          (.101, 1.4, '02  LAND & ABSORB', 'Rigid unilateral contact at the eight foot-box corners; '
                                         'the leg drives brake the descent.'),
          (1.4, 2.6, '03  CONTROLLED CROUCH', 'Lower the pelvis by 12 cm with the twelve linear drives.'),
          (2.6, 5.3, '04  SHIFT & TURN', 'Shift the weight to both sides and rotate the pelvis.'),
          (5.3, 6.4, '05  RISE', 'Extend the legs and return to the standing height.'),
          (6.4, 7., '06  SIDEWAYS PUSH', 'A 50 N peak sideways load acts at the torso centre of mass.'),
          (7., 1e9, '07  RECOVER', 'Ground reactions and the drives restore balance.')]
DRIVE_LABELS = ['Hip yaw', 'Hip drive 2', 'Hip drive 3', 'Leg length', 'Ankle 4', 'Ankle 5']
_FONTS = {}


def _font(size, bold=False):
    key = size, bold
    if key not in _FONTS:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        for path in [Path('/usr/share/fonts/truetype/dejavu') / name,
                     Path('C:/Windows/Fonts') / ('arialbd.ttf' if bold else 'arial.ttf'),
                     Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf')]:
            if path.exists():
                _FONTS[key] = ImageFont.truetype(str(path), size)
                break
        else:
            try:
                _FONTS[key] = ImageFont.truetype(name, size)
            except OSError:
                _FONTS[key] = ImageFont.load_default(size=size)
    return _FONTS[key]


def _rgb(hex_string):
    return tuple(int(hex_string[i:i + 2], 16) / 255. for i in (1, 3, 5))


def frame_schedule(fps=30):
    frames = []
    for lo, hi, speed in SEGMENTS:
        count = round((hi - lo) / speed * fps)
        if abs(count * speed / fps - (hi - lo)) > 1e-9:
            raise ValueError('Replay segment is not an integer number of frames')
        frames.extend((round(lo + i * speed / fps, 9), speed) for i in range(count))
    frames.append((SEGMENTS[-1][1], SEGMENTS[-1][2]))
    return frames


def visual_geoms(upstream):
    """Visual geoms per body, parsed exactly as v22 ``native_model.export`` does."""
    visuals = {}
    for side in ['left', 'right']:
        text = (upstream / f'parts/kangaroo.{side}_leg.xml').read_text()
        for k in ['4', '5']:
            text = text.replace(f'<!--body name="{side}_{k}_ankle_ball"', f'<body name="{side}_{k}_ankle_ball"')
        text = text.replace('</body-->', '</body>')
        for body in ET.fromstring(text).iter('body'):
            visuals[body.get('name')] = list(body.findall('geom'))
    for body in ET.parse(upstream / 'kangaroo.xml').getroot().find('worldbody').iter('body'):
        visuals[body.get('name')] = list(body.findall('geom'))
    meshes = {m.get('name'): m for m in ET.parse(upstream / 'assets/kangaroo.visual_assets.xml').getroot()
              if m.tag == 'mesh'}
    out = []
    for body, geoms in visuals.items():
        for geom in geoms:
            if geom.get('class') != 'visual':
                continue
            if any(k in geom.attrib for k in ('pos', 'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
                raise ValueError(f'Unsupported local visual placement on {body}')
            asset = meshes[geom.get('mesh')]
            out.append(dict(body=body, file=upstream / 'assets' / asset.get('file'),
                            scale=np.fromstring(asset.get('scale', '1 1 1'), sep=' '),
                            rgba=np.fromstring(geom.get('rgba', '.7 .7 .7 1'), sep=' ')))
    return out


class Scene:
    def __init__(self, root, backend, cmg, corner_bodies, corner_local):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        self.vtk, self._np = vtk, vtk_to_numpy
        self.backend = backend
        self.corner_bodies, self.corner_local = corner_bodies, corner_local
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*_rgb('#101d2a'))
        self.renderer.SetBackground2(*_rgb('#2a4256'))
        self.renderer.GradientBackgroundOn()
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(VIEW_W, VIEW_H)
        # Software OpenGL: FXAA instead of 4x MSAA (about 3.5x faster, similar edges).
        self.window.SetMultiSamples(0)
        self.renderer.UseFXAAOn()
        self.window.AddRenderer(self.renderer)
        records = {j['id']: j for j in cmg['joints']}
        ports = {records[a['joint']]['follower_body']: a['id'] for a in cmg['actuators']}
        self.actors = []
        cache = {}
        upstream = root / 'original_v22/upstream/hucebot/kangaroo_mujoco'
        for geom in visual_geoms(upstream):
            if geom['body'] not in backend.body_frame_ids:
                continue  # upstream bodies merged or absent in the CMG are not drawn
            key = (str(geom['file']), tuple(geom['scale']))
            if key not in cache:
                reader = vtk.vtkSTLReader()
                reader.SetFileName(str(geom['file']))
                transform = vtk.vtkTransform()
                transform.Scale(*geom['scale'])
                moved = vtk.vtkTransformPolyDataFilter()
                moved.SetInputConnection(reader.GetOutputPort())
                moved.SetTransform(transform)
                last = moved
                if np.prod(geom['scale']) < 0:  # mirrored mesh: restore outward winding
                    reverse = vtk.vtkReverseSense()
                    reverse.SetInputConnection(moved.GetOutputPort())
                    last = reverse
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputConnection(last.GetOutputPort())
                normals.SetFeatureAngle(35.)
                normals.Update()
                if normals.GetOutput().GetNumberOfPoints() == 0:
                    raise ValueError(f"Empty visual mesh {geom['file']}")
                cache[key] = normals.GetOutput()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(cache[key])
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            body = geom['body']
            color = geom['rgba'][:3]
            if body in ports:
                color = _rgb(AMBER) if body.startswith('left') else _rgb(BLUE)
            else:
                color = np.maximum(color, .26)
            prop = actor.GetProperty()
            prop.SetColor(*color)
            prop.SetAmbient(.2)
            prop.SetDiffuse(.78)
            prop.SetSpecular(.25)
            prop.SetSpecularPower(30.)
            self.renderer.AddActor(actor)
            self.actors.append((body, actor))
        self.mesh_count = len(self.actors)
        self.mesh_files = len(cache)
        self._floor()
        self.corners = []
        for _ in corner_bodies:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(.009)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
            self.corners.append(actor)
        self.arrows = [self._arrow(_rgb(GREEN)) for _ in range(2)] + [self._arrow(_rgb(RED))]
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(0., 0., .64)
        camera.SetPosition(2.3, -2.3, 1.32)
        camera.SetViewUp(0., 0., 1.)
        camera.SetViewAngle(30.)
        self.renderer.RemoveAllLights()
        for position, intensity in [((2., -2.5, 3.), .85), ((-2., 1.5, 2.), .45), ((0., 0., 4.), .35)]:
            light = vtk.vtkLight()
            light.SetLightTypeToSceneLight()
            light.SetPosition(*position)
            light.SetFocalPoint(0., 0., .5)
            light.SetIntensity(intensity)
            self.renderer.AddLight(light)
        self.capture = vtk.vtkWindowToImageFilter()
        self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGB()
        self.capture.ReadFrontBufferOff()

    def _floor(self):
        vtk = self.vtk
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-1.2, -1.2, 0.)
        plane.SetPoint1(1.2, -1.2, 0.)
        plane.SetPoint2(-1.2, 1.2, 0.)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*_rgb('#c9d2dc'))
        actor.GetProperty().SetAmbient(.45)
        actor.GetProperty().SetDiffuse(.55)
        self.renderer.AddActor(actor)
        for value in np.arange(-1.2, 1.21, .1):
            for a, b in [([value, -1.2, .0005], [value, 1.2, .0005]), ([-1.2, value, .0005], [1.2, value, .0005])]:
                line = vtk.vtkLineSource()
                line.SetPoint1(*a)
                line.SetPoint2(*b)
                m = vtk.vtkPolyDataMapper()
                m.SetInputConnection(line.GetOutputPort())
                act = vtk.vtkActor()
                act.SetMapper(m)
                act.GetProperty().SetColor(*_rgb('#9fb0c0'))
                self.renderer.AddActor(act)

    def _arrow(self, color):
        vtk = self.vtk
        source = vtk.vtkArrowSource()
        source.SetShaftRadius(.035)
        source.SetTipRadius(.09)
        source.SetTipLength(.25)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetAmbient(.4)
        self.renderer.AddActor(actor)
        return actor

    def _place_arrow(self, actor, start, vector):
        length = float(np.linalg.norm(vector))
        if length < 1e-3:
            actor.SetVisibility(False)
            return
        actor.SetVisibility(True)
        x = np.asarray(vector) / length
        helper = np.array([0., 0., 1.]) if abs(x[2]) < .9 else np.array([1., 0., 0.])
        y = np.cross(helper, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x * length, y * length, z * length
        T[:3, 3] = start
        actor.SetUserMatrix(self._matrix(T))

    def _matrix(self, value):
        output = self.vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                output.SetElement(i, j, float(value[i, j]))
        return output

    def render(self, q, lam_force, push, torso_com):
        poses = self.backend.poses(q)
        for body, actor in self.actors:
            actor.SetUserMatrix(self._matrix(poses[body]))
        corners = np.array([poses[b][:3, :3] @ p + poses[b][:3, 3]
                            for b, p in zip(self.corner_bodies, self.corner_local)])
        forces = lam_force.reshape(-1, 3)
        for actor, point, force in zip(self.corners, corners, forces):
            actor.SetPosition(*point)
            actor.GetProperty().SetColor(*(_rgb(GREEN) if force[2] > 1e-6 else _rgb('#6d7c8a')))
        for side in range(2):
            # Foot reaction drawn beside the foot (as in the v22 video) so the sole does not hide it.
            f = forces[4 * side:4 * side + 4].sum(axis=0)
            anchor = corners[4 * side:4 * side + 4].mean(axis=0)
            anchor = np.array([anchor[0], anchor[1] + (.13 if side == 0 else -.13), .008])
            self._place_arrow(self.arrows[side], anchor, f * .00045)
        self._place_arrow(self.arrows[2], np.asarray(torso_com) - np.asarray(push) * .006, np.asarray(push) * .006)
        self.renderer.ResetCameraClippingRange()
        self.window.Render()
        self.capture.Modified()
        self.capture.Update()
        pixels = self._np(self.capture.GetOutput().GetPointData().GetScalars())
        return pixels.reshape(VIEW_H, VIEW_W, 3)[::-1].copy(), corners

    def close(self):
        self.window.Finalize()


def _stage(t):
    for lo, hi, title, text in STAGES:
        if lo <= t < hi:
            return title, text
    return STAGES[-1][2], STAGES[-1][3]


def _wrap(draw, text, font, width):
    lines = ['']
    for word in text.split():
        trial = (lines[-1] + ' ' + word).strip()
        if draw.textlength(trial, font=font) > width and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = trial
    return lines


def _compose(view, t, speed, k, a, weight, series):
    im = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(im)
    draw.text((30, 20), 'KANGAROO  |  PINOCCHIO DYNAMICS + UNCHANGED PACDM', font=_font(34, True), fill=INK)
    draw.text((32, 68), '78 bodies  /  floating pelvis  /  24 loop cuts  /  12 linear drives  /  '
                        'rigid Coulomb contact at 8 foot corners', font=_font(20), fill=MUTED)
    im.paste(Image.fromarray(view), (20, 110))
    draw.rounded_rectangle((20, 110, 20 + VIEW_W, 110 + VIEW_H), radius=8, outline=RULE)
    draw.rounded_rectangle((34, 124, 380, 158), radius=6, fill=BG)
    draw.text((46, 131), f't = {t:5.3f} s   |   replay {speed:.2f}x', font=_font(18, True), fill=INK)
    draw.rounded_rectangle((34, 110 + VIEW_H - 44, 915, 110 + VIEW_H - 12), radius=6, fill=BG)
    draw.text((46, 110 + VIEW_H - 38), 'Saved Pinocchio/PACDM states (no interpolation)  |  green: loaded corners, '
                                      'foot reactions  |  red: push', font=_font(15), fill=MUTED)
    x = 1000
    title, text = _stage(t)
    draw.text((x, 118), title, font=_font(26, True), fill=CYAN)
    for i, line in enumerate(_wrap(draw, text, _font(18), 570)):
        draw.text((x, 156 + 24 * i), line, font=_font(18), fill='#c7d6e8')
    # Drive forces
    draw.rounded_rectangle((x - 12, 222, 1582, 520), radius=8, fill=PANEL)
    draw.text((x, 232), 'ACTUAL DRIVE FORCES  (N)', font=_font(18, True), fill=INK)
    force = series['force'][k]
    for side in range(2):
        bx = x + side * 292
        draw.text((bx, 262), 'LEFT' if side == 0 else 'RIGHT', font=_font(16, True), fill=AMBER if side == 0 else BLUE)
        for j in range(6):
            value = float(force[6 * side + j])
            y = 290 + 36 * j
            draw.text((bx, y + 6), DRIVE_LABELS[j], font=_font(14), fill=MUTED)
            draw.text((bx + 150, y + 14), f'{value:6.0f}', font=_font(14, True), fill=INK, anchor='rm')
            cx = bx + 218
            draw.line((cx, y + 4, cx, y + 26), fill=RULE, width=1)
            length = 58 * np.clip(value / 2500., -1., 1.)
            draw.rectangle((min(cx, cx + length), y + 8, max(cx, cx + length), y + 22),
                           fill=AMBER if side == 0 else BLUE)
        draw.text((x, 500), 'bars: +/-2500 N full scale', font=_font(12), fill=MUTED)
    # Contact and state
    draw.rounded_rectangle((x - 12, 532, 1582, 700), radius=8, fill=PANEL)
    feet = series['feet'][k]
    draw.text((x, 542), 'GROUND REACTION  (rigid NCP impulse / step)', font=_font(16, True), fill=INK)
    for side in range(2):
        draw.text((x + side * 292, 570), ('LEFT FOOT  ' if side == 0 else 'RIGHT FOOT  ') + f'{feet[side, 2]:6.0f} N',
                  font=_font(18, True), fill=GREEN)
    base = a['base'][k]
    tilt = np.degrees(a['tilt'][k])
    rows = [('Pelvis height', f'{base[2]:.4f} m'), ('Pelvis tilt', f'{tilt:.2f} deg'),
            ('Total normal / weight', f'{feet[:, 2].sum() / weight:.3f}'),
            ('PACDM all-row closure', f"{a['pacdm_closure'][k]:.1e}"),
            ('NCP residual (max)', f"{np.max(a['ncp'][k]):.1e}")]
    for i, (name, value) in enumerate(rows):
        draw.text((x + (i % 2) * 292, 604 + 30 * (i // 2)), f'{name}: {value}', font=_font(15), fill='#c7d6e8')
    # Height chart
    draw.rounded_rectangle((x - 12, 712, 1582, 900), radius=8, fill=PANEL)
    draw.text((x, 722), 'PELVIS HEIGHT 0.68-0.90 m', font=_font(15, True), fill=CYAN)
    draw.text((x + 260, 722), 'GROUND NORMAL / WEIGHT 0-5', font=_font(15, True), fill=GREEN)
    x0, x1, y0, y1 = x + 10, 1560, 885, 760
    draw.line((x0, y1, x0, y0, x1, y0), fill=RULE)
    tt, z, n = series['t'], series['z'], series['normal']
    zmin, zmax = .68, .90
    to = lambda ti, v, lo, hi: (x0 + (x1 - x0) * ti / 10., y0 - (y0 - y1) * (v - lo) / (hi - lo))
    draw.line([to(ti, float(np.clip(v, zmin, zmax)), zmin, zmax) for ti, v in zip(tt, z)], fill=CYAN, width=2)
    draw.line([to(ti, float(np.clip(v, 0., 5.)), 0., 5.) for ti, v in zip(tt, n)], fill=GREEN, width=1)
    cursor = to(t, zmin, zmin, zmax)[0]
    draw.line((cursor, y1, cursor, y0), fill=INK, width=1)
    draw.text((30, 920), 'CMG -> PACDM (unchanged) -> Pinocchio 3.8.0 dynamics -> native PGS/ADMM contact', font=_font(18, True),
              fill=CYAN)
    draw.text((30, 952), 'No MuJoCo module is imported for simulation or rendering. Quantitative acceptance comes '
                         'from results/validation.json.', font=_font(16), fill=MUTED)
    progress = 30 + 1540 * np.clip(t / 10., 0., 1.)
    draw.line((30, 985, 1570, 985), fill=RULE, width=2)
    draw.ellipse((progress - 5, 980, progress + 5, 990), fill=INK)
    return np.asarray(im)


def verify_encoded_video(path, expected_frames, fps):
    reader = imageio.get_reader(str(path))
    count = 0
    try:
        info = reader.get_meta_data()
        for frame in reader:
            if frame.shape != (H, W, 3):
                raise ValueError(f'Unexpected encoded frame shape {frame.shape}')
            count += 1
    finally:
        reader.close()
    if count != expected_frames:
        raise ValueError(f'Incomplete video: decoded {count}, expected {expected_frames} frames')
    if abs(float(info['fps']) - fps) > .01:
        raise ValueError(f"Unexpected encoded frame rate {info['fps']}")
    return dict(passed=True, decoded_frames=count, decoded_resolution=[W, H], decoded_fps=float(info['fps']))


def render_video(root, case='landing_nominal', fps=30, limit_frames=None):
    """Write results/<case>.mp4, a poster PNG and render metadata from saved states."""
    root = Path(root).resolve()
    source = root / 'results' / f'{case}.npz'
    with np.load(source, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    cmg_path = root / 'original_v22/data/whole_body_cmg.json'
    cmg = json.loads(cmg_path.read_text())
    from kangaroo_pin.legacy import load
    accepted = load()
    corners = accepted.foot_corners({name: np.eye(4) for name in ['left_ankle_roll', 'right_ankle_roll']})
    if [f'corner_{i}' for i in range(len(corners))] != list(a['corner_names']):
        raise ValueError('Saved corner names do not match the source foot corners')
    backend = FloatingPinBackend(cmg)
    scene = Scene(root, backend, cmg, [c[0] for c in corners], [np.asarray(c[1]) for c in corners])
    dt = float(a['timestep_s'])
    where = {int(s): i for i, s in enumerate(a['stored_step'])}
    bounds = np.array([x['force_bounds_N'] for x in cmg['actuators']])
    weight = backend.total_mass * 9.81
    stride = round(.005 / dt)
    series = dict(force=np.clip(a['act'], bounds[:, 0], bounds[:, 1]),
                  feet=a['lam'].reshape(len(a['time']), 2, 4, 3).sum(axis=2) / dt,
                  t=a['time'][::stride * 4], z=a['base'][::stride * 4, 2],
                  normal=a['lam'][::stride * 4, 2::3].sum(axis=1) / dt / weight)
    frames = frame_schedule(fps)
    if limit_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, limit_frames).astype(int)]
    suffix = '' if limit_frames is None else '_preview'
    output = root / 'results' / f'kangaroo_pinocchio_{case}{suffix}.mp4'
    poster = root / 'results' / f'kangaroo_pinocchio_{case}{suffix}_poster.png'
    writer = imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8, macro_block_size=None,
                                ffmpeg_params=['-movflags', '+faststart'])
    steps, fk_error = [], 0.
    poster_index = min(range(len(frames)), key=lambda i: abs(frames[i][0] - 2.6))
    try:
        for n, (t, speed) in enumerate(frames):
            k = int(round(t / dt))
            if abs(a['time'][k] - t) > 1e-9 or k not in where:
                raise ValueError(f'Frame time {t} is not a stored state')
            q = np.r_[a['base'][k], a['stored_z'][where[k]][:len(cmg['coordinate_ids'])]]
            view, corner_points = scene.render(q, a['lam'][k] / dt, a['push'][k], a['torso_com'][k])
            poses = backend.poses(q)
            feet = np.array([poses[f][:3, 3] for f in ['left_ankle_roll', 'right_ankle_roll']])
            fk_error = max(fk_error, float(np.max(np.abs(feet - a['foot_position'][k]))))
            frame = _compose(view, t, speed, k, a, weight, series)
            writer.append_data(frame)
            if n == poster_index:
                Image.fromarray(frame).save(poster)
            steps.append(k)
            if n % 60 == 0:
                print(f'Pinocchio/VTK: rendered {n}/{len(frames)} frames', flush=True)
        backend_name = scene.window.GetClassName()
        vtk_version = scene.vtk.vtkVersion.GetVTKVersion()
        meshes, mesh_files = scene.mesh_count, scene.mesh_files
    finally:
        writer.close()
        scene.close()
    if fk_error > 1e-9:
        raise ValueError(f'Rendered foot poses differ from the logged foot positions by {fk_error:.3g} m')
    check = verify_encoded_video(output, len(frames), fps)
    import pinocchio as pin
    metadata = dict(
        source_case=case, source_trajectory=source.name,
        source_trajectory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cmg_sha256=hashlib.sha256(cmg_path.read_bytes()).hexdigest(), video=output.name, poster=poster.name,
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), fps=fps, frames=len(frames),
        duration_s=len(frames) / fps, resolution=[W, H], replay_segments=SEGMENTS, mp4_decode_check=check,
        source_steps=steps, max_rendered_foot_fk_vs_logged_m=fk_error, pinocchio_version=pin.__version__,
        vtk_version=vtk_version, vtk_window_backend=backend_name, visual_meshes=meshes,
        visual_mesh_files=mesh_files, simulation_during_rendering=False, mujoco_imported='mujoco' in sys.modules,
        state_source='Saved pelvis pose and PACDM graph coordinates stored at the frame step; no interpolation, '
                     'no reference substitution',
        mesh_source='Original upstream STL meshes, colours and mesh scales read as XML (as in v22 native_model.export)')
    (root / 'results' / f'render_metadata{suffix}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--case', default='landing_nominal')
    parser.add_argument('--preview-frames', type=int)
    args = parser.parse_args()
    print(render_video(args.root, args.case, limit_frames=args.preview_frames))
