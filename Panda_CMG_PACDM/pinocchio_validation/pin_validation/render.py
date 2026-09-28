"""Evidence video of recorded Pinocchio dynamics, rendered with VTK.

MuJoCo is neither imported nor run. The upstream XML is read only for visual
OBJ filenames, body assignments, local visual offsets and material colours.
Every moving mesh is placed by CMG-to-Pinocchio forward kinematics of saved
actual ``q`` values. No reference state is substituted for a measured state.

Run from the package root::

    python -m pin_validation.render --case nominal

VTK 9.4+ with EGL is recommended for headless Linux. VTK 9.3 may require a
working X server (including Xvfb). Windows/macOS can use their native backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

# Direct execution is supported as well as ``python -m pin_validation.render``.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda.pin_backend import PinBackend

BG = '#0b1521'
PANEL = '#122232'
INK = '#edf4f8'
MUTED = '#95aec1'
CYAN = '#37d4d8'
AMBER = '#ffbb50'
RULE = '#2b4355'
PHASES = [(0., 3., 'Approach'), (3., 5., 'Close fingers'),
          (5., 8., 'Lift'), (8., 12., 'Translate + yaw'),
          (12., 16., 'Tilt + elbow sweep'), (16., 19., 'Lower'),
          (19., 20., 'Open fingers'), (20., 22., 'Retract')]
_FONT_CACHE = {}


def _font(size, bold=False):
    key = size, bold
    if key not in _FONT_CACHE:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        candidates = [Path('/usr/share/fonts/truetype/dejavu') / name,
                      Path('C:/Windows/Fonts') / ('arialbd.ttf' if bold else 'arial.ttf'),
                      Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf')]
        for path in candidates:
            if path.exists():
                _FONT_CACHE[key] = ImageFont.truetype(str(path), size)
                break
        else:
            try:
                _FONT_CACHE[key] = ImageFont.truetype(name, size)
            except OSError:
                # Pillow's bundled font is sufficient if no system font exists.
                _FONT_CACHE[key] = ImageFont.load_default(size=size)
    return _FONT_CACHE[key]


def _rgb(hex_string):
    return tuple(int(hex_string[i:i+2], 16)/255. for i in (1, 3, 5))


def _phase(t):
    for i, (a, b, label) in enumerate(PHASES):
        if a <= t < b:
            return i, label
    return len(PHASES)-1, PHASES[-1][2]


def _load(root, case):
    path = root / 'results' / f'{case}.npz'
    with np.load(path, allow_pickle=False) as archive:
        log = {key: archive[key] for key in archive.files}
    required = {'time': (), 'q': (9,), 'v': (9,), 'tool_pos': (3,),
                'tool_R': (3, 3), 'target_pos': (3,), 'target_R': (3, 3),
                'pose_error': (), 'angle_error': (), 'coupling_error': (),
                'wrench': (6,)}
    for key, shape in required.items():
        if key not in log:
            raise ValueError(f'Saved simulation is missing {key!r}')
        arr = np.asarray(log[key])
        if arr.shape != (len(log['time']),) + shape or not np.all(np.isfinite(arr)):
            raise ValueError(f'Invalid shape or nonfinite values in {key!r}: {arr.shape}')
    if len(log['time']) < 2 or not np.all(np.diff(log['time']) > 0.):
        raise ValueError('Recorded timestamps must increase strictly')
    return path, log


def _matrix(vtk, value):
    output = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            output.SetElement(i, j, float(value[i, j]))
    return output


def _local_visual_transform(geom):
    result = np.eye(4)
    result[:3, 3] = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
    if 'quat' in geom.attrib:
        w, x, y, z = np.fromstring(geom.get('quat'), sep=' ')
        result[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    elif any(key in geom.attrib for key in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
        raise ValueError('Unsupported visual orientation: use an explicit quaternion')
    return result


class Scene:
    """VTK visual meshes whose world poses come only from the Pin backend."""

    def __init__(self, root, backend, width=916, height=552):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        self.vtk = vtk
        self._to_numpy = vtk_to_numpy
        self.backend = backend
        self.tool_body = backend.cmg['tool']['body']
        self.tool_transform = np.asarray(backend.cmg['tool']['T_body_tool'], dtype=float)
        self.max_fk_log_position_error = 0.
        self.width, self.height = width, height
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*_rgb('#101d2a'))
        self.renderer.SetBackground2(*_rgb('#263d50'))
        self.renderer.GradientBackgroundOn()
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(width, height)
        self.window.SetMultiSamples(4)
        self.window.AddRenderer(self.renderer)
        self.actors = []
        self._read_visuals(root)
        self._ground()
        self.reference = self._line_actor([], _rgb(CYAN), .0015, .55)
        self.trail = self._line_actor([], _rgb(AMBER), .0022, 1.)
        self.axes = [self._line_actor([], color, .0015, 1.) for color in
                     ((.98, .35, .36), (.38, .91, .48), (.38, .65, 1.))]
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(.005)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(12)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        self.tool = vtk.vtkActor()
        self.tool.SetMapper(mapper)
        self.tool.GetProperty().SetColor(*_rgb(AMBER))
        self.renderer.AddActor(self.tool)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(1.40, -1.82, 1.20)
        camera.SetFocalPoint(.27, .0, .40)
        camera.SetViewUp(0., 0., 1.)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(.54)
        self.renderer.SetAmbient(.28, .28, .28)
        # Two broad lights retain contrast on the original white/black meshes.
        self.renderer.RemoveAllLights()
        for position, intensity in [((1.5, -2., 2.3), .9), ((-1.2, .8, 1.6), .6)]:
            light = vtk.vtkLight()
            light.SetLightTypeToSceneLight()
            light.SetPosition(*position)
            light.SetFocalPoint(.2, 0., .35)
            light.SetIntensity(intensity)
            self.renderer.AddLight(light)
        self.capture = vtk.vtkWindowToImageFilter()
        self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGB()
        self.capture.ReadFrontBufferOff()

    def _read_visuals(self, root):
        vtk = self.vtk
        xml_path = root / 'upstream' / 'franka_emika_panda' / 'panda.xml'
        tree = ET.parse(xml_path).getroot()
        assets = tree.find('asset')
        meshdir = xml_path.parent / tree.find('compiler').get('meshdir', '')
        meshes = {m.get('name', Path(m.get('file')).stem): m for m in assets.findall('mesh')}
        materials = {m.get('name'): np.fromstring(m.get('rgba', '0.8 0.8 0.8 1'), sep=' ')
                     for m in assets.findall('material')}
        cache = {}
        for body in tree.findall('.//worldbody//body'):
            body_name = body.get('name')
            if body_name not in self.backend.body_frame_ids:
                raise ValueError(f'Visual body {body_name!r} is absent from CMG')
            for geom in body.findall('geom'):
                if geom.get('class') != 'visual':
                    continue
                asset = meshes[geom.get('mesh')]
                path = (meshdir / asset.get('file')).resolve()
                if path.suffix.lower() != '.obj':
                    raise ValueError(f'Visual mesh is not OBJ: {path}')
                if path not in cache:
                    reader = vtk.vtkOBJReader()
                    reader.SetFileName(str(path))
                    reader.Update()
                    if reader.GetOutput().GetNumberOfPoints() == 0:
                        raise ValueError(f'Empty visual mesh: {path}')
                    cache[path] = reader.GetOutput()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(cache[path])
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                rgba = materials.get(geom.get('material'), np.array([.9, .92, .93, 1.]))
                prop = actor.GetProperty()
                prop.SetColor(*rgba[:3])
                prop.SetOpacity(float(rgba[3]))
                prop.SetAmbient(.18)
                prop.SetDiffuse(.75)
                prop.SetSpecular(.20)
                prop.SetSpecularPower(28.)
                scale = np.fromstring(asset.get('scale', '1 1 1'), sep=' ')
                actor.SetScale(*scale)
                self.actors.append((body_name, actor, _local_visual_transform(geom)))
                self.renderer.AddActor(actor)
        self.mesh_count = len(self.actors)

    def _ground(self):
        vtk = self.vtk
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-1., -1., -.013)
        plane.SetPoint1(1.3, -1., -.013)
        plane.SetPoint2(-1., 1.1, -.013)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*_rgb('#1b2b38'))
        actor.GetProperty().SetAmbient(.4)
        self.renderer.AddActor(actor)
        for value in np.arange(-.8, 1.01, .1):
            self._line_actor([[value, -.8, -.012], [value, 1., -.012]], _rgb('#355063'), .0005, .55)
            self._line_actor([[-.8, value, -.012], [1., value, -.012]], _rgb('#355063'), .0005, .55)
        cube = vtk.vtkCubeSource()
        cube.SetCenter(0., 0., -.006)
        cube.SetXLength(.23)
        cube.SetYLength(.23)
        cube.SetZLength(.014)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*_rgb('#44505d'))
        self.renderer.AddActor(actor)

    def _line_actor(self, points, color, radius, opacity):
        vtk = self.vtk
        poly = vtk.vtkPolyData()
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(poly)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(8)
        tube.CappingOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(.65)
        self.renderer.AddActor(actor)
        result = poly, actor
        self._set_line(result, points)
        return result

    def _set_line(self, item, coordinates):
        vtk = self.vtk
        poly, actor = item
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        coordinates = np.asarray(coordinates)
        if len(coordinates) >= 2:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(coordinates))
            for i, point in enumerate(coordinates):
                points.InsertNextPoint(*point)
                line.GetPointIds().SetId(i, i)
            lines.InsertNextCell(line)
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.Modified()
        actor.SetVisibility(len(coordinates) >= 2)

    def set_reference(self, positions):
        # Rendering subsamples long paths only; source logs are unchanged.
        idx = np.linspace(0, len(positions)-1, min(650, len(positions)), dtype=int)
        self._set_line(self.reference, np.asarray(positions)[idx])

    def render(self, q, tool_pos, target_pos, target_R, history):
        poses = self.backend.poses(q)
        fk_tool = poses[self.tool_body] @ self.tool_transform
        discrepancy = float(np.linalg.norm(fk_tool[:3, 3]-tool_pos))
        self.max_fk_log_position_error = max(self.max_fk_log_position_error, discrepancy)
        if discrepancy > 1e-8:
            raise ValueError(f'Logged tool position does not match rendered actual q: {discrepancy:.3g} m')
        for body, actor, local in self.actors:
            actor.SetUserMatrix(_matrix(self.vtk, poses[body] @ local))
        self.tool.SetPosition(*tool_pos)
        for k in range(3):
            self._set_line(self.axes[k], [target_pos, target_pos+.063*target_R[:, k]])
        idx = np.linspace(0, len(history)-1, min(450, len(history)), dtype=int)
        self._set_line(self.trail, history[idx])
        self.renderer.ResetCameraClippingRange()
        self.window.Render()
        self.capture.Modified()
        self.capture.Update()
        pixels = self._to_numpy(self.capture.GetOutput().GetPointData().GetScalars())
        return pixels.reshape(self.height, self.width, 3)[::-1].copy()

    def close(self):
        self.window.Finalize()


def _compose(scene, t, index, log, case):
    canvas = Image.new('RGB', (1280, 800), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 17), 'CMG / PACDM  |  INDEPENDENT BACKEND VALIDATION', font=_font(12, True), fill=CYAN)
    draw.text((26, 39), 'FRANKA PANDA', font=_font(31, True), fill=INK)
    draw.text((363, 52), 'Pinocchio dynamics | VTK rendering', font=_font(16, True), fill=MUTED)
    draw.text((1123, 38), f'{t:05.2f} s', font=_font(26, True), fill=INK)
    draw.line((28, 91, 1252, 91), fill=RULE)
    canvas.paste(Image.fromarray(scene), (28, 114))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((28, 114, 944, 666), radius=9, outline=RULE)
    phase, label = _phase(t)
    width = int(draw.textlength(label.upper(), font=_font(13, True)))+73
    draw.rounded_rectangle((44, 129, 44+width, 172), radius=6, fill=BG)
    draw.text((60, 141), f'{phase+1:02d}  /  {label.upper()}', font=_font(13, True), fill=INK)
    draw.rounded_rectangle((43, 630, 525, 652), radius=4, fill=BG)
    draw.text((53, 634), 'Recorded actual joint states  |  original Panda visual meshes', font=_font(11), fill=MUTED)
    # Legend does not call the target a measured object or imply any grasping contact.
    draw.line((46, 693, 68, 693), fill=CYAN, width=3)
    draw.text((77, 685), 'Reference tool path', font=_font(12), fill=MUTED)
    draw.line((251, 693, 273, 693), fill=AMBER, width=3)
    draw.text((283, 685), 'Measured tool trace', font=_font(12), fill=MUTED)
    draw.text((498, 685), 'RGB axes: target orientation', font=_font(12), fill=MUTED)

    left, right = 964, 1252
    draw.rounded_rectangle((left, 114, right, 403), radius=9, fill=PANEL)
    draw.text((983, 129), 'MEASURED IN SIMULATION', font=_font(11, True), fill=MUTED)
    values = [('TOOL POSITION ERROR', float(log['pose_error'][index])*1000., 'mm', '.3f'),
              ('TOOL ORIENTATION ERROR', np.rad2deg(float(log['angle_error'][index])), 'deg', '.3f'),
              ('FINGER COUPLING ERROR', abs(float(log['coupling_error'][index]))*1e6, 'um', '.3f')]
    for row, (name, value, unit, fmt) in enumerate(values):
        y = 157+row*72
        draw.text((983, y), name, font=_font(10, True), fill=MUTED)
        draw.text((981, y+17), format(value, fmt), font=_font(28, True), fill=INK)
        draw.text((1212, y+33), unit, font=_font(12), fill=MUTED)
    draw.text((983, 382), '7 arm joints + coupled fingers', font=_font(11), fill=MUTED)

    draw.rounded_rectangle((left, 419, right, 574), radius=9, fill=PANEL)
    draw.text((983, 434), 'POSITION ERROR / mm', font=_font(11, True), fill=MUTED)
    x0, x1, y0, y1 = 997, 1233, 553, 473
    all_error = np.asarray(log['pose_error'])*1000.
    ymax = max(.01, float(np.max(all_error))*1.18)
    draw.line((x0, y1, x0, y0, x1, y0), fill=RULE)
    draw.line((x0, (y0+y1)/2, x1, (y0+y1)/2), fill=RULE)
    draw.text((976, y0-5), '0', font=_font(9), fill=MUTED)
    draw.text((982, 454), f'{ymax:.3g}', font=_font(9), fill=MUTED)
    draw.text((1208, y0+5), '22 s', font=_font(9), fill=MUTED)
    ix = np.linspace(0, index, min(index+1, 400), dtype=int)
    chart = [(x0+(x1-x0)*float(log['time'][i])/float(log['time'][-1]),
              y0-(y0-y1)*float(all_error[i])/ymax) for i in ix]
    if len(chart) > 1:
        draw.line(chart, fill=AMBER, width=2)

    pulse = np.asarray(log['wrench'][index])
    # The vertical mass-equivalent load is separate from the disturbance pulse.
    perturbed = np.linalg.norm(pulse[:2]) > 1e-6 or np.linalg.norm(pulse[3:]) > 1e-6
    draw.rounded_rectangle((left, 590, right, 666), radius=9, fill='#352b1b' if perturbed else PANEL)
    draw.text((983, 602), 'PERTURBATION ACTIVE' if perturbed else 'APPLIED TOOL LOAD',
              font=_font(11, True), fill=AMBER if perturbed else CYAN)
    draw.text((983, 622), f'{np.linalg.norm(pulse[:3]):.3f} N  /  {np.linalg.norm(pulse[3:]):.3f} N m',
              font=_font(17, True), fill=INK)
    draw.text((983, 646), 'World force / moment at tool point', font=_font(10), fill=MUTED)
    draw.text((967, 685), f'Case: {case}', font=_font(12), fill=MUTED)

    draw.text((28, 715), 'Applied tool load; no object-contact simulation', font=_font(12, True), fill=INK)
    draw.text((871, 715), 'CMG -> PACDM -> Pinocchio', font=_font(12, True), fill=CYAN)
    draw.line((28, 742, 1252, 742), fill=RULE)
    short_labels = ['Approach', 'Close fingers', 'Lift', 'Translate + yaw', 'Tilt + elbow sweep', 'Lower', 'Open fingers', 'Retract']
    for k, short in enumerate(short_labels):
        x = 28+1224*k/8
        draw.rounded_rectangle((x, 753, x+145, 757), radius=2, fill=CYAN if k == phase else RULE)
        draw.text((x, 766), short, font=_font(10, k == phase), fill=INK if k == phase else MUTED)
    progress = 28+1224*np.clip(t/float(log['time'][-1]), 0., 1.)
    draw.line((28, 791, 1252, 791), fill=RULE)
    draw.ellipse((progress-3, 788, progress+3, 794), fill=INK)
    return np.asarray(canvas)


def verify_encoded_video(path, expected_frames, fps):
    """Decode every MP4 frame, rejecting incomplete or incorrectly sized output."""
    reader = imageio.get_reader(str(path))
    count = 0
    try:
        info = reader.get_meta_data()
        for frame in reader:
            if frame.shape != (800, 1280, 3):
                raise ValueError(f'Unexpected encoded frame shape: {frame.shape}')
            count += 1
    finally:
        reader.close()
    if count != expected_frames:
        raise ValueError(f'Incomplete video: decoded {count}, expected {expected_frames} frames')
    if abs(float(info['fps'])-fps) > .01:
        raise ValueError(f'Unexpected encoded frame rate: {info["fps"]}')
    return dict(passed=True, all_frames_decoded=True, decoded_frames=count,
                decoded_resolution=[1280, 800], decoded_fps=float(info['fps']),
                decoded_duration_s=count/fps)


def render_video(root, case='nominal', fps=30, *, duration=None):
    """Write a video, poster and provenance record from saved actual states.

    ``duration`` optionally creates a separately named preview. Full videos
    preserve physical time: a 22-second trajectory produces 660 frames at
    30 fps. Nearest saved samples are used without synthesizing robot states.
    """
    if int(fps) != fps or fps <= 0:
        raise ValueError('fps must be a positive integer')
    root = Path(root).resolve()
    source, log = _load(root, case)
    cmg_path = root / 'data' / 'panda_cmg.json'
    backend = PinBackend(json.loads(cmg_path.read_text(encoding='utf-8')))
    scene = Scene(root, backend)
    scene.set_reference(log['target_pos'])
    start, stop = float(log['time'][0]), float(log['time'][-1])
    if duration is not None:
        stop = min(stop, start+float(duration))
    frame_times = np.arange(start, stop-1e-10, 1./fps)
    if not len(frame_times):
        raise ValueError('The requested replay has no frames')
    suffix = '' if case == 'nominal' else '_'+case
    if duration is not None:
        suffix += '_preview'
    results = root / 'results'
    output = results / f'pinocchio_validation{suffix}.mp4'
    poster = results / f'poster{suffix}.png'
    times = log['time']
    right = np.minimum(np.searchsorted(times, frame_times), len(times)-1)
    left = np.maximum(right-1, 0)
    indices = np.where(abs(times[left]-frame_times) <= abs(times[right]-frame_times), left, right)
    poster_frame = int(np.argmin(abs(frame_times-min(10., stop*.55))))
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    writer = imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8,
                               macro_block_size=None, ffmpeg_params=['-movflags', '+faststart'])
    try:
        for n, (t, index) in enumerate(zip(frame_times, indices)):
            raw = scene.render(log['q'][index], log['tool_pos'][index], log['target_pos'][index],
                               log['target_R'][index], log['tool_pos'][:index+1])
            frame = _compose(raw, float(t), int(index), log, case)
            writer.append_data(frame)
            if n == poster_frame:
                Image.fromarray(frame).save(poster)
            if n % (fps*5) == 0:
                print(f'Pinocchio/VTK: rendered {n}/{len(frame_times)} frames', flush=True)
        backend_name = scene.window.GetClassName()
        vtk_version = scene.vtk.vtkVersion.GetVTKVersion()
    finally:
        writer.close()
        scene.close()
    decode_check = verify_encoded_video(output, len(frame_times), fps)
    import pinocchio as pin
    metadata = dict(source_case=case, source_trajectory=source.name,
        source_trajectory_sha256=source_digest, cmg_sha256=hashlib.sha256(cmg_path.read_bytes()).hexdigest(),
        video=output.name, poster=poster.name, fps=int(fps), frames=len(frame_times),
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        poster_sha256=hashlib.sha256(poster.read_bytes()).hexdigest(),
        mp4_decode_check=decode_check,
        duration_s=len(frame_times)/fps, resolution=[1280, 800],
        pinocchio_version=pin.__version__, vtk_version=vtk_version,
        vtk_window_backend=backend_name, visual_mesh_count=scene.mesh_count,
        max_rendered_fk_vs_logged_tool_position_error_m=scene.max_fk_log_position_error,
        state_source='Actual q saved by Pinocchio dynamics integration; q_ref is never used for robot meshes',
        mesh_placement='PinBackend.poses(q); original OBJ geometry with XML visual material and local offsets',
        simulation_during_rendering=False, mujoco_used_for_rendering=False,
        sample_selection='Nearest saved sample; no interpolation or reference state substitution',
        source_sample_indices=indices.tolist(), frame_times_s=frame_times.tolist(),
        max_sample_time_offset_s=float(np.max(abs(times[indices]-frame_times))),
        overlays='Saved measured errors, applied world tool wrench, reference tool path and measured tool trace',
        contact_scope='Applied tool load; no object-contact simulation',
        phase_timeline=PHASES)
    (results / f'render_metadata{suffix}.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--case', default='nominal')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--duration', type=float, help='Optional preview duration; does not replace the full video')
    args = parser.parse_args()
    print(render_video(args.root, args.case, args.fps, duration=args.duration))


if __name__ == '__main__':
    main()
