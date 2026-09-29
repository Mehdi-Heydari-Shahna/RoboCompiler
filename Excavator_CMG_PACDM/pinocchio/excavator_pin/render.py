"""Evidence video of a recorded Pinocchio-backend run, rendered with VTK.

MuJoCo is neither imported nor run.  The original ``visual_actual.xml`` is read
only for the STL file names, body assignments (``cad_body_X`` belongs to CMG
body ``body_X``), the 0.001 mesh scale and the material colours.  Every moving
mesh is placed by the Pinocchio plant forward kinematics of the *recorded*
integrated tree configuration ``q``; no reference or controller value is
substituted.  Before rendering, the kinematic tree written in the visual XML is
evaluated independently at recorded frames and compared with the plant body
frames (``visual_frame_check`` in the metadata).

Static scene elements reproduce the geometry of the original MuJoCo soil scene
(surface at z = 0 with the bed aperture, bed bottom, receiver marker and walls).
The soil is the declared surrogate: the bed is drawn as a flat brown fill, the
bucket payload as a proxy volume proportional to the surrogate payload mass and
the receiver deposit as a cone with the same bulk density.  No particles exist.

Run from the package root::

    python -m excavator_pin.render --case nominal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    # Headless Linux: EGL offscreen (Mesa llvmpipe works without a GPU or X server).
    os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW', 'vtkEGLRenderWindow')

from excavator_pin import paths, sentinel  # noqa: E402

_V26_RESULTS_EXISTED = (paths.V26 / 'results').exists()  # the original project.py creates it on import
from excavator_pin.engine import load_inputs, context  # noqa: E402
from excavator_pin.plant import ExcavatorPinModel  # noqa: E402

sentinel.install()
paths.add_original_to_path()
from soil_scene import BOUNDS  # noqa: E402  (preserved original geometry)

WIDTH, HEIGHT, FPS = 1280, 720, 25
INK = (237, 244, 248)
MUTED = (149, 174, 193)
AMBER = (255, 187, 80)
CYAN = (55, 212, 216)
PANEL = (14, 26, 38, 215)
_FONTS = {}


def _font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        for folder in (Path('/usr/share/fonts/truetype/dejavu'), Path('C:/Windows/Fonts'),
                       Path('/System/Library/Fonts/Supplemental')):
            if (folder / name).exists():
                _FONTS[key] = ImageFont.truetype(str(folder / name), size)
                break
        else:
            try:
                _FONTS[key] = ImageFont.truetype(name, size)
            except OSError:
                _FONTS[key] = ImageFont.load_default(size=size)
    return _FONTS[key]


def _matrix(vtk, value):
    out = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            out.SetElement(i, j, float(value[i, j]))
    return out


def visual_bodies():
    """(body, mesh path, rgba, specular) for every CAD geom of the original visual XML."""
    root = ET.parse(paths.VISUAL_XML).getroot()
    meshdir = paths.VISUAL_XML.parent / root.find('compiler').get('meshdir', '')
    meshes = {m.get('name'): m for m in root.find('asset').findall('mesh')}
    materials = {m.get('name'): m for m in root.find('asset').findall('material')}
    out = []
    for body in root.iter('body'):
        for geom in body.findall('geom'):
            if not geom.get('name', '').startswith('cad_'):
                continue
            if geom.get('pos', '0 0 0').split() != ['0', '0', '0'] or geom.get('quat', '1 0 0 0').split() != [
                    '1', '0', '0', '0']:
                raise ValueError(f'{geom.get("name")}: non-identity visual offset is not supported')
            mesh = meshes[geom.get('mesh')]
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            material = materials[geom.get('material')]
            out.append(dict(body=body.get('name'), path=meshdir / mesh.get('file'), scale=scale,
                            rgba=np.fromstring(material.get('rgba'), sep=' '),
                            specular=float(material.get('specular', '.3'))))
    return out


def xml_frames(q_by_joint):
    """Independent evaluation of the kinematic tree written in visual_actual.xml."""
    root = ET.parse(paths.VISUAL_XML).getroot()
    out = {}

    def pose(pos, quat):
        m = np.eye(4)
        m[:3, 3] = pos
        w, x, y, z = quat
        m[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        return m

    def visit(body, parent):
        m = parent @ pose(np.fromstring(body.get('pos', '0 0 0'), sep=' '),
                          np.fromstring(body.get('quat', '1 0 0 0'), sep=' '))
        for joint in body.findall('joint'):
            name, kind = joint.get('name'), joint.get('type', 'hinge')
            offset = q_by_joint[name] - float(joint.get('ref', '0'))
            axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
            axis = axis / np.linalg.norm(axis)
            at = np.fromstring(joint.get('pos', '0 0 0'), sep=' ')
            step = np.eye(4)
            if kind == 'hinge':
                rotation = Rotation.from_rotvec(axis * offset).as_matrix()
                step[:3, :3] = rotation
                step[:3, 3] = at - rotation @ at
            elif kind == 'slide':
                step[:3, 3] = axis * offset
            else:
                raise ValueError(kind)
            m = m @ step
        out[body.get('name')] = m
        for child in body.findall('body'):
            visit(child, m)

    for body in root.find('worldbody').findall('body'):
        visit(body, np.eye(4))
    return out


class Scene:
    def __init__(self, model, soil, receiver_center, receiver_half):
        import vtk
        self.vtk, self.model = vtk, model
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(.63, .72, .80)
        self.renderer.SetBackground2(.20, .29, .38)
        self.renderer.GradientBackgroundOn()
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(WIDTH, HEIGHT)
        self.window.SetMultiSamples(0)
        self.window.SetAlphaBitPlanes(1)
        self.renderer.SetUseDepthPeeling(1)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.UseFXAAOn()
        self.window.AddRenderer(self.renderer)
        self.mesh_actors = []
        for item in visual_bodies():
            reader = vtk.vtkSTLReader()
            reader.SetFileName(str(item['path']))
            scale = vtk.vtkTransform()
            scale.Scale(*item['scale'])
            filt = vtk.vtkTransformPolyDataFilter()
            filt.SetInputConnection(reader.GetOutputPort())
            filt.SetTransform(scale)
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(filt.GetOutputPort())
            normals.SetFeatureAngle(35.)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*item['rgba'][:3])
            prop.SetSpecular(item['specular'])
            prop.SetSpecularPower(25.)
            self.renderer.AddActor(actor)
            self.mesh_actors.append((item['body'], actor))
        self._static(soil, receiver_center, receiver_half)
        self.trail_points = vtk.vtkPoints()
        self.trail = self._polyline(self.trail_points, (1., .73, .31), 3.)
        self.force = self._arrow((.93, .25, .20))
        self.payload = self._sphere((.47, .33, .18))
        self.deposit = self._cone((.47, .33, .18))
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(10.2, -11.8, 8.4)
        camera.SetFocalPoint(3.3, 1.9, .3)
        camera.SetViewUp(0., 0., 1.)
        camera.SetViewAngle(33.)
        self.renderer.SetAmbient(.25, .25, .25)
        self.renderer.RemoveAllLights()
        for position, intensity in (((14., -10., 16.), .85), ((-6., 8., 10.), .45), ((6., -14., 2.), .25)):
            light = vtk.vtkLight()
            light.SetLightTypeToSceneLight()
            light.SetPosition(*position)
            light.SetFocalPoint(4., 1.5, 0.)
            light.SetIntensity(intensity)
            self.renderer.AddLight(light)
        self.capture = vtk.vtkWindowToImageFilter()
        self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGB()
        self.capture.ReadFrontBufferOff()

    def _box(self, low, high, color, opacity=1.):
        vtk = self.vtk
        cube = vtk.vtkCubeSource()
        cube.SetBounds(low[0], high[0], low[1], high[1], low[2], high[2])
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self.renderer.AddActor(actor)
        return actor

    def _static(self, soil, center, half):
        (x0, x1), (y0, y1), (bottom, _) = BOUNDS
        ground = (.56, .55, .50)
        # Surface slabs around the bed aperture (as in soil_scene.build_soil_scene).
        for low, high in (([-30., -30.], [x0, 30.]), ([x1, -30.], [30., 30.]),
                          ([x0, -30.], [x1, y0]), ([x0, y1], [x1, 30.])):
            self._box([low[0], low[1], -.6], [high[0], high[1], 0.], ground)
        self._box([x0, y0, bottom - .15], [x1, y1, bottom], (.40, .36, .28))
        x_hi = x1 - soil['entry_slot_m']
        if soil['enabled']:  # semi-transparent so that the submerged bucket and force arrow stay visible
            self._box([x0, y0, bottom], [x_hi, y1, soil['surface_z_m']], (.52, .38, .22), .55)
        cx, cy = center
        hx, hy = half
        self._box([cx - hx, cy - hy, 0.], [cx + hx, cy + hy, .004], (.22, .47, .43))
        wall = (.30, .37, .38)
        for low, high in (([cx - hx - .1, cy - hy - .1], [cx - hx, cy + hy + .1]),
                          ([cx + hx, cy - hy - .1], [cx + hx + .1, cy + hy + .1]),
                          ([cx - hx, cy - hy - .1], [cx + hx, cy - hy]),
                          ([cx - hx, cy + hy], [cx + hx, cy + hy + .1])):
            self._box([low[0], low[1], 0.], [high[0], high[1], .4], wall, .85)

    def _polyline(self, points, color, width):
        vtk = self.vtk
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        lines = vtk.vtkCellArray()
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        self.renderer.AddActor(actor)
        return poly

    def _arrow(self, color):
        vtk = self.vtk
        source = vtk.vtkArrowSource()
        source.SetShaftRadius(.035)
        source.SetTipRadius(.09)
        source.SetTipLength(.3)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        self.renderer.AddActor(actor)
        return actor

    def _sphere(self, color):
        vtk = self.vtk
        source = vtk.vtkSphereSource()
        source.SetThetaResolution(24)
        source.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        self.renderer.AddActor(actor)
        return actor

    def _cone(self, color):
        vtk = self.vtk
        source = vtk.vtkConeSource()
        source.SetResolution(40)
        source.SetDirection(0., 0., 1.)
        source.SetHeight(1.)
        source.SetRadius(1.)
        source.SetCenter(0., 0., .5)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        self.renderer.AddActor(actor)
        return actor

    def update(self, poses, lip, force, payload_kg, capacity_kg, centroid, deposited_kg, density,
               receiver_center, trail):
        vtk = self.vtk
        for body, actor in self.mesh_actors:
            actor.SetUserMatrix(_matrix(vtk, poses[body]))
        self.trail_points.Reset()
        lines = vtk.vtkCellArray()
        for point in trail:
            self.trail_points.InsertNextPoint(*point)
        if len(trail) > 1:
            lines.InsertNextCell(len(trail))
            for i in range(len(trail)):
                lines.InsertCellPoint(i)
        self.trail.SetLines(lines)
        self.trail.Modified()
        magnitude = float(np.linalg.norm(force))
        if magnitude > 1.:
            length = .6 + .9 * min(1., magnitude / 1500.)
            direction = force / magnitude
            transform = vtk.vtkTransform()
            transform.Translate(*(lip - direction * length))
            axis = np.cross([1., 0., 0.], direction)
            angle = np.degrees(np.arccos(np.clip(direction[0], -1., 1.)))
            if np.linalg.norm(axis) > 1e-9:
                transform.RotateWXYZ(angle, *axis)
            elif direction[0] < 0:
                transform.RotateWXYZ(180., 0., 0., 1.)
            transform.Scale(length, length, length)
            self.force.SetUserTransform(transform)
            self.force.VisibilityOn()
        else:
            self.force.VisibilityOff()
        if payload_kg > .5:
            radius = .32 * (payload_kg / capacity_kg) ** (1. / 3.)
            transform = vtk.vtkTransform()
            transform.Translate(*centroid)
            transform.Scale(radius, radius, radius)
            self.payload.SetUserTransform(transform)
            self.payload.VisibilityOn()
        else:
            self.payload.VisibilityOff()
        if deposited_kg > .5:
            # Cone with 35 deg side slope holding the deposited surrogate volume.
            volume = deposited_kg / density
            height = (3. * volume * np.tan(np.radians(35.)) ** 2 / np.pi) ** (1. / 3.)
            radius = height / np.tan(np.radians(35.))
            transform = vtk.vtkTransform()
            transform.Translate(receiver_center[0], receiver_center[1], .004)
            transform.Scale(radius, radius, height)
            self.deposit.SetUserTransform(transform)
            self.deposit.VisibilityOn()
        else:
            self.deposit.VisibilityOff()

    def frame(self):
        from vtk.util.numpy_support import vtk_to_numpy
        self.window.Render()
        self.capture.Modified()
        self.capture.Update()
        image = self.capture.GetOutput()
        width, height, _ = image.GetDimensions()
        array = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, 3)
        return np.flipud(array).copy()


def _hud(image, case, t, phase, row, peak_pressure):
    img = Image.fromarray(image).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x = WIDTH - 470
    draw.rounded_rectangle((x - 16, 18, WIDTH - 18, 262), 12, fill=PANEL)
    draw.text((x, 30), 'Excavator · Pinocchio backend', font=_font(22, True), fill=INK)
    draw.text((x, 60), f'case {case} · fixed undercarriage · soil surrogate', font=_font(15), fill=MUTED)
    lines = [('time', f'{t:6.2f} s'), ('phase', phase), ('payload (surrogate)', f"{row['payload']:7.1f} kg"),
             ('deposited in receiver', f"{row['deposited']:7.1f} kg"),
             ('cutting force at lip', f"{row['force']:7.0f} N"),
             ('tracking error', f"{row['tracking'] * 1e3:7.2f} mrad"),
             ('max chamber pressure', f'{peak_pressure / 1e6:7.2f} MPa')]
    for k, (label, value) in enumerate(lines):
        y = 92 + 23 * k
        draw.text((x, y), label, font=_font(16), fill=MUTED)
        draw.text((x + 226, y), value, font=_font(16, True), fill=AMBER if k in (2, 3, 4) else INK)
    draw.rounded_rectangle((18, HEIGHT - 58, 900, HEIGHT - 18), 10, fill=PANEL)
    draw.text((32, HEIGHT - 50), 'Meshes placed by Pinocchio FK of recorded integrated q. MuJoCo not used. '
                                 'No particles: payload/deposit are surrogate proxies.',
              font=_font(14), fill=MUTED)
    return np.asarray(Image.alpha_composite(img, overlay).convert('RGB'))


def render(case='nominal', results_dir=paths.RESULTS, output_dir=None, fps=FPS, max_frames=None):
    import imageio.v2 as imageio
    results_dir = Path(results_dir)
    output_dir = Path(output_dir or results_dir)
    report = json.loads((results_dir / f'{case}.json').read_text())
    with np.load(results_dir / f'{case}.npz', allow_pickle=False) as f:
        log = {k: f[k] for k in f.files}
    time_s = log['time']
    if len(time_s) < 2 or not np.all(np.diff(time_s) > 0):
        raise ValueError('recorded time must increase strictly')
    cmg, _ = load_inputs()
    scale = report['spec'].get('bucket_mass_scale', 1.)
    _, _, e, _, _, _ = context()
    model = ExcavatorPinModel(cmg, e.cut_ids)
    soil = report['soil_surrogate']['parameters']
    geometry = report['soil_surrogate']['geometry']
    model.add_bucket_point('centroid', geometry['centroid_local_m'])
    capacity = report['soil_surrogate']['capacity_kg']
    center = np.asarray(soil['receiver_center_xy_m'])
    half = np.asarray(soil['receiver_half_size_m'])
    scene = Scene(model, soil, center, half)
    names = report['phase_names']
    frame_times = np.arange(0., time_s[-1] + 1e-9, 1. / fps)
    if max_frames is not None:
        frame_times = frame_times[:max_frames]
    indices = np.searchsorted(time_s, frame_times - 1e-9)
    indices = np.clip(indices, 0, len(time_s) - 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / f'excavator_pinocchio_{case}.mp4'
    poster = output_dir / f'excavator_pinocchio_{case}_poster.png'
    writer = imageio.get_writer(str(video), fps=fps, codec='libx264', quality=8, macro_block_size=8,
                                pixelformat='yuv420p')
    fk_lip_error = 0.
    visual_error = 0.
    trail = []
    poster_frame, poster_index = None, int(np.argmax(log['payload_mass']))
    peak_pressure = 0.
    try:
        for count, (t, index) in enumerate(zip(frame_times, indices)):
            q = log['q'][index]
            poses = model.body_poses(q)
            pts, rotation = model.points(q, ('lip', 'centroid'))
            lip = pts['lip'][0]
            fk_lip_error = max(fk_lip_error, float(np.linalg.norm(lip - log['lip'][index])))
            if count % 50 == 0:  # independent visual-XML tree evaluation
                frames = xml_frames({j: q[model.index[j]] for j in model.tree_ids})
                for body, pose in frames.items():
                    visual_error = max(visual_error, float(np.max(np.abs(pose[:3, :] - poses[body][:3, :]))))
            trail.append(lip.copy())
            trail = trail[-int(8 * fps):]
            peak_pressure = max(peak_pressure, float(np.max(log['pressure'][index])))
            scene.update(poses, lip, log['cutting_force'][index], float(log['payload_mass'][index]), capacity,
                         pts['centroid'][0], float(log['deposited_mass'][index]),
                         soil['bulk_density_kg_m3'], center, trail)
            image = scene.frame()
            row = dict(payload=float(log['payload_mass'][index]), deposited=float(log['deposited_mass'][index]),
                       force=float(np.linalg.norm(log['cutting_force'][index])),
                       tracking=float(log['tracking_error'][index]))
            phase = names[int(log['phase'][index])]
            image = _hud(image, case, float(time_s[index]), phase, row, float(np.max(log['pressure'][index])))
            writer.append_data(image)
            if abs(index - poster_index) <= 2 and poster_frame is None:
                poster_frame = image
    finally:
        writer.close()
    if poster_frame is None:
        poster_frame = image
    Image.fromarray(poster_frame).save(poster)
    # Decode check: the written file must be readable with the expected frame count.
    reader = imageio.get_reader(str(video))
    decoded = sum(1 for _ in reader)
    reader.close()
    metadata = dict(case=case, video=video.name, poster=poster.name, fps=fps, width=WIDTH, height=HEIGHT,
                    frames_written=len(frame_times), frames_decoded=decoded,
                    simulated_duration_s=float(time_s[-1]),
                    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    source_trace=f'{case}.npz',
                    source_trace_sha256=hashlib.sha256((results_dir / f'{case}.npz').read_bytes()).hexdigest(),
                    max_recorded_lip_vs_render_FK_m=fk_lip_error,
                    visual_frame_check=dict(
                        description='visual_actual.xml kinematic tree (joint value minus ref) evaluated '
                                    'independently at every 50th frame versus plant FK body frames',
                        max_abs_difference=visual_error),
                    bucket_mass_scale=scale, mujoco=sentinel.report(),
                    placement='Pinocchio plant FK of recorded integrated q; identity CAD offsets; STL scale 0.001',
                    soil_depiction='flat surrogate fill, payload proxy sphere, deposit cone (no particles)',
                    vtk_render_window=scene.window.GetClassName())
    (output_dir / f'render_metadata_{case}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--case', default='nominal')
    parser.add_argument('--results', default=str(paths.RESULTS))
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()
    try:
        print(json.dumps(render(args.case, args.results, max_frames=args.max_frames), indent=2))
    finally:  # the original v26/project.py creates an empty v26/results directory on import
        created = paths.V26 / 'results'
        if not _V26_RESULTS_EXISTED and created.is_dir() and not any(created.iterdir()):
            created.rmdir()
