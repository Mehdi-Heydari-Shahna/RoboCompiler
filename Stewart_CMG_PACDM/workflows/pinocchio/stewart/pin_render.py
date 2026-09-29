"""Deterministic CPU mesh rendering of recorded Pinocchio/PACDM states.

This renderer is deliberately independent of any simulator viewer. All actual
body transforms are evaluated with PinBackend.poses(q) from the recorded state;
reference probe points come from the prescribed target pose. Decorative meshes
illustrate declared geometry only (they are not contact/collision geometry).
No state interpolation, replay through another engine, or model dynamics occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from .pin_backend import PinBackend
from .reference import phase

WIDTH, HEIGHT = 1280, 800
BG = (10, 18, 29)
PANEL = (17, 29, 43)
TEXT = (232, 239, 246)
MUTED = (149, 170, 188)
TEAL = (53, 225, 197)
GOLD = (246, 181, 76)
BLUE = (79, 155, 222)


def _font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    candidates = [name, '/usr/share/fonts/truetype/dejavu/' + name,
                  'C:/Windows/Fonts/' + ('arialbd.ttf' if bold else 'arial.ttf')]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


FONTS = {s: _font(s) for s in (11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 30, 36)}
BOLD = {s: _font(s, True) for s in (12, 13, 14, 15, 16, 18, 20, 22, 24, 30, 36)}


def _transform(T, pts):
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def _pose_transform(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('ZYX', pose[3:6]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def _color(color, scale):
    return tuple(int(np.clip(c * scale, 0, 255)) for c in color)


class Scene:
    """Small shaded polygon renderer; fixed orthographic camera, depth sort."""
    def __init__(self):
        self.target = np.array([0., 0., .38])
        eye = np.array([1.65, -2.05, 1.50])
        forward = self.target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.view = np.stack([right, up, forward])
        self.light = np.array([-0.4, -0.7, 1.3])
        self.light /= np.linalg.norm(self.light)
        self.scale = 370.
        self.origin = np.array([487., 382.])
        self.faces = []

    def project(self, points):
        p = (np.asarray(points) - self.target) @ self.view.T
        xy = p[..., :2] * [self.scale, -self.scale] + self.origin
        return xy, p[..., 2]

    def polygon(self, points, color, normal=None, emissive=False):
        points = np.asarray(points)
        if normal is None:
            normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal = np.asarray(normal)
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm > 1e-15 else np.array([0., 0., 1.])
        # Keep double-sided faces; closed solid geometry is depth sorted.
        lighting = 1. if emissive else .50 + .49 * max(0., float(normal @ self.light))
        xy, depth = self.project(points)
        self.faces.append((float(depth.mean()), xy, _color(color, lighting)))

    def cylinder(self, T, radius, lower, upper, color, segments=20):
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        ring = np.c_[radius*np.cos(theta), radius*np.sin(theta)]
        lo = _transform(T, np.c_[ring, np.full(segments, lower)])
        hi = _transform(T, np.c_[ring, np.full(segments, upper)])
        self.polygon(lo[::-1], color, -T[:3, 2])
        self.polygon(hi, color, T[:3, 2])
        for i in range(segments):
            j = (i+1) % segments
            normal = T[:3, :3] @ np.array([np.cos((theta[i]+np.pi/segments)),
                                          np.sin((theta[i]+np.pi/segments)), 0.])
            self.polygon([lo[i], lo[j], hi[j], hi[i]], color, normal)

    def sphere(self, center, radius, color, segments=10, rings=5):
        center = np.asarray(center)
        for j in range(rings):
            a, b = -np.pi/2+j*np.pi/rings, -np.pi/2+(j+1)*np.pi/rings
            for i in range(segments):
                p, q = 2*np.pi*i/segments, 2*np.pi*(i+1)/segments
                n = np.array([[np.cos(t)*np.cos(s), np.cos(t)*np.sin(s), np.sin(t)]
                              for t, s in [(a,p), (a,q), (b,q), (b,p)]])
                self.polygon(center + radius*n, color, np.mean(n, axis=0))

    def box(self, T, half, center, color):
        half, center = np.array(half), np.array(center)
        vertices = np.array([[x,y,z] for z in [-1,1] for y in [-1,1] for x in [-1,1]])
        world = _transform(T, center + vertices*half)
        for ids, normal in [([0,1,3,2],[0,0,-1]), ([4,6,7,5],[0,0,1]),
                            ([0,4,5,1],[0,-1,0]), ([2,3,7,6],[0,1,0]),
                            ([0,2,6,4],[-1,0,0]), ([1,5,7,3],[1,0,0])]:
            self.polygon(world[ids], color, T[:3,:3]@normal)

    def draw(self, draw):
        for _, points, color in sorted(self.faces, key=lambda face: face[0], reverse=True):
            draw.polygon([tuple(point) for point in points], fill=color)


def _line3d(draw, scene, points, color, width=1, dashed=False):
    xy, _ = scene.project(points)
    if len(xy) < 2:
        return
    if dashed:
        for start in range(0, len(xy)-1, 7):
            part = xy[start:min(start+4,len(xy))]
            if len(part)>1:
                draw.line([tuple(point) for point in part], fill=color, width=width)
    else:
        draw.line([tuple(point) for point in xy], fill=color, width=width, joint='curve')


def _round_panel(draw, xy, radius=15):
    draw.rounded_rectangle(xy, radius=radius, fill=PANEL, outline=(31, 47, 65), width=1)


def _txt(draw, xy, text, size=14, color=TEXT, bold=False, anchor=None):
    draw.text(xy, str(text), font=(BOLD if bold else FONTS)[size], fill=color, anchor=anchor)


def _plot(draw, time, error, index, bounds):
    x, y, w, h = bounds
    _round_panel(draw, (x,y,x+w,y+h))
    _txt(draw,(x+16,y+9),'POSITION TRACKING ERROR',12,MUTED,True)
    _txt(draw,(x+w-15,y+9),f'mm  |  full {time[-1]:g} s mission',12,MUTED,anchor='ra')
    left, right, top, bottom = x+56, x+w-20, y+36, y+h-20
    maximum = max(float(np.max(error))*1.15, .01)
    for value in [0, maximum/2, maximum]:
        py = bottom - (bottom-top)*value/maximum
        draw.line((left,py,right,py),fill=(41,57,73),width=1)
        _txt(draw,(left-9,py-6),f'{value:.2f}',11,MUTED,anchor='ra')
    subsample = max(1, len(time)//1700)
    indices = np.arange(0, len(time), subsample)
    points = np.c_[left+(right-left)*time[indices]/time[-1],
                   bottom-(bottom-top)*error[indices]/maximum]
    draw.line([tuple(point) for point in points],fill=(63,94,112),width=1)
    active = indices[indices<=index]
    if len(active)>1:
        actual = np.c_[left+(right-left)*time[active]/time[-1],
                       bottom-(bottom-top)*error[active]/maximum]
        draw.line([tuple(point) for point in actual],fill=TEAL,width=2)
    px = left+(right-left)*time[index]/time[-1]
    py = bottom-(bottom-top)*error[index]/maximum
    draw.line((px,top,px,bottom),fill=(111,140,158),width=1)
    draw.ellipse((px-3,py-3,px+3,py+3),fill=TEAL)
    for tick in (0,5,10,15,20,22):
        tx=left+(right-left)*tick/time[-1]
        _txt(draw,(tx,bottom+5),str(tick),11,MUTED,anchor='ma')


class Movie:
    def __init__(self, root, case):
        self.root, self.case = Path(root), case
        self.source = self.root/'results'/f'{case}.npz'
        cmg_path=self.root/'results'/f'{case}.cmg.json'
        if not cmg_path.exists():
            cmg_path=self.root/'data'/'stewart.cmg.json'
        self.cmg_path=cmg_path
        cmg_bytes=cmg_path.read_bytes()
        self.cmg_sha256=hashlib.sha256(cmg_bytes).hexdigest()
        self.cmg=json.loads(cmg_bytes)
        source_bytes=self.source.read_bytes()
        self.source_sha256=hashlib.sha256(source_bytes).hexdigest()
        with np.load(io.BytesIO(source_bytes),allow_pickle=False) as data:
            self.data={key:data[key].copy() for key in data.files}
        required={'time','q','target_pose','force','wrench','pose_error_m',
                  'angle_error_rad','closure_error_m','coordinate_ids'}
        if not required<=self.data.keys():
            raise ValueError(f'Result file is missing {sorted(required-self.data.keys())}')
        if list(self.data['coordinate_ids'])!=list(self.cmg['coordinate_ids']):
            raise ValueError('Recorded coordinate order differs from the CMG')
        self.time=np.asarray(self.data['time'],float)
        if (self.time.ndim!=1 or len(self.time)<2 or not np.all(np.isfinite(self.time))
                or np.any(np.diff(self.time)<=0) or abs(self.time[0])>1e-12):
            raise ValueError('Video requires finite, increasing sample times starting at zero')
        for key in required-{'coordinate_ids','time'}:
            value=self.data[key]
            if len(value)!=len(self.time) or not np.all(np.isfinite(value)):
                raise ValueError(f'Invalid sample array: {key}')
        self.pin=PinBackend(self.cmg)
        # Precompute display samples directly from Pinocchio body placements.
        self.probes=[]
        self.targets=[]
        for q, pose in zip(self.data['q'],self.data['target_pose']):
            body=self.pin.poses(q)['payload']
            self.probes.append(_transform(body,[[0,0,.328]])[0])
            self.targets.append(_transform(_pose_transform(pose),[[0,0,.383]])[0])
        self.probes=np.array(self.probes)
        self.targets=np.array(self.targets)
        self.force_scale=max(100.,math.ceil(float(np.max(np.abs(self.data['force'])))/100)*100.)
        self.error_mm=1000*self.data['pose_error_m']
        self.duration=float(self.time[-1])

    def index(self, t):
        right=int(np.searchsorted(self.time,t))
        if right>=len(self.time): return len(self.time)-1
        if right>0 and t-self.time[right-1]<=self.time[right]-t: return right-1
        return right

    def frame(self, t):
        k=self.index(t)
        t=float(self.time[k])
        image=Image.new('RGB',(WIDTH,HEIGHT),BG)
        draw=ImageDraw.Draw(image)
        # Quiet geometric grid and a fixed camera retain a readable sense of scale.
        scene=Scene()
        for z in np.linspace(-.95,.95,13):
            _line3d(draw,scene,[[-.95,z,-.12],[.95,z,-.12]],(21,37,51))
            _line3d(draw,scene,[[z,-.95,-.12],[z,.95,-.12]],(21,37,51))
        # Grid is confined to the robot panel; panels below cover projected tails.
        poses=self.pin.poses(self.data['q'][k])
        I=np.eye(4)
        scene.cylinder(I,.645,-.115,-.03,(58,75,92),48)
        scene.cylinder(I,.618,-.029,-.021,(77,101,120),48)
        scene.cylinder(I,.562,-.020,-.012,(36,53,68),48)
        for a in np.linspace(0,2*np.pi,36,endpoint=False):
            scene.sphere([.602*np.cos(a),.602*np.sin(a),-.012],.007,TEAL,6,3)
        for i in range(6):
            base=poses[f'leg_{i}_yoke'][:3,3]
            tip=poses[f'leg_{i}_rod'][:3,3]
            barrel=poses[f'leg_{i}_barrel']
            rod=poses[f'leg_{i}_rod']
            scene.sphere(base,.039,(113,143,160))
            scene.cylinder(barrel,.030,.027,.47,(43,143,162),16)
            scene.cylinder(barrel,.034,.444,.482,(220,170,72),16)
            scene.cylinder(barrel,.033,.026,.063,(77,105,122),16)
            scene.cylinder(rod,.015,-.39,-.012,(210,225,232),14)
            scene.sphere(tip,.026,(222,175,80))
        platform=poses['platform']
        scene.cylinder(platform,.394,-.026,.024,(100,133,155),48)
        scene.cylinder(platform,.357,.025,.031,(140,177,190),48)
        for a in self.cmg['geometry']['platform_anchors_m']:
            point=_transform(platform,[a])[0]
            scene.sphere(point,.026,(229,186,94))
        payload=poses['payload']
        scene.box(payload,[.115,.105,.10],[0,0,.10],(220,230,234))
        scene.box(payload,[.117,.025,.103],[0,0,.10],(35,62,79))
        scene.cylinder(payload,.047,.20,.31,(84,111,126),20)
        scene.sphere(self.probes[k],.021,TEAL)
        scene.draw(draw)
        # Overlay reference and actual traces at the probe location. These are
        # trajectory annotations, deliberately not occlusion-tested solid parts.
        start=max(0,k-int(round(5./np.median(np.diff(self.time)))))
        step=max(1,(k-start)//160)
        track=np.arange(start,k+1,step)
        _line3d(draw,scene,self.targets[track],GOLD,2,True)
        _line3d(draw,scene,self.probes[track],TEAL,2)
        marker,_=scene.project([self.targets[k]])
        mx,my=marker[0]
        draw.ellipse((mx-5,my-5,mx+5,my+5),outline=GOLD,width=1)
        # A small world-axis triad is an orientation aid, not platform position.
        triad_origin=np.array([-.70,-.65,-.10])
        p0,_=scene.project([triad_origin]);p0=p0[0]
        for axis,label,color in [(np.array([.15,0,0]),'X',(241,109,112)),
                                  (np.array([0,.15,0]),'Y',(115,208,140)),
                                  (np.array([0,0,.15]),'Z',(100,158,240))]:
            p1,_=scene.project([triad_origin+axis]);p1=p1[0]
            draw.line((*p0,*p1),fill=color,width=2)
            _txt(draw,(p1[0]+5,p1[1]-6),label,12,color,True)
        # Disturbance indication uses the actual recorded wrench (world force).
        force=np.asarray(self.data['wrench'][k,:3])
        moment=np.asarray(self.data['wrench'][k,3:])
        disturbed=np.linalg.norm(force)>1e-6 or np.linalg.norm(moment)>1e-6
        if np.linalg.norm(force)>1e-6:
            endpoint=platform[:3,3]
            startpoint=endpoint-.23*force/np.linalg.norm(force)
            coords,_=scene.project([startpoint,endpoint])
            draw.line([tuple(p) for p in coords],fill=(247,105,98),width=4)
            direction=coords[1]-coords[0]
            direction/=max(np.linalg.norm(direction),1e-12)
            normal=np.array([-direction[1],direction[0]])
            head=[coords[1],coords[1]-15*direction+6*normal,
                  coords[1]-15*direction-6*normal]
            draw.polygon([tuple(p) for p in head],fill=(247,105,98))
        # Header background cleans any projected floor geometry behind it.
        draw.rectangle((0,0,WIDTH,137),fill=BG)
        _txt(draw,(34,23),'STEWART PLATFORM',30,TEXT,True)
        _txt(draw,(36,65),'CMG  /  PACDM  /  PINOCCHIO',16,TEAL,True)
        _txt(draw,(36,96),'Closed-chain motion, force control and disturbance rejection',15,MUTED)
        draw.rounded_rectangle((976,25,1245,78),radius=12,fill=PANEL,outline=(44,66,83))
        _txt(draw,(993,35),'RECORDED SIMULATION',12,MUTED,True)
        _txt(draw,(993,53),f'{t:05.2f} s  /  {self.duration:05.2f} s',18,TEXT,True)
        _txt(draw,(1245,97),'6 actuators  ·  18 closure constraints',13,MUTED,anchor='ra')
        draw.line((35,132,1245,132),fill=(35,54,70),width=1)
        _txt(draw,(36,151),phase(t).upper(),14,TEXT,True)
        payload_mass=next(body['mass_kg'] for body in self.cmg['bodies'] if body['id']=='payload')
        _txt(draw,(36,176),f'{payload_mass:g} kg fixed payload  ·  world view',12,MUTED)
        # Quantitative right column.
        _round_panel(draw,(952,150,1246,430))
        _txt(draw,(971,166),'CURRENT STATE',13,MUTED,True)
        metrics=[('Position error',f'{self.error_mm[k]:.4f}','mm',TEAL),
                 ('Orientation error',f'{np.rad2deg(self.data["angle_error_rad"][k]):.4f}','deg',TEXT),
                 ('Loop closure',f'{1e6*self.data["closure_error_m"][k]:.2e}','µm',TEXT)]
        for row,(label,value,unit,color) in enumerate(metrics):
            yy=198+row*73
            _txt(draw,(972,yy),label,13,MUTED)
            _txt(draw,(972,yy+20),value,24,color,True)
            _txt(draw,(1224,yy+29),unit,13,MUTED,anchor='ra')
        _round_panel(draw,(952,444,1246,766))
        _txt(draw,(971,461),'ACTUATOR FORCES',13,MUTED,True)
        _txt(draw,(1227,486),f'display ±{self.force_scale:g} N',11,MUTED,anchor='ra')
        center=1108
        for i,value in enumerate(self.data['force'][k]):
            yy=517+i*30
            _txt(draw,(971,yy-5),f'L{i+1}',12,MUTED,True)
            draw.rounded_rectangle((1004,yy,1209,yy+8),radius=3,fill=(36,52,68))
            end=center+101*np.clip(value/self.force_scale,-1,1)
            draw.rectangle((min(center,end),yy,max(center,end),yy+8),fill=TEAL if value>=0 else BLUE)
            draw.line((center,yy-3,center,yy+11),fill=(108,132,151),width=1)
            _txt(draw,(1229,yy+11),f'{value:+.1f}',11,TEXT,anchor='ra')
        _txt(draw,(971,706),f'Force limit: ±{self.cmg["actuation"]["force_limit_N"]:g} N',12,MUTED)
        status='DISTURBANCE ACTIVE' if disturbed else 'No external disturbance'
        _txt(draw,(971,732),status,12,(247,129,105) if disturbed else MUTED,disturbed)
        if disturbed:
            draw.rounded_rectangle((36,264,303,309),radius=10,fill=(64,35,35),outline=(115,62,58))
            _txt(draw,(49,273),f'External force {np.linalg.norm(force):.1f} N',14,(255,162,133),True)
            _txt(draw,(49,292),f'Moment {np.linalg.norm(moment):.1f} N m',11,(226,147,126))
        # Trajectory legend and benchmark scope remain visible throughout.
        draw.line((38,215,68,215),fill=TEAL,width=3)
        _txt(draw,(77,206),'Actual probe path',12,MUTED)
        draw.line((38,239,47,239),fill=GOLD,width=2)
        draw.line((53,239,62,239),fill=GOLD,width=2)
        _txt(draw,(77,230),'Reference',12,MUTED)
        _txt(draw,(36,636),'Motion shown at true scale  ·  surfaces illustrative',11,MUTED)
        _plot(draw,self.time,self.error_mm,k,(35,657,894,109))
        draw.rectangle((0,776,WIDTH,HEIGHT),fill=BG)
        _txt(draw,(36,781),'Pinocchio + PACDM dynamics | CPU mesh rendering | recorded states, no interpolation',11,MUTED)
        _txt(draw,(1245,781),self.case.upper(),11,TEAL,anchor='ra')
        return image


def render_video(root, case='nominal', fps=30):
    """Render a complete recorded mission as an H.264 MP4 and source-linked poster.

    Frames are selected by nearest recorded time. The last frame is explicitly
    the final stored state. Playback covers the full simulated duration, without
    slowing the motion or manufacturing intermediate configurations.
    """
    import imageio.v2 as imageio
    import imageio_ffmpeg
    if not isinstance(fps,int) or fps<1:
        raise ValueError('fps must be a positive integer')
    movie=Movie(root,case)
    results=Path(root)/'results'
    output=results/('Stewart_Pinocchio.mp4' if case=='nominal' else f'Stewart_Pinocchio_{case}.mp4')
    poster=results/('poster.png' if case=='nominal' else f'poster_{case}.png')
    count=max(2,int(round(movie.duration*fps)))
    frame_times=np.arange(count,dtype=float)/fps
    frame_times[-1]=movie.duration
    writer=imageio.get_writer(str(output),fps=fps,codec='libx264',quality=8,
                             macro_block_size=16,pixelformat='yuv420p',
                             ffmpeg_log_level='error',output_params=['-movflags','+faststart'])
    try:
        for frame,t in enumerate(frame_times):
            writer.append_data(np.asarray(movie.frame(float(t))))
            if frame%150==0: print(f'Rendering {frame}/{count} frames',flush=True)
    finally:
        writer.close()
    # Favor the largest disturbance for a useful inspectable still.
    wrench_norm=np.linalg.norm(movie.data['wrench'][:,:3],axis=1)
    poster_time=float(movie.time[np.argmax(wrench_norm)]) if wrench_norm.max()>0 else movie.duration*.57
    movie.frame(poster_time).save(poster)
    metadata={
        'renderer':'Pinocchio + PACDM dynamics | deterministic CPU mesh rendering',
        'source_result_file':movie.source.name,
        'source_result_sha256':movie.source_sha256,
        'source_cmg_file':str(movie.cmg_path.relative_to(Path(root))),
        'source_cmg_sha256':movie.cmg_sha256,
        'video_file':output.name,'poster_file':poster.name,
        'video_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'frame_count':count,'fps':fps,'width':WIDTH,'height':HEIGHT,
        'simulated_duration_s':movie.duration,'video_duration_s':count/fps,
        'first_state_time_s':float(movie.time[0]),'last_state_time_s':float(movie.time[-1]),
        'actual_body_transforms':'stewart.pin_backend.PinBackend(cmg).poses(recorded_q)',
        'state_sampling':'nearest stored sample, no interpolation; final frame uses final state',
        'probe_actual':'Pinocchio payload body transform applied to local [0,0,0.328] m',
        'probe_reference':'target platform pose applied to local [0,0,0.383] m',
        'trail_window_s':5.0,
        'render_geometry':'Decorative meshes illustrate declared dimensions; no collision simulation.',
        'disturbance_vector':'Recorded world force, direction to scale with fixed 0.23 m annotation length.',
        'poster_state_time_s':poster_time,
        'ffmpeg_executable_name':Path(imageio_ffmpeg.get_ffmpeg_exe()).name,
        'render_scope':'Rendered motion shows the recorded simulation; numerical validation is in validation.json.'}
    path=results/('render_metadata.json' if case=='nominal' else f'render_metadata_{case}.json')
    path.write_text(json.dumps(metadata,indent=2)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--case',default='nominal')
    parser.add_argument('--fps',type=int,default=30)
    parser.add_argument('--preview',type=float,help='Save a PNG at this simulation time, without a video')
    args=parser.parse_args()
    if args.preview is not None:
        result=args.root/'results'/'preview.png'
        Movie(args.root,args.case).frame(args.preview).save(result)
        print(result)
    else:
        print(render_video(args.root,args.case,args.fps))


if __name__=='__main__':
    main()
