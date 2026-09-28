"""Render recorded native states. This script never generates a task trajectory."""
import os
import sys
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'v26'))
import argparse
import hashlib
import json
import subprocess
import shutil
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from tracked_video import font
from project import save_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(case='soil_demo',speed=2.,fps=25,preview_only=False):
    out=ROOT/'outputs'/case
    with np.load(out/'trajectory.npz',allow_pickle=False) as f:
        tr={k:f[k] for k in f.files}
    report=json.loads((out/'report.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=800
    m.vis.headlight.ambient[:]=.55;m.vis.headlight.diffuse[:]=.7
    m.vis.map.znear=.01
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[3.3,.6,1.1]
    camera.distance=14.;camera.azimuth=122.;camera.elevation=-25.
    close=mujoco.MjvCamera();close.lookat[:]=[6.65,0,-.02]
    close.distance=4.8;close.azimuth=126.;close.elevation=-38.
    bucket=m.body('body_56').id
    duration=float(tr['time'][-1]);count=max(1,int(duration*fps/speed)+1)
    frames=sorted({0,count//3,2*count//3,count-1}) if preview_only else range(count)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    pipe=None
    target=out/'soil_demo.mp4';temporary=out/'soil_demo.pending.mp4'
    if not preview_only:
        pipe=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
            '-pix_fmt','rgb24','-s','1280x800','-r',str(fps),'-i','-',
            '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(m,600,1280) as renderer, mujoco.Renderer(m,270,440) as inset:
            for frame in frames:
                t=min(duration,frame*speed/fps)
                k=min(len(tr['time'])-1,int(np.searchsorted(tr['time'],t)))
                d.qpos[:]=tr['qpos'][k];d.qvel[:]=tr['qvel'][k];d.ctrl[:]=tr['ctrl'][k]
                d.time=float(tr['time'][k])
                # Replay only needs placements; no contact solve or integration.
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)
                renderer.update_scene(d,camera,scene_option=option)
                canvas=Image.new('RGB',(1280,800),(19,33,44))
                canvas.paste(Image.fromarray(renderer.render()),(0,58))
                phase=int(tr['phase'][k]);label=report['phase_names'][phase]
                if phase>=4:
                    close.lookat[:]=d.xpos[bucket]+d.xmat[bucket].reshape(3,3)@np.array([-.108,-1.77,.85])
                    close.distance=4.
                else:
                    close.lookat[:]=[6.65,0,-.02];close.distance=4.8
                inset.update_scene(d,close,scene_option=option)
                canvas.paste(Image.fromarray(inset.render()),(820,78))
                draw=ImageDraw.Draw(canvas)
                draw.text((20,12),'RoboCompiler | Excavation-bed prototype',font=font(25,True),fill='white')
                draw.text((20,40),'Dry granular surrogate / uncalibrated material / recorded native dynamics',font=font(13),fill=(204,207,211))
                draw.rectangle((819,77,1260,348),outline=(225,190,98),width=2)
                draw.rectangle((820,349,1260,378),fill=(19,33,44))
                draw.text((830,354),'Bucket and excavation region',font=font(16),fill='white')
                draw.text((22,674),f'{label.capitalize()}  |  simulation {d.time:.2f} s  |  playback {speed:g}x',font=font(23,True),fill=(239,199,90))
                draw.text((22,714),f"Bucket-volume occupancy: {tr['bucket_mass'][k]:.1f} kg     Settled delivery: {tr['deposited_mass'][k]:.1f} kg",font=font(20),fill='white')
                draw.text((22,750),f"Soil-contact force: {np.linalg.norm(tr['soil_bucket_force'][k]):.0f} N     Run status: {report['status']}",font=font(17),fill=(192,213,224))
                draw.text((22,778),'Native contacts; no attached payload or prescribed robot motion. Material/terrain calibration pending.',font=font(13),fill=(174,185,196))
                if frame in {0,count//3,2*count//3,count-1}:
                    canvas.save(out/f'preview_{frame:04d}.png')
                if pipe is not None:pipe.stdin.write(np.asarray(canvas).tobytes())
                if frame%125==0:print(f'Rendered {frame}/{count}',flush=True)
    finally:
        if pipe is not None:
            pipe.stdin.close()
            if pipe.wait()!=0:raise RuntimeError('Video encoding failed')
    if not preview_only:
        subprocess.run([ffmpeg,'-v','error','-i',str(temporary),'-f','null','-'],check=True)
        temporary.replace(target)
        save_json(out/'video_manifest.json',dict(video=target.name,frames=count,fps=fps,speed=speed,
            native_trace_sha256=digest(out/'trajectory.npz'),scene_sha256=digest(out/'scene.xml'),
            report_sha256=digest(out/'report.json'),renderer_sha256=digest(__file__),video_sha256=digest(target),
            source='Recorded native qpos/qvel/ctrl; first saved state at or after each requested replay time',
            full_stream_decode_checked=True))
        print(target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',default='soil_demo')
    p.add_argument('--speed',type=float,default=2.);p.add_argument('--fps',type=int,default=25)
    p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    if a.speed<=0 or a.fps<=0:p.error('Positive speed and fps required')
    render(a.case,a.speed,a.fps,a.preview_only)
