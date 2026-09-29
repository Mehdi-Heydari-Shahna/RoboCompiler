"""Render saved native states; this module never advances or corrects motion."""
import os
import sys
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL','egl')
from project import ROOT, RESULTS
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def save_preview(image, target):
    """Publish a verified complete PNG, never a partially written preview."""
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.png',delete=False) as stream:
            temporary=stream.name
            image.save(stream,format='PNG');stream.flush();os.fsync(stream.fileno())
        with Image.open(temporary) as check:check.verify()
        os.replace(temporary,target);temporary=None
    finally:
        if temporary is not None:os.unlink(temporary)


def font(size,bold=False):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',
                 'C:/Windows/Fonts/'+('arialbd.ttf' if bold else 'arial.ttf')]:
        try:return ImageFont.truetype(path,size)
        except OSError:pass
    return ImageFont.load_default()


def render_video(case='nominal',fps=24,speed=2.,preview_only=False,preview_times=None,inset_mode='auto'):
    trace_path=RESULTS/f'{case}.npz';report_path=RESULTS/f'{case}.json';model_path=RESULTS/'assets'/f'{case}.xml'
    stable_inputs={p.relative_to(ROOT).as_posix():sha256(p) for p in [trace_path,model_path,ROOT/'tracked_video.py']}
    with np.load(RESULTS/f'{case}.npz',allow_pickle=False) as a:
        trace={k:a[k] for k in a.files}
    report=json.loads((RESULTS/f'{case}.json').read_text())
    report_start_hash=sha256(report_path)
    # These are every simulator-report field consumed by the renderer. Audit
    # annotations may update during a long replay without changing its content.
    report_projection={key:report.get(key) for key in ['scene','stage_names','power_order']}
    model=mujoco.MjModel.from_xml_path(str(RESULTS/'assets'/f'{case}.xml'))
    data=mujoco.MjData(model)
    model.vis.global_.offwidth=max(1280,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(800,model.vis.global_.offheight)
    model.vis.headlight.ambient[:]=.5
    model.vis.headlight.diffuse[:]=.65
    # Replay-only depth/shadow precision; no physical model file changes.
    model.vis.map.znear=.02
    model.vis.map.shadowclip=2.
    model.vis.quality.shadowsize=4096
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[2.,.1,1.55]
    camera.distance=15.8;camera.azimuth=120.;camera.elevation=-24.
    inset_camera=mujoco.MjvCamera();inset_camera.distance=4.;inset_camera.azimuth=105.;inset_camera.elevation=-20.
    bucket=model.body('body_56').id;base=model.body('body_53').id
    power_map={k:i for i,k in enumerate(report['power_order'])}
    duration=float(trace['time'][-1]);total_frames=max(1,round(duration*fps/speed)+1)
    if preview_times is None:preview_times=[0.,min(10.,duration),min(24.,duration),min(40.,duration)]
    preview_frames={min(total_frames-1,round(t*fps/speed)) for t in preview_times}
    frame_ids=sorted(preview_frames) if preview_only else range(total_frames)
    out=ROOT/'Excavator_Tracked_v26.mp4'
    pipe=None
    temporary_video=None
    if not preview_only:
        try:
            import imageio_ffmpeg
            exe=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            exe=shutil.which('ffmpeg')
            if not exe:raise RuntimeError('Install imageio-ffmpeg or make ffmpeg available on PATH.')
        descriptor,temporary_name=tempfile.mkstemp(prefix='.v26_video_',suffix='.mp4',dir=out.parent)
        os.close(descriptor);temporary_video=Path(temporary_name)
        pipe=subprocess.Popen([exe,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x800','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary_video)],stdin=subprocess.PIPE)
    large=font(29,True);medium=font(22,True);small=font(17);tiny=font(14)
    try:
        with mujoco.Renderer(model,800,1280) as renderer,mujoco.Renderer(model,245,390) as inset:
            for frame in frame_ids:
                requested=min(duration,frame*speed/fps)
                k=int(np.argmin(abs(trace['time']-requested)));t=float(trace['time'][k])
                data.qpos[:]=trace['qpos'][k];data.qvel[:]=trace['qvel'][k]
                if 'ctrl' in trace:data.ctrl[:]=trace['ctrl'][k]
                data.time=t;mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera,scene_option=option)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,1280,85),fill=(20,39,53))
                draw.text((24,12),'ROBOIR  |  Articulated-track excavator',font=large,fill='white')
                shoes=2*report.get('scene',{}).get('tracks',{}).get('shoes_per_side',50)
                draw.text((25,51),f'{shoes} articulated shoes  ·  hydraulic travel drives  ·  PACDM arm closure  ·  native contact',font=small,fill=(177,213,232))
                stage_id=int(trace['stage'][k]);names=report.get('stage_names',[])
                stage=names[stage_id] if stage_id<len(names) else f'Stage {stage_id}'
                bucket_view=inset_mode=='bucket' or (inset_mode=='auto' and any(word in stage for word in ('dig','bucket','receiver','delivered','unload','clear')))
                receiver_view=inset_mode=='auto' and any(word in stage for word in ('position closed','open above','settle delivered','close bucket'))
                if receiver_view and 'depot_center_m' in report.get('scene',{}):
                    center=np.asarray(report['scene']['depot_center_m'])
                    inset_camera.lookat[:]=[center[0],center[1],.85]
                    inset_camera.distance=4.7;inset_camera.azimuth=70.;inset_camera.elevation=-35.
                    inset_title='Bucket opening / native receiving-bay contact'
                elif bucket_view:
                    inset_camera.lookat[:]=data.xpos[bucket]+data.xmat[bucket].reshape(3,3)@np.array([-.108,-1.77,.85])
                    inset_camera.distance=3.;inset_camera.azimuth=35+np.rad2deg(trace['independent'][k,0]-trace['independent'][0,0])+np.rad2deg(trace['pose'][k,2]);inset_camera.elevation=-28.
                    inset_title='Bucket cavity / native particle contact'
                else:
                    inset_camera.lookat[:]=data.xpos[base]+data.xmat[base].reshape(3,3)@np.array([.35,2.1034,.33])
                    inset_camera.distance=4.6;inset_camera.azimuth=95+np.rad2deg(trace['pose'][k,2]);inset_camera.elevation=-18.
                    inset_title='Articulated belt / sprocket engagement'
                inset.update_scene(data,inset_camera,scene_option=option)
                im.paste(Image.fromarray(inset.render()),(866,105));draw=ImageDraw.Draw(im)
                draw.rectangle((865,104,1257,351),outline=(89,184,203),width=2)
                draw.rectangle((866,351,1256,381),fill=(20,39,53));draw.text((877,358),inset_title,font=tiny,fill='white')
                draw.rectangle((0,621,1280,800),fill=(20,39,53))
                draw.text((24,634),f'{stage.capitalize()}  |  t = {t:5.2f} s  |  playback {speed:g}×',font=medium,fill=(247,204,93))
                displacement=np.linalg.norm(trace['pose'][k,:2]-trace['pose'][0,:2])
                drive_power=trace['power'][k,power_map['drive_mechanical']]/1000
                supply=sum(trace['work'][k,power_map[x]] for x in ['arm_supply','drive_supply'])/1000
                values=[('Inside / at destination*',f"{trace['captured_mass'][k]:.1f} / {trace['delivered_mass'][k]:.1f} kg"),
                        ('Base displacement',f'{displacement:.2f} m'),('Drive mechanical power',f'{drive_power:+.1f} kW'),('Hydraulic supply work',f'{supply:.1f} kJ')]
                for i,(label,value) in enumerate(values):
                    x=24+315*i;draw.text((x,677),label,font=small,fill=(177,213,232));draw.text((x,706),value,font=large,fill='white')
                draw.text((24,759),'*Preliminary occupancy, not audited settled delivery. Native recorded states; no field calibration.',font=tiny,fill=(174,191,203))
                draw.text((24,779),'Assumed track / hydraulic parameters and coarse rigid aggregate.',font=font(12),fill=(174,191,203))
                if frame in preview_frames:
                    suffix='' if inset_mode=='auto' else '_'+inset_mode
                    target=RESULTS/f'{case}_video_t{t:05.2f}{suffix}.png';save_preview(im,target);print(target,flush=True)
                if pipe is not None:pipe.stdin.write(np.asarray(im).tobytes())
                if frame%120==0:print(f'Rendered {frame}/{total_frames} frames',flush=True)
    finally:
        if pipe is not None:
            pipe.stdin.close()
            status=pipe.wait()
            if status!=0:
                temporary_video.unlink(missing_ok=True)
                raise RuntimeError(f'Video encoder failed with status {status}')
    if not preview_only:
        try:
            if not temporary_video.exists() or temporary_video.stat().st_size < 1024:
                raise RuntimeError('Video encoder did not produce a valid nonempty artifact.')
            # Decode the entire stream before atomically publishing the final MP4.
            verified=subprocess.run([exe,'-v','error','-i',str(temporary_video),'-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True)
            if verified.returncode or verified.stderr.strip():
                raise RuntimeError('Encoded video decode verification failed: '+verified.stderr)
            decoded_counts=[int(line.split('=',1)[1]) for line in verified.stdout.splitlines() if line.startswith('frame=')]
            if not decoded_counts or decoded_counts[-1]!=total_frames:
                raise RuntimeError(f'Encoded video frame count differs: expected {total_frames}, decoded {decoded_counts[-1] if decoded_counts else None}')
            changed=[name for name,expected in stable_inputs.items() if sha256(ROOT/name)!=expected]
            if changed:raise RuntimeError('Replay inputs changed during rendering: '+', '.join(changed))
            current_report=json.loads(report_path.read_text())
            if {key:current_report.get(key) for key in report_projection}!=report_projection:
                raise RuntimeError('A simulator-report field used by the video changed during rendering')
            with temporary_video.open('rb') as stream:os.fsync(stream.fileno())
            os.replace(temporary_video,out)
        finally:
            temporary_video.unlink(missing_ok=True)
        inputs=[RESULTS/f'{case}.npz',RESULTS/f'{case}.json',RESULTS/'assets'/f'{case}.xml',ROOT/'tracked_video.py']
        receipt=dict(case=case,video=out.name,video_sha256=sha256(out),complete_stream_decode_verified=True,
                     frames=total_frames,decoded_frames=decoded_counts[-1],fps=fps,playback_speed=speed,source_duration_s=duration,
                     report_fields_used=list(report_projection),report_start_sha256=report_start_hash,
                     rendered_report_projection_sha256=hashlib.sha256(json.dumps(report_projection,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                     report_unused_metadata_changed_during_render=report_start_hash!=sha256(report_path),
                     width=1280,height=800,input_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in inputs})
        (RESULTS/'video_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print(out,flush=True)
        return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',default='nominal');p.add_argument('--fps',type=int,default=24)
    p.add_argument('--speed',type=float,default=2.);p.add_argument('--preview-only',action='store_true')
    p.add_argument('--preview-times',type=float,nargs='*')
    p.add_argument('--inset-mode',choices=['auto','track','bucket'],default='auto')
    args=p.parse_args()
    if args.fps<=0 or args.speed<=0:p.error('fps and speed must be positive')
    render_video(args.case,args.fps,args.speed,args.preview_only,args.preview_times,args.inset_mode)
