"""Render recorded native states, with no motion editing or particle attachment."""
import os,sys,shutil,subprocess,json
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from benchmark_model import ROOT,load_trace
from digging_path import stage_at

def font(size,bold=False):
 choices=['/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf','C:/Windows/Fonts/'+('arialbd.ttf' if bold else 'arial.ttf')]
 for path in choices:
  try:return ImageFont.truetype(path,size)
  except OSError:pass
 return ImageFont.load_default()

def render_video(name='nominal',fps=30,speed=.5):
 a=load_trace(name);report=json.loads((ROOT/'results'/f'{name}.json').read_text());path=json.loads((ROOT/'results/digging_path.json').read_text())
 m=mujoco.MjModel.from_xml_path(str(ROOT/'results/assets'/f'{name}.xml'));d=mujoco.MjData(m);m.vis.headlight.ambient[:]=.5
 opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
 cam=mujoco.MjvCamera();cam.lookat[:]=[3.5,.2,1.8];cam.distance=13.5;cam.azimuth=115;cam.elevation=-24
 close=mujoco.MjvCamera();close.distance=2.8;close.azimuth=35;close.elevation=-30
 movie=ROOT/'Excavator_v21_digging.mp4';exe=shutil.which('ffmpeg')
 if not exe:
  import imageio_ffmpeg
  exe=imageio_ffmpeg.get_ffmpeg_exe()
 command=[exe,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1440x900','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
 pipe=subprocess.Popen(command,stdin=subprocess.PIPE);duration=float(a['time'][-1]);frames=round(duration*fps/speed);bucket=m.body('body_56').id
 fsmall=font(21);fmid=font(25,True);flarge=font(34,True)
 try:
  with mujoco.Renderer(m,900,1440) as renderer,mujoco.Renderer(m,320,500) as inset:
   for frame in range(frames):
    t=min(duration,frame*speed/fps);k=int(np.argmin(abs(a['time']-t)));d.qpos[:]=a['qpos'][k];d.qvel[:]=a['qvel'][k];mujoco.mj_forward(m,d)
    renderer.update_scene(d,cam,scene_option=opt);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,1440,96),fill=(18,31,44));draw.text((28,13),'ROBOIR  |  Full-body excavator',font=flarge,fill='white');draw.text((30,58),'Native MuJoCo contact · six hydraulic cylinders · two rotary drives',font=fsmall,fill=(177,209,227))
    # Track the bucket cavity, not its CAD origin. The inset uses the same native state.
    center=d.xpos[bucket]+d.xmat[bucket].reshape(3,3)@np.array([-.108,-1.77,.85]);close.lookat[:]=center;close.azimuth=35+np.degrees(a['independent'][k,0]-a['independent'][0,0])
    inset.update_scene(d,close,scene_option=opt);im.paste(Image.fromarray(inset.render()),(914,116));draw=ImageDraw.Draw(im);draw.rectangle((913,115,1414,436),outline=(238,192,71),width=3)
    draw.rectangle((914,436,1414,478),fill=(18,31,44));draw.text((930,446),'Bucket close-up · recorded contact motion',font=fsmall,fill='white')
    draw.rectangle((0,710,1440,900),fill=(18,31,44));stage=stage_at(t,path).capitalize();draw.text((28,725),f'{stage}  |  t = {t:4.1f} s  |  playback {speed:g}×',font=fmid,fill=(255,210,80))
    values=[('Material inside bucket',f"{a['captured_mass'][k]:.1f} kg"),('Bucket contact force',f"{np.linalg.norm(a['wrenches'][k,0,:3])/1000:.1f} kN"),('Maximum chamber pressure',f"{a['pressure'][k].max()/1e6:.1f} MPa"),('Hydraulic supply work',f"{a['work'][k,0]/1000:.1f} kJ")]
    for i,(label,value) in enumerate(values):
     x=28+i*355;draw.text((x,777),label,font=fsmall,fill=(177,209,227));draw.text((x,811),value,font=flarge,fill='white')
    draw.text((28,867),'Synthetic hydraulic parameters and coarse aggregate; simulation verification, not field calibration.',font=font(18),fill=(177,190,204))
    if frame in [0,round(8*fps/speed),round(11.5*fps/speed)]:im.save(ROOT/'results'/f'video_frame_{frame:04d}.png')
    pipe.stdin.write(np.asarray(im).tobytes())
    if frame%150==0:print(f'Rendered {frame}/{frames} frames',flush=True)
 finally:pipe.stdin.close()
 if pipe.wait()!=0:raise RuntimeError('Video encoder failed')
 print(movie,flush=True);return movie
if __name__=='__main__':render_video(sys.argv[1] if len(sys.argv)>1 else 'nominal')
