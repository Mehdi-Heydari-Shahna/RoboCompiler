"""Render actual logged full-body trajectories; rendering is explicitly replay."""
import os,sys,subprocess,shutil,json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from native_model import export
from reconstructed_model import whole_body
ROOT=Path(__file__).resolve().parent
W,H=1440,900

def font(size,bold=False):
    for path in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf'),Path('C:/Windows/Fonts/'+('arialbd.ttf' if bold else 'arial.ttf'))]:
        if path.exists():return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)
FONTS={n:font(n) for n in [15,16,17,18,20,23]};BOLD={n:font(n,True) for n in [17,18,20,25,31,36]}

def scene(c,floating):
    path=export(c,ROOT/'models'/('standing_visual.xml' if floating else 'whole_body_visual.xml'),floating=floating,drive=True,visual=True,contact=floating,timestep=.000025 if floating else .00005,solref=.0001 if floating else .0004)
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);m.vis.headlight.ambient[:]=.25;m.vis.headlight.diffuse[:]=.35;m.light_diffuse[:]*=.7
    joints={j['id']:j for j in c['joints']};ports={joints[a['joint']]['follower_body']:a['joint'] for a in c['actuators']};housing={joints[a['joint']]['base_body'] for a in c['actuators']}
    for i in range(m.ngeom):
        body=m.body(m.geom_bodyid[i]).name
        if body in ports:m.geom_rgba[i]=[1,.61,.20,1] if 'left' in body else [.25,.72,1,1]
        elif body in housing and m.geom_group[i]==2:m.geom_rgba[i]=[.55,.64,.74,.32]
        elif body in ['base_link','torso'] and m.geom_group[i]==2:m.geom_rgba[i]=[.32,.38,.46,1]
    cam=mujoco.MjvCamera();cam.lookat[:]=[-.01,0,.65 if floating else -.18];cam.distance=2.20;cam.azimuth=132;cam.elevation=-9
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.tendongroup[:]=0;opt.geomgroup[0]=1
    renderer=mujoco.Renderer(m,height=710,width=840)
    order=np.array([m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in c['coordinate_ids']])
    return m,d,cam,opt,renderer,order

def canvas(rgb,force,t,stage,metrics):
    im=Image.new('RGB',(W,H),'#0c1420');im.paste(Image.fromarray(rgb),(0,111));dr=ImageDraw.Draw(im)
    dr.text((28,20),'KANGAROO | WHOLE-BODY PACDM',font=BOLD[31],fill='#f1f5fa')
    dr.text((28,68),'12 linear force actuators  |  76 internal coordinates  |  pelvis, torso and both legs',font=FONTS[20],fill='#b7c9df')
    x=868;dr.text((x,125),'01  ALL ACTUATORS MOVING' if stage==0 else '02  STANDING ON TWO FEET',font=BOLD[20],fill='#9ce2cc')
    dr.text((x,161),'Pelvis held fixed; force commands only' if stage==0 else 'Floating pelvis; floor contact supports weight',font=FONTS[17],fill='#d2dfed')
    dr.text((x,195),'ACTUAL MOTOR FORCES (N)',font=BOLD[17],fill='#b4c5db')
    names=['Hip yaw','Hip drive 2','Hip drive 3','Leg length','Ankle 4','Ankle 5']
    for side in range(2):
        bx=x+side*272;color='#ffb75f' if side==0 else '#65c6ff';dr.text((bx,229),'LEFT LEG' if side==0 else 'RIGHT LEG',font=BOLD[18],fill=color)
        for j in range(6):
            yy=267+j*46;value=float(force[side*6+j]);dr.text((bx,yy),names[j],font=FONTS[16],fill='#c4d1e1');dr.text((bx+175,yy),f'{value:+.0f}',font=BOLD[17],fill=color)
            dr.rectangle((bx,yy+27,bx+236,yy+30),fill='#29374b');center=bx+118;end=center+np.clip(value/800,-1,1)*118;dr.rectangle((min(center,end),yy+25,max(center,end),yy+32),fill=color)
    dr.line((x,568,1409,568),fill='#304157',width=2)
    if stage==0:
        dr.text((x,590),f"Motor mechanical power: {metrics['power']:+.2f} W",font=BOLD[20],fill='#f1f5fa')
        dr.text((x,629),f"Loop point gap: {metrics['gap']*1e6:.3f} micrometres",font=FONTS[18],fill='#a2dfce')
        dr.text((x,661),f"Energy ledger error: {metrics['balance']*1e3:+.3f} mJ",font=FONTS[18],fill='#a2dfce')
    else:
        dr.text((x,590),f"Ground force: {metrics['ground']:.2f} N",font=BOLD[25],fill='#f1f5fa')
        dr.text((x,631),'Robot weight: 414.13 N  |  mass: 42.21 kg',font=FONTS[18],fill='#a2dfce')
        dr.text((x,665),f"Loop point gap: {metrics['gap']*1e6:.3f} micrometres",font=FONTS[18],fill='#a2dfce')
    dr.text((x,710),'Source force bounds and joint limits enabled',font=FONTS[17],fill='#bfcee0')
    dr.text((x,747),f'Simulation time {t:.2f} s  |  playback 0.5x',font=FONTS[18],fill='#92a9c5')
    dr.rectangle((0,826,W,H),fill='#223146');dr.text((28,837),'Published-constraint reconstruction; joint assumptions are documented in the CMG metadata.',font=FONTS[18],fill='#f7d4a1');dr.text((28,870),'Native MuJoCo trajectory replay. Amber / blue highlight left / right actuator components.',font=FONTS[16],fill='#b7c9df')
    return im

def render(preview=False):
    c=whole_body();motion=dict(np.load(ROOT/'results/motion_dt_5e-05.npz'));standing=dict(np.load(ROOT/'results/standing_2.5e-05.npz'));proc=None
    if not preview:
        if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg required on PATH')
        proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate','30','-i','-','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'videos/Kangaroo_full_body_PACDM.mp4')],stdin=subprocess.PIPE)
    for stage,(arr,nframes,duration) in enumerate([(motion,240,4.),(standing,120,2.)]):
        m,d,cam,opt,renderer,order=scene(c,bool(stage))
        for frame in ([0,nframes//2,nframes-1] if preview else range(nframes)):
            t=duration*frame/(nframes-1);i=int(np.argmin(abs(arr['sample_t']-t)))
            if stage:d.qpos[:]=arr['q'][i]
            else:d.qpos[order]=arr['q'][i]
            d.qvel[:]=0;d.ctrl[:]=arr['force'][i];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam,scene_option=opt)
            if stage:
                j=int(np.argmin(abs(arr['history'][:,0]-t)));h=arr['history'][j];metrics=dict(ground=h[8],gap=h[9])
            else:
                j=int(np.argmin(abs(arr['time']-t)));h=arr['history'][j];metrics=dict(power=h[2],gap=h[5],balance=arr['balance'][j])
            im=canvas(renderer.render(),arr['force'][i],t,stage,metrics)
            if frame in [0,nframes//2,nframes-1]:im.save(ROOT/'videos'/f'frame_{stage}_{frame}.png')
            if proc:proc.stdin.write(np.asarray(im).tobytes())
        renderer.close()
    if proc:
        proc.stdin.close()
        if proc.wait():raise RuntimeError('ffmpeg failed')
    print('Full-body video '+('preview' if preview else 'render')+' complete',flush=True)
if __name__=='__main__':render('--preview' in sys.argv)
