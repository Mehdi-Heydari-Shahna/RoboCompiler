"""Render the actual recorded native trajectory, with synchronized force data."""
import os
if os.name!='nt':os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,json,subprocess,shutil
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from render_whole_body import font
from contact_task import make_model
from contact_audit import load_trace

ROOT=Path(__file__).resolve().parent
W,H=1600,1000
F={n:font(n) for n in [16,18,20,22,24]};B={n:font(n,True) for n in [18,20,22,26,34]}

def stage(t):
    if t<.101:return '01  RELEASE & FALL','Both feet start 5 cm above the floor.'
    if t<1.4:return '02  LAND & ABSORB','The feet meet the floor; leg drives brake the descent.'
    if t<2.6:return '03  CONTROLLED CROUCH','Lower the pelvis by 12 cm using the leg drives.'
    if t<5.3:return '04  SHIFT & TURN','Shift weight to both sides and rotate the pelvis.'
    if t<6.4:return '05  RISE','Extend the legs and return to standing height.'
    if t<7.:return '06  SIDEWAYS PUSH','Apply a 50 N peak load at the torso center of mass.'
    return '07  RECOVER','Ground forces and the twelve drives restore balance.'

def arrow(scene,start,vector,rgba,width=.007):
    if np.linalg.norm(vector)<.003:return
    if scene.ngeom>=scene.maxgeom:return
    g=scene.geoms[scene.ngeom];scene.ngeom+=1
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3),np.zeros(3),np.eye(3).flatten(),np.array(rgba,dtype=np.float32))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,width,np.array(start),np.array(start)+vector)

def render(preview=False,name='landing_nominal'):
    a=load_trace(name);r=json.loads((ROOT/'results'/f'{name}.json').read_text())
    c,path=make_model('contact_visual',r['timestep_s'],r['configuration'],visual=True)
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    m.vis.global_.offwidth=max(1600,m.vis.global_.offwidth);m.vis.global_.offheight=max(1000,m.vis.global_.offheight)
    m.vis.headlight.ambient[:]=.35;m.vis.headlight.diffuse[:]=.65;m.vis.headlight.specular[:]=.15
    records={j['id']:j for j in c['joints']};ports={records[x['joint']]['follower_body'] for x in c['actuators']}
    for i in range(m.ngeom):
        body=m.body(m.geom_bodyid[i]).name
        if body in ports:m.geom_rgba[i]=[1,.62,.2,1] if 'left' in body else [.2,.72,1,1]
        elif body in ['base_link','torso'] and m.geom_group[i]==2:m.geom_rgba[i]=[.55,.62,.70,1]
        elif m.geom_group[i]==2:m.geom_rgba[i,:3]=np.maximum(m.geom_rgba[i,:3],.24)
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.70];cam.distance=2.4;cam.azimuth=135;cam.elevation=-9
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.tendongroup[:]=0
    renderer=mujoco.Renderer(m,height=785,width=950)
    feet=[m.body(n+'_ankle_roll').id for n in ['left','right']];torso=m.body('torso').id
    # Explicit variable replay speed: the short landing event needs slow motion.
    segments=[(0,.4,.12),(.4,1.4,.6),(1.4,6.4,.7),(6.4,7.5,.4),(7.5,10.,.7)]
    frames=[]
    for lo,hi,speed in segments:
        frames.extend((float(t),speed) for t in np.linspace(lo,hi,round((hi-lo)/speed*30),endpoint=False))
    frames.append((10.,.7));process=None
    if not preview:
        if not shutil.which('ffmpeg'):raise RuntimeError('Install ffmpeg and add it to PATH')
        process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'videos/Kangaroo_v22_contact_task.mp4')],stdin=subprocess.PIPE)
    selected=[(t,.5) for t in [.08,.14,2.6,3.5,6.825,9.9]] if preview else frames
    for fi,(t,speed) in enumerate(selected):
        k=int(np.argmin(abs(a['time']-t)));t=float(a['time'][k]);d.qpos[:]=a['q'][k];d.qvel[:]=a['v'][k];d.act[:]=a['act'][k];d.ctrl[:]=a['command'][k];d.xfrc_applied[:]=0;d.xfrc_applied[torso,:3]=a['push'][k]
        mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam,scene_option=opt)
        for j,bid in enumerate(feet):
            start=np.r_[d.xpos[bid,:2],.008]+[0,.13 if j==0 else -.13,0]
            arrow(renderer.scene,start,a['foot_force'][k,j]*.00028,[.13,.88,.6,1])
        arrow(renderer.scene,d.xipos[torso]-np.array([0,.30,0]),a['push'][k]*.005,[1,.28,.28,1],.011)
        im=Image.new('RGB',(W,H),'#0d1521');im.paste(Image.fromarray(renderer.render()),(0,110));draw=ImageDraw.Draw(im)
        draw.text((30,20),'KANGAROO  |  CONTACT & WHOLE-BODY DYNAMICS',font=B[34],fill='#f3f6fa')
        draw.text((32,69),'78 bodies  /  floating pelvis  /  12 linear force actuators  /  PACDM references',font=F[22],fill='#b5c8df')
        x=978;title,description=stage(t);draw.text((x,124),title,font=B[26],fill='#8ce5c5')
        # Split prose naturally to fit the narrow telemetry panel.
        words=description.split();lines=['']
        for word in words:
            proposed=(lines[-1]+' '+word).strip()
            if draw.textlength(proposed,font=F[18])>582:lines.append(word)
            else:lines[-1]=proposed
        for j,line in enumerate(lines):draw.text((x,164+25*j),line,font=F[18],fill='#c7d6e8')
        draw.text((x,229),'ACTUAL DRIVE FORCES  (N)',font=B[20],fill='#e6eef8')
        labels=['Hip yaw','Hip drive 2','Hip drive 3','Leg length','Ankle 4','Ankle 5']
        for side in range(2):
            bx=x+side*300;color='#ffb760' if side==0 else '#62caff';draw.text((bx,265),'LEFT' if side==0 else 'RIGHT',font=B[18],fill=color)
            for j,label in enumerate(labels):
                yy=303+j*48;value=a['act'][k,side*6+j]
                draw.text((bx,yy),label,font=F[18],fill='#c7d6e8');draw.text((bx+181,yy),f'{value:+.0f}',font=B[18],fill=color)
                draw.rectangle((bx,yy+29,bx+266,yy+32),fill='#28374b');mid=bx+133;end=mid+np.clip(value/3000,-1,1)*133
                draw.rectangle((min(mid,end),yy+27,max(mid,end),yy+34),fill=color)
        draw.line((x,610,1570,610),fill='#33445b',width=2)
        draw.text((x,631),f"Ground: {a['foot_force'][k,:,2].sum():.0f} N",font=B[26],fill='#8ce5c5')
        draw.text((x+310,638),'Weight: 414 N',font=F[20],fill='#c7d6e8')
        draw.text((x,679),f"Motor power: {a['motor_power'][k].sum():+.1f} W",font=B[22],fill='#f3f6fa')
        draw.text((x,721),f"Motor work: {a['motor_work'][k].sum():+.2f} J",font=F[20],fill='#c7d6e8')
        draw.text((x,758),f"Energy ledger residual: {a['ledger'][k]*1000:+.1f} mJ",font=F[20],fill='#c7d6e8')
        draw.text((x,807),f'Simulation {t:5.2f} s  |  playback {speed:.2f}x',font=F[20],fill='#90aac9')
        draw.text((30,903),'Green: ground reaction (arrows offset beside feet)    Red: applied torso disturbance',font=F[20],fill='#a7e4ce')
        draw.rectangle((0,944,W,H),fill='#243246')
        draw.text((30,960),'Published-cut reconstruction. Assumed drive response and compliant contact; hardware validation remains separate.',font=F[20],fill='#f0d5ac')
        if preview:im.save(ROOT/'videos'/f'contact_preview_{t:.3f}.png')
        elif fi%150==0:im.save(ROOT/'videos/contact_poster.png')
        if process:process.stdin.write(np.asarray(im).tobytes())
    renderer.close()
    if process:
        process.stdin.close()
        if process.wait():raise RuntimeError('ffmpeg failed')
    print('Contact video '+('preview' if preview else 'complete'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--name',default='landing_nominal');a=p.parse_args();render(a.preview,a.name)
