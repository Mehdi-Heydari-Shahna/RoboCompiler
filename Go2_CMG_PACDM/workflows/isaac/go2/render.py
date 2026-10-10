"""Polished native-state replay; the renderer never advances the simulation.

All geometry comes from MuJoCo. HUD quantities are the nearest saved sample;
free-joint interpolation uses MuJoCo's manifold difference and integration.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' or sys.platform == 'darwin' else 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .task import PHASES

INK='#eaf2f7'; MUTED='#91aabd'; CYAN='#44d4d1'; AMBER='#ffba50'
BG='#0b1521'; PANEL='#132333'; LINE='#2a4559'
LEG_COLORS=['#45d0cb','#f6ba58','#8eabef','#e99ac1']
LEG_NAMES=['FL','FR','RL','RR']
LIMITS=np.tile([23.7,23.7,45.43],4)
_FONT_CACHE={}

def _font(size,bold=False):
    key=(size,bold)
    if key not in _FONT_CACHE:
        name='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        try: _FONT_CACHE[key]=ImageFont.truetype(str(Path('/usr/share/fonts/truetype/dejavu')/name),size)
        except OSError:
            import matplotlib
            _FONT_CACHE[key]=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf'/name),size)
    return _FONT_CACHE[key]

def phase_at(t):
    for i,(a,b,label) in enumerate(PHASES):
        if a<=t<b:return i,label
    return len(PHASES)-1,PHASES[-1][2]

def _load(root,case):
    with np.load(root/'results'/f'{case}.npz',allow_pickle=False) as saved:
        log={k:saved[k] for k in saved.files}
    for key in ('time','qpos','qvel','q','q_ref','normal_force','torque','push','stance'):
        if key not in log:raise ValueError(f'Missing replay quantity {key}')
    if len(log['time'])<2 or not np.all(np.diff(log['time'])>0):
        raise ValueError('Replay timestamps must increase strictly')
    return log

def _connector(scene,a,b,rgba,radius=.0015):
    a=np.asarray(a);b=np.asarray(b)
    if scene.ngeom>=scene.maxgeom or np.linalg.norm(a-b)<1e-7:return
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.full(3,radius),np.zeros(3),np.eye(3).ravel(),np.asarray(rgba,np.float32))
    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,a,b)
    scene.ngeom+=1

def _mini_trace(draw,t,log,index,rect):
    x0,y0,x1,y1=rect
    start=max(0,t-4);stop=max(4,t)
    ids=np.flatnonzero((log['time']>=start)&(log['time']<=t))
    for h in (.24,.32):
        y=y1-(h-.20)/.18*(y1-y0)
        draw.line((x0,y,x1,y),fill=LINE,width=1)
        draw.text((x0+3,y-13),f'{h:.2f}',font=_font(9),fill=MUTED)
    for key,color in [('q_ref',MUTED),('q',CYAN)]:
        points=[(x0+(log['time'][i]-start)/(stop-start)*(x1-x0),y1-(log[key][i,2]-.20)/.18*(y1-y0)) for i in ids]
        if len(points)>1:draw.line(points,fill=color,width=2 if key=='q' else 1)

def _compose(main,t,index,log,width=1280,height=800):
    canvas=Image.new('RGB',(1280,800),BG);d=ImageDraw.Draw(canvas)
    d.text((28,17),'CMG / PACDM  ·  ROBOT GENERALITY',font=_font(12,True),fill=CYAN)
    d.text((27,39),'UNITREE GO2',font=_font(32,True),fill=INK)
    d.text((301,53),'FLOATING BASE  /  CHANGING CONTACT',font=_font(15,True),fill=MUTED)
    d.text((1115,35),f'{t:05.2f} s',font=_font(26,True),fill=INK)
    d.line((28,91,1252,91),fill=LINE,width=1)
    canvas.paste(Image.fromarray(main),(28,112));d=ImageDraw.Draw(canvas)
    d.rounded_rectangle((28,112,958,719),radius=10,outline=LINE,width=1)
    phase,label=phase_at(t)
    pw=min(600,max(255,int(d.textlength(label.upper(),font=_font(13,True)))+74))
    d.rounded_rectangle((45,128,45+pw,175),radius=7,fill=BG)
    d.text((61,141),f'{phase+1:02d}  /  {label.upper()}',font=_font(13,True),fill=INK)
    d.rounded_rectangle((44,682,655,709),radius=5,fill=BG)
    d.text((55,690),'Recorded MuJoCo dynamics  ·  12 joint torques  ·  unilateral foot contact',font=_font(11),fill=MUTED)
    push=np.asarray(log['push'][index,:3]);mag=np.linalg.norm(push)
    if mag>1e-8:
        d.rounded_rectangle((45,185,307,225),radius=7,fill='#543e28')
        d.text((60,197),f'EXTERNAL PUSH  {mag:.0f} N',font=_font(13,True),fill=AMBER)
    x=979
    d.rounded_rectangle((x,112,1252,382),radius=10,fill=PANEL)
    d.text((x+17,130),'MEASURED FOOT SUPPORT',font=_font(12,True),fill=INK)
    d.text((x+17,153),'Vertical reaction  ·  support > 2 N',font=_font(10),fill=MUTED)
    # A schematic contact indicator, not synthetic robot geometry.
    d.rounded_rectangle((x+95,183,x+176,262),radius=14,fill=BG,outline=LINE,width=1)
    d.polygon([(x+129,189),(x+123,198),(x+135,198)],fill=MUTED)
    for k,(fx,fy) in enumerate([(x+80,197),(x+191,197),(x+80,249),(x+191,249)]):
        force=float(log.get('support_force',log['normal_force'])[index,k]);on=force>2
        color=LEG_COLORS[k] if on else LINE
        d.line((fx,fy,x+106 if k%2==0 else x+165,fy),fill=LINE,width=3)
        d.ellipse((fx-8,fy-8,fx+8,fy+8),fill=color,outline=LEG_COLORS[k])
        d.text((fx-10,fy-27),LEG_NAMES[k],font=_font(10,True),fill=MUTED)
        d.text((fx-16,fy+13),f'{force:4.0f} N',font=_font(10),fill=INK)
    for k in range(4):
        yy=298+k*17;force=max(0,float(log.get('support_force',log['normal_force'])[index,k]))
        d.text((x+18,yy-3),LEG_NAMES[k],font=_font(10,True),fill=LEG_COLORS[k])
        d.rounded_rectangle((x+49,yy,x+188,yy+5),radius=2,fill=BG)
        length=139*np.clip(force/150,0,1)
        if length>0:d.rounded_rectangle((x+49,yy,x+49+length,yy+5),radius=2,fill=LEG_COLORS[k])
        planned='STANCE' if log['stance'][index,k] else 'SWING'
        d.text((x+198,yy-3),planned,font=_font(8,True),fill=MUTED)
    d.rounded_rectangle((x,395,1252,544),radius=10,fill=PANEL)
    d.text((x+17,411),'BASE HEIGHT',font=_font(12,True),fill=INK)
    d.text((x+170,405),f'{log["q"][index,2]*1000:3.0f}',font=_font(24,True),fill=CYAN)
    d.text((x+226,420),'mm',font=_font(10),fill=MUTED)
    _mini_trace(d,t,log,index,(x+17,446,x+252,513))
    d.text((x+17,525),'Cyan: measured  ·  grey: reference',font=_font(9),fill=MUTED)
    d.rounded_rectangle((x,557,1252,719),radius=10,fill=PANEL)
    d.text((x+17,574),'MOTOR LIMIT USAGE',font=_font(12,True),fill=INK)
    ratios=np.max(np.abs(log['torque'][index].reshape(4,3))/LIMITS.reshape(4,3),axis=1)
    for k,value in enumerate(ratios):
        yy=604+21*k
        d.text((x+17,yy-3),LEG_NAMES[k],font=_font(10,True),fill=LEG_COLORS[k])
        d.rounded_rectangle((x+48,yy,x+202,yy+7),radius=3,fill=BG)
        length=154*np.clip(value,0,1)
        if length>0:d.rounded_rectangle((x+48,yy,x+48+length,yy+7),radius=3,fill=AMBER if value>.9 else LEG_COLORS[k])
        d.text((x+215,yy-3),f'{100*value:3.0f}%',font=_font(10),fill=INK)
    d.text((x+17,697),'Per leg: largest |torque| / source limit',font=_font(9),fill=MUTED)
    duration=float(log['time'][-1]);phasewidth=1224/len(PHASES)
    for k,(a,b,name) in enumerate(PHASES):
        xx=28+k*phasewidth
        d.rectangle((xx,741,xx+phasewidth-6,745),fill=CYAN if k==phase else LINE)
        d.text((xx,756),name,font=_font(11,k==phase),fill=INK if k==phase else MUTED)
    progress=28+1224*np.clip(t/duration,0,1)
    d.line((28,790,1252,790),fill=LINE,width=1);d.ellipse((progress-3,787,progress+3,793),fill=INK)
    if (width,height)!=(1280,800):canvas=canvas.resize((width,height),Image.Resampling.LANCZOS)
    return np.asarray(canvas)

def render(root,case='nominal',*,fps=30,width=1280,height=800,duration=None):
    """Write demo.mp4 and poster.png from the recorded trajectory.

    A duration-limited or non-nominal preview uses separate output filenames.
    """
    root=Path(root);results=root/'results';log=_load(root,case);times=log['time']
    model=mujoco.MjModel.from_xml_path(str((results/f'{case}.xml').resolve()))
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,930)
    model.vis.global_.offheight=max(model.vis.global_.offheight,607)
    model.vis.headlight.diffuse[:]=.4;model.vis.headlight.ambient[:]=.25;model.vis.headlight.specular[:]=.05
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=607,width=930,max_geom=5000)
    option=mujoco.MjvOption();option.sitegroup[:]=0
    for flag in (mujoco.mjtVisFlag.mjVIS_CONSTRAINT,mujoco.mjtVisFlag.mjVIS_JOINT,mujoco.mjtVisFlag.mjVIS_ACTUATOR):option.flags[flag]=False
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance=1.4;camera.azimuth=140;camera.elevation=-26
    suffix='' if case=='nominal' else '_'+case
    if duration is not None:suffix+='_preview'
    output=results/f'demo{suffix}.mp4';poster=results/f'poster{suffix}.png'
    stop=min(float(times[-1]),duration) if duration is not None else float(times[-1])
    frame_times=np.arange(float(times[0]),stop+1e-8,1/fps)
    # Prefer the recorded state at the gate; the scene determines its location.
    try:
        gate_x=float(model.geom('low_gate').pos[0])
        desired=float(times[np.argmin(abs(log['q'][:,0]-gate_x))])
    except KeyError:
        desired=float(PHASES[min(2,len(PHASES)-1)][0])+.15
    poster_index=int(np.argmin(abs(frame_times-min(stop,desired))))
    tangent=np.zeros(model.nv)
    writer=imageio.get_writer(str(output),fps=fps,codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-movflags','+faststart'])
    try:
        for frame_index,t in enumerate(frame_times):
            right=min(int(np.searchsorted(times,t)),len(times)-1);left=max(0,right-1)
            blend=(t-times[left])/(times[right]-times[left]) if right!=left else 0
            data.qpos[:]=log['qpos'][left]
            mujoco.mj_differentiatePos(model,tangent,1.,log['qpos'][left],log['qpos'][right])
            mujoco.mj_integratePos(model,data.qpos,tangent,blend)
            data.qvel[:]=(1-blend)*log['qvel'][left]+blend*log['qvel'][right]
            mujoco.mj_forward(model,data)
            camera.lookat[:]=data.xpos[model.body('base').id]+np.array([.09,0,-.03])
            renderer.update_scene(data,camera=camera,scene_option=option)
            # A short ground-projected base trail is an annotation derived from
            # recorded states. It does not alter the model or physical contacts.
            start=np.searchsorted(times,max(0,t-5));ids=np.linspace(start,right,min(70,max(2,right-start+1)),dtype=int)
            for a,b in zip(ids[:-1],ids[1:]):
                pa=np.r_[log['q'][a,:2],.003];pb=np.r_[log['q'][b,:2],.003]
                _connector(renderer.scene,pa,pb,[.18,.83,.81,.75],.002)
            nearest=left if blend<.5 else right
            frame=_compose(renderer.render().copy(),float(t),nearest,log,width,height)
            writer.append_data(frame)
            if frame_index==poster_index:Image.fromarray(frame).save(poster)
            if frame_index%(5*fps)==0:print(f'Rendered {frame_index}/{len(frame_times)} frames',flush=True)
    finally:
        writer.close();renderer.close()
    metadata=dict(source_case=case,video=output.name,poster=poster.name,fps=fps,frames=len(frame_times),resolution=[width,height],
                  render_backend=os.environ.get('MUJOCO_GL'),source='MuJoCo replay of saved native qpos/qvel; no simulation steps during rendering',
                  interpolation='MuJoCo tangent-space difference/integration, including the native free quaternion',
                  camera='Follows the measured source base body; no model coordinates changed for framing',
                  trail='Last five seconds of recorded base position projected onto the ground',
                  hud='Nearest logged sample; world-vertical support-force bars capped visually at 150 N, support indicated above 2 N',
                  lighting='Headlight adjusted for mesh contrast; physics and source scene unchanged')
    (results/f'render_metadata{suffix}.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return output

render_video=render
