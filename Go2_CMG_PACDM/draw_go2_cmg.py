from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.path import Path as MPath

ROOT=Path(__file__).resolve().parent
cmg=json.loads((ROOT/'data/go2_cmg.json').read_text())
bodies={b['id']:b for b in cmg['bodies']}
joints={j['id']:j for j in cmg['joints']}
assert len(bodies)==19 and len(joints)==18
assert sum(b['mass_kg']>0 for b in bodies.values())==13
assert sum(j['type']=='revolute' for j in joints.values())==15
assert sum(j['type']=='prismatic' for j in joints.values())==3
assert not cmg['closures']

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                     'svg.fonttype':'none','ps.fonttype':42})
fig,ax=plt.subplots(figsize=(15.1,12.5))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(0,15.1),ylim=(0,12.5),aspect='equal');ax.axis('off')
navy='#17384F'; blue='#23729A'; grey='#778693'; ink='#304F65'
positions={'world':(1.5,10),'chart_0':(4.5,10),'chart_1':(7.5,10),
           'chart_2':(10.5,10),'chart_3':(13.5,10),
           'chart_4':(13.5,8.15),'base':(7.5,8.15)}
for leg,x in zip(['FL','FR','RL','RR'],[1.8,5.6,9.4,13.2]):
    for part,y in [('hip',5.85),('thigh',3.95),('calf',2.05)]:
        positions[f'{leg}_{part}']=(x,y)
assert set(positions)==set(bodies)

def route(points,color=grey):
    p=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=16,
        linewidth=1.85,color=color,zorder=2))
def text(x,y,s,size=13,color=ink,weight='normal',ha='center',bg=None):
    kw={} if bg is None else {'bbox':dict(facecolor=bg,edgecolor='none',pad=2)}
    ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,
            fontweight=weight,zorder=5,**kw)
def rectangle(name,label,w=2.6,h=.86,fill='#EDF5FA',edge='#8AAEC1'):
    x,y=positions[name]
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=.015,rounding_size=.10',linewidth=1.45,
        facecolor=fill,edgecolor=edge,zorder=3))
    text(x,y,label,17,navy,'bold')

text(.6,11.98,'Unitree Go2 canonical mechanism graph',25,navy,'bold','left')
text(.6,11.43,'13 physical bodies  •  12 motorized leg joints  •  6 floating-base coordinates',16,ink,ha='left')

# The six scalar stages are the actual unactuated base chart in go2_cmg.json.
ax.add_patch(FancyBboxPatch((.45,7.43),14.1,3.48,
    boxstyle='round,pad=.015,rounding_size=.13',linewidth=1.2,
    edgecolor='#CFD9E0',facecolor='#F8FAFC',zorder=0))
text(.72,10.63,'FLOATING-BASE CHART',12.2,grey,'bold','left')
rectangle('world','World',1.8,.74,'#E9EDF1','#94A4B1')
for i in range(5):
    x,y=positions[f'chart_{i}']
    ax.add_patch(Circle((x,y),.34,linewidth=1.5,edgecolor=grey,
                       facecolor='white',zorder=3))
    text(x,y,rf'$c_{i}$',17,grey)
    if i==3:
        text(x+.40,y-.60,f'chart_{i}',11.8,grey,ha='left')
    else:
        text(x,y-.60,f'chart_{i}',11.8,grey)
rectangle('base','Trunk / base',2.65,.92,'#DCECF5','#6699B6')
text(2.7,8.43,'Five massless intermediate frames',13.4,ink)
text(2.7,8.04,'Six unactuated scalar joints',13.4,ink)
text(2.7,7.67,'XYZ translation + ZYX orientation',12.2,grey)

base_edges=[
 ('base_x',[(2.43,10),(4.11,10)],(3.20,10.25),r'$P_x$: base $x$'),
 ('base_y',[(4.87,10),(7.11,10)],(6,10.25),r'$P_y$: base $y$'),
 ('base_z',[(7.87,10),(10.11,10)],(9,10.25),r'$P_z$: base $z$'),
 ('base_yaw',[(10.87,10),(13.11,10)],(12,10.25),r'$R_z$: yaw'),
 ('base_pitch',[(13.5,9.63),(13.5,8.53)],(12.67,9.04),r'$R_y$: pitch'),
 ('base_roll',[(13.11,8.15),(8.88,8.15)],(10.8,8.43),r'$R_x$: roll')
]
seen=[]
for name,points,labelpos,label in base_edges:
    assert name in joints
    seen.append(name);route(points)
    text(*labelpos,label,13.1,grey,bg='#F8FAFC')

# All leg edges are motorized revolute joints: hip x, thigh y, calf y.
for leg,x in zip(['FL','FR','RL','RR'],[1.8,5.6,9.4,13.2]):
    yhip=positions[f'{leg}_hip'][1]
    for part,axis in [('hip','x'),('thigh','y'),('calf','y')]:
        jid=f'{leg}_{part}_joint';j=joints[jid];seen.append(jid)
        parent=j['base_body'];child=j['follower_body']
        assert child==f'{leg}_{part}'
        xp,yp=positions[parent];xc,yc=positions[child]
        if part=='hip':
            points=[(7.5,7.66),(7.5,6.94),(x,6.94),(x,yc+.48)]
            ly=6.59
        else:
            points=[(xp,yp-.47),(xc,yc+.48)]
            ly=(yp+yc)/2
        route(points,blue)
        text(x+.42,ly,rf'$R_{axis}$',15,blue,bg='white')
    for part in ['hip','thigh','calf']:
        rectangle(f'{leg}_{part}',f'{leg} {part}')
    meanings={'FL':'Front left','FR':'Front right','RL':'Rear left','RR':'Rear right'}
    text(x,1.34,meanings[leg],12.6,ink)
assert len(seen)==18 and set(seen)==set(joints)

text(.62,.81,'Blue edges: actuated revolute joints',12.7,blue,ha='left')
text(7.02,.81,'Gray edges: unactuated chart joints',12.7,grey,ha='left')
text(.62,.36,'Expanded CMG: 19 nodes = 13 bodies + 5 chart frames + world   |   18 edges = 15 R + 3 P   |   Structural cycle rank: 0',12.5,navy,ha='left')

out=ROOT/'figures';out.mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:
    fig.savefig(out/f'Go2_Graph.{ext}',
                dpi=300 if ext=='png' else 100,facecolor='white')
print('Graph checked against CMG: 19 nodes, 18 edges, 13 physical bodies, 12 actuated leg joints.')
