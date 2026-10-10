from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import argparse

BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Draw the Stewart CMG from its model records.')
parser.add_argument('--cmg',type=Path,default=BASE.parent/'data/stewart.cmg.json')
parser.add_argument('--output',type=Path,default=BASE/'figures')
args=parser.parse_args()
cmg=json.loads(args.cmg.read_text())
bodies={b['id']:b for b in cmg['bodies']}
joints={j['id']:j for j in cmg['joints']}
cuts={c['id']:c for c in cmg['closures']}
assert len(bodies)==26 and len(joints)==25 and len(cuts)==6
assert len(cmg['coordinate_ids'])==24 and len(cmg['independent_ids'])==6
assert sum(b['mass_kg']>0 for b in bodies.values())==20
for i in range(6):
    for suffix,a,b,kind in [('u_x','world',f'leg_{i}_yoke','revolute'),
                           ('u_y',f'leg_{i}_yoke',f'leg_{i}_barrel','revolute'),
                           ('length',f'leg_{i}_barrel',f'leg_{i}_rod','prismatic')]:
        j=joints[f'leg_{i}_{suffix}']
        assert (j['base_body'],j['follower_body'],j['type'])==(a,b,kind)
    assert (cuts[f'cut_{i}']['body1'],cuts[f'cut_{i}']['body2'])==(f'leg_{i}_rod','platform')

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                     'mathtext.fontset':'dejavusans'})
W,H=1600,1220
fig,ax=plt.subplots(figsize=(16,12.2))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
navy='#173850';ink='#253D50';muted='#607483';blue='#477C9F'
teal='#00877D';orange='#C7791C';gray='#8A9AA7'

def txt(x,y,s,size=12,color=ink,weight='normal',ha='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,
                   ha=ha,va='center',linespacing=1.45,**kw)

def box(x,y,w,h,title,body='',color=blue,fill='#F3F8FC',dash=False,size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=7',
                 linewidth=1.3,edgecolor=color,facecolor=fill,
                 linestyle=(0,(3,2)) if dash else '-',zorder=3))
    if body:
        txt(x+w/2,y+21,title,size,color,'bold',zorder=4)
        txt(x+w/2,y+h-17,body,10.5,muted,zorder=4)
    else:txt(x+w/2,y+h/2,title,size,color,'bold',zorder=4)

def edge(x,y1,y2,label,color=gray,style='-',lw=1.6,label_x=None):
    ax.plot([x,x],[y1,y2],color=color,linestyle=style,linewidth=lw,zorder=1)
    txt(x+13 if label_x is None else label_x,(y1+y2)/2,label,12,color,ha='left',zorder=4)

txt(55,43,'Stewart platform: canonical mechanism graph',25,navy,'bold',ha='left')
txt(55,84,'Six UPS limbs • Body instances and joint connections from the CMG model',13,muted,ha='left')
txt(55,116,'20 moving massive bodies + world  |  12 revolute + 6 actuated prismatic + 6 spherical + 1 fixed connection',12,ink,ha='left')

# Physical graph panel. The wide bars are single nodes with six attachment ports.
ax.add_patch(FancyBboxPatch((45,148),1100,890,boxstyle='round,pad=0,rounding_size=12',
             linewidth=1,edgecolor='#DAE3E9',facecolor='white',zorder=0))
txt(70,177,'PHYSICAL BODY–JOINT GRAPH',13,navy,'bold',ha='left')
txt(1120,177,'Joint axes are local',10.5,muted,ha='right')
box(85,220,1020,60,'world  /  fixed base',color=navy,fill='#EAF1F6',size=15)

xs=[170,340,510,680,850,1020]
for i,x in enumerate(xs):
    # A universal base joint is two successive revolute joints.
    edge(x,280,362,r'$R_x$',color=gray)
    ax.plot(x,280,'o',color=navy,markersize=3,zorder=4)
    for suffix,y in [('yoke',362),('barrel',508),('rod',670)]:
        ident=f'leg_{i}_{suffix}'
        box(x-75,y,150,66,ident,f"{bodies[ident]['mass_kg']:g} kg",size=11)
    edge(x,428,508,r'$R_y$',color=gray)
    edge(x,574,670,r'$P_z$'+'\n'+rf'$\ell_{i},\ f_{i}$',color=teal,lw=2.5)
    ax.add_patch(Rectangle((x-6,612),12,18,edgecolor=teal,facecolor='white',linewidth=1.8,zorder=4))
    edge(x,736,828,rf'$S_{i}$'+'\n'+f'cut_{i}',color=orange,style=(0,(5,3)),lw=1.9)
    ax.plot(x,828,'o',color=orange,markersize=3,zorder=4)

box(85,828,1020,65,'platform  |  12 kg',color=navy,fill='#EAF1F6',size=15)
edge(595,893,950,'Fixed',color=gray)
box(460,950,270,66,'payload','8 kg nominal',color=blue,size=13)
txt(820,977,'Universal joint: two revolute axes',10.5,muted,ha='left')
txt(820,1003,'Each limb: universal–prismatic–spherical',10.2,muted,ha='left')

# The implementation introduces a pose-chart branch. Display it separately so
# its scalar joints are never confused with physical world-platform joints.
ax.add_patch(FancyBboxPatch((1175,148),380,890,boxstyle='round,pad=0,rounding_size=12',
             linewidth=1,edgecolor='#DAE3E9',facecolor='#FBFCFD',zorder=0))
txt(1365,178,'COMPUTATIONAL POSE CHART',12.5,navy,'bold')
txt(1365,207,'Five massless intermediate frames',10.5,muted)

chart_x=1275;chart_w=180;chart_c=1365
chart=['world','chart_x','chart_y','chart_z','chart_yaw','chart_pitch','platform']
ys=[240,338,436,534,632,730,828]
labels=[r'$P_x\ (x)$',r'$P_y\ (y)$',r'$P_z\ (z)$',r'$R_z\ (\mathrm{yaw})$',
        r'$R_y\ (\mathrm{pitch})$',r'$R_x\ (\mathrm{roll})$']
for k,(name,y) in enumerate(zip(chart,ys)):
    endpoint=k in [0,6]
    box(chart_x,y,chart_w,44,name,color=navy if endpoint else gray,
        fill='#EAF1F6' if endpoint else 'white',dash=not endpoint,size=11.5)
    if k<6:
        edge(chart_c,y+44,ys[k+1],labels[k],color=gray,style=(0,(2,3)),lw=1.4)
txt(1365,924,'world and platform repeat\nthe same physical graph nodes.',11,muted)
txt(1365,984,'Six pose coordinates;\nall chart frames have zero inertia.',11,muted)

# Compact legend, followed by the numerical structure actually used in PACDM.
legend_y=1073
for x,color,style,label in [(65,gray,'-','Revolute / fixed joint'),
    (430,teal,'-','Actuated prismatic joint'),
    (830,orange,(0,(5,3)),'Spherical point cut'),
    (1200,gray,(0,(2,3)),'Pose-chart connection')]:
    ax.plot([x,x+48],[legend_y,legend_y],color=color,linestyle=style,linewidth=2)
    txt(x+61,legend_y,label,11,ink,ha='left')
box(45,1110,1510,85,'PACDM coordinate structure',
    '24 tree coordinates + 18 free cut-orientation coordinates = 42 augmented coordinates'
    '   |   36 closure rows   |   6 independent leg lengths',
    color=navy,fill='#F1F5F8',size=13)

out=args.output
out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'Stewart_Graph.pdf',facecolor='white')
fig.savefig(out/'Stewart_Graph.png',dpi=260,facecolor='white')
plt.close(fig)
