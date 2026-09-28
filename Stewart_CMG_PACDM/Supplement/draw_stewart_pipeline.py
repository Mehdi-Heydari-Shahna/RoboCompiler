from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import argparse

BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Draw the Stewart CMG-PACDM implementation diagram.')
parser.add_argument('--output',type=Path,default=BASE/'figures')
args=parser.parse_args()
OUT=args.output
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                     'mathtext.fontset':'dejavusans'})
W,H=1400,1240
fig,ax=plt.subplots(figsize=(14,12.4))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
fig.patch.set_facecolor('white')
navy='#16324B'; ink='#233A4B'; muted='#576B7B'; line='#8294A1'
teal='#137D81'; tealbg='#EEF8F7'; blue='#315D96'; bluebg='#F0F5FC'

def text(x,y,s,size=12,color=ink,weight='normal',ha='center',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,
                   ha=ha,va=va,linespacing=1.48,**kw)

def box(x,y,w,h,title,body='',color=navy,fill='white',title_size=14,body_size=11.5):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=9',
                    linewidth=1.25,edgecolor=color,facecolor=fill,zorder=3)
    ax.add_patch(p)
    text(x+w/2,y+25,title,title_size,color,'bold',zorder=4)
    if body: text(x+w/2,y+(h+35)/2,body,body_size,ink,zorder=4)
    return p

def path(points,color=ink,dashed=False,arrow=True,lw=1.6):
    xs,ys=zip(*points)
    ax.plot(xs,ys,color=color,linewidth=lw,
            linestyle=(0,(4,3)) if dashed else '-',zorder=2,
            solid_capstyle='round',solid_joinstyle='round')
    if arrow:
        a,b=points[-2],points[-1]
        dx,dy=b[0]-a[0],b[1]-a[1]
        n=(dx*dx+dy*dy)**.5
        start=(b[0]-min(14,n)*dx/n,b[1]-min(14,n)*dy/n)
        ax.add_patch(FancyArrowPatch(start,b,arrowstyle='-|>',mutation_scale=12,
                                    color=color,linewidth=lw,zorder=5,
                                    shrinkA=0,shrinkB=0))

text(55,42,'Stewart CMG–PACDM implementation',24,navy,'bold',ha='left')
text(55,78,'Shared mechanism model  /  Separate force-feedback rollouts  /  Numerical validation',
     12,muted,ha='left')

box(325,115,750,98,'Canonical mechanism graph (CMG)',
    'Bodies and inertias • Joint transforms • Six point cuts • Six actuator ports',
    color=navy,fill='#F1F5F8',title_size=17,body_size=12)

# Common physical source independently supplies the three computational views.
path([(700,213),(700,245)],color=line,dashed=True,arrow=False)
path([(255,245),(1145,245)],color=line,dashed=True,arrow=False)
for x in [255,700,1145]:path([(x,245),(x,280)],color=line,dashed=True)
text(1225,227,'Model compilation',10,muted)

box(55,280,400,112,'PACDM closure',
    'Ordered paths and closure residuals\nTangent lift $E$ and passive reconstruction',
    color=teal,fill=tealbg)
box(500,280,400,112,'Pinocchio backend',
    'Tree mass $M_T$ and bias $h_T$\nInverse dynamics (RNEA)',
    color=teal,fill=tealbg)
box(945,280,400,112,'MuJoCo model',
    '24 tree coordinates\nSix native point-connect equalities',
    color=blue,fill=bluebg)

path([(255,392),(255,446)],color=teal)
path([(700,392),(700,446)],color=teal)
box(180,446,745,100,'PACDM reference + projected inverse dynamics',
    'Assembled motion • Compatible velocities and accelerations\n'
    r'Reference signals: $\ell_r$, $\dot{\ell}_r$, $f_{\mathrm{ff}}$',
    color=teal,fill=tealbg,title_size=14,body_size=12)

# Reference outputs split; each plant closes its own state-feedback loop.
path([(552.5,546),(552.5,590)],arrow=False)
path([(365,590),(1035,590)],arrow=False)
for x in [365,1035]:path([(x,590),(x,710)])
text(720,572,'Same reference signals',11,muted)

for x,c,f,title in [(55,teal,tealbg,'A   Direct PACDM rollout'),
                    (725,blue,bluebg,'B   Native MuJoCo rollout')]:
    ax.add_patch(FancyBboxPatch((x,632),620,402,boxstyle='round,pad=0,rounding_size=12',
                 linewidth=1,edgecolor=c,facecolor=f,zorder=0))
    text(x+25,666,title,13,c,'bold',ha='left')

for x,c in [(140,teal),(810,blue)]:
    box(x,710,450,76,'Leg-force controller',
        'Feedforward + leg feedback + force saturation',
        color=c,title_size=13,body_size=11)
    path([(x+225,786),(x+225,854)],color=c)
    text(x+238,820,r'Leg forces $f$',11,c,ha='left')

box(140,854,450,122,'Reduced dynamics + closure',
    'Six independent leg lengths\n'
    'Pinocchio dynamics • PACDM reconstruction\n'
    r'Semi-implicit Euler • $\Delta t=2$ ms',
    color=teal,title_size=13,body_size=11.2)
box(810,854,450,122,'MuJoCo native dynamics',
    '24 coordinates • Six connect equalities\n'
    r'implicitfast • $\Delta t=2$ ms'+'\n'
    'Time-step refinement: 1 ms',
    color=blue,title_size=13,body_size=11.2)

# Separate feedback states; never connect the two controllers to one state.
path([(140,915),(102,915),(102,748),(140,748)],color=teal)
text(84,833,r'$\ell,\dot{\ell}$',12,teal,rotation=90)
path([(1260,915),(1303,915),(1303,748),(1260,748)],color=blue)
text(1323,833,r'$\ell,\dot{\ell}$',12,blue,rotation=90)

# The same prescribed world wrench enters each dynamics block.
path([(652,916),(590,916)],color=teal)
text(624,895,r'$w(t)$',12,teal)
path([(748,916),(810,916)],color=blue)
text(777,895,r'$w(t)$',12,blue)

# The CMG-derived MuJoCo model supplies the native integrator, not Pinocchio.
path([(1145,392),(1145,425),(1376,425),(1376,995),
      (1218,995),(1218,976)],color=line,dashed=True)
text(1360,720,'Compiled MuJoCo model',10,muted,rotation=90)

path([(365,976),(365,1104)],color=teal)
path([(1035,976),(1035,1104)],color=blue)
text(700,1068,'Recorded states, forces and closure residuals',11,muted)
box(55,1104,1290,98,'Numerical verification and reporting',
    'Matching-state mechanics • Closure and rank • Actuator virtual power\n'
    'Trajectory agreement • Payload increase • Feedforward ablation • Time-step refinement',
    color=navy,fill='#F1F5F8',title_size=15,body_size=12)

fig.savefig(OUT/'Stewart_Diagram.pdf',facecolor='white')
fig.savefig(OUT/'Stewart_Diagram.png',dpi=260,facecolor='white')
plt.close(fig)
