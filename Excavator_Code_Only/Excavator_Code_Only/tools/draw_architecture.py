from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath

import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Draw the CMG-PACDM implementation and validation flow.')
parser.add_argument('--output',type=Path,default=ROOT/'figures')
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42})
fig, ax = plt.subplots(figsize=(12.4, 10.1))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(-0.1,15.2)
ax.set_ylim(12.4,-0.05)
ax.axis('off')

ink='#172B40'; line='#45596D'; blue='#356F9E'; teal='#23766C'
amber='#A67526'; gray='#66768A'

def box(cx,cy,w,h,title,detail,stroke,fill):
    x,y=cx-w/2,cy-h/2
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.018,rounding_size=0.10',
        linewidth=1.25,edgecolor=stroke,facecolor=fill,zorder=3))
    if '\n' in detail:
        ty=cy-h/2+0.38
        dy=cy+0.20
    else:
        ty=cy-0.22
        dy=cy+0.25
    ax.text(cx,ty,title,ha='center',va='center',fontsize=14.4,weight='bold',color=ink,zorder=4)
    ax.text(cx,dy,detail,ha='center',va='center',fontsize=12.2,color=ink,linespacing=1.6,zorder=4)

def route(points,color=line,style='->',dash=False,lw=1.5):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle=style,mutation_scale=13,
       linewidth=lw,color=color,linestyle=(0,(4,3)) if dash else '-',zorder=2,
       capstyle='round',joinstyle='round'))

def label(x,y,s,rotation=0,ha='center',size=11.3,color=line):
    ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,rotation=rotation,
        linespacing=1.3,zorder=5,bbox=dict(facecolor='white',edgecolor='none',pad=1.8))

box(5,1,5.4,1.15,'CMG: physical mechanism graph',
    'Bodies · joints · frames · inertias · ports',blue,'#EEF5FB')
box(5,3.35,5.4,1.2,'PACDM closure and reduction',
    'Closed configuration · tangent lift E',blue,'#EEF5FB')
box(12,3.35,5.2,1.2,'Pinocchio differential backend',
    'Point-closure Jacobians and derivatives',teal,'#F0F8F6')
box(5,6,5.4,1.6,'Floating-arm inverse dynamics',
    'Source-body mass and bias\nMeasured base coupling',blue,'#EEF5FB')
box(5,8.5,5.4,1.2,'Actuation',
    'Hydraulic cylinders · rotary / travel drives',blue,'#EEF5FB')
box(5,11,5.4,1.35,'MuJoCo native plant',
    'Floating robot · tracks · granular material',amber,'#FFF8EC')
box(12,11,5.2,1.35,'Contact environment',
    'Granular bed · terrain · receiving bay',amber,'#FFF8EC')
box(12,7,5.2,2.3,'Numerical validation and reporting',
    'Closure and tangent agreement\nForce / energy balance · material transfer\nJSON · NPZ · trajectory replay',gray,'#F3F5F8')

# Main implementation chain.
route([(5,1.575),(5,2.75)])
route([(5,3.95),(5,5.2)])
route([(5,6.8),(5,7.9)])
label(5,7.34,'arm effort requests')
route([(5,9.1),(5,10.325)])
label(5,9.72,'actual efforts')

# Shared physical description and independent differential evaluation.
route([(7.7,1),(12,1),(12,2.75)],color=teal)
label(10.0,0.75,'shared physical model',color=teal)
route([(7.7,3.35),(9.4,3.35)],color=teal)
label(8.55,3.03,'closed q',size=10.8,color=teal)
route([(9.4,3.95),(8.55,4.40),(7.7,5.2)],color=teal)
label(10.65,4.67,'point differentials\nand tangent comparison',color=teal)

# One measured-state bus feeds both reconstruction and base-conditioned dynamics.
route([(2.3,11),(0.85,11),(0.85,3.35),(2.3,3.35)],color=blue)
route([(0.85,6.45),(2.3,6.45)],color=blue)
ax.plot([0.85],[6.45],'o',markersize=3.7,color=blue,zorder=3)
label(0.34,8.5,'measured state and base motion',rotation=90,color=blue,size=11.4)

# Task reference is separate from measured-state feedback.
route([(1.25,5.55),(2.3,5.55)])
label(1.58,5.02,'task\nreference',size=11.1)
route([(1.25,8.5),(2.3,8.5)])
label(1.58,7.99,'travel-servo\nrequests',size=10.8)

# Dashed links carry recorded numerical evidence.
route([(7.7,6.35),(9.4,6.35)],dash=True)
label(8.55,5.83,'controller\nresiduals',size=10.6)
route([(7.7,10.65),(8.65,10.65),(8.65,7.70),(9.4,7.70)],dash=True)
label(8.42,9.28,'native trajectory',rotation=90,size=10.9)

# Both sides of contact are solved by the native plant.
route([(7.7,11),(9.4,11)],color=amber,style='<->')
label(8.55,11.57,'native\ncontact',size=10.6,color=amber)

label(7.7,12.15,'Solid arrows: model / control / contact flow     Dashed arrows: recorded evidence',size=10.8)
fig.savefig(OUT/'ex_diag.png',dpi=320,facecolor='white')
fig.savefig(OUT/'ex_diag.pdf',facecolor='white',metadata={
  'Title':'CMG–PACDM excavator implementation and validation flow',
  'Subject':'Source-verified mechanism, controller, native simulation and evidence relationships'})
plt.close(fig)
