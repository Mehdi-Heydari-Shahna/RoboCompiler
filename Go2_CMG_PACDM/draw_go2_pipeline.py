from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(15.5, 9.9))
ax.set(xlim=(0,15.5), ylim=(0,9.9), aspect='equal')
ax.axis('off')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
navy='#18354B'; ink='#243B4D'; line='#5B7181'; teal='#177E83'

def box(x,y,w,h,title,lines,fill='#F6F9FC',border='#A8B7C4',title_color=navy):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.10',
                    linewidth=1.45,edgecolor=border,facecolor=fill,zorder=3)
    ax.add_patch(p)
    ax.text(x+w/2,y+h-.37,title,ha='center',va='center',fontsize=17,
            fontweight='bold',color=title_color,zorder=4)
    ax.text(x+w/2,y+h-.97,'\n'.join(lines),ha='center',va='center',
            fontsize=13.7,linespacing=1.55,color=ink,zorder=4)

def route(points,color=line,width=1.8,dashed=False):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=15,
                               linewidth=width,color=color,
                               linestyle=(0,(5,4)) if dashed else '-',zorder=2))

def label(x,y,s,fontsize=12.6,**kw):
    ax.text(x,y,s,ha='center',va='center',fontsize=fontsize,color=ink,
            bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=5,**kw)

ax.text(.8,9.45,'Go2: CMG–PACDM implementation and validation',
        fontsize=23.5,fontweight='bold',color=navy,ha='left',va='center')

# State-dependent mechanics and control share one clearly marked runtime group.
ax.add_patch(FancyBboxPatch((5.4,3.10),4.4,5.58,
             boxstyle='round,pad=0.015,rounding_size=.15',
             linewidth=1.35,edgecolor='#9AB3C7',facecolor='#F2F7FC',zorder=0))
ax.text(7.6,8.39,'ONLINE CONTROLLER',ha='center',va='center',fontsize=12.4,
        color='#42627D',fontweight='bold')
ax.text(7.6,3.35,'Current state drives dynamics and control',
        ha='center',va='center',fontsize=11.6,color='#42627D')

box(.8,6.50,4.0,1.60,'Canonical mechanism graph',
    ['Body–joint topology and inertias','12 motor ports and joint limits'])
box(.8,3.70,4.0,1.60,'PACDM reference generation',
    ['Foot-task assembly and tangent lift','Joint position, velocity, acceleration'],
    fill='#EFF9F7',border='#8DBFB8',title_color='#146E6D')
box(5.6,6.50,4.0,1.60,'Pinocchio dynamics backend',
    [r'$M(q),\ h(q,\dot{q})$','Compiled from the CMG'],
    fill='white',border='#A7BED1')
box(5.6,3.70,4.0,1.60,'Whole-body QP',
    ['Base dynamics and predicted foot forces','Motor limits and friction constraints'],
    fill='white',border='#A7BED1')
box(10.7,6.50,4.0,1.60,'Task and environment',
    ['Floor, rails, and low gate','Scheduled external pushes'],
    fill='#FFFAEF',border='#D6C092',title_color='#8A6420')
box(10.7,3.70,4.0,1.60,'MuJoCo simulation',
    ['Robot motion from 12 motor torques','Unilateral frictional contact'],
    fill='#F0F8F8',border='#96BBBC',title_color='#146E6D')

# Model and reference inputs.
label(2.8,8.67,'Authored Go2 model (MJCF)',fontsize=13)
route([(2.8,8.44),(2.8,8.13)])
route([(4.83,7.3),(5.55,7.3)])
route([(2.8,6.47),(2.8,5.34)])
label(2.8,5.90,'CMG kinematics',fontsize=12.2)
route([(4.83,4.5),(5.55,4.5)],color=teal)
route([(7.6,6.47),(7.6,5.34)])
label(7.6,5.90,r'$M(q),\ h(q,\dot q)$',fontsize=14)
route([(9.63,4.5),(10.65,4.5)],color=teal,width=2.0)
label(10.14,4.78,'12 torques',fontsize=11.4)
route([(12.7,6.47),(12.7,5.34)])
label(12.7,5.90,'Contacts / loads',fontsize=12.2)

ax.add_patch(FancyBboxPatch((.8,2.25),4,.80,
             boxstyle='round,pad=.015,rounding_size=.10',
             linewidth=1.2,facecolor='white',edgecolor='#8DBFB8',zorder=3))
ax.text(2.8,2.65,'Prescribed body and foot trajectories',
        ha='center',va='center',fontsize=13.1,color='#146E6D',zorder=4)
route([(2.8,3.08),(2.8,3.66)],color=teal)

# Feedback is routed below the forward path and enters the controller group.
route([(12.7,3.67),(12.7,2.17),(7.6,2.17),(7.6,3.07)],
      color=teal,width=1.9)
label(10.1,2.40,r'State feedback $(q,\dot q)$',fontsize=13)

# Validation panel states all evidence sources; outer dashed paths stay clear
# of model/control links and the feedback loop.
ax.add_patch(FancyBboxPatch((.8,.35),13.9,1.16,
             boxstyle='round,pad=.015,rounding_size=.11',
             linewidth=1.3,edgecolor='#A8B7C4',facecolor='#F6F8FA',zorder=3))
ax.text(7.75,1.13,'Numerical validation and reporting',ha='center',va='center',
        fontsize=17,fontweight='bold',color=navy,zorder=4)
ax.text(7.75,.69,'PACDM closure and tangent maps   •   Pinocchio–MuJoCo mechanics   •   Contact and tracking metrics',
        ha='center',va='center',fontsize=13.1,color=ink,zorder=4)
route([(.77,4.5),(.35,4.5),(.35,.93),(.76,.93)],dashed=True,width=1.4)
route([(14.73,4.5),(15.15,4.5),(15.15,.93),(14.74,.93)],dashed=True,width=1.4)

for ext in ['png','pdf','svg']:
    kw={'dpi':300} if ext=='png' else {}
    fig.savefig(OUT/f'Go2_Diagram.{ext}',facecolor='white',**kw)
plt.close(fig)
print('Saved PNG, PDF, and SVG diagram.')
