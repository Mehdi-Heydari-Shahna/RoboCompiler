#!/usr/bin/env python3
"""Draw the Kangaroo CMG/PACDM implementation and validation flow."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath


def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'supplement_figures',
                        help='Directory for PDF, PNG, and SVG figures')
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(16.4,9.55))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0,16.4); ax.set_ylim(9.55,0); ax.axis('off')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    INK='#172B42'; FLOW='#42556B'; AUDIT='#725B8C'

    nodes={}
    def box(key,x,y,w,h,title,lines,fill,edge):
        nodes[key]=(x,y,w,h)
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.10',
            linewidth=1.35,edgecolor=edge,facecolor=fill,zorder=3))
        ax.text(x+w/2,y+.32,title,ha='center',va='center',fontsize=15.1,fontweight='bold',color=INK,zorder=4)
        spacing=.30
        top=y+.72 if len(lines)<=2 else y+.68
        for i,t in enumerate(lines):
            ax.text(x+w/2,top+i*spacing,t,ha='center',va='center',fontsize=12.3,color=INK,zorder=4)

    def arrow(points,dashed=False):
        path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
        p=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=17,
                         linewidth=1.65,linestyle=(0,(5,3.5)) if dashed else '-',
                         color=AUDIT if dashed else FLOW,shrinkA=0,shrinkB=0,
                         capstyle='round',joinstyle='round',zorder=2)
        ax.add_patch(p)

    def label(x,y,text,fontsize=10.5,color=FLOW,rotation=0):
        ax.text(x,y,text,ha='center',va='center',fontsize=fontsize,color=color,
                rotation=rotation,zorder=6,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.2))

    ax.text(8.2,.38,'CMG–PACDM implementation for Kangaroo',ha='center',va='center',
            fontsize=22,fontweight='bold',color=INK)

    box('cmg',5.4,1.02,5.6,1.35,'Canonical mechanism graph (CMG)',
        ['78 bodies · 77 tree joints','24 closure cuts · 12 actuator ports'],'#EFF3F8','#567087')
    box('pacdm',.75,3.25,4.3,1.65,'PACDM + NumPy dynamics',
        ['Closure and tangent mapping','Foot-pose IK and feedforward'],'#EEF5FD','#527CAC')
    box('control',6.05,3.25,4.3,1.65,'Physical force control',
        ['1 kHz feedforward + PD','12 prismatic force ports'],'#EEF5FD','#527CAC')
    box('mujoco',11.35,3.25,4.3,1.65,'Native MuJoCo simulation',
        ['Floating pelvis and closed loops','Foot contacts and torso push','Force-driven time integration'],'#EDF7F4','#4C8979')
    box('pin',.75,6.55,4.3,1.65,'Pinocchio mechanics',
        ['Independent free-flyer tree','Mass, bias, kinematics and energy'],'#F4F0F8','#897399')
    box('audit',6.05,6.55,4.3,1.65,'Independent numerical audits',
        ['Source / Pinocchio / MuJoCo agreement','Force, power and energy balances'],'#FAF5EA','#A88C52')
    box('report',11.35,6.55,4.3,1.65,'Results and validation',
        ['Closure and task tracking','Force limits, recovery and refinement','Numerical gates and figures'],'#F0F6F0','#6D9071')

    # Three separately generated views of the same CMG mechanism records.
    arrow([(6.15,2.37),(6.15,2.78),(2.90,2.78),(2.90,3.25)])
    label(4.5,2.59,'closure + inertia data')
    arrow([(10.25,2.37),(10.25,2.78),(13.50,2.78),(13.50,3.25)])
    label(11.91,2.59,'native model')
    arrow([(5.40,1.65),(.25,1.65),(.25,7.375),(.75,7.375)])
    label(2.85,1.44,'independent model')

    # Runtime control and native state feedback.
    arrow([(5.05,4.075),(6.05,4.075)])
    label(5.55,3.69,'reference\n+ FF',9.7)
    arrow([(10.35,4.075),(11.35,4.075)])
    label(10.85,3.69,'force\ncommands',10.0)
    arrow([(13.50,4.90),(13.50,5.50),(8.20,5.50),(8.20,4.90)])
    label(10.85,5.74,'motor position / velocity + contact status',10.3)

    # Numerical comparison inputs, deliberately kept outside the runtime path.
    arrow([(2.90,4.90),(2.90,5.87),(6.73,5.87),(6.73,6.55)],True)
    label(4.78,5.65,'source mechanics',10.5,AUDIT)
    arrow([(5.05,7.375),(6.05,7.375)],True)
    label(5.55,7.06,'mechanics',9.7,AUDIT)
    arrow([(15.65,4.28),(16.13,4.28),(16.13,8.66),(8.20,8.66),(8.20,8.20)],True)
    label(12.15,8.88,'recorded states, contacts, forces and energy',10.7,AUDIT)
    arrow([(10.35,7.375),(11.35,7.375)])

    # Compact, unambiguous visual key.
    arrow([(3.00,9.26),(3.72,9.26)])
    ax.text(3.86,9.26,'Model, control and simulation data',ha='left',va='center',fontsize=10.6,color=FLOW)
    arrow([(9.28,9.26),(10.00,9.26)],True)
    ax.text(10.14,9.26,'Numerical-audit inputs',ha='left',va='center',fontsize=10.6,color=AUDIT)

    stem=OUT/'kang_dia'
    fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'),facecolor='white',metadata={'Title':'Kangaroo CMG-PACDM implementation flow','Author':''})
    fig.savefig(stem.with_suffix('.svg'),facecolor='white')
    print('\n'.join(str(stem.with_suffix(e)) for e in ['.png','.pdf','.svg']))

    plt.close(fig)


if __name__ == "__main__":
    main()
