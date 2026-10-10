#!/usr/bin/env python3
"""Draw the exact Kangaroo CMG topology in a fixed schematic layout.

All vertices and edges are read from the CMG; positions and short labels
are chosen for this 78-body Kangaroo reconstruction.
"""
from pathlib import Path
import argparse
import collections
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cmg', type=Path, default=ROOT / 'data' / 'whole_body_cmg.json',
                        help='Kangaroo canonical mechanism graph JSON')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'supplement_figures',
                        help='Directory for PDF, PNG, SVG, and graph audit JSON')
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    c = json.loads(args.cmg.read_text(encoding='utf-8'))
    # Positions encode only topology; all displayed vertices and edges come from CMG.
    spec={
     'yaw_motor':('YM',-2.4,-1.9), 'yaw_rod':('YR',-2.4,-.6),
     'hip_yaw':('HY',0,0), 'hip_pitch':('HP',0,1.6), 'hip_roll':('HR',0,3.3),
     'femour_pitch':('FP',0,5.7), 'leg_length_slidinig_nut':('LN',-2.0,6.9),
     'leg_length_short_bars':('LS',-2.0,8.3), 'knee':('K',0,9.4),
     'ankle_pitch':('AP',0,11.0), 'ankle_roll':('AR',0,12.6),
     'rear_short_bar':('RS',1.75,5.7), 'rear_triangle':('RT',1.75,7.25),
     'rear_long_bar':('RL',1.75,8.8),
    }
    for k,sgn in [(2,-1),(3,1)]:
     for suffix,code,y in [('cradle','C',1.2),('motor','M',2.6),('rod','R',4.0)]:
      spec[f'hip_differential_{k}_{suffix}']=(f'H{k}{code}',sgn*4.4,y)
     spec[f'hip_differential_{k}_univ']=(f'H{k}U',sgn*2.2,2.5)
    for k,sgn in [(4,-1),(5,1)]:
     for suffix,code,x,y in [('motor','M',7,5.7),('rod','R',7,7.2),('pendulum','P',5,7.2)]:
      spec[f'ankle_{k}_{suffix}']=(f'A{k}{code}',sgn*x,y)
     for suffix,code,x,y in [('higher_ankle_bar',f'A{k}H',5,9.1),('butterfly',f'B{k}',3.2,10),
                               ('knee_ball',f'K{k}',3.2,11.5),('lower_ankle_bar',f'L{k}',3.2,13),
                               ('ankle_ball',f'A{k}B',1.7,14.1)]:
      spec[f'{k}_{suffix}']=(code,sgn*x,y)
    assert len(spec)==38
    pos={'base_link':(0,-4.5),'torso':(0,-6.0)}
    labels={'base_link':'Pelvis','torso':'Torso'}
    for side,offset,mirror in [('left',-9.2,1),('right',9.2,-1)]:
     for name,(lab,x,y) in spec.items():pos[side+'_'+name]=(offset+mirror*x,y);labels[side+'_'+name]=lab
    assert set(pos)=={b['id'] for b in c['bodies']}
    edges=[]
    for j in c['joints']:
     edges.append(dict(id=j['id'],a=j['base_body'],b=j['follower_body'],kind=j['type']))
    for j in c['closures']:
     edges.append(dict(id=j['id'],a=j['body1'],b=j['body2'],kind=j['type']))
    counts=collections.Counter(e['kind'] for e in edges)
    assert counts=={'revolute':64,'prismatic':12,'fixed':1,'point_coincidence':16,'universal':8}
    assert len(edges)-len(pos)+1==24
    # Check connectivity and the CMG spanning tree using disjoint sets.
    parent={n:n for n in pos}
    def find(x):
     while parent[x]!=x:x=parent[x]
     return x
    for e in edges[:77]:
     a,b=find(e['a']),find(e['b']);assert a!=b;parent[a]=b
    assert len({find(n) for n in pos})==1

    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(24,18.5));fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(-18.6,18.6);ax.set_ylim(20.2,-8.5);ax.axis('off')
    INK='#17324A';R='#718092';P='#13769E';F='#24364A';C='#C97524';U='#8756A5'
    style={'revolute':(R,'-',1.6),'prismatic':(P,'-',3.5),'fixed':(F,'-',3.3),
           'point_coincidence':(C,(0,(6,3)),2.5),'universal':(U,(0,(6,2.4,1.1,2.4)),2.5)}
    for side,x in [('left',-17.5),('right',.9)]:
     ax.add_patch(FancyBboxPatch((x,-2.8),16.6,18.2,boxstyle='round,pad=0,rounding_size=.23',
                 facecolor='#F8FAFD' if side=='left' else '#F6FAF8',edgecolor='#D8E1E8',linewidth=1.0,zorder=0))
     hx=-15 if side=='left' else 15
     ax.text(hx,-1.96,side.upper()+' LEG',ha='center',va='center',fontsize=20,fontweight='bold',color=INK)
     ax.text(hx,-1.22,'38 body vertices\n6 actuated joints · 12 cuts',ha='center',va='center',fontsize=13,color=R,linespacing=1.5)

    ax.text(0,-7.92,'Kangaroo canonical mechanism graph',ha='center',va='center',fontsize=30,fontweight='bold',color=INK)
    ax.text(0,-7.19,'78 rigid-body vertices  |  77 tree joints  |  24 closure chords  |  24 independent graph cycles',
            ha='center',va='center',fontsize=16.5,color=R)
    ax.text(3.05,-4.78,'Floating pelvis\n6 free-root velocities',ha='left',va='center',fontsize=14.4,color=INK,linespacing=1.4)

    # Edge endpoint clipping at body-box boundaries.
    sizes={n:(2.2,.80) if n in ['base_link','torso'] else (1.36,.78) for n in pos}
    def clip(p,q,n):
     dx=q[0]-p[0];dy=q[1]-p[1];w,h=sizes[n]
     t=min((w/2)/abs(dx) if dx else float('inf'),(h/2)/abs(dy) if dy else float('inf'))
     return p[0]+t*dx,p[1]+t*dy
    segments=[]
    for e in edges:
     p,q=pos[e['a']],pos[e['b']];s,t=clip(p,q,e['a']),clip(q,p,e['b'])
     color,ls,lw=style[e['kind']]
     ax.plot([s[0],t[0]],[s[1],t[1]],color=color,linestyle=ls,linewidth=lw,solid_capstyle='round',zorder=1)
     segments.append((e,s,t))
     if e['kind']=='prismatic':
      mid=((s[0]+t[0])/2,(s[1]+t[1])/2)
      token='Pℓ' if 'length' in e['id'] else 'P'+e['id'].split('_')[-2]
      dx=.53 if abs(s[0]-t[0])<.01 else -.20
      dy=0 if abs(s[0]-t[0])<.01 else -.46
      if e['a'].startswith('right') and abs(s[0]-t[0])<.01:dx=-.53
      ax.text(mid[0]+dx,mid[1]+dy,token,ha='center',va='center',fontsize=13.0,fontweight='bold',color=P,
              bbox=dict(facecolor='white',edgecolor='none',pad=.5),zorder=5)
     if e['kind']=='fixed':
      ax.text(.4,(s[1]+t[1])/2,'F',fontsize=13.5,fontweight='bold',color=F,ha='left',va='center')

    for name,(x,y) in pos.items():
     w,h=sizes[name]
     main=name in ['base_link','torso']
     fill='#E7EDF4' if main else ('#E9F2FD' if name.startswith('left') else '#EAF4EE')
     border='#637B93' if main else ('#7C9BBB' if name.startswith('left') else '#79A28D')
     ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0,rounding_size=.13',
                  facecolor=fill,edgecolor=border,linewidth=1.35,zorder=3))
     ax.text(x,y,labels[name],ha='center',va='center',fontsize=17.0 if not main else 18.5,
             fontweight='semibold',color=INK,zorder=4)

    # Verify no edge passes through any unrelated node box.
    def intersects_rect(a,b,xmin,xmax,ymin,ymax):
     dx=b[0]-a[0];dy=b[1]-a[1];t0,t1=0.,1.
     for p,q in [(-dx,a[0]-xmin),(dx,xmax-a[0]),(-dy,a[1]-ymin),(dy,ymax-a[1])]:
      if p==0:
       if q<0:return False
      else:
       r=q/p
       if p<0:t0=max(t0,r)
       else:t1=min(t1,r)
       if t0>t1:return False
     return True
    hits=[]
    for e,s,t in segments:
     for n,(x,y) in pos.items():
      if n in [e['a'],e['b']]:continue
      w,h=sizes[n]
      if intersects_rect(s,t,x-w/2-.04,x+w/2+.04,y-h/2-.04,y+h/2+.04):hits.append((e['id'],n))
    if hits:
     raise ValueError(f'Graph edges intersect unrelated node boxes: {hits}')

    # Styles and abbreviation key remain readable when zoomed in vector form.
    legend=[('Revolute tree joint (64)','revolute'),('Actuated prismatic joint (12)','prismatic'),
            ('Fixed joint (1)','fixed'),('Point closure (16)','point_coincidence'),('Universal closure (8)','universal')]
    xs=[-16.55,-9.5,-1.45,4.45,10.65]
    for x,(txt,kind) in zip(xs,legend):
     col,ls,lw=style[kind]
     ax.plot([x,x+1.1],[16.20,16.20],color=col,linestyle=ls,linewidth=lw)
     ax.text(x+1.32,16.2,txt,ha='left',va='center',fontsize=12.4,color=INK)

    ax.text(-17.15,17.0,'BODY LABEL KEY',ha='left',va='center',fontsize=14.3,fontweight='bold',color=INK)
    key=[
     'HY / HP / HR: hip yaw / pitch / roll    •    FP: femur pitch    •    K: knee    •    AP / AR: ankle pitch / roll',
     'YM / YR: yaw motor / rod    •    H2 / H3: hip differential branch; C = cradle, M = motor, R = rod, U = universal body',
     'LN / LS: leg-length sliding nut / short bars    •    RS / RT / RL: rear short bar / triangle / long bar',
     'A4 / A5: ankle branch; M = motor, R = rod, P = pendulum, H = higher bar, B = ankle ball',
     'B4 / B5: butterfly    •    K4 / K5: knee ball    •    L4 / L5: lower ankle bar    •    Pℓ: actuated leg-length slide',
    ]
    for i,line in enumerate(key):ax.text(-17.15,17.52+.45*i,line,ha='left',va='center',fontsize=13.2,color=INK)
    ax.text(17.15,19.94,'Exact CMG connectivity · node positions are schematic · left/right prefixes follow the panel',
            ha='right',va='center',fontsize=11.5,color=R)
    stem=OUT/'Kangaroo_Graph'
    fig.savefig(stem.with_suffix('.png'),dpi=260,facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'),facecolor='white',metadata={'Title':'Kangaroo canonical mechanism graph','Author':''})
    fig.savefig(stem.with_suffix('.svg'),facecolor='white')
    # Internal deterministic audit of the generated model view.
    (OUT/'Kangaroo_Graph_audit.json').write_text(json.dumps({'bodies':len(pos),'edges':len(edges),'types':dict(counts),'cycles':len(edges)-len(pos)+1,'edge_node_collisions':hits,'node_ids':sorted(pos)},indent=2))
    print('Counts',dict(counts),'nodes',len(pos),'cycles',len(edges)-len(pos)+1)

    plt.close(fig)
    print('Saved:', ', '.join(str(stem.with_suffix(ext)) for ext in ('.pdf', '.png', '.svg')))


if __name__ == "__main__":
    main()
