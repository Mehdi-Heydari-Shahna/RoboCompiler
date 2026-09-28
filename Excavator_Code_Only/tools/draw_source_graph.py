from pathlib import Path
import json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Draw the source CMG and ordered closure comparisons.')
parser.add_argument('--output',type=Path,default=ROOT/'figures')
args=parser.parse_args()
SRC=ROOT/'v26/source/Excavator_RoboIR_full_body_v21/data'
OUT=args.output; OUT.mkdir(parents=True,exist_ok=True)
c=json.loads((SRC/'accepted_cmg_v04.json').read_text())
m=json.loads((SRC/'accepted_mujoco_mapping.json').read_text())
nodes={b['id']:b for b in c['bodies']}
joints={j['id']:j for j in c['joints']}
tree={j['id'] for j in m['tree']}; cuts=set(m['cut_joint_ids'])
assert len(nodes)==26 and len(joints)==34
assert collections.Counter(j['type'] for j in joints.values())=={'revolute':26,'prismatic':6,'fixed':2}
assert len(tree)==25 and len(cuts)==9 and tree|cuts==set(joints) and not tree&cuts
assert sum(joints[j]['type']!='fixed' for j in tree)==23
adj={n:[] for n in nodes}
for k in tree:
 j=joints[k];a,b=j['base_body'],j['follower_body'];adj[a].append((b,k,1));adj[b].append((a,k,-1))
def ordered_path(a,b):
 queue=collections.deque([(a,[])]);seen={a}
 while queue:
  u,p=queue.popleft()
  if u==b:return p
  for v,k,s in adj[u]:
   if v not in seen:seen.add(v);queue.append((v,p+[(k,s)]))
 raise ValueError('Disconnected graph')
paths={k:ordered_path(joints[k]['base_body'],joints[k]['follower_body']) for k in cuts}
assert all(ordered_path('world',n) is not None for n in nodes)

P={
 'world':(-10,0),'body_53':(-6.5,0),'body_57':(-3,0),
 'body_1':(-7,1.9),'body_3':(-7,3.8),'body_2':(1,1.9),'body_4':(1,3.8),
 'body_54':(-3,5.8),'body_44':(-7,7.7),'body_45':(-7,9.6),'body_60':(-3,11),
 'body_5':(1,11),'body_6':(4.5,11),'body_59':(8,11),
 'body_7':(1,6.8),'body_9':(8,6.8),'body_8':(1,15.8),'body_10':(8,15.8),
 'body_61':(12,11),'body_48':(14,8),'body_46':(18,8),
 'body_49':(14,14),'body_47':(18,14),'body_58':(20,11),
 'body_55':(24,11),'body_56':(28,11)
}
N={
 'world':'World','body_53':'Base','body_57':'Cabin',
 'body_1':'Boom cylinder 1','body_2':'Boom cylinder 2',
 'body_3':'Boom piston 1','body_4':'Boom piston 2','body_54':'Boom',
 'body_44':'Stick cylinder','body_45':'Stick piston','body_60':'Stick',
 'body_5':'Bucket cylinder','body_6':'Bucket piston','body_59':'Pivot',
 'body_7':'DF link 1','body_8':'DF link 2','body_9':'EF link 1','body_10':'EF link 2',
 'body_61':'Top housing','body_58':'Middle housing','body_55':'Bottom housing','body_56':'Bucket',
 'body_46':'Tilt cylinder 1','body_47':'Tilt cylinder 2','body_48':'Tilt piston 1','body_49':'Tilt piston 2'
}
assert set(P)==set(nodes)==set(N)
L={
 'fixed_world_base':(-8.25,-.88),'q23':(-4.75,-.88),
 'q24':(-5.6,.95),'q25':(-.6,.45),'q7':(-2.4,2.9),
 'p3':(-7.6,2.85),'p4':(1.6,2.85),'q5':(-5.55,5.18),'q6':(-.45,5.18),
 'q8':(-5.6,6.48),'p5':(-7.65,8.65),'q9':(-5.4,10.96),'q4':(-2.4,8.5),
 'q10':(-1,11.8),'p2':(2.75,11.8),'q22':(6.25,11.8),
 'q13':(-1.68,8.0),'q14':(4.5,6.03),'q15':(5.45,8.50),
 'q16':(-1.72,13.9),'q17':(4.5,16.58),'q18':(5.55,13.85),
 'q19':(10.45,8.05),'q20':(10.52,14.48),'q0':(4.5,18.5),
 'q1':(16,10.2),'q2':(13.8,9.5),'q3':(13.8,12.6),
 'p0':(16,7.23),'p1':(16,14.77),'q11':(20.02,8.70),'q12':(20.04,13.3),
 'q21':(22,10.2),'fixed_bucket_mount':(26,10.2)
}
assert set(L)==set(joints)
colors={'revolute':'#285F88','prismatic':'#008071','fixed':'#697482'}
cut_color='#C45B16';ink='#1C2F42'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(21,12.5));fig.subplots_adjust(0,0,1,1)
ax.set_xlim(-12,30);ax.set_ylim(21.3,-3.5);ax.axis('off')

ax.text(-11.2,-2.65,'Excavator source CMG and selected computational tree',fontsize=22,weight='bold',color=ink)
ax.text(-11.2,-1.7,'25 physical bodies + world   |   34 physical joints: 26 revolute, 6 prismatic, 2 fixed',fontsize=14,color='#4C6072')

W,H=3.0,1.10
def clip(a,b):
 a=np.array(a,float);d=np.array(b,float)-a
 t=min((W/2)/abs(d[0]) if d[0] else np.inf,(H/2)/abs(d[1]) if d[1] else np.inf)
 return tuple(a+t*d)
edge_records=[]
for k,j in joints.items():
 a,b=P[j['base_body']],P[j['follower_body']]
 route=[a,b] if k!='q0' else [a,(-3,18.5),(12,18.5),b]
 route[0]=clip(a,route[1]);route[-1]=clip(b,route[-2])
 col=cut_color if k in cuts else colors[j['type']]
 xs,ys=zip(*route)
 ax.plot(xs,ys,color=col,lw=1.9 if k in cuts else 1.7,
   linestyle=(0,(5,3)) if k in cuts else '-',solid_capstyle='round',zorder=1)
 lab=(r'$F_{WB}$' if k=='fixed_world_base' else r'$F_{BM}$' if k=='fixed_bucket_mount' else
       '$'+k[0]+'_{'+k[1:]+'}$')
 lx,ly=L[k]
 ax.text(lx,ly,lab,fontsize=13,weight='bold',ha='center',va='center',color=col,zorder=5,
   bbox={'facecolor':'white','edgecolor':'none','pad':1.4})
 edge_records.append({'joint':k,'a':j['base_body'],'b':j['follower_body'],'polyline':route})

main={'body_53','body_57','body_54','body_60','body_61','body_58','body_55','body_56'}
for k,(x,y) in P.items():
 fill='#EEF4F9' if k in main else '#FFFFFF'
 if k=='world':fill='#E9EDF1'
 ax.add_patch(FancyBboxPatch((x-W/2,y-H/2),W,H,boxstyle='round,pad=0.018,rounding_size=.1',
   facecolor=fill,edgecolor='#77899B',linewidth=1.05,zorder=3))
 ax.text(x,y-.18,N[k],fontsize=11.0,ha='center',va='center',weight='bold',color=ink,zorder=4)
 ax.text(x,y+.24,k if k!='world' else 'reference node',fontsize=9.7,ha='center',va='center',color='#52677A',zorder=4)

# Ordered path table derived directly from the selected tree.
tx,ty,tw,th=10.0,-.10,19.0,6.03
ax.add_patch(FancyBboxPatch((tx,ty),tw,th,boxstyle='round,pad=.02,rounding_size=.12',
   facecolor='#F7F9FB',edgecolor='#CED8E0',lw=.9,zorder=0))
ax.text(tx+.45,ty+.47,'Nine ordered closure comparisons',fontsize=15,weight='bold',color=ink)
ax.text(tx+.45,ty+1.05,'Cut joint',fontsize=11.5,weight='bold',color=cut_color)
ax.text(tx+4.3,ty+1.05,'Ordered tree path (same endpoints)',fontsize=11.5,weight='bold',color=ink)
order=['q5','q6','q9','q15','q18','q19','q20','q2','q3']
for i,k in enumerate(order):
 yy=ty+1.53+i*.465
 ax.text(tx+1.15,yy,'$q_{'+k[1:]+'}$',fontsize=12.0,color=cut_color,ha='center')
 s=r'\, ,\; '.join(('+' if sign>0 else '-')+joint[0]+'_{'+joint[1:]+'}' for joint,sign in paths[k])
 ax.text(tx+4.3,yy,'$'+s+'$',fontsize=12.0,color=ink)
ax.text(tx+.45,ty+5.82,'Paths run from each cut-joint base to follower; signs follow source joint orientation.',fontsize=10.0,color='#52677A')

# Two independent encodings: joint type (color / ID), tree membership (line style).
ly=19.75
entries=[(-10.9,'#285F88','-','Revolute (q)'),(-2.65,'#008071','-','Prismatic (p)'),
         (5.65,'#697482','-','Fixed (F)'),(12.8,'#285F88','-','Tree edge'),
         (20.9,cut_color,(0,(5,3)),'Physical cut joint')]
for x,col,sty,text in entries:
 ax.plot([x,x+1.0],[ly,ly],color=col,lw=2,ls=sty)
 ax.text(x+1.3,ly,text,fontsize=12.0,va='center',color=ink)
ax.text(-10.9,20.55,r'$34-26+1=9$ cycles    |    Tree: 25 edges = 23 moving + 2 fixed    |    Augmented coordinates: $23+9=32$',fontsize=13,color=ink)

fig.savefig(OUT/'graph_ex.png',dpi=320,facecolor='white')
fig.savefig(OUT/'graph_ex.pdf',facecolor='white',metadata={'Title':'Excavator source CMG and selected computational tree'})
plt.close(fig)
report={'nodes':len(nodes),'physical_bodies':sum(x['kind']=='rigid_body' for x in nodes.values()),
 'joints':len(joints),'joint_types':dict(collections.Counter(x['type'] for x in joints.values())),
 'tree_edges':len(tree),'tree_moving_coordinates':23,'chords':sorted(cuts),'cycle_rank':9,
 'augmented_coordinates':32,'ordered_tree_paths':paths,'rendered_edges':edge_records}
(OUT/'source_graph_check.json').write_text(json.dumps(report,indent=2))
print({k:report[k] for k in ['nodes','physical_bodies','joints','joint_types','tree_edges','tree_moving_coordinates','cycle_rank','augmented_coordinates']})
