#!/usr/bin/env python3
"""Publication-style physical CMG diagram, driven by Panda CMG JSON."""
from collections import Counter
from pathlib import Path
import json
from xml.etree import ElementTree as ET

BASE = Path(__file__).resolve().parents[1]
cmg = json.loads((BASE/'data/panda_cmg.json').read_text())
outdir = BASE/'figures'
outdir.mkdir(exist_ok=True)

bodies={b['id']:b for b in cmg['bodies']}
joints=cmg['joints']
assert len(bodies)==12 and len(joints)==11
assert Counter(j['type'] for j in joints)=={'revolute':7,'prismatic':2,'fixed':2}
assert len(cmg['coordinate_ids'])==9 and len(cmg['coordinate_couplings'])==1
assert len(cmg['actuation']['actuators'])==8 and not cmg['closures']
assert cmg['coordinate_couplings'][0]['joint1']=='finger_joint1'
assert cmg['coordinate_couplings'][0]['joint2']=='finger_joint2'
seen={'world'}
for j in joints:
    assert j['base_body'] in seen and j['follower_body'] not in seen
    seen.add(j['follower_body'])
assert seen==set(bodies)

S='http://www.w3.org/2000/svg'
ET.register_namespace('', S)
root=ET.Element('{%s}svg'%S,attrib={'viewBox':'0 0 3200 1640','width':'3200','height':'1640','role':'img','aria-label':'Franka Panda canonical mechanism graph: world, seven-link serial arm, hand and two coupled prismatic fingers'})

def el(tag,**attrs):
 return ET.SubElement(root,'{%s}%s'%(S,tag),**{k.replace('_','-'):str(v) for k,v in attrs.items()})

def rect(x,y,w,h,fill,stroke='none',r=18,sw=2):
 return el('rect',x=x,y=y,width=w,height=h,rx=r,fill=fill,stroke=stroke,stroke_width=sw)

def line(x1,y1,x2,y2,color,sw=5,dash=None):
 a=dict(x1=x1,y1=y1,x2=x2,y2=y2,stroke=color,stroke_width=sw,stroke_linecap='round')
 if dash:a['stroke_dasharray']=dash
 return el('line',**a)

def txt(x,y,s,size=32,color='#17384D',weight=450,anchor='start',spacing=None):
 a={'x':str(x),'y':str(y),'fill':color,'font-family':'DejaVu Sans, Arial, sans-serif','font-size':str(size),'font-weight':str(weight),'text-anchor':anchor}
 if spacing is not None:a['letter-spacing']=str(spacing)
 t=ET.SubElement(root,'{%s}text'%S,**a);t.text=s;return t

def pill(cx,cy,w,h,fill,stroke,primary,secondary=None,primary_size=30):
 rect(cx-w/2,cy-h/2,w,h,fill,stroke,18,2)
 txt(cx,cy+(2 if secondary else 10)-(15 if secondary else 0),primary,primary_size,stroke,700,'middle')
 if secondary:txt(cx,cy+27,secondary,28,stroke,490,'middle')

def body(name,cx,cy,kind):
 if kind=='world':w,h,fill,stroke,fg=188,80,'#193C51','#193C51','#FFFFFF'
 elif kind=='hand':w,h,fill,stroke,fg=195,82,'#D8F0EA','#197863','#144839'
 elif kind=='finger':w,h,fill,stroke,fg=255,80,'#F5FBF9','#228A77','#174B42'
 else:w,h,fill,stroke,fg=164,78,'#F1F6FA','#5F839A','#17384D'
 rect(cx-w/2,cy-h/2,w,h,fill,stroke,18,3)
 if kind=='finger':rect(cx-w/2+8,cy-h/2+10,8,h-20,'#228A77',r=4)
 txt(cx,cy+12,name,38 if kind=='finger' else 37,fg,680,'middle')
 return (w,h)

# Palette reflects edge semantics, not the link's physical material.
FIX='#8B9AA5';REV='#445FC0';PRI='#168F9A';COUP='#AB619F';INK='#153648';MUT='#60798A'
rect(0,0,3200,1640,'#FFFFFF',r=0)
rect(84,55,166,48,'#E6F1F6',r=24)
txt(167,89,'FRANKA',26,'#386679',750,'middle',1.2)
txt(83,178,'Canonical mechanism graph',75,INK,780)
txt(86,234,'Panda body–joint tree, actuator ports, and finger-coordinate coupling',36,MUT,460)
rect(2040,67,1075,144,'#F3F8FA','#DBE8ED',24,2)
txt(2080,120,'12 body nodes  ·  11 joint edges',39,'#214C60',700)
txt(2080,174,'7 R  ·  2 P  ·  2 fixed  ·  0 graph cycles',34,'#5B7481',490)
line(84,273,3115,273,'#DCE7EC',3)

# Source IDs determine every solid body–joint edge.
chain=['world','link0','link1','link2','link3','link4','link5','link6','link7','hand']
coords={name:(164+278*i,650) for i,name in enumerate(chain)}
coords['left_finger']=(2446,1128);coords['right_finger']=(2925,1128)
assert len(coords)==12
joint_by_pair={(j['base_body'],j['follower_body']):j for j in joints}
for parent,child in zip(chain,chain[1:]):
 j=joint_by_pair[(parent,child)]
 px,py=coords[parent];cx,cy=coords[child]
 w1=188 if parent=='world' else 195 if parent=='hand' else 164
 w2=195 if child=='hand' else 164
 color=FIX if j['type']=='fixed' else REV
 line(px+w1/2,py,cx-w2/2,cy,color,7)
for child in ('left_finger','right_finger'):
 j=joint_by_pair[('hand',child)]
 assert j['type']=='prismatic'
 x1,y1=coords['hand'];x2,y2=coords[child]
 line(x1,y1+41,x2,y2-40,PRI,7)

# Draw the 12 source body instances.
for name in chain:body(name,*coords[name],'world' if name=='world' else 'hand' if name=='hand' else 'arm')
for name in ('left_finger','right_finger'):body(name,*coords[name],'finger')

# Joint labels and actuation use the source actuator transmission map.
act={next(iter(a['transmission'])):a['id'] for a in cmg['actuation']['actuators'][:7]}
for parent,child in zip(chain,chain[1:]):
 j=joint_by_pair[(parent,child)]
 x=(coords[parent][0]+coords[child][0])/2
 col=FIX if j['type']=='fixed' else REV
 sub='fixed' if j['type']=='fixed' else 'R  ·  '+act[j['id']].replace('actuator','A')
 pill(x,493,248 if j['type']=='fixed' else 208,84,'#FFFFFF',col,j['id'],sub,29 if j['type']=='fixed' else 33)
 line(x,537,x,601,'#CBD7DC',2)
 line(x,601,x,615,col,3)
for child in ('left_finger','right_finger'):
 j=joint_by_pair[('hand',child)]
 x=2490 if child=='left_finger' else 2920
 pill(x,936,348,87,'#FFFFFF',PRI,j['id'],'P  ·  split tendon A8',31)

# Coupling is a dashed coordinate relation, deliberately not a solid joint edge.
xL,yL=coords['left_finger'];xR,yR=coords['right_finger']
line(xL,yL+41,xL,1242,COUP,5,'13 11')
line(xL,1242,xR,1242,COUP,5,'13 11')
line(xR,1242,xR,yR+41,COUP,5,'13 11')
pill((xL+xR)/2,1242,410,64,'#FFFFFF',COUP,'q_f1 = q_f2',None,38)
txt(2688,1325,'scalar finger coupling · not a joint edge',28,COUP,540,'middle')

# Tool site: an attached frame, not an extra body.
line(coords['hand'][0]+100,650,2834,650,'#B8914B',4,'7 8')
el('polygon',points='2854,650 2834,630 2814,650 2834,670',fill='#F3D49C',stroke='#A8752C',stroke_width='3')
txt(2874,635,'tool site',30,'#8A642C',650)
txt(2874,674,'hand +z: 0.1029 m',25,'#806C53',470)

# Explanatory cards separated from graph edges.
rect(95,883,1700,472,'#F6F9FB','#DDE8ED',25,2)
txt(142,949,'PHYSICAL GRAPH',30,'#4D697A',760,spacing=1.3)
txt(142,1009,'Solid edges form one connected tree.',37,INK,630)
txt(142,1063,'Nine joint coordinates; one finger equality;',32,'#3F5D70')
txt(142,1113,'eight independent physical coordinates.',32,'#3F5D70')
line(142,1151,1745,1151,'#D9E4EA',2)
txt(142,1211,'PACDM TASK LAYER',30,'#4D697A',760,spacing=1.3)
txt(142,1262,'Six massless target coordinates and a tool-pose comparison.',31,'#3F5D70')
txt(142,1314,'Task edges are virtual; the CMG tree remains the source topology.',30,'#627B89')

rect(1916,1370,1200,144,'#EFF8F5','#D7EBE3',22,2)
txt(1960,1427,'A8  ·  SPLIT TENDON',31,'#246D59',760,spacing=0.8)
txt(1960,1475,'Transmission: 0.5 finger_joint1 + 0.5 finger_joint2',30,'#3D6D5D')
line(84,1541,3116,1541,'#DCE7EC',3)
txt(92,1591,'KEY',28,'#567281',760,spacing=1.2)
for x,col,label in [(230,FIX,'fixed joint'),(746,REV,'R · arm servo'),(1417,PRI,'P · finger slide')]:
 line(x,1580,x+81,1580,col,8);txt(x+104,1592,label,29,'#426273')
line(2190,1580,2273,1580,COUP,6,'13 11');txt(2295,1592,'scalar coordinate coupling',29,'#765274')

file=outdir/'Franka_Panda_Canonical_Mechanism_Graph.svg'
ET.indent(root, space='  ')
file.write_bytes(ET.tostring(root,xml_declaration=True,encoding='UTF-8'))
print(file)
print('validated',len(coords),'bodies,',len(joints),'solid joints; kinds',dict(Counter(j['type'] for j in joints)),'; physical cycle rank',len(joints)-len(coords)+1)
