#!/usr/bin/env python3
"""Implementation-flow diagram for the Franka Panda CMG/PACDM benchmark."""
from pathlib import Path
from xml.etree import ElementTree as ET
import json

ROOT=Path(__file__).resolve().parents[1]
cmg=json.loads((ROOT/'data/panda_cmg.json').read_text())
assert len(cmg['bodies'])==12 and len(cmg['joints'])==11
assert len(cmg['actuation']['actuators'])==8
assert (ROOT/'panda/model.py').is_file() and (ROOT/'panda/task.py').is_file()
assert (ROOT/'panda/pin_backend.py').is_file() and (ROOT/'panda/simulation.py').is_file()
assert (ROOT/'panda/validation.py').is_file() and (ROOT/'panda/task_validation.py').is_file()
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
S='http://www.w3.org/2000/svg'
ET.register_namespace('', S)
svg=ET.Element('{%s}svg'%S,attrib={'width':'3400','height':'1830','viewBox':'0 0 3400 1830','role':'img','aria-label':'Franka Panda implementation flow with solid model, control and simulation data arrows and dashed numerical audit arrows'})

def sub(tag,parent=svg,**attrs):
 return ET.SubElement(parent,'{%s}%s'%(S,tag),**{k.replace('_','-'):str(v) for k,v in attrs.items()})

def rect(x,y,w,h,fill,stroke='none',sw=2,r=20):
 return sub('rect',x=x,y=y,width=w,height=h,rx=r,fill=fill,stroke=stroke,stroke_width=sw)

def txt(x,y,s,size=32,color='#1A394C',weight=450,anchor='start',spacing=None):
 a={'x':x,'y':y,'fill':color,'font-family':'DejaVu Sans, Arial, sans-serif','font-size':size,'font-weight':weight,'text-anchor':anchor}
 if spacing is not None:a['letter-spacing']=spacing
 t=sub('text',**a);t.text=s;return t

def edge(d,color='#59768A',width=6,dashed=False,tip='arrow-solid'):
 a={'d':d,'fill':'none','stroke':color,'stroke_width':width,'stroke_linecap':'round','stroke_linejoin':'round','marker_end':f'url(#{tip})'}
 if dashed:a['stroke_dasharray']='16 12'
 return sub('path',**a)

def box(cx,cy,w,h,title,detail,fill,accent,title_size=37,detail_size=28):
 rect(cx-w/2,cy-h/2,w,h,fill,'#D8E4E9',2,22)
 rect(cx-w/2,cy-h/2,10,h,accent,r=5)
 txt(cx,cy-13,title,title_size,'#204359',750,'middle')
 txt(cx,cy+35,detail,detail_size,'#526B7C',480,'middle')

# Arrow definitions.
defs=sub('defs')
for id_,color in [('arrow-solid','#59768A'),('arrow-purple','#7762AF'),('arrow-blue','#4164A3'),('arrow-green','#167E68'),('arrow-orange','#C4843C'),('arrow-audit','#A76399'),('arrow-scene','#B1874B')]:
 m=sub('marker',defs,id=id_,viewBox='0 0 12 12',refX='10',refY='6',markerWidth='12',markerHeight='12',orient='auto-start-reverse')
 sub('path',m,d='M 1 1 L 11 6 L 1 11 z',fill=color)

# White canvas and header.
rect(0,0,3400,1830,'#FFFFFF',r=0)
rect(82,56,136,43,'#E8F3F4',r=21)
txt(150,87,'PANDA',24,'#287380',750,'middle',1.0)
txt(82,163,'CMG--PACDM implementation flow',69,'#163A4D',770)
txt(86,217,'Source model  →  graph compilation  →  reference and dynamics  →  contact rollout  →  evidence',33,'#587286')
sub('line',x1=84,y1=249,x2=3317,y2=249,stroke='#DCE7EC',stroke_width=3)

# Faint stage underlays for a layered but uncluttered layout.
rect(80,282,3220,755,'#FBFDFE','#E9EFF2',2,27)
txt(123,325,'MODEL  ·  REFERENCE  ·  CONTROL  ·  SIMULATION',27,'#7D91A1',760,spacing=1.3)
rect(80,1110,3220,595,'#FCFBFD','#EFE6EF',2,27)
txt(122,1154,'NUMERICAL AUDITS AND REPORTING',27,'#98788F',760,spacing=1.3)

# Main data paths, always solid.
edge('M 475 618 H 575',tip='arrow-solid')
edge('M 1010 618 H 1115',tip='arrow-solid')
edge('M 1550 574 C 1640 517 1655 408 1715 408',color='#7762AF',tip='arrow-purple')
edge('M 1550 663 C 1620 706 1640 846 1715 846',color='#4164A3',tip='arrow-blue')
edge('M 2018 408 H 2143',color='#7762AF',tip='arrow-purple')
edge('M 2355 486 C 2430 535 2485 656 2490 753',color='#7762AF',tip='arrow-purple')
edge('M 2022 846 H 2235',color='#4164A3',tip='arrow-blue')
edge('M 2745 846 H 2826',color='#C4843C',tip='arrow-orange')
# The source MJCF also builds the native contact scene, via an elevated model-data route.
edge('M 280 546 V 510 H 100 V 270 H 3070 V 385',color='#B1874B',width=5,tip='arrow-scene')
edge('M 3070 504 V 749',color='#B1874B',width=5,tip='arrow-scene')
# State feedback from native MuJoCo to inverse dynamics and bounded servos.
edge('M 3070 932 V 1010 H 2490 V 940',color='#167E68',width=5,tip='arrow-green')
txt(2780,1000,'measured q, v feedback',26,'#238372',580,'middle')
# Native logs to the reporting panel, routed outside the audit row.
edge('M 3295 846 H 3342 V 1590 H 2954',color='#167E68',width=5,tip='arrow-green')
txt(3331,1104,'logs',25,'#167E68',600,'end')

# Dashed numerical audit arrows.
edge('M 1825 937 C 1710 1030 1530 1127 1410 1211',color='#A76399',width=5,dashed=True,tip='arrow-audit')
edge('M 2300 487 C 2185 691 2170 1015 2170 1211',color='#A76399',width=5,dashed=True,tip='arrow-audit')
edge('M 3070 934 V 1211',color='#A76399',width=5,dashed=True,tip='arrow-audit')
# Audit outputs converge into the saved validation record.
for x in [1250,2170,3070]:
 edge(f'M {x} 1365 V 1453',color='#A76399',width=4,dashed=True,tip='arrow-audit')
sub('line',x1=1250,y1=1455,x2=3070,y2=1455,stroke='#BA8BAF',stroke_width=4,stroke_dasharray='15 10')
edge('M 2630 1455 V 1506',color='#A76399',width=5,dashed=True,tip='arrow-audit')

# File provenance and graph data.
box(280,618,392,144,'Pinned Panda MJCF','Menagerie panda.xml','#F4F8FA','#7295A7',36,27)
box(790,618,440,144,'CMG importer','panda/model.py','#EFF6F8','#27889A',37,27)
box(1332,618,438,178,'Canonical graph','12 bodies · 11 joints','#E7F4F5','#188093',38,27)
txt(1332,687,'finger equality',25,'#526B7C',480,'middle')
box(1858,408,470,158,'TaskGraph + PACDM','virtual SE(3) target · rank 7','#F3F0FA','#7762AF',35,26)
box(2353,408,430,158,'Joint reference','q*, v*, a* · 100 Hz','#F4F1FA','#7762AF',38,29)
box(1858,846,470,177,'Pinocchio backend','CMG tree · FK / CRBA / RNEA','#EEF2FA','#4164A3',36,27)
box(2490,846,500,177,'Bounded arm servos','reference + feedforward + state','#FFF4EA','#C4843C',37,26)
box(3070,445,475,118,'MuJoCo scene','free cartridge · barrier · socket','#FBF7EC','#B1874B',36,26)
box(3070,846,475,175,'MuJoCo rollout','native contact · 1-ms physics','#EAF7F2','#167E68',37,27)

# Audit cards record precise validation classes; these are data endpoints of dashed arrows.
box(1250,1285,690,156,'Mechanics audit','FK · Jacobians · mass · RNEA · ports','#FAF2F8','#A76399',35,27)
box(2170,1285,690,156,'PACDM task audit','closure · rank · tangent · route','#FAF2F8','#A76399',35,27)
box(3070,1285,530,156,'Mission audit','contact · tracking · placement','#FAF2F8','#A76399',35,25)
box(2630,1592,650,157,'Validation & report','NPZ / JSON · plots · 229 checks','#EEF6F6','#477987',37,27)

# Legend at foot.
sub('line',x1=85,y1=1742,x2=3316,y2=1742,stroke='#DCE7EC',stroke_width=3)
txt(96,1791,'KEY',26,'#526D7D',750,spacing=1.1)
sub('line',x1=213,y1=1781,x2=320,y2=1781,stroke='#59768A',stroke_width=7)
txt(348,1792,'model, control, and simulation data',29,'#536E7E')
sub('line',x1=1395,y1=1781,x2=1501,y2=1781,stroke='#A76399',stroke_width=7,stroke_dasharray='16 12')
txt(1528,1792,'numerical audit',29,'#8A6184')
txt(2380,1792,'Pinocchio: rigid-body backend  ·  MuJoCo: contact physics',27,'#6B8290')

path=OUT/'Franka_Panda_Implementation_Flow.svg'
ET.indent(svg, space='  ')
path.write_bytes(ET.tostring(svg,xml_declaration=True,encoding='UTF-8'))
print(path)
