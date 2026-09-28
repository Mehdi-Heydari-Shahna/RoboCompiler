"""Explicit reconstruction of published four-constraint cut joints.

This is not an author-verified source repair. Original hinge cuts and
commented-out ankle bodies differ from this explicit modeling hypothesis.
"""
import sys,copy,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from import_full_model import build,UP,transform,rotation,numbers
from mechanism_backend import Mechanism
from source_bias import SourceBiasDynamics
from pacdm import PointGraph,PACDM,exp,log,inv,adj,inv_left,rank


def make(side='left'):
    raw=build()
    c=copy.deepcopy(raw)
    c['coordinate_ids']=[];c['joints']=[];c['bodies']=c['bodies'][:1];c['closures']=[]
    c['armature']={};c['joint_dissipation']={}
    text=(UP/f'parts/kangaroo.{side}_leg.xml').read_text()
    text=text.replace(f'<!--body name="{side}_5_ankle_ball"',f'<body name="{side}_5_ankle_ball"').replace(f'<!--body name="{side}_4_ankle_ball"',f'<body name="{side}_4_ankle_ball"').replace('</body-->','</body>')
    def visit(el,parent):
        name=el.get('name');ie=el.find('inertial');R=rotation(ie)
        I=R@np.diag(numbers(ie,'diaginertia',''))@R.T
        c['bodies'].append(dict(id=name,kind='rigid_body',mass_kg=float(ie.get('mass')),com_m=numbers(ie,'pos','0 0 0').tolist(),inertia_com_kg_m2=I.tolist()))
        j=el.find('joint');jid=j.get('name');axis=numbers(j,'axis','0 0 1');axis/=np.linalg.norm(axis)
        kind='prismatic' if j.get('type')=='slide' else 'revolute';lo,hi=numbers(j,'range','')
        c['coordinate_ids'].append(jid)
        c['joints'].append(dict(id=jid,base_body=parent,follower_body=name,type=kind,coordinate_unit='m' if kind=='prismatic' else 'rad',axis=axis.tolist(),T_BJ=transform(el).tolist(),T_FJ=np.eye(4).tolist(),limits=dict(lower=float(lo),upper=float(hi))))
        c['armature'][jid]=float(j.get('armature',0));c['joint_dissipation'][jid]=dict(damping=float(j.get('damping',3)),frictionloss=float(j.get('frictionloss',.1)))
        for child in el.findall('body'):visit(child,name)
    for body in ET.fromstring(text).findall('body'):visit(body,'base_link')
    c['actuators']=[]
    for a in ET.parse(UP/f'controllers/kangaroo.{side}_leg.motor_controllers.xml').getroot().find('actuator'):
        c['actuators'].append(dict(id=a.get('name'),joint=a.get('joint'),unit='N',gear=1.,force_bounds_N=numbers(a,'ctrlrange','').tolist()))
    c['independent_ids']=[a['joint'] for a in c['actuators']]
    zero=np.zeros(len(c['coordinate_ids']));P=SourceBiasDynamics(c,c['coordinate_ids']).evaluate(zero,zero,[0,0,0])['poses']
    # Retain selected physical source tree masses/frames; record planar axes.
    changes=[]
    groups=[(f'leg_{side}_1_joint',[f'{side}_yaw_motor_joint',f'leg_{side}_1_motor',f'leg_{side}_1_joint']),
            (f'leg_{side}_femour_joint',[f'leg_{side}_femour_joint',f'leg_{side}_length_motor',f'{side}_leg_length_short_bars_joint',f'leg_{side}_knee_joint',f'{side}_rear_short_bar_joint',f'{side}_rear_triangle_joint',f'{side}_rear_long_bar_joint']+
             [x for k in ['4','5'] for x in [f'{side}_ankle_{k}_motor_joint',f'leg_{side}_{k}_motor',f'{side}_ankle_{k}_pendulum_joint',f'{side}_{k}_higher_ankle_bar_joint',f'{side}_{k}_butterfly_joint']])]
    records={j['id']:j for j in c['joints']}
    for ref,names in groups:
        j=records[ref];normal=P[j['follower_body']][:3,:3]@j['axis'];normal/=np.linalg.norm(normal)
        for name in names:
            j=records[name];R=P[j['follower_body']][:3,:3];old=R@j['axis']
            new=normal*np.sign(old@normal) if j['type']=='revolute' else old-normal*(old@normal)
            new/=np.linalg.norm(new);angle=np.arctan2(np.linalg.norm(np.cross(old,new)),old@new)
            if angle>1e-12:
                changes.append(dict(joint=name,old=j['axis'],new=(R.T@new).tolist(),angle_rad=float(angle)))
                j['axis']=(R.T@new).tolist()
    eq=ET.parse(UP/f'constraints/kangaroo.{side}_leg.equality_constraints.xml').getroot().find('equality')
    for el in eq:
        name=el.get('name')
        if 'differential' in name and name.endswith('b'):continue
        a,b=el.get('body1'),el.get('body2');p=numbers(el,'anchor','0 0 0')
        universal='differential' in name or name in [f'{side}_ankle_4_closed',f'{side}_ankle_5_closed']
        if 'differential' in name:p=np.zeros(3)
        if name in [f'{side}_ankle_4_closed',f'{side}_ankle_5_closed']:
            b=f'{side}_{name.split("_")[2]}_ankle_ball'
        world=P[a][:3,:3]@p+P[a][:3,3];pb=P[b][:3,:3].T@(world-P[b][:3,3])
        item=dict(id=name,type='universal' if universal else 'point_coincidence',body1=a,body2=b,point1_m=p.tolist(),point2_m=pb.tolist())
        if universal:
            item['frame1_R']=np.eye(3).tolist();item['frame2_R']=(P[b][:3,:3].T@P[a][:3,:3]).tolist()
            if 'ankle_ball' in b:
                joint=next(j for j in c['joints'] if j['follower_body']==b)
                normal=P[b][:3,:3]@joint['axis'];normal/=np.linalg.norm(normal)
                first=P[a][:3,:3][:,0].copy();first-=normal*(normal@first);first/=np.linalg.norm(first)
                basis=np.column_stack([first,np.cross(normal,first),normal])
                item['frame1_R']=(P[a][:3,:3].T@basis).tolist();item['frame2_R']=(P[b][:3,:3].T@basis).tolist()
            for key in ['frame1_R','frame2_R']:
                R=np.array(item[key]);assert np.linalg.norm(R.T@R-np.eye(3))<1e-12 and np.linalg.det(R)>0
        c['closures'].append(item)
    original_ids=[j.get('name') for j in ET.parse(UP/f'parts/kangaroo.{side}_leg.xml').getroot().iter('joint')];original_seed=np.loadtxt(UP/'initial_q_pos.csv')[7:43] if side=='left' else np.loadtxt(UP/'initial_q_pos.csv')[43:79]
    c['initial_seed']=[float(original_seed[original_ids.index(j)]) if j in original_ids else 0. for j in c['coordinate_ids']]
    c['source']['geometry_modifications']=changes
    c['source']['model']='EXPERIMENTAL published-cut reconstruction; not author-verified'
    return c


class UniversalMechanism(Mechanism):
    def __init__(self,cmg):
        point=copy.deepcopy(cmg)
        for c in point['closures']:c['type']='point_coincidence'
        super().__init__(point);self.cmg=copy.deepcopy(cmg)
    def closure(self,e):
        rs=[];js=[];ds=[]
        for c in self.cmg['closures']:
            a,ja,da=self.source._point(e,c['body1'],np.array(c['point1_m']))
            b,jb,db=self.source._point(e,c['body2'],np.array(c['point2_m']))
            rs.append(a-b);js.append(ja-jb);ds.append(da-db)
            if c['type']=='universal':
                a=e['poses'][c['body1']][:3,:3]@np.asarray(c['frame1_R'])[:,0]
                b=e['poses'][c['body2']][:3,:3]@np.asarray(c['frame2_R'])[:,1]
                JA=e['jacobians'][c['body1']][3:];JB=e['jacobians'][c['body2']][3:]
                DA=e['body_jacobian_dots'][c['body1']][3:];DB=e['body_jacobian_dots'][c['body2']][3:]
                v=e['velocity'];ad=np.cross(JA@v,a);bd=np.cross(JB@v,b);cross=np.cross(a,b)
                rs.append(np.array([a@b]));js.append((cross@(JA-JB))[None,:]);ds.append((np.cross(ad,b)@(JA-JB)+np.cross(a,bd)@(JA-JB)+cross@(DA-DB))[None,:])
        return np.concatenate(rs),np.vstack(js),np.vstack(ds)


class CutGraph(PointGraph):
    def __init__(self,c,seed):
        super().__init__(c,seed)
        self.n=self.nt+sum(2 if x['type']=='universal' else 3 for x in self.cuts)
        self.passive=np.setdiff1d(np.arange(self.n),self.active)
        self.lower=self.lower[:self.n];self.upper=self.upper[:self.n]
        self.frames=[];self.chart_columns=[];at=self.nt
        for k,cut in enumerate(self.cuts):
            A=np.eye(4);B=np.eye(4);A[:3,3]=cut['point1_m'];B[:3,3]=cut['point2_m']
            if cut['type']=='universal':
                A[:3,:3]=cut['frame1_R'];B[:3,:3]=cut['frame2_R'];n=2
            else:A[:3,:3]=self.references[k];n=3
            self.frames.append((A,B));self.chart_columns.append(list(range(at,at+n)));at+=n
        for cut,path,cols in zip(self.cuts,self.paths,self.chart_columns):
            path['cut_type']=cut['type'];path['chart_columns']=cols
    def augment(self,q):return np.r_[q,np.zeros(self.n-self.nt)]
    def lift(self,q):
        out=self.augment(q);P,_=self.poses(out)
        for cut,(A,B),cols in zip(self.cuts,self.frames,self.chart_columns):
            relative=(P[cut['body1']]@A)[:3,:3].T@(P[cut['body2']]@B)[:3,:3]
            out[cols]=Rotation.from_matrix(relative).as_euler('XYZ')[:len(cols)]
        return out
    def residual(self,q,defects=None):
        P,E=self.poses(q);rs=[];js=[];deltas=[]
        for k,(cut,(A,B),cols) in enumerate(zip(self.cuts,self.frames,self.chart_columns)):
            minus=P[cut['body1']]@A;eta=E[cut['body1']].copy()
            for i,axis in zip(cols,np.eye(3)):
                twist=np.r_[axis,[0,0,0]];eta[:,i]+=adj(minus)@twist;minus=minus@exp(twist*q[i])
            plus=P[cut['body2']]@B;D=np.eye(4) if defects is None else defects[k]
            reverse=inv(minus@D);delta=reverse@plus;r=log(delta)
            rs.append(r);js.append(inv_left(r)@adj(reverse)@(E[cut['body2']]-eta));deltas.append(delta)
        return np.concatenate(rs),np.vstack(js),np.asarray(deltas)


def whole_body():
    """Source prototype: pelvis, fixed torso, two reconstructed 38-DOF legs."""
    left,right=make('left'),make('right');c=copy.deepcopy(left)
    root=ET.parse(UP/'kangaroo.xml').getroot().find('worldbody/body')
    def inertia(el):
        i=el.find('inertial');xx,yy,zz,xy,xz,yz=numbers(i,'fullinertia','')
        return dict(id=el.get('name'),kind='rigid_body',mass_kg=float(i.get('mass')),
                    com_m=numbers(i,'pos','0 0 0').tolist(),
                    inertia_com_kg_m2=[[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    c['bodies'][0]=inertia(root);torso=root.find('body');c['bodies'].append(inertia(torso))
    c['joints'].append(dict(id='torso_fixed',base_body='base_link',follower_body='torso',type='fixed',axis=[0,0,1],
                            T_BJ=transform(torso).tolist(),T_FJ=np.eye(4).tolist()))
    c['bodies']+=right['bodies'][1:]
    for key in ['coordinate_ids','independent_ids','joints','closures','actuators','initial_seed']:c[key]+=right[key]
    for key in ['armature','joint_dissipation']:c[key].update(right[key])
    c['source']['geometry_modifications']+=right['source']['geometry_modifications']
    c['source']['model']='Full published-cut reconstruction: pelvis, torso and two complete legs'
    c['source']['author_verified']=False
    c['source']['topology_modifications']=[
        'Each original hip pair of point connects (hinge cut) becomes one universal cut at their midpoint.',
        'Restore both commented ankle-ball bodies per leg, with source inertia, joint axis and bounds.',
        'Each terminal ankle point cut moves from ankle_roll to restored ankle_ball and becomes universal.',
        'Universal frame axes are explicit reconstruction assumptions; not supplied author CAD frames.']
    c['floating_base']={'coordinates':'world translation and quaternion xyzw',
                        'velocity':'world linear then world angular, followed by scalar joints',
                        'actuated':False}
    return c


def polish(graph,q):
    """Tighter Newton refinement AFTER unchanged PACDM acceptance.

    A 1e-9 closure tolerance can generate spurious 1e-9 singular values near
    a symmetric pose. Refine the same selected square Jacobian, then require
    all rows and the original PACDM rank gates. No rows/limits are removed.
    """
    solver=PACDM(graph);q=q.copy();_,info=solver.mapping(q)
    if not info['success']:raise ValueError(info)
    rows=info['rows']
    for _ in range(8):
        r,J,D=graph.residual(q)
        if np.max(abs(r))<5e-13:break
        q[graph.passive]-=np.linalg.solve(J[np.ix_(rows,graph.passive)],r[rows])
    _,info=solver.mapping(q)
    if not info['success'] or np.max(abs(r))>=5e-13 or np.any(q<graph.lower) or np.any(q>graph.upper):raise ValueError('Tighter all-row Newton refinement failed')
    return q
