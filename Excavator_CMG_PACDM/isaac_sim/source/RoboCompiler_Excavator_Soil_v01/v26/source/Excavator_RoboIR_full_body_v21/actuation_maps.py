"""Physical effort-port maps with explicit unresolved source inputs.

Port efforts have the SI unit of their joint: N or N m. Source channel numbers
are interpreted only by the supplied numerical gains; no calibrated hydraulic
transmission or declared unit on a Simulink signal is inferred.
"""
from copy import deepcopy
from collections.abc import Mapping

import numpy as np

from constrained_dynamics import finite_vector
from source_dynamics import _skew
from source_reference import cylinder_attachment_records


class ActuationMaps:
    """Compile eight physical ports and six known numerical source channels.

    E maps port efforts into tree efforts. For qdot=N udot, A=E.T N maps
    independent velocity into port velocity; A.T maps physical effort back.
    D maps known source channel values into port effort contributions. The
    missing p0 effort is a separate required experiment input, never a default.
    """

    def __init__(self, cmg, tree_ids):
        self.cmg = deepcopy(cmg)
        self.tree_ids = list(tree_ids)
        self.joints = {j['id']:j for j in cmg['joints']}
        if len(self.joints)!=len(cmg['joints']) or len(set(self.tree_ids))!=len(self.tree_ids):
            raise ValueError('Joint and tree coordinate IDs must be unique')
        records = sorted(cmg['actuators'],key=lambda x:x['drive_joint_id'])
        self.port_ids = [a['drive_joint_id'] for a in records]
        if self.port_ids != ['p0','p1','p2','p3','p4','p5','q21','q23']:
            raise ValueError('This source must retain its eight distinct mechanical effort ports')
        if any(j not in self.tree_ids for j in self.port_ids):
            raise ValueError('This lowering requires all effort joints in its tree')
        command=cmg['command_map']
        self.channel_ids=list(command['channel_ids'])
        if len(set(self.channel_ids))!=6 or len(self.channel_ids)!=6:
            raise ValueError('Expected six unique known source channels')
        self.E=np.zeros((len(self.tree_ids),len(records)))
        self.D=np.zeros((len(records),6))
        self.unknown_ids=[]
        self.port_units=[]
        for col,a in enumerate(records):
            jid=a['drive_joint_id'];joint=self.joints[jid]
            expected=('linear_force','N') if joint['type']=='prismatic' else ('rotary_torque','N*m')
            if joint['type'] not in ('prismatic','revolute') or (a['type'],a['effort_unit'])!=expected:
                raise ValueError('Effort type/unit must agree with its mechanical joint')
            self.E[self.tree_ids.index(jid),col]=1.
            self.port_units.append(a['effort_unit'])
            gain=finite_vector([a['source_signal_gain']],1,'Source gain')[0]
            channel=a['source_command_channel']
            if a['input_status']=='unconnected':
                if not a['unknown_runtime_effort'] or channel!=0 or gain!=0:
                    raise ValueError('Unconnected effort must remain explicitly unknown')
                self.unknown_ids.append(jid)
            elif a['input_status']=='connected':
                if a['unknown_runtime_effort'] or not isinstance(channel,int) or isinstance(channel,bool) or not 1<=channel<=6:
                    raise ValueError('Connected actuator channel metadata is invalid')
                self.D[col,channel-1]=gain
            else:
                raise ValueError('Unknown actuator input status')
        if self.unknown_ids!=['p0'] or command['unconnected_effort_joint_ids']!=self.unknown_ids:
            raise ValueError('The source p0 finding must be preserved')

        # Check a second representation of the source wiring, not a copy of D.
        branches=np.zeros_like(self.D);seen=set();channels_seen=set()
        for channel in command['channels']:
            index=channel['index']
            if not isinstance(index,int) or isinstance(index,bool) or not 1<=index<=6:
                raise ValueError('Invalid channel index')
            if channel['id']!=self.channel_ids[index-1] or index in channels_seen:
                raise ValueError('Duplicate/inconsistent channel identity')
            channels_seen.add(index)
            for b in channel['branches']:
                jid=b['joint_id']
                if jid not in self.port_ids or jid in seen:
                    raise ValueError('Duplicate or unknown source effort branch')
                seen.add(jid);row=self.port_ids.index(jid);a=records[row]
                if b['unit']!=a['effort_unit'] or b['converter_sid']!=a['source_converter_sid']:
                    raise ValueError('Source converter/unit provenance disagrees')
                branches[row,index-1]=finite_vector([b['gain']],1,'Branch gain')[0]
        if channels_seen!=set(range(1,7)) or seen!=set(self.port_ids)-set(self.unknown_ids) or not np.array_equal(branches,self.D):
            raise ValueError('Actuator and channel-branch records disagree')
        full_ids=command['coordinate_ids']
        if len(set(full_ids))!=len(full_ids) or set(full_ids)!={j['id'] for j in cmg['joints'] if j['type']!='fixed'}:
            raise ValueError('Full effort-coordinate record is incomplete')
        full=np.asarray(command['B_SI'],dtype=float)
        generated=np.zeros((len(full_ids),6))
        for i,jid in enumerate(self.port_ids):generated[full_ids.index(jid)]=self.D[i]
        if full.shape!=generated.shape or not np.all(np.isfinite(full)) or not np.array_equal(full,generated):
            raise ValueError('Full command matrix disagrees with physical actuator records')
        self.unknown_E=self.E[:,[self.port_ids.index(j) for j in self.unknown_ids]]
        self.attachment_records=cylinder_attachment_records(cmg)

    def source_efforts(self, channels, *, unresolved_efforts):
        """Complete experiment effort; p0 must be supplied by name explicitly."""
        values=finite_vector(channels,6,'Six source channel numbers')
        if not isinstance(unresolved_efforts,Mapping) or set(unresolved_efforts)!=set(self.unknown_ids):
            raise ValueError('Complete experiment needs an explicit p0 effort; source value remains unknown')
        result=self.D@values
        for jid in self.unknown_ids:
            result[self.port_ids.index(jid)]=finite_vector([unresolved_efforts[jid]],1,'Explicit unresolved-port experiment effort')[0]
        return result

    def evaluate(self, tangent_map, physical_efforts):
        if np.iscomplexobj(tangent_map):raise ValueError('Tangent map must be real')
        n=np.asarray(tangent_map,dtype=float)
        if n.shape!=(len(self.tree_ids),7) or not np.all(np.isfinite(n)):
            raise ValueError('Expected a finite seven-coordinate tangent map')
        effort=finite_vector(physical_efforts,len(self.port_ids),'Eight physical joint effort ports')
        a=self.E.T@n
        return {'port_velocity_map':a,'port_to_tree_effort_map':self.E.copy(),
                'port_to_reduced_effort_map':a.T,'known_channel_to_port_effort_map':self.D.copy(),
                'known_channel_to_reduced_effort_map':a.T@self.D,
                'unresolved_p0_to_reduced_effort_map':n.T@self.unknown_E,
                'physical_efforts':effort,'tree_effort':self.E@effort,'reduced_effort':a.T@effort}

    @staticmethod
    def _point(evaluation,body,local):
        pose=evaluation['poses'][body];jac=evaluation['jacobians'][body]
        offset=pose[:3,:3]@local
        return pose[:3,3]+offset,jac[:3]-_skew(offset)@jac[3:]

    def source_geometry(self,evaluation):
        """Independent body-wrench projection and attachment-distance gradients.

        A distance-port effort is conjugate to the Euclidean attachment
        distance. It need not equal the effort conjugate to signed joint p.
        Neither coordinate is asserted to be calibrated hydraulic extension.
        """
        forces=[];coordinates=[]
        for jid in self.port_ids:
            j=self.joints[jid];b,f=j['base_body'],j['follower_body']
            tb=np.asarray(j['T_BJ']);tf=np.asarray(j['T_FJ'])
            wb=evaluation['poses'][b]@tb;wf=evaluation['poses'][f]@tf
            axis=wb[:3,:3]@np.asarray(j['axis'])
            if j['type']=='prismatic':
                pb,jb=self._point(evaluation,b,tb[:3,3]);pf,jf=self._point(evaluation,f,tf[:3,3])
                forces.append((jf-jb).T@axis)
                coordinates.append(float(axis@(pf-pb)))
            else:
                forces.append((evaluation['jacobians'][f][3:]-evaluation['jacobians'][b][3:]).T@axis)
                relative=wb[:3,:3].T@wf[:3,:3]
                skew=(relative-relative.T)/2
                sine=float(np.asarray(j['axis'])@np.array([skew[2,1],skew[0,2],skew[1,0]]))
                angle=float(np.arctan2(sine,(np.trace(relative)-1)/2))
                coordinates.append(angle)
        distances=[];distance_forces=[]
        for r in self.attachment_records:
            points=[self._point(evaluation,e['body'],e['point']) for e in r['endpoints']]
            delta=points[1][0]-points[0][0];distance=np.linalg.norm(delta)
            if not np.isfinite(distance) or distance<=1e-12:raise ValueError('Attachment distance is degenerate')
            distances.append(float(distance));distance_forces.append((points[1][1]-points[0][1]).T@(delta/distance))
        return {'joint_effort_map':np.column_stack(forces),
                'attachment_distance_map':np.column_stack(distance_forces),
                'distances':np.asarray(distances),'joint_coordinates':np.asarray(coordinates)}


def scaled_rank(matrix,tolerance=1e-10):
    """Structural rank after column normalization, not a physical conditioning metric."""
    value=np.asarray(matrix,dtype=float)
    norms=np.linalg.norm(value,axis=0)
    scaled=value/np.where(norms>1e-14,norms,1.)[None,:]
    singular=np.linalg.svd(scaled,compute_uv=False)
    return int(np.count_nonzero(singular>tolerance)),singular
