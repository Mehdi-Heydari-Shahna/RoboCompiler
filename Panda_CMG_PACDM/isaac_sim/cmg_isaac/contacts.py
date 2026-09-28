"""Native per-collider PhysX contacts, not inferred actuator loads."""
import numpy as np


class ContactObserver:
    def __init__(self, labels, dt):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        self.labels=labels;self.dt=dt;self.decode=PhysicsSchemaTools.intToSdfPath
        self.error=None;self.total_contact_points=0;self.callback_count=0
        self.last_bad=[];self.clear()
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(self._callback)

    def clear(self):
        self.left=0.;self.right=0.;self.pad_contacts=0;self.bad=0
        self.support=0.;self.contact_points=0;self.left_count=0;self.right_count=0
        self.last_bad=[]

    def _callback(self, headers, data):
        try:
            self.callback_count+=1
            for h in headers:
                if h.num_contact_data==0:
                    continue  # includes CONTACT_LOST
                p0,p1=str(self.decode(h.collider0)),str(self.decode(h.collider1))
                if p0 not in self.labels or p1 not in self.labels:
                    raise RuntimeError(f'Unmapped native collider pair: {p0}, {p1}')
                a,b=self.labels[p0],self.labels[p1]
                robot=[x for x in (a,b) if x['kind']=='robot']
                obj=any(x['kind']=='object' for x in (a,b))
                scene=next((x for x in (a,b) if x['kind']=='scene'),None)
                bad=(len(robot)==2 or
                     bool(robot and scene and (scene['name']!='floor' or robot[0]['body']!='link0')) or
                     bool(obj and scene and scene['name']=='barrier') or
                     bool(obj and robot and robot[0]['body'] not in ('left_finger','right_finger')))
                if bad:
                    self.bad+=1
                    if len(self.last_bad)<5:self.last_bad.append([p0,p1])
                for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                    c=data[i]
                    normal=np.asarray(c.normal,dtype=float)
                    impulse=np.asarray(c.impulse,dtype=float)
                    if normal.shape!=(3,) or impulse.shape!=(3,):
                        raise RuntimeError('Unexpected PhysX contact-vector representation')
                    if not np.all(np.isfinite(np.r_[normal,impulse])):
                        raise FloatingPointError('Nonfinite native contact impulse')
                    force=abs(float(normal@impulse))/self.dt
                    self.contact_points+=1;self.total_contact_points+=1
                    if obj and robot and robot[0].get('pad',False):
                        self.pad_contacts+=1
                        if robot[0]['body']=='left_finger':
                            self.left+=force;self.left_count+=1
                        elif robot[0]['body']=='right_finger':
                            self.right+=force;self.right_count+=1
                    if obj and scene and scene['name']=='pick_plinth':
                        self.support+=abs(float(impulse[2]))/self.dt
        except Exception as exc:
            # Native callbacks must not swallow an instrumentation failure.
            self.error=repr(exc)

    def values(self):
        if self.error:
            raise RuntimeError('Contact reporting failed: '+self.error)
        return dict(left_normal_force=self.left,right_normal_force=self.right,
                    normal_force=self.left+self.right,pad_contacts=self.pad_contacts,
                    bilateral_contact=int(self.left_count>0 and self.right_count>0),
                    unexpected_contacts=self.bad,pick_support_N=self.support)
