"""Bounded sprocket-speed PI shared by plant control and component validation.

The production controller is sampled at a fixed 1 kHz simulated rate. Pressure,
flow, motor losses and torque application remain in the hydraulic/native plant.
"""
import numpy as np


class ShaftSpeedPI:
    """Two independent PI regulators with bounded conditional integration."""
    def __init__(self, Kp=6500., Ki=4500., integral_bound=2.5, torque_bound=20000.):
        self.Kp=float(Kp);self.Ki=float(Ki)
        self.integral_bound=float(integral_bound);self.torque_bound=float(torque_bound)
        if not all(np.isfinite(x) and x>0 for x in [self.Kp,self.Ki,self.integral_bound,self.torque_bound]):
            raise ValueError('PI gains and bounds must be positive and finite')
        self.integral=np.zeros(2)

    def evaluate(self,desired_speed_rad_s,speed_rad_s,dt):
        desired=np.asarray(desired_speed_rad_s,dtype=float)
        speed=np.asarray(speed_rad_s,dtype=float)
        if desired.shape!=(2,) or speed.shape!=(2,) or not np.all(np.isfinite(desired)) or not np.all(np.isfinite(speed)):
            raise ValueError('Shaft-speed references and measurements must be finite two-element arrays')
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('PI sample interval must be positive and finite')
        err=desired-speed
        trial=np.clip(self.integral+dt*err,-self.integral_bound,self.integral_bound)
        command=self.Kp*err+self.Ki*trial
        limited=np.clip(command,-self.torque_bound,self.torque_bound)
        self.integral=np.where((abs(command)<=self.torque_bound)|(command*err<0),trial,self.integral)
        return limited
