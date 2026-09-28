"""Explicit local verification experiments, not hydraulic/hardware commands."""
import numpy as np


def experiments():
    zero=[0.]*8
    common={'duration_s':.08,'steps_s':[.01,.005,.0025],
            'port_order':['p0','p1','p2','p3','p4','p5','q21','q23'],
            'port_units':['N']*6+['N*m']*2,
            'source_p0_effort_resolved':False,
            'p0_semantics':'The first effort entry is an explicit experiment input; it does not set the missing source-runtime force.',
            'hydraulic_model_used':False,'contact_model_used':False,
            'force_equation':'e(t)=bias+amplitude*sin(2*pi*t/period+phase)',
            'wave_period_s':.04,'wave_phase_rad':[0.,.4,.8,1.2,1.6,2.,2.4,2.8]}
    return [{**common,'id':'free_zero_gravity','gravity_m_s2':[0.,0.,0.],
             'initial_independent_velocity_rad_s':[.08,-.04,.05,-.07,.06,-.05,.12],
             'effort_bias_SI':zero.copy(),'effort_amplitude_SI':zero.copy(),
             'purpose':'Conservative free motion from nonzero velocity; all eight applied effort ports explicitly zero.'},
            {**common,'id':'gravity_release','gravity_m_s2':[0.,0.,-9.81],
             'initial_independent_velocity_rad_s':[0.]*7,
             'effort_bias_SI':zero.copy(),'effort_amplitude_SI':zero.copy(),
             'purpose':'Conversion of potential to kinetic energy; all eight applied effort ports explicitly zero.'},
            {**common,'id':'driven_ports','gravity_m_s2':[0.,0.,-9.81],
             'initial_independent_velocity_rad_s':[.03,-.02,.025,-.03,.02,-.015,.09],
             'effort_bias_SI':[125.,-250.,500.,-1000.,-1000.,750.,4.,6.],
             'effort_amplitude_SI':[200.,250.,600.,700.,500.,800.,5.,8.],
             'purpose':'Smooth time-varying mechanical effort with explicit p0 forcing and observable integration refinement; not a measured hydraulic bandwidth.'}]


def effort_at(case,time_s):
    if not np.isscalar(time_s) or not np.isfinite(time_s):raise ValueError('Time must be finite')
    values=np.asarray(case['effort_bias_SI'])+np.asarray(case['effort_amplitude_SI'])*np.sin(
        2*np.pi*time_s/case['wave_period_s']+np.asarray(case['wave_phase_rad']))
    if values.shape!=(8,) or not np.all(np.isfinite(values)):raise ValueError('Every test needs eight finite physical port efforts')
    return values
