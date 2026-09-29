"""Source contact compliance, mapped to the PhysX compliant-contact model.

The original v22 MuJoCo task (``vendor_v22/contact_task.py``) does not use rigid
contacts. It declares soft sole/floor contacts with

    solref = (contact_time_constant_s = 0.003 s, dampratio = 1)
    solimp = (dmin = 0.95, dmax = 0.99, width = 0.001 m; default midpoint/power)

Earlier Isaac ports authored rigid PhysX contacts instead. With this robot
(links of 2-4 g inside closed chains next to the feet) the rigid TGS contact
solve produced step-to-step normal-impulse noise at the soles: in the recorded
23.0.4 runs each resting sole reported zero normal force at roughly 40 % of the
1-kHz controller ticks (25 us step) and at roughly 65 % of the 5-ms telemetry
samples (12.5 us step), and the feet crept (13.8 mm and 22.6 mm maximum drift).
The source trace creeps 0.19-0.50 mm.

PhysX offers an implicit compliant contact (``physxMaterial:compliantContact*``).
With ``compliantContactAccelerationSpring`` the spring/damper act per unit
effective contact mass, the same normalisation MuJoCo uses for its reference
acceleration. MuJoCo's constraint-space soft-contact dynamics are

    a1 + d (b v + k r) = (1 - d) a0,   b = 2/(dmax T),   k = d/(dmax^2 T^2 zeta^2)

At rest (a1 = v = 0) this is a spring of acceleration stiffness
d k / (1 - d) acting on the violation r. At the small-violation impedance
d = dmin this gives

    k_contact = dmin^2 / ((1 - dmin) dmax^2 T^2 zeta^2)      [1/s^2]

and the damping is chosen for the source damping ratio around that stiffness:

    c_contact = 2 zeta sqrt(k_contact)                        [1/s]

A linear PhysX spring cannot also reproduce MuJoCo's 3 ms dynamic time
constant (that would need an acceleration stiffness 20x lower and gives
millimetre-scale resting penetration). This is therefore a documented
approximation, not an exact equivalence. The values are derived only from the
source solref/solimp; they are not tuned against a task gate.
"""
from __future__ import annotations
import math

CONTACT_MODELS = ('source_compliance', 'rigid')

# Exactly the values declared by vendor_v22/contact_task.py (DEFAULTS and make_model).
SOURCE_CONTACT = {
    'time_constant_s': 0.003,
    'damping_ratio': 1.0,
    'solimp_dmin': 0.95,
    'solimp_dmax': 0.99,
    'solimp_width_m': 0.001,
    'origin': 'vendor_v22/contact_task.py: contact_time_constant_s=.003, solref "<T> 1", solimp ".95 .99 .001"',
}


def source_compliance(source=SOURCE_CONTACT):
    """Return (stiffness 1/s^2, damping 1/s) of the PhysX acceleration spring."""
    T = float(source['time_constant_s']); zeta = float(source['damping_ratio'])
    dmin = float(source['solimp_dmin']); dmax = float(source['solimp_dmax'])
    values = (T, zeta, dmin, dmax)
    if not all(math.isfinite(v) for v in values) or T <= 0 or zeta <= 0 or not 0 < dmin <= dmax < 1:
        raise ValueError('Invalid source contact parameters')
    stiffness = dmin**2 / ((1 - dmin) * dmax**2 * T**2 * zeta**2)
    damping = 2 * zeta * math.sqrt(stiffness)
    return stiffness, damping


SOURCE_STIFFNESS_PER_S2, SOURCE_DAMPING_PER_S = source_compliance()


def describe(cfg) -> dict:
    """Configuration record written with every native result."""
    if cfg.contact_model == 'rigid':
        return {'model': 'rigid', 'note': 'legacy rigid PhysX contact (23.0.4 and earlier); not source-faithful'}
    return {'model': 'source_compliance',
            'physx_material': {'compliantContactAccelerationSpring': True,
                               'compliantContactStiffness_per_s2': cfg.contact_stiffness_per_s2,
                               'compliantContactDamping_per_s': cfg.contact_damping_per_s},
            'derivation': 'k = dmin^2/((1-dmin) dmax^2 T^2 zeta^2), c = 2 zeta sqrt(k)',
            'source': dict(SOURCE_CONTACT),
            'equivalence_claim': False,
            'scope': 'same small-violation static stiffness and damping ratio as the MuJoCo source contact; '
                     'not the same 3 ms dynamic time constant, cone shape or impedance nonlinearity'}
