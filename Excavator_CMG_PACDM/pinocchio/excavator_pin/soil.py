"""Declared soil-interaction surrogate for the Pinocchio backend.

Pinocchio has no granular contact solver.  In this backend the dry-granular
bed of the original MuJoCo prototype is replaced by a deterministic, smooth,
state-dependent *prescribed wrench and payload model*.  It is a numerical
test load, not a soil constitutive law, not a calibration and not a replacement
for the native particle evidence.

Geometry is taken from the preserved source: bed bounds from
``soil_scene.BOUNDS``, the optional cleared entry slot, the CAD bucket section
and width from ``digging_export`` and the lip/heel points from ``digging_path``.

Model (all SI):

* depth ``d = footprint * ramp(z_surface - z_lip, 0.01 m)`` of the bucket lip
  below the bed surface ``z = 0``; the C2 ramp equals ``x - 0.005 m`` beyond
  1 cm, and the footprint is a C2 indicator of the material region;
* cutting resistance at the lip ``F = -N_gamma * rho * g * w * d**2 * v/|v|_reg``
  with ``|v|_reg = sqrt(|v|**2 + 0.02**2)`` (the dry, cohesionless form of the
  fundamental earthmoving relation, velocity-regularized direction);
* material entering the bucket ``mdot_in = eta * rho * w * d * ramp(v.n, 0.005)``
  where ``n`` is the outward bucket-mouth normal (C2 positive part of ``v.n``);
* retention capacity ``C = rho * V_struck * (1 - (1 - s_n) * (1 - s_d))`` with
  ``s_n = smoothstep((n_z + 0.2) / 0.5)`` and ``s_d = smoothstep(d / 0.02)``:
  material is kept while the mouth faces up or the lip is submerged; material
  above capacity leaves with time constant ``tau_release``; released material
  is counted as deposited while the mouth centre is inside the registered
  receiver footprint, otherwise as spilled;
* payload weight ``(0, 0, -m g)`` at the CAD cavity centroid.  Payload inertia is
  not modelled; the payload is an applied load, not an attached body.

Every smooth switch is a quintic smoothstep or a C2 ramp so that the RK4
refinement study is meaningful.  All constants below are declared assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np

GRAVITY = 9.81


def smoothstep(s):
    s = np.clip(s, 0., 1.)
    return s * s * s * (10. - 15. * s + 6. * s * s)


def smoothstep_derivative(s):
    inside = (s > 0.) & (s < 1.)
    return np.where(inside, 30. * s * s * (1. - s) ** 2, 0.)


def ramp(x, width):
    """C2 positive part: 0 for x<=0, x-width/2 for x>=width."""
    x = np.asarray(x, dtype=float)
    y = np.where(x <= 0., 0., np.where(x >= width, x - .5 * width,
                                       x ** 3 / width ** 2 - x ** 4 / (2. * width ** 3)))
    return y if y.ndim else float(y)


def box(point, lower, upper, width):
    """C2 indicator of an axis-aligned box in the first len(lower) coordinates."""
    value = 1.
    for x, lo, hi in zip(point, lower, upper):
        value *= smoothstep((x - lo) / width) * smoothstep((hi - x) / width)
    return float(value)


@dataclass(frozen=True)
class SoilParameters:
    enabled: bool = True
    bulk_density_kg_m3: float = 1200.   # 0.60 packing x 2000 kg/m3 source particle density
    resistance_factor_N_gamma: float = 3.5
    fill_efficiency: float = 1.0
    surface_z_m: float = 0.0
    bed_x_m: tuple = (5.45, 7.95)
    bed_y_m: tuple = (-.72, .72)
    entry_slot_m: float = 0.0
    footprint_transition_m: float = .05
    depth_ramp_m: float = .01
    velocity_regularization_m_s: float = .02
    inflow_ramp_m_s: float = .005
    retention_normal_z: tuple = (-.20, .30)
    release_time_s: float = .25
    release_ramp_kg: float = .5
    receiver_center_xy_m: tuple = (0., 0.)
    receiver_half_size_m: tuple = (1., 1.)
    receiver_transition_m: float = .05

    def scaled(self, **changes):
        return replace(self, **changes)

    def as_dict(self):
        return asdict(self)


class BucketGeometry:
    """CAD-derived cavity section, width, mouth normal and centroid (bucket frame)."""

    def __init__(self, section, bucket_x, lip, heel):
        section = np.asarray(section, dtype=float)
        y, z = section[:, 0], section[:, 1]
        cross = y * np.roll(z, -1) - np.roll(y, -1) * z
        area = .5 * cross.sum()
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6. * area)
        cz = ((z + np.roll(z, -1)) * cross).sum() / (6. * area)
        self.width_m = float(bucket_x[1] - bucket_x[0])
        self.section_area_m2 = float(abs(area))
        self.struck_volume_m3 = self.section_area_m2 * self.width_m
        self.centroid_local = np.array([.5 * (bucket_x[0] + bucket_x[1]), cy, cz])
        lip, heel = np.asarray(lip, dtype=float), np.asarray(heel, dtype=float)
        mouth = heel - lip
        normal = np.array([0., -mouth[2], mouth[1]])
        normal /= np.linalg.norm(normal)
        middle = .5 * (lip + heel)
        if normal @ (self.centroid_local - middle) > 0.:
            normal = -normal
        self.mouth_normal_local = normal
        self.lip_local, self.heel_local = lip, heel
        self.mouth_local = middle

    def as_dict(self):
        return dict(width_m=self.width_m, section_area_m2=self.section_area_m2,
                    struck_volume_m3=self.struck_volume_m3,
                    centroid_local_m=self.centroid_local.tolist(),
                    mouth_normal_local=self.mouth_normal_local.tolist(),
                    lip_local_m=self.lip_local.tolist(), heel_local_m=self.heel_local.tolist())


class SoilSurrogate:
    """Evaluate the declared wrench and payload rates at one plant state."""

    def __init__(self, parameters: SoilParameters, geometry: BucketGeometry):
        self.p = parameters
        self.g = geometry
        self.capacity_kg = parameters.bulk_density_kg_m3 * geometry.struck_volume_m3

    def evaluate(self, lip, lip_velocity, mouth, normal_world, payload_kg):
        p = self.p
        lip = np.asarray(lip, dtype=float)
        velocity = np.asarray(lip_velocity, dtype=float)
        x_hi = p.bed_x_m[1] - p.entry_slot_m
        footprint = box(lip[:2], (p.bed_x_m[0], p.bed_y_m[0]), (x_hi, p.bed_y_m[1]),
                        p.footprint_transition_m) if p.enabled else 0.
        depth = footprint * ramp(p.surface_z_m - lip[2], p.depth_ramp_m)
        speed = float(np.sqrt(velocity @ velocity + p.velocity_regularization_m_s ** 2))
        weight_rho_g_w = p.bulk_density_kg_m3 * GRAVITY * self.g.width_m
        magnitude = p.resistance_factor_N_gamma * weight_rho_g_w * depth ** 2
        cutting_force = -magnitude * velocity / speed
        inflow_speed = ramp(float(velocity @ normal_world), p.inflow_ramp_m_s)
        fill = p.fill_efficiency * p.bulk_density_kg_m3 * self.g.width_m * depth * inflow_speed
        # Material cannot leave while the lip remains submerged in the bed.
        submerged = smoothstep(depth / .02)
        orientation = smoothstep((normal_world[2] - p.retention_normal_z[0])
                                 / (p.retention_normal_z[1] - p.retention_normal_z[0]))
        capacity = self.capacity_kg * (1. - (1. - orientation) * (1. - submerged))
        release = ramp(payload_kg - capacity, p.release_ramp_kg) / p.release_time_s
        receiver = box(np.asarray(mouth)[:2],
                       np.asarray(p.receiver_center_xy_m) - np.asarray(p.receiver_half_size_m),
                       np.asarray(p.receiver_center_xy_m) + np.asarray(p.receiver_half_size_m),
                       p.receiver_transition_m)
        return dict(footprint=footprint, depth_m=float(depth), cutting_force_N=cutting_force,
                    weight_force_N=np.array([0., 0., -payload_kg * GRAVITY]),
                    fill_rate_kg_s=float(fill), release_rate_kg_s=float(release),
                    deposit_rate_kg_s=float(release * receiver),
                    spill_rate_kg_s=float(release * (1. - receiver)),
                    capacity_kg=float(capacity), receiver_fraction=receiver,
                    orientation_retention=float(orientation))
