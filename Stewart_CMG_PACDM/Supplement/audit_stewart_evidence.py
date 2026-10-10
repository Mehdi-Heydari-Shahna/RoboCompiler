#!/usr/bin/env python3
"""Audit saved Stewart trajectories and PACDM reference states.

No simulation integration is run. Requires NumPy and SciPy; the unchanged
PointGraph/PACDM core is imported from the accompanying benchmark directory.

Usage:
    python audit_stewart_evidence.py
    python audit_stewart_evidence.py --root /path/to/Stewart_CMG_PACDM_Final \
        --output /path/to/audit_metrics.json

The default root is the repository directory above Supplement/. The
JSON destination defaults to audit_metrics.json beside this script. All input
paths written to the JSON are relative to the benchmark root.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path,
        default=script_dir.parent,
        help='Benchmark directory containing data/, results/, stewart/, and vendor/.')
    parser.add_argument('--output', type=Path,
        default=script_dir / 'audit_metrics.json', help='Output JSON file.')
    args = parser.parse_args()
    R = args.root.expanduser().resolve()
    required = ['data/stewart.cmg.json', 'vendor/pacdm_original.py',
                'stewart/model.py', 'results/reference.npz']
    required += ['results/' + n + '.npz' for n in
                 ['nominal', 'fine', 'heavy_payload', 'no_feedforward', 'pacdm']]
    for name in required:
        if not (R / name).is_file():
            parser.error('Missing input: ' + str(R / name))
    sys.path.insert(0, str(R))
    from vendor.pacdm_original import PointGraph, PACDM
    from stewart.model import inverse_seed

    def load(p):
     with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    data={n:load(R/'results'/f'{n}.npz') for n in ['nominal','fine','heavy_payload','no_feedforward']}
    data['pacdm']=load(R/'results/pacdm.npz')
    ref=load(R/'results/reference.npz')
    cmg=json.loads((R/'data/stewart.cmg.json').read_text())
    ids=cmg['coordinate_ids']; active=[ids.index(n) for n in cmg['independent_ids']]
    j={x['id']:x for x in cmg['joints']}
    lo=np.array([j[n]['limits']['lower'] for n in ids]);hi=np.array([j[n]['limits']['upper'] for n in ids])
    mi=np.array([k for k,n in enumerate(ids) if j[n]['type']=='prismatic'])
    ri=np.array([k for k,n in enumerate(ids) if j[n]['type']=='revolute'])
    audit={'sources':{},'case_metrics':{},'reference':{},'comparisons':{},'notes':[]}
    for n,d in data.items():
     p=R/'results'/f'{n}.npz'
     audit['sources'][n]={'path':p.relative_to(R).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
     t,q,target=d['time'],d['q'],d['target_pose']; epos=np.linalg.norm(q[:,:3]-target[:,:3],axis=1)
     er=Rotation.from_euler('ZYX',target[:,3:6]).inv()*Rotation.from_euler('ZYX',q[:,3:6])
     eang=np.rad2deg(er.magnitude());dock=t>=21
     limit_margin=np.minimum(q-lo,hi-q)
     m={'samples':len(t),'timestep_s':float(np.median(np.diff(t))),
     'rms_position_mm':float(np.sqrt(np.mean(epos**2))*1000),'max_position_mm':float(epos.max()*1000),
     'rms_orientation_deg':float(np.sqrt(np.mean(eang**2))),'max_orientation_deg':float(eang.max()),
     'peak_position_time_s':float(t[np.argmax(epos)]),'peak_orientation_time_s':float(t[np.argmax(eang)]),
     'docking_max_position_mm':float(epos[dock].max()*1000),'docking_max_orientation_deg':float(eang[dock].max()),
     'max_point_closure_m':float(d['closure_error_m'].max()),'max_force_N':float(abs(d['force']).max()),
     'force_utilization_percent':float(abs(d['force']).max()/900*100),'force_limit_touches_count':int(np.sum(np.any(abs(d['force'])>=900,axis=1))),
     'all_joint_limits_respected':bool(np.all(limit_margin>=0)),
     'min_linear_coordinate_limit_margin_m':float(limit_margin[:,mi].min()),
     'min_revolute_coordinate_limit_margin_rad':float(limit_margin[:,ri].min()),
     'leg_length_range_m':[float(q[:,active].min()),float(q[:,active].max())],
     'per_axis_rms_position_mm':(np.sqrt(np.mean((q[:,:3]-target[:,:3])**2,axis=0))*1000).tolist(),
     'per_axis_maxabs_position_mm':(np.max(abs(q[:,:3]-target[:,:3]),axis=0)*1000).tolist(),
     'max_error_recomputation_difference_m':float(abs(epos-d['pose_error_m']).max()),
     'max_angle_error_recomputation_difference_rad':float(abs(np.deg2rad(eang)-d['angle_error_rad']).max()),
     'rms_position_by_phase_mm':{},'pose_coordinate_min_xyz_yaw_pitch_roll':q[:,:6].min(axis=0).tolist(),
     'pose_coordinate_max_xyz_yaw_pitch_roll':q[:,:6].max(axis=0).tolist()}
     if 'actuator_error_m' in d:
      m['max_abs_actuator_error_mm']=float(np.max(abs(d['actuator_error_m']))*1000)
      m['rms_actuator_error_mm']=float(np.sqrt(np.mean(d['actuator_error_m']**2))*1000)
     if 'reduced_equation_residual' in d:m['max_reduced_equation_residual_N']=float(d['reduced_equation_residual'].max())
     for label,a,b in [('initial_hold',0,2),('helical',2,9),('figure_eight',9,17),('docking',17,20),('hold',20,22.00001)]:
      ix=(t>=a)&(t<b);m['rms_position_by_phase_mm'][label]=float(np.sqrt(np.mean(epos[ix]**2))*1000)
     audit['case_metrics'][n]=m

    n=data['nominal']
    for name,other in [('direct_vs_mujoco','pacdm'),('dt1_vs_dt2','fine')]:
     d=data[other];q=np.column_stack([np.interp(n['time'],d['time'],d['q'][:,k]) for k in range(24)])
     dp=np.linalg.norm(n['q'][:,:3]-q[:,:3],axis=1)
     dr=(Rotation.from_euler('ZYX',n['q'][:,3:6]).inv()*Rotation.from_euler('ZYX',q[:,3:6])).magnitude()
     audit['comparisons'][name]={'max_position_difference_um':float(dp.max()*1e6),'rms_position_difference_um':float(np.sqrt(np.mean(dp**2))*1e6),
     'max_orientation_difference_deg':float(np.rad2deg(dr.max())),
     'rms_orientation_difference_deg':float(np.rad2deg(np.sqrt(np.mean(dr**2)))),
     'max_position_difference_time_s':float(n['time'][np.argmax(dp)]),'max_orientation_difference_time_s':float(n['time'][np.argmax(dr)])}
    m=audit['case_metrics']
    audit['comparisons']['feedforward_rms_position_reduction_percent']=100*(1-m['nominal']['rms_position_mm']/m['no_feedforward']['rms_position_mm'])
    audit['comparisons']['step_refinement_rms_position_relative_change_percent']=100*(m['fine']['rms_position_mm']/m['nominal']['rms_position_mm']-1)
    for k in ['closure_residual','tangent_residual','acceleration_residual','selected_block_condition','reduced_mass_min_eigenvalue']:
     audit['reference'][k]={'minimum':float(ref[k].min()),'maximum':float(ref[k].max()),'zero_samples':int((ref[k]==0).sum())}
    audit['reference']['samples']=len(ref['time'])
    audit['reference']['max_pose_reconstruction_component_mixed_SI']=float(np.max(abs(ref['q'][:,:6]-ref['target_pose'])))
    audit['reference']['max_position_reconstruction_norm_m']=float(np.max(np.linalg.norm(ref['q'][:,:3]-ref['target_pose'][:,:3],axis=1)))
    audit['reference']['max_orientation_reconstruction_deg']=float(np.rad2deg((Rotation.from_euler('ZYX',ref['target_pose'][:,3:6]).inv()*Rotation.from_euler('ZYX',ref['q'][:,3:6])).magnitude().max()))
    audit['reference']['max_feedforward_force_N']=float(abs(ref['feedforward_force']).max())
    audit['reference']['target_pose_range']=np.c_[ref['target_pose'].min(axis=0),ref['target_pose'].max(axis=0)].tolist()
    g=PointGraph(cmg,inverse_seed(cmg,cmg['geometry']['nominal_pose']));s=PACDM(g)
    rank_full=[];rank_passive=[];consistency=[];res=[]
    for q,M in zip(ref['q_augmented'],ref['mapping']):
     N,info=s.mapping(q)
     if not info['success']:raise RuntimeError(info)
     rank_full.append(info['rank_full']);rank_passive.append(info['rank_passive'])
     consistency.append(abs(N[:g.nt]-M).max());res.append(info['residual_inf'])
    audit['reference']['recomputed_all_reference_mapping_checks_pass']=True
    audit['reference']['recomputed_full_rank_range']=[min(rank_full),max(rank_full)]
    audit['reference']['recomputed_passive_rank_range']=[min(rank_passive),max(rank_passive)]
    audit['reference']['recomputed_mapping_max_abs_difference']=float(max(consistency))
    audit['reference']['recomputed_closure_max_abs_difference']=float(np.max(abs(np.array(res)-ref['closure_residual'])))
    audit['notes']=[
     'Orientation coordinate order is yaw,pitch,roll and matrix convention is intrinsic ZYX.',
     'Tracking positions are platform-body origin; rotation error is geodesic norm of log(R_reference^T R_actual).',
     'Augmented residual summaries combine angular and translational SI components.',
     'Selected block condition is condition number in the matrix 1-norm.',
     'Controller runtime q,velocity,feedforward are independent CubicSpline interpolants of the stored reference arrays, not online inverse-dynamics recomputation.',
     'No-feedforward keeps the same PACDM reference and PD controller and removes the precomputed analytical force feedforward only.',
     'Heavy-payload run uses unchanged 8-kg reference/feedforward, with actual payload 14 kg and unchanged controller gains.',
     'All joint-limit checks combine computational platform-chart limits and limb limits. Per-type margins distinguish m and rad.',
     'Nominal physical closure point separation uses max over six Euclidean endpoint distances, while mechanics.physical_point_closure_max_abs uses maximum component.',
     'The 25-configuration mechanics record contains aggregate maxima and minima.',
     'This audit reevaluates closure ranks and tangent mappings at every saved augmented reference state.'
    ]
    for label, name in [('reference', 'results/reference.npz'),
                        ('CMG', 'data/stewart.cmg.json'),
                        ('PACDM_core', 'vendor/pacdm_original.py'),
                        ('model_module', 'stewart/model.py')]:
        p = R / name
        audit['sources'][label] = {'path': name,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Audited {len(data)} saved rollouts and {len(ref["time"])} reference states.')
    print('Reference full/passive rank ranges:',
          audit['reference']['recomputed_full_rank_range'],
          audit['reference']['recomputed_passive_rank_range'])
    print('Maximum mapping reproduction difference:',
          audit['reference']['recomputed_mapping_max_abs_difference'])
    print(f'Wrote {out.name}')


if __name__ == '__main__':
    main()
