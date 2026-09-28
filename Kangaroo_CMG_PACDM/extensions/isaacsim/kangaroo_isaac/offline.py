"""Executable CPU checks. These are NOT Isaac Sim or PhysX results."""
from __future__ import annotations
from pathlib import Path
import csv,sys,platform,time,json
import numpy as np
from .model import ROOT,Model
from .control import Reference
from .cut_graph import CutGraph
from .pacdm import PACDM
from .io_utils import write_json,sha256


def audit_reference(output,verbose=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();m=Model();r=Reference();g=CutGraph(m.c,np.asarray(m.c['initial_seed']))
    solver=PACDM(g);rows=[];all_ok=True;gimbal_margin=90.
    for i,(t,aug,q) in enumerate(zip(r.data['t'],r.data['qaug'],r.data['q'])):
        N,info=solver.mapping(aug)
        residual,J,_=g.residual(aug)
        P=m.fk(q,r.data['base'][i]);gap,dots,dist=m.closures(P)
        # For actor0=B and actor1=A: the unconstrained swing about B.Y
        # becomes singular when A.X is parallel to B.Z.
        for k in m.universal:
            c=m.c['closures'][k]
            A=P[m.bi[c['body1']],:3,:3]@np.asarray(c['frame1_R'])
            B=P[m.bi[c['body2']],:3,:3]@np.asarray(c['frame2_R'])
            rel=B.T@A
            beta=np.arctan2(-rel[2,0],rel[0,0])
            margin=abs(90-abs(np.degrees(beta)))
            gimbal_margin=min(gimbal_margin,margin)
        row={'sample':i,'time_s':float(t),'mapping_success':bool(info['success']),
             'rank_full':int(info['rank_full']),'rank_passive':int(info['rank_passive']),
             'rcond':float(info['rcond']),'augmented_residual_inf':float(abs(residual).max()),
             'tangent_residual_inf':float(abs(J@N).max()),
             'physical_point_gap_m':float(np.linalg.norm(gap,axis=1).max()),
             'universal_dot':float(abs(dots).max()),'distance_equivalent_residual_m':float(abs(dist).max()),
             'reference_tangent_difference':float(abs(N[:76]-r.data['tangent'][i]).max())}
        rows.append(row)
        all_ok&=bool(info['success'] and row['augmented_residual_inf']<1e-8 and row['tangent_residual_inf']<1e-8
                     and row['physical_point_gap_m']<1e-8 and row['universal_dot']<1e-8)
        if verbose and (i%100==0 or i==400):print(f'CPU reference audit: {i+1}/401 samples',flush=True)
    with (output/'reference_samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    extrema={k:max(x[k] for x in rows) for k in ('augmented_residual_inf','tangent_residual_inf','physical_point_gap_m',
            'universal_dot','distance_equivalent_residual_m','reference_tangent_difference')}
    original=(ROOT/'vendor_v22/pacdm.py').read_bytes();actual=(ROOT/'kangaroo_isaac/pacdm.py').read_bytes()
    all_ok&=actual==original
    report={'status':'PASS' if all_ok else 'FAIL','scope':'OFFLINE_CPU_REFERENCE_AND_COMPILATION_ONLY',
            'isaac_sim_execution':'NOT_RUN','native_performance':'NOT_MEASURED','certified_ready':False,
            'samples_checked':len(rows),'augmented_coordinates':g.n,'physical_coordinates':m.n,
            'independent_actuators':len(m.active),'body_count':m.nb,'mass_kg':float(m.mass.sum()),
            'physical_cuts':len(m.ca),'point_cuts':len(m.ca)-len(m.universal),'universal_cuts':len(m.universal),
            'rank_full_values':sorted(set(x['rank_full'] for x in rows)),
            'rank_passive_values':sorted(set(x['rank_passive'] for x in rows)),
            'minimum_rcond':min(x['rcond'] for x in rows),'maximum_errors':extrema,
            'minimum_reference_universal_gimbal_margin_deg':gimbal_margin,
            'unmodified_pacdm_core':actual==original,'pacdm_sha256':sha256(ROOT/'kangaroo_isaac/pacdm.py'),
            'source_archive_sha256':json.loads((ROOT/'data/port_provenance.json').read_text())['source_archive_sha256'],
            'elapsed_s':time.perf_counter()-start,'python':sys.version,'platform':platform.platform(),
            'numpy':np.__version__}
    write_json(output/'offline_reference_audit.json',report)
    return report
