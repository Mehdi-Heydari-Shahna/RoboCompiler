"""End-to-end check: the delivered rigid-contact rollout with the accepted vs the generated evaluator.

New extension code.  It runs the unchanged delivered simulation
(``original/kangaroo_pin/pin_simulation.run_case``, landing_nominal, 1 ms)
in the counterbalanced order accepted, generated, generated, accepted: twice
with the delivered cached accepted ``CutGraph`` and twice with
``SuppliedLayoutGraph`` (the generated evaluator in the supplied layout, with
the same exact single-entry cache).  Everything else (Pinocchio dynamics,
native contact solver, drive law, polish) is identical.  The full per-step
arrays are written to a temporary folder by the delivered code; compact
arrays of the first run of each evaluator and all summaries are retained, and
the second run of each evaluator is checked to be bitwise identical.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import bootstrap  # noqa: F401
from .evaluator import SuppliedLayoutGraph
from .io_utils import save_json

KEEP = ['time', 'base', 'motor', 'xi', 'act', 'lam', 'pacdm_closure', 'polish_shift', 'tangent', 'rcond']
ORDER = ('accepted_cutgraph', 'generated_evaluator', 'generated_evaluator', 'accepted_cutgraph')


def _compact(arrays, stride):
    out = {k: np.asarray(arrays[k]) for k in KEEP}
    keep = np.flatnonzero(np.isin(arrays['stored_step'], np.arange(0, len(arrays['time']), stride)))
    out['stored_step'] = np.asarray(arrays['stored_step'])[keep]
    out['stored_z'] = np.asarray(arrays['stored_z'])[keep]
    return out


def _identical(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in KEEP + ['stored_step', 'stored_z'])


def run_rollouts(comp, accepted, cmg0, duration, out, order=ORDER):
    from kangaroo_pin import pin_simulation as ps
    runs = []
    first = {}
    scratch = Path(tempfile.mkdtemp(prefix='kangaroo_rollout_'))
    try:
        for index, label in enumerate(order):
            plant = ps.KangarooPlant()
            if label == 'generated_evaluator':
                plant.graph = SuppliedLayoutGraph(comp, accepted, cmg0)
                plant.solver = accepted.PACDM(plant.graph)
            summary, a = ps.run_case('landing_nominal', dt=.001, duration=duration, plant=plant,
                                     output=scratch / f'{index}_{label}', progress=False)
            compact = _compact(a, 50)
            record = dict(run=index + 1, evaluator=label, elapsed_s=summary['elapsed_s'],
                          graph_evaluations=summary['graph_evaluations'], graph_cache_hits=summary['graph_cache_hits'],
                          fallback_count=summary['fallback_count'], corrector_iterations=summary['corrector_iterations'])
            if label in first:
                record['bitwise_identical_to_first_run'] = bool(_identical(first[label], compact))
            else:
                first[label] = compact
                np.savez_compressed(out / f'rollout_{label}.npz', **compact)
                save_json(out / f'rollout_{label}.json', summary)
            save_json(out / f'rollout_run{index + 1}_{label}_summary.json', summary)
            runs.append(record)
            print(f'rollout run {index + 1} {label}: {summary["elapsed_s"]:.1f} s, fresh evaluations '
                  f'{summary["graph_evaluations"]}', flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    A, G = first['accepted_cutgraph'], first['generated_evaluator']
    same_steps = len(A['time']) == len(G['time'])
    stored = np.intersect1d(A['stored_step'], G['stored_step'])
    za = A['stored_z'][np.searchsorted(A['stored_step'], stored)]
    zg = G['stored_z'][np.searchsorted(G['stored_step'], stored)]
    ra = next(r for r in runs if r['evaluator'] == 'accepted_cutgraph')
    rg = next(r for r in runs if r['evaluator'] == 'generated_evaluator')
    import json
    sa = json.loads((out / 'rollout_accepted_cutgraph.json').read_text())
    sg = json.loads((out / 'rollout_generated_evaluator.json').read_text())
    elapsed = {k: [r['elapsed_s'] for r in runs if r['evaluator'] == k] for k in ('accepted_cutgraph',
                                                                                 'generated_evaluator')}
    mean = {k: float(np.mean(v)) for k, v in elapsed.items()}
    metrics = ['maximum_motor_force_N', 'maximum_ground_normal_N', 'touchdown_s', 'achieved_crouch_m',
               'achieved_lateral_excursion_m', 'maximum_tilt_deg', 'final_position_error_m',
               'maximum_pacdm_closure', 'maximum_polish_shift', 'minimum_mapping_rcond']
    comparison = dict(
        duration_s=duration, steps=int(len(A['time']) - 1), identical_step_count=bool(same_steps),
        run_order=list(order), runs=runs,
        elapsed_s=elapsed, mean_elapsed_s=mean,
        wall_time_reduction_percent=100. * (1. - mean['generated_evaluator'] / mean['accepted_cutgraph']),
        repeat_runs_bitwise_identical=bool(all(r.get('bitwise_identical_to_first_run', True) for r in runs)),
        fresh_graph_evaluations={k: ra['graph_evaluations'] if k == 'accepted_cutgraph' else rg['graph_evaluations']
                                 for k in elapsed},
        cache_hits={k: ra['graph_cache_hits'] if k == 'accepted_cutgraph' else rg['graph_cache_hits']
                    for k in elapsed},
        fallback_count={'accepted_cutgraph': ra['fallback_count'], 'generated_evaluator': rg['fallback_count']},
        corrector_iterations={'accepted_cutgraph': ra['corrector_iterations'],
                              'generated_evaluator': rg['corrector_iterations']},
        max_base_pose_difference=float(np.max(np.abs(A['base'] - G['base']))) if same_steps else None,
        max_motor_difference_m=float(np.max(np.abs(A['motor'] - G['motor']))) if same_steps else None,
        max_velocity_difference=float(np.max(np.abs(A['xi'] - G['xi']))) if same_steps else None,
        max_per_foot_impulse_difference_Ns=float(np.max(np.abs(
            A['lam'].reshape(len(A['lam']), 2, 4, 3).sum(axis=2) - G['lam'].reshape(len(G['lam']), 2, 4, 3).sum(axis=2)
        ))) if same_steps else None,
        max_corner_impulse_difference_Ns=float(np.max(np.abs(A['lam'] - G['lam']))) if same_steps else None,
        corner_impulse_note='Each foot has four coplanar corners; individual corner impulses are not unique, '
                            'per-foot impulses and post-contact velocities are.',
        max_stored_state_difference=float(np.max(np.abs(za - zg))) if len(stored) else None,
        task_metrics={m: dict(accepted=sa.get(m), generated=sg.get(m)) for m in metrics},
        note='Same delivered simulation code and settings; only the PACDM graph object differs. Two runs per '
             'evaluator in the counterbalanced order A, G, G, A on the benchmark host; wall times are descriptive.')
    save_json(out / 'rollout_comparison.json', comparison)
    return comparison
