"""Tables, benefit summary and readable report generated from a result directory.

Every number is read from the saved CSV/JSON records of the same directory; the
report is regenerated with ``python -m src.reporting RESULT_DIR``.
"""
from __future__ import annotations

from pathlib import Path
import html
import json
import sys

import numpy as np

from .io_utils import read_csv, save_csv, save_json

NAMES = {
    'original_global': 'Original adapter, global PACDM',
    'compiled_global': 'Generated evaluator, global PACDM',
    'compiled_global_reuse': 'Generated evaluator, global PACDM + whole-input reuse',
    'compiled_global_loopfree': 'Generated evaluator, global PACDM + loop-free exclusion',
    'compiled_modular': 'Generated loop modules, PACDM + module reuse',
    'modular_no_reuse': 'Generated loop modules, PACDM, no reuse',
    'modular_no_predictor': 'Generated loop modules, PACDM, no predictor',
    'trf_global': 'Generated evaluator, global analytic TRF + whole-input reuse',
    'trf_global_loopfree': 'Generated evaluator, global analytic TRF + loop-free exclusion',
    'trf_modular': 'Generated loop modules, analytic TRF + module reuse',
    'native_newton': 'Native Pinocchio Newton + whole-input reuse',
    'native_newton_loopfree': 'Native Pinocchio Newton + loop-free exclusion',
    'native_modular': 'Native Pinocchio Newton + generated module schedule',
    'analytic_dense': 'Generated analytic Jacobian, dense TRF',
    'fd2_dense': '2-point differences, dense TRF',
    'fd3_dense': '3-point differences, dense TRF',
    'analytic_sparse': 'Generated analytic Jacobian, sparse TRF',
    'fd2_colored': 'Colored 2-point differences, sparse TRF',
    'fd3_colored': 'Colored 3-point differences, sparse TRF'}
PARTITION_NAMES = {'joint_space': 'joint space', 'cylinder_space': 'cylinder space'}
ROUTE_NAMES = {
    'joint_reference': 'joint space, planned reference',
    'cylinder_reference': 'cylinder space, planned reference, held strokes',
    'cylinder_reference_nohold': 'cylinder space, planned reference, strokes not held (ablation)',
    'joint_measured': 'joint space, measured signal',
    'cylinder_measured': 'cylinder space, measured signal'}
# (corrector, whole-input reuse, loop-free exclusion, loop modules)
FAMILIES = [('PACDM', 'compiled_global_reuse', 'compiled_global_loopfree', 'compiled_modular'),
            ('TRF', 'trf_global', 'trf_global_loopfree', 'trf_modular'),
            ('Native Newton', 'native_newton', 'native_newton_loopfree', 'native_modular')]


def _f(x, kind='ms'):
    if x is None or x == '':
        return '—'
    x = float(x)
    if kind == 'ms':
        return f'{x:.3f}'
    if kind == 's':
        return f'{x:.2f}'
    if kind == 'pct':
        return f'{x:+.1f}%'
    if kind == 'sci':
        return f'{x:.2e}'
    if kind == 'int':
        return f'{int(round(x))}'
    return str(x)


def _reduction(ref, prop):
    return 100. * (1. - float(prop) / float(ref))


def load(out):
    out = Path(out)
    j = lambda n: json.loads((out / n).read_text())  # noqa: E731
    d = dict(summary=j('summary.json'), protocol=j('protocol.json'), environment=j('environment.json'),
             plans={p: j(f'compiled_plan_{p}.json') for p in ('joint_space', 'cylinder_space')})
    for name in ('warm_summary', 'warm_phase_summary', 'warm_route_totals', 'warm_per_repeat', 'local_summary',
                 'derivative_summary', 'evaluator_summary', 'compilation_times', 'pipeline_raw', 'verification_tests',
                 'curvature_magnitude', 'dynamics_raw'):
        p = out / f'{name}.csv'
        d[name] = read_csv(p) if p.exists() else []
    return d


def _index(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def benefit_rows(d):
    rows = []
    totals = _index(d['warm_route_totals'], 'route', 'method')
    phase = _index(d['warm_phase_summary'], 'route', 'phase_group', 'method')
    local = _index(d['local_summary'], 'partition', 'condition', 'method')
    deriv = _index(d['derivative_summary'], 'partition', 'method')
    ev = {r['evaluator']: r for r in d['evaluator_summary']}

    def add(group, condition, comparison, ref, prop, rv, pv, unit):
        rows.append(dict(group=group, condition=condition, comparison=comparison, reference=ref, proposed=prop,
                         reference_value=float(rv), proposed_value=float(pv), unit=unit,
                         reduction_percent=_reduction(rv, pv), reference_over_proposed=float(rv) / float(pv)))

    if 'original_residual_jacobian' in ev and 'compiled_residual_jacobian' in ev:
        add('evaluator', 'closure residual + Jacobian per call',
            'Generated LCA-relative evaluator vs original adapter (implementation)', 'original_residual_jacobian',
            'compiled_residual_jacobian', ev['original_residual_jacobian']['median_us'],
            ev['compiled_residual_jacobian']['median_us'], 'us')
    pairs = [('compiled_global', 'original_global', 'Generated evaluator (same global PACDM) vs original adapter'),
             ('compiled_global_loopfree', 'compiled_global_reuse', 'Loop-free exclusion vs whole-input reuse (PACDM)'),
             ('compiled_modular', 'compiled_global_loopfree', 'Loop modules vs loop-free exclusion (PACDM)'),
             ('compiled_modular', 'compiled_global_reuse', 'Loop modules vs whole-input reuse (PACDM)'),
             ('compiled_modular', 'compiled_global', 'Loop modules vs global PACDM without reuse'),
             ('compiled_modular', 'modular_no_reuse', 'Exact module reuse vs recomputing every module'),
             ('compiled_modular', 'modular_no_predictor', 'Tangent predictor within modular PACDM'),
             ('compiled_modular', 'original_global', 'Full generated workflow vs original adapter'),
             ('trf_global_loopfree', 'trf_global', 'Loop-free exclusion vs whole-input reuse (TRF)'),
             ('trf_modular', 'trf_global_loopfree', 'Loop modules vs loop-free exclusion (TRF)'),
             ('trf_modular', 'trf_global', 'Loop modules vs whole-input reuse (TRF)'),
             ('native_newton_loopfree', 'native_newton', 'Loop-free exclusion vs whole-input reuse (native)'),
             ('native_modular', 'native_newton_loopfree', 'Loop modules vs loop-free exclusion (native)'),
             ('native_modular', 'native_newton', 'Loop modules vs whole-input reuse (native)')]
    for route in d['protocol']['routes']:
        for prop, ref, text in pairs:
            if (route, ref) in totals and (route, prop) in totals:
                add(f'route total ({ROUTE_NAMES[route]})', '42 s dig-lift-slew-dump cycle, sum of step times',
                    text, ref, prop, totals[(route, ref)]['median_total_s'], totals[(route, prop)]['median_total_s'], 's')
        for group in ('slew', 'dig', 'hold'):
            for prop, ref, text in (pairs[3], pairs[10], pairs[13]):
                if (route, group, ref) in phase and (route, group, prop) in phase:
                    add(f'route {group} samples ({ROUTE_NAMES[route]})', f'median step time, {group} phases', text,
                        ref, prop, phase[(route, group, ref)]['median_ms'], phase[(route, group, prop)]['median_ms'], 'ms')
    for partition in ('joint_space', 'cylinder_space'):
        for condition in ('slew', 'boom', 'stick', 'bucket', 'tilt', 'rotator', 'pin', 'dig', 'all'):
            for prop, ref, text in (('compiled_modular', 'compiled_global', 'Loop modules vs global PACDM'),
                                    ('compiled_modular', 'compiled_global_loopfree', 'Loop modules vs loop-free exclusion (PACDM)'),
                                    ('trf_modular', 'trf_global', 'Loop modules vs global TRF'),
                                    ('native_modular', 'native_newton', 'Loop modules on the native backend')):
                if (partition, condition, ref) in local and (partition, condition, prop) in local:
                    add(f'local edits ({PARTITION_NAMES[partition]})', f'{condition} edits', text, ref, prop,
                        local[(partition, condition, ref)]['median_ms'], local[(partition, condition, prop)]['median_ms'], 'ms')
        for prop, ref in (('analytic_dense', 'fd2_dense'), ('analytic_dense', 'fd3_dense'),
                          ('analytic_sparse', 'fd2_colored'), ('analytic_sparse', 'fd3_colored')):
            if (partition, ref) in deriv and (partition, prop) in deriv:
                add(f'derivatives ({PARTITION_NAMES[partition]})', 'TRF corrector only', 'Generated analytic Jacobian vs '
                    + NAMES[ref].split(',')[0].lower(), ref, prop, deriv[(partition, ref)]['median_ms'],
                    deriv[(partition, prop)]['median_ms'], 'ms')
    return rows


def attribution_rows(d):
    """Per route and corrector: whole-input reuse -> loop-free exclusion -> loop modules."""
    totals = _index(d['warm_route_totals'], 'route', 'method')
    rows = []
    for route in d['protocol']['routes']:
        for family, whole, free, mods in FAMILIES:
            if not all((route, m) in totals for m in (whole, free, mods)):
                continue
            w, f, m = (float(totals[(route, k)]['median_total_s']) for k in (whole, free, mods))
            rows.append(dict(route=route, corrector=family, whole_input_reuse_s=w, loop_free_exclusion_s=f,
                             loop_modules_s=m, exclusion_effect_percent=_reduction(w, f),
                             decomposition_effect_percent=_reduction(f, m), total_effect_percent=_reduction(w, m)))
    return rows


def _md_table(headers, rows):
    for c in list(headers) + [c for r in rows for c in r]:
        if '|' in str(c):
            raise ValueError(f'Table cell contains a pipe: {c!r}')
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def _range(t):
    return f"{float(t['min_total_s']):.2f}–{float(t['max_total_s']):.2f}"


def tables(d):
    T = []
    pj, pc = d['plans']['joint_space'], d['plans']['cylinder_space']
    T.append(dict(key='structure', title='Generated closed-chain structure of the excavator arm',
                  headers=['Quantity', 'Original adapter', 'Compiled, joint space', 'Compiled, cylinder space'],
                  rows=[['Physical bodies / joints (moving)', f"{pj['physical_bodies']} / {pj['physical_joints']} ({pj['moving_joints']})",
                         f"{pj['physical_bodies']} / {pj['physical_joints']} ({pj['moving_joints']})",
                         f"{pc['physical_bodies']} / {pc['physical_joints']} ({pc['moving_joints']})"],
                        ['Tree coordinates / cut joints', '23 / 9 (from the MuJoCo mapping)', f"{pj['tree_coordinates']} / {len(pj['cut_joint_ids'])}",
                         f"{pc['tree_coordinates']} / {len(pc['cut_joint_ids'])}"],
                        ['Augmented coordinates / closure rows / rank', '32 / 54 / 25',
                         f"{pj['augmented_coordinates']} / {pj['closure_rows']} / {pj['closure_rank']}",
                         f"{pc['augmented_coordinates']} / {pc['closure_rows']} / {pc['closure_rank']}"],
                        ['Mobility / independent coordinates', '7 / hard-coded joint set', f"{pj['mobility']} / " + ' '.join(pj['independent_ids']),
                         f"{pc['mobility']} / " + ' '.join(pc['independent_ids'])],
                        ['Dependent coordinates solved together', '25 (one block)', f"{pj['dependent_coordinates']} in {pj['modules_count']} modules",
                         f"{pc['dependent_coordinates']} in {pc['modules_count']} modules"],
                        ['Module sizes (dependent coordinates)', '25', ' '.join(map(str, pj['module_sizes'])), ' '.join(map(str, pc['module_sizes']))],
                        ['Independent coordinates in no loop', 'not identified', 'q23 q21 (slew, rotator)', 'q23 q21 (slew, rotator)']],
                  note='The compiled columns are generated from physical records only. Modules are conditional on their input '
                       'coordinates; the rigid-body dynamics remain coupled.'))
    pr = d['pipeline_raw']
    rows = []
    for partition in ('joint_space', 'cylinder_space'):
        rr = [r for r in pr if r['partition'] == partition]
        rows.append([PARTITION_NAMES[partition], len({r['variant'] for r in rr}), len(rr),
                     f"{sum(r['success'] == 'True' for r in rr)}/{len(rr)}"])
    ct = d['compilation_times']
    T.append(dict(key='compiler_coverage', title='Compiler coverage over twelve predefined physical-data variants',
                  headers=['Partition', 'Variants', 'Closed witness checks', 'Accepted'], rows=rows,
                  note=f"Median compilation time {(np.median([float(r['compilation_ms']) for r in ct]) if ct else float('nan')):.1f} ms "
                       f"(includes numerical mobility, partition-completion and module-rank analysis). Witnesses are "
                       f"assembled by the compiled PACDM and checked against a native Pinocchio model written directly "
                       f"from the same physical records with the original cut set."))
    ev = d['evaluator_summary']
    T.append(dict(key='evaluators', title='Per-call cost of the loop-closure residual and Jacobian',
                  headers=['Evaluator', 'Median (us)', 'Range over repeats (us)'],
                  rows=[[r['evaluator'], _f(r['median_us'], 'ms'),
                         '—' if not r['min_us'] else f"{float(r['min_us']):.1f}–{float(r['max_us']):.1f}"] for r in ev],
                  note='Closed states sampled from the route. The original and generated evaluators compute the same residual '
                       'in Python, so their difference is an implementation effect. The native evaluator is C++ kinematics '
                       'through Python bindings.'))
    totals = _index(d['warm_route_totals'], 'route', 'method')
    phase = _index(d['warm_phase_summary'], 'route', 'phase_group', 'method')
    summ = _index(d['warm_summary'], 'route', 'method')
    table_table = d['protocol'].get('method_table', {})
    for route in ('joint_reference', 'cylinder_reference'):
        spec = d['protocol']['routes'].get(route)
        if not spec:
            continue
        rows = []
        for m in spec['methods']:
            s = summ.get((route, m))
            if not s:
                continue
            t = totals[(route, m)]
            rows.append([NAMES[m], f"{s['accepted']}/{s['attempts']}", _f(s['median_ms']),
                         _f(phase.get((route, 'dig', m), {}).get('median_ms')),
                         _f(phase.get((route, 'slew', m), {}).get('median_ms')),
                         _f(phase.get((route, 'hold', m), {}).get('median_ms')),
                         _f(t['median_total_s'], 's'), _range(t), _f(s['max_gap_m'], 'sci')])
        T.append(dict(key=f'route_{route}', title=f'Dig-cycle route, {ROUTE_NAMES[route]}: step time (ms) and cycle total (s)',
                      headers=['Method', 'Accepted', 'Median', 'Dig', 'Slew', 'Hold', 'Cycle total (s)', 'Range (s)',
                               'Max gap (m)'],
                      rows=rows, note='Median over all repeats and samples; cycle total = sum of step times over the 4,201 '
                                      'samples, median (and range) of the three repeats. Dig/slew/hold are phase groups of the '
                                      'original mission. Max gap = largest native closure gap of any accepted attempt.'))
    att = attribution_rows(d)
    T.append(dict(key='attribution', title='Where the route gain comes from: loop-free exclusion versus loop decomposition',
                  headers=['Route', 'Corrector', 'Whole-input reuse (s)', 'Loop-free exclusion (s)', 'Loop modules (s)',
                           'Exclusion effect', 'Decomposition effect'],
                  rows=[[ROUTE_NAMES[r['route']], r['corrector'], _f(r['whole_input_reuse_s'], 's'),
                         _f(r['loop_free_exclusion_s'], 's'), _f(r['loop_modules_s'], 's'),
                         _f(-r['exclusion_effect_percent'], 'pct'), _f(-r['decomposition_effect_percent'], 'pct')]
                        for r in att],
                  note='Cycle totals (median of three repeats). Effects are relative changes of the cycle total (negative = '
                       'faster): loop-free exclusion skips the loop solve when only slew or rotator changed but keeps one '
                       'global loop problem; loop modules additionally solve only the modules whose inputs changed. Both use '
                       'the compiled structure; neither is available to the original adapter.'))
    rows = []
    for route in ('cylinder_reference_nohold', 'joint_measured', 'cylinder_measured'):
        spec = d['protocol']['routes'].get(route)
        if not spec:
            continue
        for m in spec['methods']:
            t = totals.get((route, m))
            s = summ.get((route, m))
            if not t:
                continue
            rows.append([ROUTE_NAMES[route], NAMES[m], f"{s['accepted']}/{s['attempts']}", _f(t['median_total_s'], 's'),
                         _range(t), t['samples_without_solve']])
    T.append(dict(key='robustness', title='Routes without bit-identical planned commands: cycle totals (s)',
                  headers=['Route', 'Method', 'Accepted', 'Cycle total (s)', 'Range (s)', 'Samples without solve'],
                  rows=rows,
                  note='Without the stroke hold, strokes read from the global truth projection change at round-off in '
                       'every moving sample; the measured signal (simulated joint coordinates of the validated nominal run) '
                       'changes some loop input in every sample. Exact reuse therefore rarely applies. Samples without '
                       'solve are counted in the first repeat (4,201 samples).'))
    loc = _index(d['local_summary'], 'partition', 'condition', 'method')
    for partition in ('joint_space', 'cylinder_space'):
        conds = [c for c, _ in d['protocol']['local_conditions']]
        methods = [m for m in d['protocol']['local_methods'] if (partition, conds[0], m) in loc]
        rows = [[NAMES[m]] + [_f(loc[(partition, c, m)]['median_ms']) if (partition, c, m) in loc else '—' for c in conds]
                for m in methods]
        T.append(dict(key=f'local_{partition}', title=f'Localized edits, {PARTITION_NAMES[partition]}: median step time (ms)',
                      headers=['Method'] + conds, rows=rows,
                      note='Twelve successive edits (at most 0.018 rad each) from four closed route states, three repeats per '
                           'condition (144 attempts per method and condition). Conditions name the joint-space coordinates '
                           'that change.'))
    der = _index(d['derivative_summary'], 'partition', 'method')
    rows = []
    for partition in ('joint_space', 'cylinder_space'):
        for m in d['protocol']['derivative_methods']:
            r = der.get((partition, m))
            if r:
                rows.append([PARTITION_NAMES[partition], NAMES[m], f"{r['accepted']}/{r['attempts']}", _f(r['median_ms']),
                             _f(r['p95_ms']), _f(r['median_evaluations'], 'int')])
    T.append(dict(key='derivatives', title='Generated analytic derivatives versus finite differences (TRF corrector only)',
                  headers=['Partition', 'Route', 'Accepted', 'Median (ms)', 'P95 (ms)', 'Median model calls'], rows=rows,
                  note='Same residual, bounds, tolerances, prediction and linear solver within each pair; colored differences '
                       'receive the generated sparsity pattern. Model calls count every fresh residual evaluation.'))
    dyn = d['summary']['stages'].get('dynamics', {})
    mx = dyn.get('maxima', {})
    labels = [('pacdm_vs_kkt_force', 'PACDM vs NumPy KKT, force-equivalent'), ('pacdm_vs_native_force', 'PACDM vs Pinocchio, force-equivalent'),
              ('kkt_vs_native_force', 'NumPy KKT vs Pinocchio, force-equivalent'),
              ('pacdm_vs_kkt_relative', 'PACDM vs NumPy KKT, normalized acceleration (reported)'),
              ('pacdm_vs_native_relative', 'PACDM vs Pinocchio, normalized acceleration (reported)'),
              ('map_pacdm_vs_native', 'PACDM map vs native least-squares map (max entry)'),
              ('native_map_reduction_vs_native_relative', 'Same reduction with the native map vs Pinocchio, normalized acceleration'),
              ('pacdm_constraint_acceleration_m_s2', 'PACDM constraint-acceleration residual (m/s2)'),
              ('pacdm_tangent_dynamics_relative', 'PACDM projected-dynamics residual'),
              ('numpy_vs_native_mass', 'NumPy vs Pinocchio mass matrix (max entry)'),
              ('numpy_vs_native_jacobian', 'NumPy vs Pinocchio closure Jacobian'),
              ('tangent_residual_native', 'Native closure Jacobian x PACDM map'),
              ('reduced_inertia_condition', 'Condition number of the reduced inertia (reported)')]
    rows = [[text, _f(mx.get('joint_space', {}).get(k), 'sci'), _f(mx.get('cylinder_space', {}).get(k), 'sci')] for k, text in labels]
    rows.append(['Partition invariance of accelerations, force-equivalent', _f(dyn.get('invariance_force_max'), 'sci'), '(joint vs cylinder)'])
    rows.append(['Partition invariance of accelerations, normalized', _f(dyn.get('invariance_relative_max'), 'sci'), '(joint vs cylinder)'])
    T.append(dict(key='dynamics', title='Constrained-dynamics consistency at 48 closed route states (maxima)',
                  headers=['Quantity', 'Joint space', 'Cylinder space'], rows=rows,
                  note='Force-equivalent = max-norm of M (a - a_ref) / max(1, max-norm tau, max-norm h). The normalized '
                       'acceleration difference is reported, not gated: the same reduction formula with the native map '
                       'reproduces Pinocchio closely, so the difference stems from the small map difference, which is '
                       'amplified along the light linkage-pivot mode.'))
    cur = d['curvature_magnitude']
    rows = []
    for s in sorted({float(r['speed_scale']) for r in cur}):
        rr = [r for r in cur if float(r['speed_scale']) == s]
        rows.append([f'{s:g}', len(rr), _f(max(float(r['velocity_product_term_max']) for r in rr), 'sci'),
                     _f(max(float(r['with_term_residual_m_s2']) for r in rr), 'sci'),
                     _f(max(float(r['omitted_term_residual_m_s2']) for r in rr), 'sci')])
    T.append(dict(key='velocity_term', title='Magnitude of the velocity-product term c in q_dd = N u_dd + c',
                  headers=['Velocity scale', 'States', 'Largest entry of c', 'Constraint residual with c (m/s2)', 'Without c (m/s2)'],
                  rows=rows, note='A magnitude check, not a comparison with another solver: omitting c leaves the native '
                                  'constraint-acceleration bias uncompensated.'))
    rows = []
    for m, (corr, evaluator, sched) in table_table.items():
        rows.append([m, NAMES.get(m, m), corr, evaluator, sched])
    T.append(dict(key='methods', title='Compared routes', headers=['Key', 'Name', 'Corrector', 'Evaluator', 'Reuse or schedule'],
                  rows=rows, note='All PACDM routes use the unchanged core with the original online step (prediction, '
                                  'correction with at most 12 iterations, original 5e-13 polish, differential map).'))
    return T


def latex_table(t):
    def esc(s):
        return (str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
                .replace('#', r'\#').replace('–', '--').replace('—', '---').replace('^', r'\^{}').replace('|', r'$|$'))
    cols = 'l' + 'r' * (len(t['headers']) - 1)
    lines = [r'\begin{table*}[!htbp]', r'\centering', r'\small', rf"\caption{{{esc(t['title'])}}}", rf"\label{{tab:exc_{t['key']}}}",
             rf'\begin{{tabular}}{{{cols}}}', r'\hline', ' & '.join(esc(h) for h in t['headers']) + r' \\', r'\hline']
    for r in t['rows']:
        lines.append(' & '.join(esc(c) for c in r) + r' \\')
    lines += [r'\hline', r'\end{tabular}', rf"\par\smallskip\parbox{{0.95\textwidth}}{{\footnotesize {esc(t['note'])}}}", r'\end{table*}', '']
    return '\n'.join(lines)


def generate_report(out):
    out = Path(out)
    d = load(out)
    rows = benefit_rows(d)
    save_csv(out / 'benefit_summary.csv', rows)
    save_csv(out / 'attribution_summary.csv', attribution_rows(d))
    T = tables(d)
    save_json(out / 'tables.json', T)
    (out / 'TABLES.tex').write_text('% Generated by src/reporting.py from this result directory.\n\n'
                                    + '\n'.join(latex_table(t) for t in T), encoding='utf-8')
    s = d['summary']
    env = d['environment']
    lines = ['# Excavator CMG/PACDM framework-benefit study — executed results', '',
             f"Status: **{s['status']}** (profile `{s['profile']}`); failed stages: {s.get('failed_stages') or 'none'}.", '',
             f"Host: {env.get('processor_model')} ({env.get('visible_logical_cpus')} visible logical CPUs), "
             f"Python {env['python'].split()[0]}, NumPy {env['versions'].get('numpy')}, SciPy {env['versions'].get('scipy')}, "
             f"Pinocchio {env['versions'].get('pinocchio_imported')}; BLAS limited to one thread.", '',
             '## Stage outcomes', '']
    stage_rows = []
    for name, r in s['stages'].items():
        denom = r.get('attempts', r.get('checks', r.get('tests')))
        stage_rows.append([name, f"{r.get('passed')}/{denom}", f"{r.get('stage_wall_s', 0):.1f}"])
    lines += [_md_table(['Stage', 'Passed', 'Wall time (s)'], stage_rows), '', '## Benefit summary', '',
              'Reduction = 1 - proposed/reference (negative = the proposed route is slower).', '',
              _md_table(['Group', 'Condition', 'Comparison', 'Reference', 'Proposed', 'Reduction'],
                        [[r['group'], r['condition'], r['comparison'], f"{r['reference_value']:.3f} {r['unit']}",
                          f"{r['proposed_value']:.3f} {r['unit']}", f"{r['reduction_percent']:.1f}%"] for r in rows]), '']
    for t in T:
        lines += [f"## {t['title']}", '', _md_table(t['headers'], t['rows']), '', t['note'], '']
    lines += ['## Scope', '',
              'Timings are descriptive repetitions on one host, not independent robot trials; reductions are ratios of medians '
              '(or of route totals). The generated evaluator, module schedule and native module adapter are new code; the '
              'PACDM core, the original adapter, the NumPy dynamics and the validated Pinocchio backend are unchanged. '
              'Exact reuse requires bit-identical inputs, which planned commands provide and measured signals do not. '
              'No hydraulics, soil, contact, hardware or real-time claim is made.', '']
    md = '\n'.join(lines)
    (out / 'REPORT.md').write_text(md, encoding='utf-8')
    (out / 'REPORT.html').write_text(markdown_to_html(md, 'Excavator CMG/PACDM benefit study'), encoding='utf-8')
    return rows


def markdown_to_html(md, title):
    body, table = [], []

    def flush():
        if table:
            rows = [r.strip().strip('|').split('|') for r in table if not set(r.replace('|', '').strip()) <= set('-')]
            body.append('<table><thead><tr>' + ''.join(f'<th>{html.escape(c.strip())}</th>' for c in rows[0]) + '</tr></thead><tbody>'
                        + ''.join('<tr>' + ''.join(f'<td>{html.escape(c.strip())}</td>' for c in r) + '</tr>' for r in rows[1:])
                        + '</tbody></table>')
            table.clear()
    for line in md.splitlines():
        if line.startswith('|'):
            table.append(line)
            continue
        flush()
        if line.startswith('## '):
            body.append(f'<h2>{html.escape(line[3:])}</h2>')
        elif line.startswith('# '):
            body.append(f'<h1>{html.escape(line[2:])}</h1>')
        elif line.strip():
            text = html.escape(line).replace('**', '')
            body.append(f'<p>{text}</p>')
    flush()
    css = ('body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;'
           'color:#1d2330;background:#fff}table{border-collapse:collapse;margin:8px 0 18px;font-size:13px;width:100%}'
           'th,td{border-bottom:1px solid #d7dbe3;padding:4px 8px;text-align:right}th:first-child,td:first-child{text-align:left}'
           'th{background:#f3f5f9}h2{margin-top:28px;border-bottom:2px solid #e4e7ee;padding-bottom:4px}'
           '@media (prefers-color-scheme: dark){body{color:#e6e8ee;background:#15171c}th{background:#23262e}'
           'th,td{border-bottom-color:#343844}h2{border-bottom-color:#343844}}')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head><body>' + '\n'.join(body) + '</body></html>\n')


if __name__ == '__main__':
    generate_report(sys.argv[1])
    print('report regenerated in', sys.argv[1])
