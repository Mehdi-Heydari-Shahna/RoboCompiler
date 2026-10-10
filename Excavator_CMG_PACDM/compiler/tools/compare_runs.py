#!/usr/bin/env python
"""Compare two result directories of run_excavator.py (e.g. shipped vs reproduced).

    python tools/compare_runs.py results_here ../results_repro [--out comparison.json]

Deterministic content (compiled structures, generated variants, every non-timing
column of every raw record, stored states, dynamics) must be identical on the same
platform; it is compared value by value. Timing is machine- and load-dependent and is
reported as a table of route totals and reduction percentages side by side, not gated.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import read_csv, sha  # noqa: E402

TIMING = {'time_ms', 'compilation_ms', 'us_per_call', 'stage_wall_s', 'total_wall_s', 'median_ms', 'p95_ms',
          'median_total_s', 'min_total_s', 'max_total_s', 'median_us', 'min_us', 'max_us', 'min_ms', 'max_ms'}


def rows(d, name):
    for candidate in (name, name + '.gz'):
        if (d / candidate).exists():
            return read_csv(d / candidate)
    return None


def compare(a, b):
    a, b = Path(a), Path(b)
    out = dict(identical_files=[], differing_files=[], raw_records={}, timing={})
    for name in ['compiled_cmg_joint_space.json', 'compiled_cmg_cylinder_space.json', 'compiled_plan_joint_space.json',
                 'compiled_plan_cylinder_space.json', 'route_inputs.npz', 'warm_states.npz', 'dynamics_states.json',
                 'derivative_inputs.json', 'local_inputs.json']:
        if (a / name).exists() and (b / name).exists():
            (out['identical_files'] if sha(a / name) == sha(b / name) else out['differing_files']).append(name)
    for f in sorted((a / 'generated').rglob('*.json')):
        rel = f.relative_to(a)
        (out['identical_files'] if (b / rel).exists() and sha(f) == sha(b / rel) else out['differing_files']).append(str(rel))
    warm_files = sorted(str(f.relative_to(a)).replace('\\', '/')[:-3] for f in (a / 'warm_raw').glob('*.csv.gz'))
    for name in ['pipeline_raw.csv', 'verification_tests.csv', 'dynamics_raw.csv', 'curvature_magnitude.csv',
                 'local_raw.csv', 'derivative_raw.csv'] + warm_files:
        ra, rb = rows(a, name), rows(b, name)
        if ra is None or rb is None:
            continue
        diff = 0
        fields = [k for k in ra[0] if k not in TIMING]
        if len(ra) != len(rb):
            out['raw_records'][name] = dict(records=[len(ra), len(rb)], differing_values=None)
            continue
        # rows are written in a deterministic order except the per-sample method permutation,
        # which is seeded: compare after sorting on the identifying fields
        key = lambda r: tuple(r.get(k, '') for k in ('route', 'partition', 'condition', 'method', 'variant', 'case',  # noqa: E731
                                                     'block', 'step', 'repeat', 'sample', 'witness', 'test',
                                                     'speed_scale'))
        for x, y in zip(sorted(ra, key=key), sorted(rb, key=key)):
            diff += sum(x[k] != y.get(k) for k in fields)
        out['raw_records'][name] = dict(records=len(ra), compared_fields=len(fields), differing_values=diff)
    ta = {(r['route'], r['method']): float(r['median_total_s']) for r in read_csv(a / 'warm_route_totals.csv')}
    tb = {(r['route'], r['method']): float(r['median_total_s']) for r in read_csv(b / 'warm_route_totals.csv')}
    out['timing'] = {f'{p}/{m}': dict(a_total_s=ta[(p, m)], b_total_s=tb.get((p, m))) for p, m in ta}
    aa = {(r['route'], r['corrector']): r for r in read_csv(a / 'attribution_summary.csv')}
    ab = {(r['route'], r['corrector']): r for r in read_csv(b / 'attribution_summary.csv')}
    out['attribution_percent'] = [dict(route=k[0], corrector=k[1],
                                       a_exclusion=float(v['exclusion_effect_percent']),
                                       b_exclusion=float(ab[k]['exclusion_effect_percent']) if k in ab else None,
                                       a_decomposition=float(v['decomposition_effect_percent']),
                                       b_decomposition=float(ab[k]['decomposition_effect_percent']) if k in ab else None)
                                  for k, v in aa.items()]
    ba = {(r['group'], r['condition'], r['comparison']): float(r['reduction_percent']) for r in read_csv(a / 'benefit_summary.csv')}
    bb = {(r['group'], r['condition'], r['comparison']): float(r['reduction_percent']) for r in read_csv(b / 'benefit_summary.csv')}
    out['benefit_reduction_percent'] = [dict(group=k[0], condition=k[1], comparison=k[2], a=v, b=bb.get(k)) for k, v in ba.items()]
    out['summary'] = dict(identical_files=len(out['identical_files']), differing_files=len(out['differing_files']),
                          differing_raw_values=sum(v['differing_values'] or 0 for v in out['raw_records'].values()),
                          record_count_mismatch=[k for k, v in out['raw_records'].items() if v['differing_values'] is None])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    result = compare(args.a, args.b)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(json.dumps(result['summary'], indent=2))
    ok = not result['differing_files'] and result['summary']['differing_raw_values'] == 0 \
        and not result['summary']['record_count_mismatch']
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(main())
