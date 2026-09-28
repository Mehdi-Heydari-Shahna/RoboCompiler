"""Unit and regression tests for the Kangaroo framework-benefit benchmark.

Run from ``benchmark/``:  python -m unittest discover -s tests -v
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import unittest  # noqa: E402
from copy import deepcopy

import numpy as np

from src import bootstrap  # noqa: F401
from src.bootstrap import ROOT
from src.compiler import (compile_graph, variant_inputs, support_sets, components, ModelError, FORBIDDEN)
from src.evaluator import (GeneratedLoopGraph, GeneratedSupportGraph, ModularSolver, CutSetEvaluator,
                           SuppliedLayoutGraph, METHODS, create_solver)
from src.physics import NumpyReference, dynamics_witness, normmax
from src.runner import Layout, Acceptance, supplied_cmg, supplied_reference, accepted, chart_base_state, MODES
from pacdm import PACDM


class KangarooTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc = accepted()
        cls.source = json.loads((ROOT / 'inputs/physical_graph.json').read_text())
        cls.comp = compile_graph(cls.source)
        cls.ref = supplied_reference()
        cls.layout = Layout(cls.comp, cls.acc)
        cls.X = np.array([cls.layout(x) for x in cls.ref['qaug']])
        cls.motors = np.array(cls.comp.plan['motor_indices'])

    def test_original_sources_unchanged(self):
        manifest = json.loads((ROOT / 'original/original_v22/MANIFEST_SHA256.json').read_text())['files']
        for path in sorted((ROOT / 'original/original_v22').rglob('*')):
            if path.is_file() and path.name != 'MANIFEST_SHA256.json':
                name = path.relative_to(ROOT / 'original/original_v22').as_posix()
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest[name], name)
        origin = json.loads((ROOT / 'original/ORIGIN_VERIFICATION.json').read_text())
        for name, digest in origin['kangaroo_pin_files'].items():
            self.assertEqual(hashlib.sha256((ROOT / 'original' / name).read_bytes()).hexdigest(), digest, name)

    def test_plan_structure(self):
        p = self.comp.plan
        self.assertEqual((p['physical_bodies'], p['physical_coordinates'], p['loop_cuts'], p['point_cuts'],
                          p['universal_cuts']), (78, 76, 24, 16, 8))
        self.assertEqual((p['chart_coordinates'], p['augmented_coordinates'], p['augmented_residual_rows'],
                          p['augmented_rank_expected']), (64, 140, 144, 128))
        self.assertEqual((p['physical_closure_rows'], p['physical_closure_rank_expected'],
                          p['redundant_physical_rows']), (80, 64, 16))
        self.assertEqual(sorted(p['module_sizes']), [5, 5, 12, 12, 47, 47])
        self.assertEqual(p['largest_module_block'], 47)
        self.assertEqual(len(p['motor_indices']), 12)

    def test_input_is_not_preauthored(self):
        self.assertFalse(FORBIDDEN & set(self.source))
        for key in ('modules', 'coordinate_ids', 'chart_columns'):
            s = deepcopy(self.source)
            s[key] = []
            with self.assertRaises(ModelError):
                compile_graph(s)

    def test_generated_equals_accepted_cutgraph(self):
        g0 = self.acc.CutGraph(self.comp.cmg, np.asarray(self.comp.cmg['initial_seed']))
        g1 = GeneratedLoopGraph(self.comp)
        rng = np.random.default_rng(7)
        for k in (0, 90, 150, 230, 400):
            for scale in (0., .05):
                x = self.X[k] + rng.normal(0., scale, g1.n)
                D = None if scale == 0. else np.array([self.acc.pacdm.exp(rng.normal(0, .02, 6)) for _ in range(24)])
                a, b = g0.residual(x, D), g1.residual(x, D)
                for u, v in zip(a, b):
                    self.assertLess(normmax(u - v), 1e-12)

    def test_module_graphs_equal_global_rows(self):
        g = GeneratedLoopGraph(self.comp)
        x = self.X[160]
        r, J, _ = g.residual(x)
        for m in self.comp.plan['modules']:
            gm = GeneratedLoopGraph(self.comp, m['cuts'], m)
            rm, Jm, _ = gm.residual(x[gm.columns])
            rows = np.array(m['rows'])
            self.assertLess(normmax(rm - r[rows]), 1e-14)
            self.assertLess(normmax(Jm - J[np.ix_(rows, gm.columns)]), 1e-14)
            others = np.setdiff1d(np.arange(g.n), gm.columns)
            self.assertEqual(normmax(J[np.ix_(rows, others)]), 0.)

    def test_generated_derivative_off_manifold(self):
        g = GeneratedLoopGraph(self.comp)
        x = self.X[120] + np.random.default_rng(3).normal(0., .03, g.n)
        _, J, _ = g.residual(x)
        fd = np.empty_like(J)
        h = 1e-6
        for i in range(g.n):
            d = np.zeros(g.n)
            d[i] = h
            fd[:, i] = (g.residual_only(x + d) - g.residual_only(x - d)) / (2 * h)
        self.assertLess(normmax(fd - J), 1e-6)

    def test_residual_only_does_not_form_jacobian(self):
        g = GeneratedLoopGraph(self.comp)
        calls = []
        original = CutSetEvaluator._twists

        def spy(*a, **k):
            calls.append(1)
            return original(*a, **k)

        CutSetEvaluator._twists = staticmethod(spy)
        try:
            g.residual_only(self.X[100])
            self.assertEqual(calls, [])
            g.residual(self.X[100])
            self.assertEqual(len(calls), 2)
        finally:
            CutSetEvaluator._twists = staticmethod(original)

    def test_structural_sparsity(self):
        g = GeneratedLoopGraph(self.comp)
        x = self.X[100] + .01
        J = g.residual(x)[1][:, g.passive]
        pattern = np.array(self.comp.plan['passive_sparsity'])
        self.assertEqual(int(pattern.sum()), self.comp.plan['sparsity_nonzeros'])
        self.assertEqual(normmax(J[pattern == 0]), 0.)

    def test_all_solver_interfaces(self):
        check = Acceptance(self.comp)
        target = self.X[101][self.motors]
        for name in METHODS:
            q, N, _ = create_solver(name, self.comp, self.X[100], self.acc).step(target)
            r = check(q, N, target, self.X[101][:76])
            self.assertTrue(r['success'], (name, r))

    def test_exact_reuse_and_invalidation(self):
        s = ModularSolver(self.comp, self.X[150])
        a = self.X[150][self.motors].copy()
        self.assertEqual(s.step(a)[2]['skipped_modules'], 6)
        a[list(self.motors).index(self.comp.plan['modules'][2]['motors'][0])] += 2e-4
        self.assertEqual(s.step(a)[2]['solved_modules'], 1)
        self.assertEqual(s.step(self.X[152][self.motors])[2]['solved_modules'], 6)

    def test_support_partitions_and_rank(self):
        x = chart_base_state(self.ref, 150, self.X[150])
        chart = NumpyReference(self.comp.chart_cmg())
        from src.physics import sole_frame_records
        for sites in MODES:
            frames = sole_frame_records(self.comp, sites)
            anchors = chart.weld_geometry(x[:82], None, frames)['poses']
            g = GeneratedSupportGraph(self.comp, sites, anchors)
            N, info = PACDM(g).mapping(x)
            self.assertTrue(info['success'], sites)
            self.assertEqual(info['rank_full'], 128 + 6 * len(sites))
            self.assertEqual(N.shape, (146, 18 - 6 * len(sites)))

    def test_dynamics_witness_all_modes_numpy(self):
        x = chart_base_state(self.ref, 200, self.X[200])
        rng = np.random.default_rng(11)
        for sites in MODES:
            plan = self.comp.support_plan(sites)
            r, _ = dynamics_witness(self.comp, x, sites, rng.uniform(-.1, .1, len(plan['active'])),
                                    rng.uniform(-150, 150, 12), rng.uniform(-5, 5, 6))
            self.assertTrue(r['success'], (sites, r))
            self.assertEqual(r['base_motor_effort_max'], 0.)

    def test_twelve_variants_compile_with_same_structure(self):
        variants = variant_inputs(self.source)
        self.assertEqual(len(variants), 12)
        for name, s in variants:
            c = compile_graph(s)
            self.assertEqual(sorted(c.plan['module_sizes']), [5, 5, 12, 12, 47, 47], name)
            self.assertEqual(c.plan['augmented_rank_expected'], 128, name)
            self.assertEqual(len(support_sets(c)), 3, name)

    def test_components_use_shared_dependent_coordinates(self):
        self.assertEqual(components([[1, 2], [3], [2, 4], [5, 3]]), [[0, 2], [1, 3]])

    def test_supplied_layout_adapter_equals_accepted(self):
        cmg0 = supplied_cmg()
        g0 = self.acc.CutGraph(cmg0, np.asarray(cmg0['initial_seed']))
        ga = SuppliedLayoutGraph(self.comp, self.acc, cmg0)
        x = self.ref['qaug'][180]
        for u, v in zip(g0.residual(x), ga.residual(x)):
            self.assertLess(normmax(u - v), 1e-12)
        self.assertEqual(ga.residual(x)[0].tobytes(), ga.residual(x)[0].tobytes())
        self.assertEqual(ga.cache_hits, 2)


if __name__ == '__main__':
    unittest.main()
