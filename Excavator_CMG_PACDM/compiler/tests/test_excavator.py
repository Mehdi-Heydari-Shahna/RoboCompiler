"""Functional tests (python -m unittest discover -s tests -v). About 15 seconds."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import bootstrap  # noqa: E402,F401
from src.compiler import compile_graph, ModelError  # noqa: E402
from src.context import Context  # noqa: E402
from src.loop_graph import GlobalLoopGraph  # noqa: E402
from src.solvers import create_solver, polish, SOLVERS, command_drivers, loop_input_positions  # noqa: E402
from src.runner import route_truth, cylinder_commands, up4, native_cmg_from_physical  # noqa: E402
from plant import ExcavatorPinModel  # noqa: E402
from src.variants import variant_inputs  # noqa: E402
from excavator_pacdm import compile_pacdm  # noqa: E402


class ExcavatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = Context()
        cls.physical = cls.ctx.physical

    def test_generated_evaluator_equals_original_adapter(self):
        ctx = self.ctx
        go, _ = compile_pacdm(ctx.cmg, ctx.mapping, ctx.mapping['tree_joint_ids'])
        comp = compile_graph(self.physical, 'joint_space', cut_override=ctx.mapping['cut_joint_ids'])
        g = GlobalLoopGraph(comp)
        perm = [go.ids.index(k) for k in g.ids]
        rng = np.random.default_rng(3)
        for trial in range(10):
            qo = go.seed + (rng.normal(0, .05, go.seed.size) if trial else 0.)
            ro, Jo, _ = go.residual(qo)
            rc, Jc, _ = g.residual(qo[perm])
            for i, cid in enumerate(comp.plan['cut_joint_ids']):
                j = ctx.mapping['cut_joint_ids'].index(cid)
                self.assertLess(np.max(np.abs(rc[6 * i:6 * i + 6] - ro[6 * j:6 * j + 6])), 1e-12)
                self.assertLess(np.max(np.abs(Jc[6 * i:6 * i + 6] - Jo[6 * j:6 * j + 6][:, perm])), 1e-12)

    def test_polish_is_the_original_polish(self):
        stub = types.ModuleType('mujoco')
        stub.mjtJoint = types.SimpleNamespace(mjJNT_FREE=0)
        previous = sys.modules.get('mujoco')
        sys.modules['mujoco'] = stub
        sys.path.insert(0, str(ROOT / 'original/v26'))
        try:
            import tracked_arm
        finally:
            if previous is None:
                sys.modules.pop('mujoco', None)
            else:
                sys.modules['mujoco'] = previous
        g, pacdm = compile_pacdm(self.ctx.cmg, self.ctx.mapping, self.ctx.mapping['tree_joint_ids'])
        N, info = pacdm.mapping(g.seed)
        rows = np.array(info['rows'])
        rng = np.random.default_rng(5)
        for _ in range(5):
            q = g.seed.copy()
            q[g.passive] += rng.normal(0, 1e-6, len(g.passive))
            original = tracked_arm.TrackedArmController._polish(types.SimpleNamespace(graph=g), q, rows)
            self.assertTrue(np.array_equal(original, polish(g, q, rows)))

    def test_generated_structure(self):
        j, c = self.ctx.compiled['joint_space'], self.ctx.compiled['cylinder_space']
        self.assertEqual(sorted(j.plan['module_sizes']), [3, 3, 3, 3, 3, 10])
        self.assertEqual(sorted(c.plan['module_sizes']), [3, 6, 6, 10])
        for comp in (j, c):
            self.assertEqual(comp.plan['mobility'], 7)
            used = {k for m in comp.plan['modules'] for k in m['dependent_ids'] + m['input_ids']}
            self.assertNotIn('q23', used)
            self.assertNotIn('q21', used)

    def test_rejects_preauthored_structure_and_redundancy(self):
        bad = dict(self.physical, cut_joint_ids=['q5'])
        with self.assertRaises(ModelError):
            compile_graph(bad, 'joint_space')
        with self.assertRaises(ModelError):
            compile_graph(self.physical, 'joint_space', requested=['q23', 'p3', 'p4', 'p5', 'p2', 'q21', 'q22'])

    def test_solvers_agree_along_route(self):
        ctx = self.ctx
        truth = route_truth(ctx, ctx.joint_route()[:600])
        route = ctx.joint_route()[:600]
        q0 = ctx.full_state(truth[0])
        names = list(SOLVERS)
        solvers = {n: create_solver(n, ctx, q0, 'joint_space') for n in names}
        for i in range(0, 600, 12):
            out = {n: s.step(route[i]) for n, s in solvers.items()}
            for n in names:
                self.assertLess(np.max(np.abs(out[n][0] - truth[i])), 1e-9, n)
                self.assertLess(np.max(np.abs(out[n][1] - out['native_newton'][1])), 1e-8, n)

    def test_cylinder_commands_hold_at_round_off(self):
        truth = route_truth(self.ctx, self.ctx.joint_route())
        _, deviation = cylinder_commands(self.ctx, self.ctx.joint_route(), truth)
        self.assertLess(deviation, 1e-12)

    def test_loop_free_reuse_is_bit_exact(self):
        ctx = self.ctx
        q32 = ctx.full_state(ctx.route['tree'][0])
        u = np.array([q32[k] for k in ctx.independent_ids('joint_space')])
        self.assertEqual(list(loop_input_positions(ctx.compiled['joint_space'])), [1, 2, 3, 4, 6])
        s = create_solver('compiled_global_loopfree', ctx, q32, 'joint_space')
        x0 = s.x.copy()
        u[0] += .4
        u[5] -= .3
        q, N, info = s.step(u)
        self.assertEqual(info['solved_modules'], 0)
        self.assertTrue(np.array_equal(s.g.residual(x0)[0], s.g.residual(s.x)[0]))
        self.assertTrue(np.array_equal(s.solver.mapping(s.x)[0][s.out.take], N))
        self.assertTrue(ctx.reference.check(q, N, u, 'joint_space')['success'])
        native = create_solver('native_newton_loopfree', ctx, q32, 'joint_space')
        qn, Nn, info = native.step(u)
        self.assertEqual(info['solved_modules'], 0)
        self.assertTrue(ctx.reference.check(qn, Nn, u, 'joint_space')['success'])

    def test_native_whole_input_reuse_skips_the_plant(self):
        ctx = self.ctx
        q32 = ctx.full_state(ctx.route['tree'][0])
        s = create_solver('native_newton', ctx, q32, 'joint_space')
        u = np.array([q32[k] for k in ctx.independent_ids('joint_space')])
        u[1] += .01
        first = s.step(u)
        count = s.plant.stats['projections']
        second = s.step(u.copy())
        self.assertEqual(s.plant.stats['projections'], count)
        self.assertTrue(np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1]))

    def test_failed_step_commits_nothing(self):
        ctx = self.ctx
        q32 = ctx.full_state(ctx.route['tree'][0])
        for name in ('compiled_modular', 'native_modular'):
            s = create_solver(name, ctx, q32, 'cylinder_space')
            before = (s.q.copy(), s.N.copy())
            u = np.array([q32[k] for k in ctx.independent_ids('cylinder_space')])
            u[4] += .005
            u[1] += 1.
            with self.assertRaises(RuntimeError):
                s.step(u)
            self.assertTrue(np.array_equal(before[0], s.q) and np.array_equal(before[1], s.N), name)

    def test_command_drivers_are_derived_from_modules(self):
        self.assertEqual(command_drivers(self.ctx), {'p3': ('q7',), 'p5': ('q4',), 'p2': ('q0', 'q22'), 'p1': ('q1',)})

    def test_error_metrics_are_rounded_up(self):
        rng = np.random.default_rng(11)
        for x in rng.uniform(0, 1, 2000) * 10.0 ** rng.integers(-17, -5, 2000):
            v = up4(x)
            self.assertGreaterEqual(v, x)
            self.assertLessEqual(v, x * 1.001)

    def test_reference_model_written_from_physical_records(self):
        ctx = self.ctx
        ref = ExcavatorPinModel(native_cmg_from_physical(self.physical), ctx.mapping['cut_joint_ids'])
        self.assertEqual(list(ref.tree_ids), ctx.ref_ids)
        self.assertLess(np.max(np.abs(ref.closure_residual(np.asarray(ctx.route['tree'][0], float)))), 1e-9)
        with self.assertRaisesRegex(ModelError, 'Disconnected physical graph'):
            s = json.loads(json.dumps(self.physical))
            s['joints'] = [j for j in s['joints'] if j['id'] != 'q21']
            s['seed']['joints'].pop('q21')
            s['actuators'] = [a for a in s['actuators'] if a['joint_id'] != 'q21']
            s['partition_requests'] = {k: [x for x in v if x != 'q21'] for k, v in s['partition_requests'].items()}
            compile_graph(s, 'joint_space')

    def test_all_variants_compile(self):
        for name, s in variant_inputs(self.physical):
            for partition in ('joint_space', 'cylinder_space'):
                comp = compile_graph(s, partition)
                self.assertEqual(comp.plan['mobility'], 7, name)


if __name__ == '__main__':
    unittest.main()
