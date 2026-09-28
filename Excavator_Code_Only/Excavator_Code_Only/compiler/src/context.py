"""Shared inputs: original records, compiled partitions, route and the reference."""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np

from .bootstrap import ROOT, V21
from .compiler import compile_graph
from excavator_pacdm import compile_pacdm
from plant import ExcavatorPinModel  # validated Pinocchio backend (unchanged)

PARTITIONS = ('joint_space', 'cylinder_space')


class Context:
    def __init__(self, physical=None):
        self.cmg = json.loads((V21 / 'data/accepted_cmg_v04.json').read_text())
        self.mapping = json.loads((V21 / 'data/accepted_mujoco_mapping.json').read_text())
        self.physical = physical or json.loads((ROOT / 'inputs/physical_graph.json').read_text())
        self.compiled = {p: compile_graph(self.physical, p) for p in PARTITIONS}
        with np.load(ROOT / 'inputs/nominal_route.npz', allow_pickle=False) as z:
            self.route = {k: z[k].copy() for k in z.files}
        self.ref_ids = [str(x) for x in self.route['tree_ids']]
        self.reference = Reference(self)

    def independent_ids(self, partition):
        return list(self.compiled[partition].plan['independent_ids'])

    @cached_property
    def original_graph(self):
        return compile_pacdm(self.cmg, self.mapping, self.mapping['tree_joint_ids'])[0]

    def full_state(self, tree_ref):
        """All 32 joint values from 23 closed reference-tree values (original lift)."""
        g = self.original_graph
        values = dict(zip(self.ref_ids, np.asarray(tree_ref, float)))
        q = g.lift(np.array([values[k] for k in g.ids[:g.nt]]))
        return dict(zip(g.ids, q))

    def joint_route(self):
        return np.asarray(self.route['desired'], float)


class Reference:
    """Independent acceptance with a separate native Pinocchio model (original cuts).

    Closure gap over the 18 native point pairs, exact independent coordinates,
    branch window, tangent residual of the returned map, passive rank and the
    difference to the native least-squares map of the same partition.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.model = ExcavatorPinModel(ctx.cmg, ctx.mapping['cut_joint_ids'])
        if list(self.model.tree_ids) != ctx.ref_ids:
            raise RuntimeError('Route tree order differs from the native model order')
        s = ctx.compiled['joint_space'].structure
        ids = list(s['coordinate_ids'])
        self.lower = np.array([s['lower'][ids.index(k)] for k in ctx.ref_ids])
        self.upper = np.array([s['upper'][ids.index(k)] for k in ctx.ref_ids])
        self.active = {p: np.array([ctx.ref_ids.index(k) for k in ctx.independent_ids(p)], int)
                       for p in PARTITIONS}
        self.passive = {p: np.setdiff1d(np.arange(len(ctx.ref_ids)), a) for p, a in self.active.items()}

    def check(self, q, N, target, partition, known=None):
        a, p = self.active[partition], self.passive[partition]
        residual = self.model.closure_residual(q)
        J = self.model.closure_jacobian(q)
        gap = float(np.max(np.abs(residual)))
        target_error = float(np.max(np.abs(q[a] - target)))
        eye = float(np.max(np.abs(N[a] - np.eye(len(a)))))
        tangent = float(np.max(np.abs(J @ N)))
        Jp = J[:, p]
        s = np.linalg.svd(Jp, compute_uv=False)
        rank = int(np.sum(s > 1e-10 * s[0]))
        native_map = np.linalg.lstsq(Jp, -J[:, a], rcond=None)[0]
        map_error = float(np.max(np.abs(N[p] - native_map)))
        inside = bool(np.all(q >= self.lower - 1e-12) and np.all(q <= self.upper + 1e-12))
        out = dict(max_gap_m=gap, target_error=target_error, tangent_residual_ref=tangent, map_error=map_error,
                   passive_rank=rank, inside_branch_window=inside)
        ok = (np.all(np.isfinite(q)) and np.all(np.isfinite(N)) and gap <= 1e-8 and target_error <= 1e-12
              and eye < 1e-12 and tangent < 1e-8 and map_error < 1e-8 and rank == len(p) and inside)
        if known is not None:
            out['known_branch_error'] = float(np.max(np.abs(q - known)))
            ok = ok and out['known_branch_error'] < 1e-6
        out['success'] = bool(ok)
        return out
