import hashlib,json,unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from src import bootstrap
from src.bootstrap import ROOT
from src.compiler import compile_graph,export_physical,variant_inputs,support_sets,components,ModelError
from src.task_graph import GeneratedTaskGraph,CompiledStanceGraph,ModularSolver,create_solver
from src.physics import PhysicalReference,dynamics_witness,normmax,mapping_curvature
from vendor.pacdm_original import PACDM


class Go2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads((ROOT/'original/data/go2_cmg.json').read_text());cls.source=export_physical(cls.original);cls.comp=compile_graph(cls.source)
        cls.q=np.array(cls.comp.cmg['q_reference']);cls.ref=PhysicalReference(cls.comp.cmg)

    def test_original_core_unchanged(self):
        m=json.loads((ROOT/'original/ORIGINAL_HASHES.json').read_text());p=ROOT/'original/vendor/pacdm_original.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m['vendor/pacdm_original.py'])

    def test_no_physical_dimension_or_actuator_reduction(self):
        p=self.comp.plan;self.assertEqual((p['physical_cycles'],p['physical_coordinates'],p['physical_actuators'],p['task_augmented_coordinates']),(0,18,12,30))
        self.assertEqual(p['conditional_subproblem_sizes'],[3]*4)

    def test_nominal_source_equivalence(self):
        original=deepcopy(self.original);original['physical_root']='base';r=PhysicalReference(original)
        q=self.q.copy();q[3:6]=[.2,.1,-.1]
        a=self.ref.forward(q);b=r.forward(q)
        for name in a:self.assertLess(normmax(a[name]['p']-b[name]['p']),1e-12);self.assertLess(normmax(a[name]['R']-b[name]['R']),1e-12)
        M,h,_=r.mass_bias(q,np.ones(18)*.03);Mc,hc,_=self.ref.mass_bias(q,np.ones(18)*.03)
        self.assertLess(normmax(M-Mc),1e-11);self.assertLess(normmax(h-hc),1e-11)

    def test_all_twelve_data_variants(self):
        variants=variant_inputs(self.source);self.assertEqual(len(variants),12)
        for name,s in variants:
            c=compile_graph(s);q=np.array(c.cmg['q_reference']);p,J=GeneratedTaskGraph(c).points_jacobian(q);rr=PhysicalReference(c.cmg).feet_geometry(q)
            self.assertLess(normmax(p-rr['points']),1e-12,name);self.assertLess(normmax(J-rr['jacobian']),1e-12,name)

    def test_generated_task_derivative_off_manifold(self):
        g=GeneratedTaskGraph(self.comp);x=g.lift(self.q);x[:6]+=[.02,.01,0,.1,.04,-.02];x[7]+=.03;x[21]+=.005
        r,J,D=g.residual(x);fd=np.empty_like(J);eps=1e-6
        for i in range(g.n):
            d=np.eye(g.n)[i]*eps;fd[:,i]=(g.residual_only(x+d)-g.residual_only(x-d))/(2*eps)
        self.assertLess(normmax(fd-J),1e-7)

    def test_fd_residual_does_not_compute_jacobian(self):
        g=GeneratedTaskGraph(self.comp);x=g.lift(self.q);record=[];original=g.points_jacobian
        def spy(q,need_jac=True):record.append(need_jac);return original(q,need_jac)
        g.points_jacobian=spy;g.residual_only(x);self.assertEqual(record,[False])

    def test_structural_sparsity(self):
        g=GeneratedTaskGraph(self.comp);x=g.lift(self.q);x[:18]+=.02
        J=g.residual(x)[1][:,g.passive];pattern=np.array(self.comp.plan['passive_sparsity'])
        self.assertEqual(int(pattern.sum()),36);self.assertEqual(normmax(J[pattern==0]),0.)

    def test_all_solver_interfaces(self):
        g=GeneratedTaskGraph(self.comp);p,_=g.points_jacobian(self.q);qa=np.r_[self.q[:6],p.ravel()];qa[8]+=.003
        for name in ['original_monolithic','compiled_monolithic','compiled_modular','modular_no_predictor','modular_no_reuse','trf_compiled_predictor','trf_modular_predictor']:
            q,N,_=create_solver(name,self.comp,self.q).step(qa);geo=self.ref.feet_geometry(q)
            self.assertLess(normmax(geo['points'].ravel()-qa[6:]),1e-8,name)
            self.assertLess(normmax(geo['jacobian'].reshape(12,18)@N-np.c_[np.zeros((12,6)),np.eye(12)]),1e-9,name)

    def test_exact_reuse_and_invalidation(self):
        s=ModularSolver(self.comp,self.q);g=GeneratedTaskGraph(self.comp);p,_=g.points_jacobian(self.q);qa=np.r_[self.q[:6],p.ravel()]
        _,_,info=s.step(qa);self.assertEqual(info['skipped_modules'],4)
        qa[8]+=.001;_,_,info=s.step(qa);self.assertEqual(info['solved_modules'],1)
        qa[1]+=.001;_,_,info=s.step(qa);self.assertEqual(info['solved_modules'],4)

    def test_support_partitions_and_rank(self):
        for sites in support_sets():
            g=CompiledStanceGraph(self.comp,self.q,sites);N,info=PACDM(g).mapping(self.q)
            self.assertTrue(info['success']);self.assertEqual(info['rank_full'],3*len(sites));self.assertEqual(N.shape,(18,18-3*len(sites)))

    def test_unactuated_base_in_dynamics(self):
        for sites in [[0,3],[0,1,2],[0,1,2,3]]:
            g=CompiledStanceGraph(self.comp,self.q,sites);r,_=dynamics_witness(self.comp,self.q,sites,np.linspace(-.07,.1,len(g.active)),np.linspace(-2,2,12),np.zeros(6))
            self.assertTrue(r['success']);self.assertEqual(r['base_motor_effort_max'],0.)

    def test_quadratic_curvature_scaling(self):
        g=CompiledStanceGraph(self.comp,self.q,[0,1,2,3]);va=np.array([.1,-.08,.05,.02,.04,-.03]);values=[]
        for speed in (1,2):
            N,v,c,_=mapping_curvature(g,self.q,va*speed);geo=self.ref.feet_geometry(self.q,v);J=geo['jacobian'].reshape(12,18);gamma=geo['gamma'].ravel()
            self.assertLess(normmax(J@c+gamma),2e-6);values.append(normmax(gamma))
        self.assertAlmostEqual(values[1]/values[0],4.,places=9)

    def test_dependency_coupling_not_just_four_leg_names(self):
        supports=[p['dependent_indices']+[0]for p in self.comp.plan['point_paths']]
        self.assertEqual(components(supports),[[0,1,2,3]])

    def test_invalid_input_not_silently_repaired(self):
        s=deepcopy(self.source);s['joints'][0]['axis']=[3,0,0]
        with self.assertRaises(ModelError):compile_graph(s)

    def test_supplied_graph_is_physical_only(self):
        s=json.loads((ROOT/'inputs/physical_graph.json').read_text());self.assertNotIn('joint_ids',s['point_sites'][0]);self.assertNotIn('coordinate_ids',s);self.assertEqual(len(s['bodies']),13)

if __name__=='__main__':unittest.main(verbosity=2)
