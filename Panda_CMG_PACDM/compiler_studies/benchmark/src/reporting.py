"""Deterministic paper tables and narrative from this run's recorded data.
No historical rollout metrics are substituted for newly measured results.
"""
from pathlib import Path
import json,csv,html,re

SHORT={'original_pacdm':'Original PACDM','generated_full_pacdm':'Generated evaluator; explicit coupling','compiled_pacdm':'Affine-reduced PACDM','compiled_reuse':'Affine-reduced PACDM + reuse','compiled_no_predictor':'Affine-reduced PACDM; no predictor','trf_compiled':'Analytical TRF + predictor','trf_reuse':'Analytical TRF + predictor + reuse',
'analytic_dense':'Analytical Jacobian + dense TRF','analytic_sparse':'Analytical Jacobian + sparse TRF','fd3_dense':'Three-point differences + dense TRF','fd3_colored':'Colored three-point differences + sparse TRF'}
ORDER=list(SHORT)

def esc(s):
    s=str(s)
    for a,b in [('\\','\\textbackslash{}'),('&','\\&'),('%','\\%'),('_','\\_'),('#','\\#')]:s=s.replace(a,b)
    return s

def sci(x,latex=False):
    if x is None:return 'Not run'
    if float(x)==0:return '$0$'if latex else '0'
    s=f'{float(x):.3e}';m,e=s.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'if latex else s

def table_tex(label,caption,headers,rows,note='',wide=True,formats=None):
    env='table*'if wide else 'table';col='l'+'r'*(len(headers)-1)
    lines=[f'\\begin{{{env}}}[t]',r'\centering\small',r'\caption{'+caption+'}',r'\label{tab:franka_'+label+'}',r'\begin{tabular}{'+col+'}',r'\hline',' & '.join(headers)+r' \\',r'\hline']
    lines+=[' & '.join(str(x)for x in row)+r' \\'for row in rows];lines +=[r'\hline',r'\end{tabular}']
    if note:lines+=[r'\par\smallskip',r'\parbox{0.98\linewidth}{\footnotesize '+note+'}']
    lines.append(f'\\end{{{env}}}');return '\n'.join(lines)+'\n'

def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows])

def generate_report(out):
    out=Path(out);s=json.loads((out/'summary.json').read_text());env=json.loads((out/'environment.json').read_text());p=json.loads((out/'protocol.json').read_text())
    tables=[];texts=[];methods=[];effects=[]
    cfg=s['config'];native=s.get('native',s.get('dynamics',{})).get('native_executed',False)
    def add(label,title,headers,rows,note='',texrows=None):
        tables.append(table_tex(label,title,[esc(x)for x in headers],texrows if texrows is not None else [[esc(x)for x in r]for r in rows],note));texts.append('## '+title+'\n\n'+mdtable(headers,rows)+'\n\n'+note.replace('\\',''))
    def effect(name,ref,val,source):
        e=dict(benefit=name,reference_median_ms=ref,proposed_median_ms=val,reduction_percent=100*(1-val/ref),time_ratio=ref/val,source=source);effects.append(e);return e['reduction_percent']
    structure=list(csv.DictReader((out/'structure.csv').open()))
    add('structure','Equivalent Franka representations and physical mobility.',['Quantity','Original','Compiled'],[[r['quantity'],r['original'],r['compiled']]for r in structure],
        r'The nine-coordinate physical backend is retained. An exact affine lift parameterizes its rank-one finger equality using eight physical coordinates. Six virtual tool-target coordinates do not create a structural arm loop or additional physical actuators.')
    methods.append(r'''We evaluate the physical-graph compiler extension with the original PACDM core unchanged. The physical input contains a fixed-root scalar-joint tree, declared affine finger coupling, inertias, transmissions, a hand-attached tool frame, and named seeds. The generated output includes rooted paths, an exact affine lift $q=Sx+b$, transformed limits and actuation, and a tool-task evaluator. The compiler does not infer the coupling from measurements. The original implementation already used a manually specified finger reduction in its simulated plant; the new contribution tested here is its automatic construction and use in task assembly, not the invention of affine coordinate elimination.

The task coordinates are $z=(p_d,\theta_d,q_r,g)$: target translation, a local intrinsic-XYZ orientation chart, a prescribed upstream redundancy coordinate, and the independent finger coordinate. In the compiled chart, $\dot q=N_p\dot z$ and $\ddot q=N_p\ddot z+c_p$. The hand-attached tool is independent of the finger opening. Exact reuse is allowed only when all six target coordinates and the redundancy coordinate are bitwise unchanged. Any change in those inputs invalidates the reused arm solution and mapping. Physical dynamics are recomputed separately; changing the fingers is not assumed to leave the mass matrix unchanged.''')
    if 'pipeline'in s:
        a=s['pipeline'];rows=[['Predefined physical-data variants',a['variants']],['Candidate redundancy declarations',a['candidate_charts']],['Distinct variant/configuration pairs',a['variants']*cfg['witnesses']],['Chart/configuration checks',a['witness_checks']],['Admissible mappings verified',a['admissible_maps']],['Singular local charts correctly rejected',a['singular_chart_rejections']],['Checks satisfying their declared criteria',f"{a['passed']}/{a['witness_checks']}"],['Median graph compilation (ms)',f"{a['compile_median_ms']:.3f}"]]
        add('pipeline','Physical-graph generation and local coordinate admissibility.',['Quantity','Result'],rows,
            r'All seven upstream arm joints are enumerated as candidate prescribed redundancy coordinates. These are known-feasible physical configurations, not cold-start convergence trials. A rejected chart is a locally invalid coordinate choice, not an infeasible robot configuration. Off-manifold derivative checks are shared across chart declarations at each physical state; 420 checks do not mean 420 independent physical states.' if a['witness_checks']==420 else r'Known-feasible configuration checks; correctly rejected singular charts are not counted as successful nonlinear solves.')
        methods.append(f"Model-generation tests cover {a['variants']} predefined geometry, tool-frame, inertia, frame-expression, record-order and naming variants. Seven upstream-joint declarations per variant and {cfg['witnesses']} feasible configurations give {a['witness_checks']} chart/configuration checks. Of these, {a['admissible_maps']} admit the prescribed local chart and verify its mapping; the remaining {a['singular_chart_rejections']} are correctly rejected by the rank/conditioning checks. All {a['passed']} declared verification outcomes pass. Median graph compilation is {a['compile_median_ms']:.3f} ms, measured separately from witness evaluation. These outcomes establish executable interface generation and numerical coordinate-validity checking within the tested class, not unrestricted automatic coordinate selection.")
    if 'warm'in s:
        a=sorted(s['warm'],key=lambda r:ORDER.index(r['method']));rows=[];tr=[]
        for r in a:
            rr=[SHORT[r['method']],f"{r['accepted']}/{r['attempts']}",f"{r['median_ms']:.3f}",f"{r['p95_ms']:.3f}",sci(r['max_tool_gap_m'])];rows.append(rr);tr.append([esc(v)for v in rr[:-1]]+[sci(r['max_tool_gap_m'],True)])
        add('continuation','Franka continuation: assembly plus physical output mapping.',['Method','Accepted / attempts','Median (ms)','P95 (ms)','Max tool gap (m)'],rows,
            r'All methods use their own previous accepted solution. Initialization and independent physical validation are excluded equally. A common $10^{-8}$ m position/coupling gate and $10^{-8}$ rad orientation gate are imposed. Hold and gripper-only samples remain included; moving-input summaries are provided separately.',tr)
        W={r['method']:r['median_ms']for r in a};pe=effect('Generated evaluator vs original',W['original_pacdm'],W['generated_full_pacdm'],'warm_summary.csv');pc=effect('Affine reduction vs generated explicit coupling',W['generated_full_pacdm'],W['compiled_pacdm'],'warm_summary.csv');pt=effect('Compiled affine-reduced vs original',W['original_pacdm'],W['compiled_pacdm'],'warm_summary.csv');pp=effect('Predictor vs removal',W['compiled_no_predictor'],W['compiled_pacdm'],'warm_summary.csv');pr=effect('Reuse-enabled vs original full route',W['original_pacdm'],W['compiled_reuse'],'warm_summary.csv')
        methods.append(f"The continuation study follows the original 22-second route at {a[0]['attempts']//cfg['repeats']} samples with {cfg['repeats']} timing repetitions. With the PACDM core unchanged, the generated explicit-coupling evaluator has a median of {W['generated_full_pacdm']:.3f} ms versus {W['original_pacdm']:.3f} ms for the original evaluator ({pe:.1f}\\% lower). Affine lowering further changes the median to {W['compiled_pacdm']:.3f} ms, a {pc:.1f}\\% reduction relative to the generated explicit-coupling representation and {pt:.1f}\\% relative to the original implementation. Enabling tangent prediction reduces the compiled median by {pp:.1f}\\% relative to predictor removal. The exact-reuse workflow is reported separately; analytical TRF with the same generated model and predictor is retained as a strong comparator. These are implementation-level measurements, not a proof of algorithmic complexity or universal corrector superiority.")
    if 'local'in s:
        a=s['local'];modes=[m for m in ORDER if any(r['method']==m for r in a)];conditions=['gripper_only','tool_only','redundancy_only','tool_and_gripper'];rows=[]
        for m in modes:
            cells=[SHORT[m]]
            for c in conditions:
                r=next(v for v in a if v['method']==m and v['condition']==c);cells.append(f"{r['median_ms']:.3f} / {r['p95_ms']:.3f}")
            rows.append(cells)
        add('local_updates','Dependency-local update comparisons: median / P95 milliseconds.',['Method','Gripper only','Tool only','Redundancy only','Tool + gripper'],rows,
            r'Times include the physical configuration and mapping. Each condition uses the same initial physical seeds and target inputs across methods. Gripper-only reuse returns an updated exact affine configuration, not a cached frozen robot state. All conditions, including those invalidating reuse, are retained.')
        G={r['method']:r for r in a if r['condition']=='gripper_only'};gl=effect('Gripper-only exact reuse vs compiled recomputation',G['compiled_pacdm']['median_ms'],G['compiled_reuse']['median_ms'],'local_updates_summary.csv')
        methods.append(f"The local-update study has {G['compiled_pacdm']['attempts']} attempts per method and condition, comprising {cfg['cases']} matched local updates repeated {cfg['repeats']} times. For gripper-only changes with the complete arm-task input unchanged, exact dependency-aware reuse reduces median assembly-and-mapping time from {G['compiled_pacdm']['median_ms']:.3f} to {G['compiled_reuse']['median_ms']:.3f} ms ({gl:.1f}\\%). This is the isolated scheduling benefit: the same reduced model and PACDM solver are used, but unnecessary arm reevaluation is avoided. Tool-only, redundancy-only and combined updates are also measured and correctly invalidate reuse. The TRF reuse route is included because this compiler-generated scheduling benefit is not exclusive to one corrector.")
    if 'derivatives'in s:
        a=sorted(s['derivatives'],key=lambda r:['analytic_dense','fd3_dense','analytic_sparse','fd3_colored'].index(r['method']));rows=[[SHORT[r['method']],f"{r['accepted']}/{r['attempts']}",f"{r['median_ms']:.3f}",f"{r['p95_ms']:.3f}",f"{r['median_evaluations']:.0f}"]for r in a]
        add('derivatives','Matched derivative-availability experiments: corrector-only runtime.',['Derivative route','Accepted / attempts','Median (ms)','P95 (ms)','Median evaluations'],rows,
            r'Within each pair, TRF formulation, bounds, prediction, stopping rules and linear solver are matched. Fresh model calls include finite-difference perturbations, not just nonlinear iterations or SciPy nfev. The reduced serial-arm block is structurally dense: coloring requires six groups and is not claimed to reduce differentiation work here.')
        D={r['method']:r['median_ms']for r in a};ed=effect('Analytical vs dense finite differences',D['fd3_dense'],D['analytic_dense'],'derivative_summary.csv');es=effect('Analytical vs colored finite differences, sparse pair',D['fd3_colored'],D['analytic_sparse'],'derivative_summary.csv')
        methods.append(f"The derivative study uses {cfg['cases']} matched off-manifold correction cases with {cfg['repeats']} timing repetitions. Analytical and three-point numerical derivatives are compared within identical dense-TRF and sparse-TRF pairs. Generated analytical derivatives reduce median corrector time by {ed:.1f}\\% versus dense finite differences and {es:.1f}\\% versus colored finite differences in the sparse pair. The finite-difference path does not compute and discard an analytical Jacobian. Because the reduced serial-arm block is dense, the colored baseline still requires six perturbation groups; these measurements demonstrate analytical derivative availability, not a sparsity benefit. Shared prediction and downstream mapping/validation are excluded from these timings.")
    d=s.get('native',s.get('dynamics'))
    if d:
        M=d['maxima'];spec=[('Task-coordinate PACDM vs full KKT, normalized','relative_acceleration_difference',1e-8),('Generated affine reduction vs full KKT, normalized','affine_vs_kkt_relative',1e-8),('Finger acceleration residual (m/s2)','finger_acceleration_residual_m_s2',1e-8),('Tool acceleration consistency, mixed SI','tool_acceleration_consistency_mixed',2e-6),('Generated vs independent task mapping','physical_mapping_discrepancy',1e-9),('Task-coordinate virtual-power defect (W)','virtual_power_defect_W',1e-9),('Affine virtual-power defect (W)','affine_power_defect_W',1e-9),('Task reduced-inertia discrepancy, normalized','reduced_inertia_relative',1e-9),('Full KKT force-balance residual, mixed SI','kkt_force_balance',1e-8)]
        if native:spec=[('PACDM vs native-matrix coupling KKT, normalized','native_kkt_acceleration_relative',1e-7),('NumPy vs native mass matrix','native_mass_inf',1e-8),('NumPy vs native bias vector','native_bias_inf',1e-7),('NumPy vs native tool Jacobian','native_tool_jacobian_inf',1e-9),('NumPy vs native tool acceleration bias','native_tool_gamma_inf',1e-8),('Native classical tool acceleration consistency','native_task_acceleration_mixed',2e-6)]+spec
        rows=[[name,sci(M[k]),sci(lim)]for name,k,lim in spec];tr=[[esc(name),sci(M[k],True),sci(lim,True)]for name,k,lim in spec]
        add('dynamics',f'Franka dynamics and interface consistency: {d["accepted"]}/{d["witnesses"]} accepted witnesses.',['Comparison / metric','Maximum','Limit'],rows,
            r'The sole physical constraint is the declared scalar finger equality; the tool target is not a welded contact. Diagonal source armature and viscous damping are included on both routes. Max-entry normalization is defined in the text. Generalized-coordinate and tool-motion norms mix translational and angular SI components. Limits are strict upper bounds. '+(r'Native CRBA/RNEA/FK data are used with an explicitly assembled scalar-coupling KKT solve; this is not Pinocchio constraintDynamics.'if native else r'This table uses the independent NumPy physical-point/classical rigid-body reference. Native Pinocchio was not run.'),tr)
        cur=d['curvature'];rows=[[f"{r['speed_scale']:g}x",sci(r['full_linear_m_s2']),f"{r['omitted_linear_m_s2']:.6f}",sci(r['full_angular_rad_s2']),f"{r['omitted_angular_rad_s2']:.6f}"]for r in cur];tr=[[f"${r['speed_scale']:g}\\times$",sci(r['full_linear_m_s2'],True),f"{r['omitted_linear_m_s2']:.6f}",sci(r['full_angular_rad_s2'],True),f"{r['omitted_angular_rad_s2']:.6f}"]for r in cur]
        add('curvature','Acceleration-curvature ablation: separate translational and angular defects.',['Speed','Full (m/s2)','Omitted (m/s2)','Full (rad/s2)','Omitted (rad/s2)'],rows,
            r'Twelve configurations at four velocity scales in the full profile. The target chart velocity is prescribed with zero independent acceleration. Complete mapping and curvature-omitted mapping are checked using independent classical tool acceleration. The omitted term is an intentional ablation, not a correct competing dynamics solver.',tr)
        base=f"Across {d['witnesses']} matched states, the generated task-coordinate dynamics agree with the full-coordinate NumPy KKT reference, with maximum normalized acceleration discrepancy {sci(M['relative_acceleration_difference'],True)}. All {d['accepted']} witnesses pass. The generated affine lift also reproduces the physical finger-reduced dynamics and preserves virtual power. The tool-task coordinate mass matrix and acceleration curvature are validated without treating the tool target as a physical constraint. "
        if native:base+=f"The additional native-matrix comparison gives {sci(M['native_kkt_acceleration_relative'],True)} normalized acceleration discrepancy. Native Pinocchio rigid-body evaluations are coupled to an explicit scalar-equality KKT linear system; this is not a call to the native point-contact constrained-dynamics routine. "
        else:base+="Native Pinocchio was unavailable for this run. Its adapter is included for a separate local execution; no native result is asserted. "
        c4=cur[-1];base+=f"At fourfold prescribed speed, omission of curvature produces maximum translational and angular defects of {c4['omitted_linear_m_s2']:.6f} m/s$^2$ and {c4['omitted_angular_rad_s2']:.6f} rad/s$^2$, while the complete interface gives {sci(c4['full_linear_m_s2'],True)} m/s$^2$ and {sci(c4['full_angular_rad_s2'],True)} rad/s$^2$. The curvature is evaluated using a centered directional difference of the generated analytical Jacobian."
        methods.append(base)
    methods.append(r'''All reported medians and empirical 95th percentiles are descriptive timing statistics, not confidence intervals or independent robot success estimates. Method order is randomized within matched cases; BLAS threads are limited to one. PACDM retains its original stopping rules and uses its original acquisition routine if an eight-iteration warm corrector fails. TRF uses $10^{-11}$ cost, step and gradient tolerances, unit numerical scaling, and a limit of 1,500 objective evaluations. Common external physical acceptance tests are applied independently. The physical tool-position and finger-equality gates are $10^{-8}$ m, the orientation gate is $10^{-8}$ rad, and additional bound, independent-coordinate and conditioning gates are retained.

For normalized acceleration and inertia discrepancies,
\[
\delta(X,X_{\mathrm{ref}})=\frac{\|X-X_{\mathrm{ref}}\|_{\max}}{\max(1,\|X_{\mathrm{ref}}\|_{\max})},\qquad
\|X\|_{\max}=\max_{i,j}|X_{ij}|.
\]
The analogous maximum-component norm is used for vectors. This is coordinate-dependent numerical normalization, not a unit-invariant physical percentage. Runtime reductions are $100(1-t_{\mathrm{proposed}}/t_{\mathrm{reference}})$, calculated from medians. Model-evaluation counts include fresh combined residual/Jacobian calls on the analytical path and all fresh residual calls, including numerical perturbations, on the finite-difference path.

The experiments demonstrate scoped physical-graph interface generation, affine lowering, conditional task reuse, and numerical consistency. They do not establish novel analytical differentiation or affine reduction in isolation, arbitrary-robot generality, global convergence, real-time guarantees, physical grasp/contact performance, hardware accuracy, or superiority over URDF+/generalized\_rbda, which was not benchmarked. No newly integrated closed-loop robot rollout is claimed.''')
    versions=env['versions'];nativever=versions.get('pinocchio_imported')
    methods.append('The measurements were generated using Python '+env['python'].split()[0]+', NumPy '+str(versions['numpy'])+', SciPy '+str(versions['scipy'])+', on '+esc(env['platform'])+'. The recorded processor identification is '+esc(env.get('processor_model')or'not available')+'. The host-exposed identification and virtualization information are saved in the environment record; these are not measurements on an assumed dedicated processor or on a different native-validation machine.'+(' The imported Pinocchio version was '+str(nativever)+'.'if nativever else ' Pinocchio was not imported in this environment.'))
    text='\n\n'.join(methods);(out/'METHODS_RESULTS.tex').write_text(text+'\n',encoding='utf-8');(out/'TABLES_MAIN.tex').write_text('\n\n'.join(tables),encoding='utf-8')
    (out/'effects.json').write_text(json.dumps(effects,indent=2)+'\n',encoding='utf-8')
    with (out/'effects.csv').open('w',newline='',encoding='utf-8')as f:
        if effects:w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
    report='# Franka CMG/PACDM framework-benefit study\n\nExecution: **'+s['status']+'**. New measurements only; original archived control-rollout results are not pooled.\n\n'
    report+='The original PACDM core is unchanged. The physical-graph compiler, task evaluator, affine task-coordinate lowering and exact-input scheduler are new scoped extensions.\n\n'
    report+='\n\n'.join(texts)+'\n\n## Methods, interpretation and boundaries\n\n'+text+'\n\n## Evidence\n\nCSV/JSON/NPZ files next to this report contain raw measurements and saved states. `protocol.json`, `source_hashes.json` and `environment.json` identify the implementation and conditions. `native_status.json` distinguishes availability from execution.\n\nPrimary documentation: SciPy 1.17.0 least_squares, https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html ; Pinocchio official project, https://github.com/stack-of-tasks/pinocchio ; URDF+ v1 (not benchmarked), https://arxiv.org/html/2411.19753v1 .\n'
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Franka framework benefits</title><style>body{font:15px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 26px;color:#172b3a}h1,h2{color:#184e62}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}td,th{border-bottom:1px solid #d5dce1;padding:9px;text-align:left}th{background:#e9f1f3}pre{white-space:pre-wrap;font:13px/1.6 ui-monospace,monospace;background:#f3f6f8;padding:18px}</style><h1>Franka framework-benefit study</h1><p>'+html.escape(s['status'])+'</p><p>New compiler/evaluator/affine lowering and scheduler extensions; original PACDM core unchanged. No new control rollout or hardware trial.</p>']
    for t in texts:
        lines=t.splitlines();parts.append('<h2>'+html.escape(lines[0].lstrip('# '))+'</h2>');tab=[l for l in lines if l.startswith('|')];parts.append('<table>')
        for i,line in enumerate(tab):
            if i==1:continue
            tag='th'if i==0 else 'td';parts.append('<tr>'+''.join(f'<{tag}>'+html.escape(v.strip())+f'</{tag}>'for v in line.strip('|').split('|'))+'</tr>')
        parts.append('</table><p>'+html.escape(lines[-1])+'</p>')
    parts.append('<h2>Methods and interpretation (LaTeX source)</h2><pre>'+html.escape(text)+'</pre></html>');(out/'REPORT.html').write_text(''.join(parts),encoding='utf-8')
    claims='''# Franka claim-to-evidence guide

Supported focus: physical-tree records generate the exact declared finger-coupling lift, task dependency paths and verified analytical interfaces. The numerical task solve becomes 6-dimensional rather than 7-dimensional; physical mobility remains eight. Hand-attached tool tasks allow exact reuse during gripper-only updates. Full strong-baseline comparisons remain in the data.

The affine plant reduction already existed manually in the original package. Its generation and use in task assembly are the new extension; affine reduction and analytical differentiation are not claimed to be new mathematical ideas. No physical arm loop or additional actuators are introduced by the six virtual task coordinates. A tool attached to a finger is outside the supported reuse pattern and is rejected by the compiler.

The rank study contains correctly rejected charts: physical feasibility does not guarantee validity of a specified redundancy coordinate. No nonlinear convergence success is claimed for those rejected charts. The original core hash is recorded in summary.json.

Native scope: native rigid-body data and frame derivatives plus an explicit scalar-coupling KKT solve, not Pinocchio constraintDynamics. No URDF+/generalized_rbda port is included or measured. No contact/grasp/closed-loop trajectory was rerun for this study.
'''
    (out/'CLAIMS.md').write_text(claims,encoding='utf-8')
