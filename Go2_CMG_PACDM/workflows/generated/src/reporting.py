"""Paper-facing tables generated from measured records, never preset outcomes."""
from __future__ import annotations
from pathlib import Path
import html,json,csv
from .io_utils import save_csv

NAMES={
 'original_monolithic':'Original global PACDM',
 'compiled_monolithic':'Compiled global PACDM',
 'compiled_modular':'Compiled modular PACDM + reuse',
 'modular_no_predictor':'Modular PACDM, predictor removed',
 'modular_no_reuse':'Modular PACDM, reuse removed',
 'trf_compiled_predictor':'Global TRF + generated predictor',
 'trf_modular_predictor':'Modular TRF + predictor + reuse',
 'analytic_dense':'Generated analytical + dense TRF',
 'fd3_dense':'Three-point FD + dense TRF',
 'analytic_sparse':'Generated analytical + sparse TRF',
 'fd3_colored':'Colored three-point FD + sparse TRF'}
METHOD_ORDER=list(NAMES)[:7]


def read(path):
    if not Path(path).exists():return []
    with Path(path).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))

def f(x,n=3):return f'{float(x):.{n}f}'
def sci(x):return 'not run' if x is None else f'{float(x):.3e}'
def escape(s):
    s=str(s)
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c)for c in s)

def value_tex(v):
    text=str(v)
    if 'e-' in text or 'e+' in text:
        try:
            a,b=text.split('e');float(a);int(b);return f'${a}\\times10^{{{int(b)}}}$'
        except ValueError:pass
    return escape(text)


def generate_report(out):
    out=Path(out);summary=json.loads((out/'summary.json').read_text());st=summary['stages'];env=json.loads((out/'environment.json').read_text())
    plan=json.loads((out/'compiled_plan.json').read_text());protocol=json.loads((out/'protocol.json').read_text());cfg=protocol['configuration']
    tables=[];paragraphs=[];benefits=[]
    def table(key,title,heads,rows,note,source):
        tables.append(dict(key=key,title=title,headers=heads,rows=rows,note=note,source=source))
    table('dimensions','Equivalent Go2 foot-task representations',
        ['Quantity','Original global solve','Compiled conditional modules'],[
            ['Physical coordinates',18,18],['Virtual Cartesian target coordinates',12,12],['Total augmented coordinates',30,30],
            ['Permanent physical closure constraints',0,0],['Task residual rows / numerical rank','24 / 12','24 / 12'],
            ['Total dependent joint coordinates',12,12],['Largest dependent solve block',12,3],
            ['Executed task subproblems',1,4],['Physical actuators / unactuated base coordinates','12 / 6','12 / 6']],
        'No reduction in physical mobility or total coordinates is claimed. Four kinematic blocks are independent only conditional on prescribed base6. The rigid-body dynamics remain coupled. Twenty-four SE(3)-subgroup rows include twelve identically zero rotational rows.',
        'compiled_plan.json; src/compiler.py; original/go2/contact.py')
    if 'pipeline'in st:
        r=st['pipeline'];v=r['variants'];w=cfg['witnesses']
        table('compiler','Physical-graph compilation and mode-plan verification',
              ['Generated plan','Plans','Configurations / plan','Accepted / checks','Rank / chart dimension'],[
                ['Four Cartesian foot tasks',v,w,f'{v*w}/{v*w}','12 / 18'],
                ['Two-foot ideal support',6*v,w,f'{6*v*w}/{6*v*w}','6 / 12'],
                ['Three-foot ideal support',4*v,w,f'{4*v*w}/{4*v*w}','9 / 9'],
                ['Four-foot ideal support',v,w,f'{v*w}/{v*w}','12 / 6']],
              f"{v} predefined physical/data variants; {r['mode_plans']} total mode plans; actual aggregate {r['passed']}/{r['checks']} accepted. Known-feasible consistency checks, not cold-start convergence or stable-support trials. The source supplies physical data, sites, actuators and named seeds, not paths, membership lists, Jacobians or support partitions.",
              'pipeline_raw.csv; compilation_times.csv; pipeline_states.json; generated/')
        # Per-group pass numbers are read from raw rather than assumed above.
        raw=read(out/'pipeline_raw.csv')
        for i,m in enumerate((0,2,3,4)):
            rr=[x for x in raw if int(x['support_count'])==m];passed=sum(x['success']=='True'for x in rr)
            tables[-1]['rows'][i][3]=f'{passed}/{len(rr)}'
        paragraphs.append(f"The compiler generated {r['physical_models']} physical models and {r['mode_plans']} task/support mode plans. All recorded outcomes are retained: {r['passed']}/{r['checks']} configuration checks passed. Median graph compilation took {r['median_compilation_ms']:.3f} ms (separate from solving).")
    if 'warm'in st:
        data={r['method']:r for r in st['warm']['methods']}
        table('continuation','Go2 continuation: assembly plus physical output mapping',
              ['Method','Accepted / attempts','Median ms','P95 ms','Max foot gap (m)'],[
                [NAMES[m],f"{data[m]['accepted']}/{data[m]['attempts']}",f(data[m]['median_ms']),f(data[m]['p95_ms']),sci(data[m]['max_gap_m'])]for m in METHOD_ORDER if m in data],
              f"Original 26-second route, {cfg['route_samples']} samples and {cfg['repeats']} timing repetitions. Each method uses its own previous solution; common initialization and independent auditing are outside timing. Hold samples are retained; moving-only and per-repeat summaries are also supplied. P95 is not a confidence interval.",
              'warm_raw.csv; warm_summary.csv; warm_per_repeat.csv; warm_states.npz; warm_inputs.json')
        for label,ref,proposed in [
            ('Generated evaluator versus supplied evaluator','original_monolithic','compiled_monolithic'),
            ('Full compiled module workflow versus original','original_monolithic','compiled_modular'),
            ('Predictor within compiled modular PACDM','modular_no_predictor','compiled_modular'),
            ('Exact reuse within compiled modular PACDM','modular_no_reuse','compiled_modular')]:
            a,b=data[ref]['median_ms'],data[proposed]['median_ms']
            benefits.append(dict(condition='26-second full route',comparison=label,reference=ref,proposed=proposed,reference_median_ms=a,proposed_median_ms=b,reduction_percent=100*(1-b/a),reference_over_proposed=a/b))
        paragraphs.append('The comparison separates evaluator lowering, conditional scheduling and the predictor. The original and compiled global routes use the same unchanged PACDM core and total coordinate count. Full-route modular overhead is measured rather than presumed beneficial. Both global and modular analytical TRF alternatives receive the generated derivatives and the same predictor/scheduling information.')
    if 'incremental'in st:
        data={(int(r['changed_feet']),r['method']):r for r in st['incremental']['methods']}
        header=['Changed feet','Global PACDM ms','Modular PACDM ms','No reuse ms','Global TRF ms','Modular TRF ms']
        rows=[]
        for count in (1,2,4):
            rows.append([count]+[f(data[(count,m)]['median_ms'])for m in ('compiled_monolithic','compiled_modular','modular_no_reuse','trf_compiled_predictor','trf_modular_predictor')])
            for reference in ('compiled_monolithic','modular_no_reuse','trf_compiled_predictor'):
                a=data[(count,reference)]['median_ms'];b=data[(count,'compiled_modular')]['median_ms']
                benefits.append(dict(condition=f'{count} changed feet, prescribed base fixed',comparison='Conditional task update',reference=reference,proposed='compiled_modular',reference_median_ms=a,proposed_median_ms=b,reduction_percent=100*(1-b/a),reference_over_proposed=a/b))
        table('incremental','Conditional task edits: generated dependency-local recomputation',header,rows,
              f"Median assembly-plus-mapping time; {cfg['increment_blocks']} initial configurations, {cfg['increment_steps']} edits and {cfg['repeats']} repeats per group. Every one-, two- and four-foot condition is retained. Same generated evaluator and unchanged PACDM core for global/modular PACDM. Skips require exact equality of the independent block; changing base pose invalidates all four blocks. This is a task-update test, not a fixed-base physical robot.",
              'incremental_raw.csv; incremental_summary.csv; incremental_inputs.json')
        paragraphs.append('The incremental comparison is the direct module-scheduling experiment: generated dependency support identifies which leg coordinates and mapping rows must be recomputed after localized foot-target edits. The same generated evaluator is used by the compiled global and modular routes. The no-reuse ablation recomputes all four modules, isolating exact-cache scheduling from evaluator implementation differences. A strong modular TRF route is retained too.')
    if 'derivatives'in st:
        data={r['method']:r for r in st['derivatives']['methods']}
        order=['analytic_dense','fd3_dense','analytic_sparse','fd3_colored']
        table('derivatives','Matched availability of generated analytical derivatives',
              ['Derivative route','Accepted / attempts','Median ms','P95 ms','Median model evaluations'],[
                  [NAMES[m],f"{data[m]['accepted']}/{data[m]['attempts']}",f(data[m]['median_ms']),f(data[m]['p95_ms']),f(data[m]['median_evaluations'],1)]for m in order],
              f"Corrector-only time; {cfg['derivative_cases']} cases with {cfg['repeats']} repetitions. Each dense/sparse pair uses the same TRF equations, prediction, bounds, tolerances and linear solver. FD residuals do not calculate and discard an analytical Jacobian. The colored baseline receives the generated sparsity. Counts include all finite-difference perturbations; they are not iterations or SciPy nfev.",
              'derivative_raw.csv; derivative_summary.csv; derivative_inputs.json')
        for ref,pr in [('fd3_dense','analytic_dense'),('fd3_colored','analytic_sparse')]:
            a,b=data[ref]['median_ms'],data[pr]['median_ms']
            benefits.append(dict(condition='Matched corrector-only derivative test',comparison='Generated derivative interface',reference=ref,proposed=pr,reference_median_ms=a,proposed_median_ms=b,reduction_percent=100*(1-b/a),reference_over_proposed=a/b))
        paragraphs.append('The derivative experiment measures the practical value of the automatically available derivative/sparsity interface. It is not a claim that analytical differentiation is new or superior to all automatic-differentiation implementations, and not a PACDM-versus-TRF corrector comparison.')
    dyn=st.get('native',st.get('dynamics'))
    if dyn:
        native=dyn['native']['status']=='completed';maxima=dyn['maxima']
        table('support_dynamics','Ideal-support dynamics: PACDM reduction versus physical-point KKT',
            ['Supporting feet','Accepted / states','Rank / mobility','Max normalized acceleration error','Max point residual (m/s2)'],[
                [r['support_count'],f"{r['passed']}/{r['attempts']}",f"{r['expected_rank']} / {r['expected_mobility']}",sci(r['acceleration_relative']),sci(r['constraint_residual_m_s2'])]for r in dyn['support_modes']],
            'Matched rigid-body states and inputs, twelve leg motors with zero base motor effort. Ideal fixed-anchor toe centers; no friction cones, detachment, impacts, support-transition integration or stable-gait claim. Geometry, mass and bias reference uses an independent classical NumPy recursion.',
            'dynamics_raw.csv; dynamics_states.json; support_dynamics_summary.csv')
        checks=[('PACDM versus NumPy KKT acceleration (normalized)','relative_acceleration_difference',1e-8),
                ('Physical point acceleration residual (m/s2)','point_acceleration_residual_m_s2',2e-6),
                ('Physical point tangent residual','tangent_residual',1e-9),
                ('Independent support-mapping discrepancy','mapping_discrepancy',1e-9),
                ('Virtual-power defect (W)','virtual_power_defect_W',1e-9),
                ('Reduced-inertia discrepancy (normalized)','reduced_inertia_relative',1e-9)]
        if native:checks=[('PACDM versus native Pinocchio acceleration (normalized)','native_acceleration_relative',1e-7),
                          ('NumPy versus native mass matrix','native_mass_inf',1e-8),('NumPy versus native bias vector','native_bias_inf',1e-7),
                          ('NumPy versus native point Jacobian','native_point_jacobian_inf',1e-9),('NumPy versus native point acceleration bias','native_point_bias_inf',1e-8)]+checks
        table('dynamics_consistency',('Native and ' if native else '')+'sampled Go2 dynamics consistency',
              ['Quantity','Maximum','Limit'],[[label,sci(maxima.get(key)),sci(limit)]for label,key,limit in checks],
              f"{dyn['passed']}/{dyn['attempts']} state/support witnesses pass the recorded gates. Native Pinocchio status: {dyn['native']['status']}"+(f" (version {dyn['native']['version']})." if native else '; no native values substituted.')+
              ' Max-entry normalized metrics are coordinate-dependent. Point residuals are m/s2; force, mass and joint-acceleration entries otherwise mix scalar-coordinate SI units. Shared physical data verify formulation consistency, not hardware parameter truth.',
              'dynamics_raw.csv; summary.json; native_status.json')
        c=read(out/'curvature_summary.csv')
        table('curvature','Acceleration-curvature ablation at prescribed velocity scales',
              ['Speed scale','Configurations','Complete mapping max (m/s2)','Curvature omitted max (m/s2)'],[
                  [r['speed_scale'],r['configurations'],sci(r['full_max_m_s2']),f(r['omitted_max_m_s2'],6)]for r in c],
              f"{cfg['curvature_states']} configurations across two/three/four-foot support charts, reused at four speed scales. Both columns are physical acceleration-constraint residuals. The omitted-curvature route is an intentional component ablation, not a correctly formulated competing dynamics method. Full-term gate: 2e-6 m/s2.",
              'curvature_ablation.csv; curvature_inputs.json; curvature_summary.csv')
        paragraphs.append(f"The new sampled support-dynamics run accepted {dyn['passed']}/{dyn['attempts']} witnesses; the maximum normalized PACDM/KKT acceleration discrepancy is {sci(maxima.get('relative_acceleration_difference'))}. Native status is {dyn['native']['status']}. This is a new numerical dynamics computation at supplied reference configurations, not a rerun of the archived locomotion simulations.")
    save_csv(out/'benefit_summary.csv',benefits)
    # All generated tables have a structured source for workbook regeneration.
    (out/'tables.json').write_text(json.dumps(tables,indent=2)+'\n',encoding='utf-8')
    md=['# Go2 CMG/PACDM framework-benefit study',f"\nExecution status: **{summary['status']}**. Profile: **{summary['profile']}**.",
        '\n## Scope and provenance',
        'New compiler/evaluator/module-scheduler extension with the supplied PACDM core unchanged. The existing source package already has a Go2 importer and point-task/support adapters; the claim is not that they were absent. The new input removes authored foot joint memberships and support partitions and generates pruned task kernels and dependency-based scheduling. No Go2 physical coordinate reduction is claimed.',
        'The physical Go2 model is a floating tree. Foot tasks are virtual constraints. Support charts are ideal mode-dependent constraints and are not permanent structural loops. Conditional per-leg kinematics do not make coupled rigid-body dynamics or real contact independent.',
        '\n## Executed evidence']+paragraphs
    for t in tables:
        md+=['\n## '+t['title'],'| '+' | '.join(map(str,t['headers']))+' |','| '+' | '.join(['---']*len(t['headers']))+' |']
        md+=['| '+' | '.join(map(str,row))+' |'for row in t['rows']]
        md+=['\n'+t['note'],'\nSource: `'+t['source']+'`.']
    md+=['\n## Timing and numerical protocol',
        f"Route: 26 s / {cfg['route_samples']} samples / {cfg['repeats']} repeats. Method order randomized with saved seeds; one BLAS thread. PACDM keeps its original 1e-9 correction threshold and 1e-8 physical acceptance, TRF cost/step/gradient tolerances are 1e-11. Independent toe-gap gate is 1e-8 m, with bound, rank and tangent checks. Different solver stopping rules are not treated as identical work. No significance or real-time claim.",
        f"Recorded platform: {env['platform']}. Python {env['python'].split()[0]}, NumPy {env['versions']['numpy']}, SciPy {env['versions']['scipy']}, threadpoolctl {env['versions']['threadpoolctl']}. Reported processor: {env['processor_model']}; source: {env['processor_source']}; visible logical CPUs: {env['visible_logical_cpus']}. Host-exposed virtual CPU identification does not establish dedicated core allocation. Timing records remain associated with this recorded environment.",
        'Normalized error is max(abs(X-Xref))/max(1,max(abs(Xref))). This is coordinate-dependent numerical normalization, not an invariant physical percentage. Each analytical model call returns a combined residual/Jacobian; FD calls include every perturbed residual. Computation reuse at an identical argument is counted once. This differs from SciPy nfev.',
        '\n## Benefits and boundaries',
        'Benefits supported where observed: executable physical-graph/task/support compilation, verified generated derivatives, conditional small-block kinematics, exact local recomputation and predictor effects, and consistent constrained dynamics. All full-route and one/two/four-foot comparisons, including faster alternatives, remain in the tables and raw files.',
        'Not established: novel analytical differentiation or tree traversal in isolation; arbitrary-robot/compiler generality; global convergence; faster full coupled dynamics; correct contact switching, stable locomotion or hardware accuracy; superiority over native URDF+/generalized_rbda, which has not been ported or run here.',
        '\n## Reproduce',
        '```bat\nconda activate go2-native\npython -m pip install -r requirements.txt\npython run_go2.py --profile full --native required --out results_local\n```',
        'To run only the new native support-dynamics check: `python run_go2.py --profile full --stages native --native required --out results_native`. To reproduce the executed NumPy/SciPy path: `python run_go2.py --profile full --native off --out results_numpy`. No visualization, MuJoCo, ROS or GPU dependencies are needed.',
        '\n## Primary references',
        '- SciPy 1.17.0: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html',
        '- Pinocchio: https://github.com/stack-of-tasks/pinocchio (the shipped original backend is retained; native Go2 support adapter is new).',
        '- Go2 source provenance and pinned commit: `original/data/go2_cmg.json`; upstream XML and license retained.',
        '- URDF+ arXiv v1: https://arxiv.org/html/2411.19753v1 (context only; no executable comparison claimed).']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    # Direct HTML construction avoids dependence on markdown rendering packages.
    htmls=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Go2 framework benefit study</title><style>body{font:15px/1.6 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 26px;color:#193344}h1{font-size:30px}h2{margin-top:34px;border-bottom:1px solid #ccd7df;padding-bottom:7px}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #d9e1e7;text-align:left;vertical-align:top}th{background:#e7f0f3}tbody tr:nth-child(even){background:#f7fafb}.note{font-size:13px;color:#48616d}.source{font-family:monospace;font-size:11px;overflow-wrap:anywhere}pre{background:#eef3f6;padding:18px;white-space:pre-wrap}@media print{body{margin:15px}h2{break-after:avoid}tr{break-inside:avoid}}</style><body><h1>Go2 CMG/PACDM framework benefits</h1>',
           '<p><strong>Execution: '+html.escape(summary['status'])+'</strong> | profile '+html.escape(summary['profile'])+'</p>']
    for p in md[3:8]:
        if not p.startswith('#'):htmls.append('<p>'+html.escape(p)+'</p>')
    for p in paragraphs:htmls.append('<p>'+html.escape(p)+'</p>')
    for t in tables:
        htmls+=['<h2>'+html.escape(t['title'])+'</h2><table><thead><tr>',''.join('<th>'+html.escape(str(h))+'</th>'for h in t['headers']),'</tr></thead><tbody>']
        for row in t['rows']:htmls.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>')
        htmls+=['</tbody></table><p class="note">'+html.escape(t['note'])+'</p><p class="source">'+html.escape(t['source'])+'</p>']
    htmls+=['<h2>Protocol, environment and interpretation</h2><pre>'+html.escape('\n\n'.join(md[md.index('\n## Timing and numerical protocol')+1:]))+'</pre></body></html>']
    (out/'REPORT.html').write_text('\n'.join(htmls),encoding='utf-8')
    tex=['% Generated directly from this run. No special table packages required.','% Use in the paper after defining the framework/compiler extension.']
    for t in tables:
        tex+=[r'\begin{table*}[t]',r'\centering\small',r'\caption{'+escape(t['title'])+'.}',r'\label{tab:go2_'+t['key']+'}',
            r'\begin{tabular}{'+'l'+'r'*(len(t['headers'])-1)+'}',r'\hline',' & '.join(escape(h)for h in t['headers'])+r' \\',r'\hline']
        tex+=[' & '.join(value_tex(v)for v in row)+r' \\'for row in t['rows']]
        tex+=[r'\hline',r'\end{tabular}',r'\par\smallskip',r'\parbox{0.98\linewidth}{\footnotesize '+escape(t['note'])+'}',r'\end{table*}','']
    (out/'TABLES.tex').write_text('\n'.join(tex),encoding='utf-8')
    (out/'PAPER_TEXT.tex').write_text(paper_text(summary,env,protocol,benefits),encoding='utf-8')


def paper_text(summary,env,protocol,benefits):
    st=summary['stages'];cfg=protocol['configuration']
    text=[r'We evaluate a Go2 physical-graph compiler and conditional task-scheduling extension with the original PACDM core unchanged. The physical model is a floating tree: the twelve foot-target coordinates define virtual tasks, whereas selected stationary toe centers define temporary ideal-support charts. Unlike the Stewart representation, both Go2 formulations retain 18 physical and 30 augmented coordinates. Conditioning on the prescribed base pose permits the twelve dependent leg coordinates to be organized into four three-coordinate blocks; this does not decouple the rigid-body dynamics.']
    if 'pipeline'in st:
        r=st['pipeline'];text.append(f"Model-generation tests cover {r['variants']} predefined data/representation variants, each with one four-foot task plan and all eleven two-, three-, and four-foot ideal-support modes. The compiler generated {r['mode_plans']} mode plans, and {r['passed']}/{r['checks']} known-feasible configuration checks passed. Physical attachment data, inertias, actuator declarations and named seeds are supplied; leg memberships, point Jacobians and mode-specific independent/dependent partitions are generated.")
    if 'warm'in st:
        text.append(f"The continuation test uses the original 26-second reference route with {cfg['route_samples']:,} samples and {cfg['repeats']} timing repetitions. Every method continues from its own previous accepted solution. Times include assembly and the output mapping; initialization and independent physical checks are excluded equally. Method order is randomized and the BLAS thread limit is one. Medians and 95th percentiles are descriptive statistics, not independent robot success estimates or confidence intervals.")
    if 'derivatives'in st:
        text.append(f"The matched derivative-availability experiment uses {cfg['derivative_cases']} cases and {cfg['repeats']} repetitions and measures only the corrector. Each analytical/finite-difference pair uses the same TRF formulation, predictor, bounds, tolerances and linear solver. The sparse finite-difference route receives generated structural sparsity. PACDM retains its original stopping rules, while TRF cost, step and gradient tolerances are $10^{{-11}}$. Physical acceptance includes a $10^{{-8}}\\,\\mathrm{{m}}$ toe-gap gate and additional bounds, rank and tangent checks.")
    chosen=[b for b in benefits if (b['comparison']=='Generated evaluator versus supplied evaluator' or b['comparison']=='Predictor within compiled modular PACDM' or b['comparison']=='Generated derivative interface' or (b['condition'].startswith(('1 changed','2 changed')) and b['reference']=='modular_no_reuse'))]
    for b in chosen:
        direction='reduced' if b['reduction_percent']>=0 else 'increased'
        context={'Generated evaluator versus supplied evaluator':'Using the generated pruned point evaluator in the global PACDM route',
                 'Predictor within compiled modular PACDM':'Enabling the tangent predictor within the compiled modular route',
                 'Generated derivative interface':'Supplying the generated analytical derivative interface',
                 'Conditional task update':'Reusing unaffected task modules for '+b['condition'].split(',')[0]+' updates'}[b['comparison']]
        text.append(context+f" {direction} the measured median runtime by {abs(b['reduction_percent']):.1f}\\% relative to "+escape(NAMES.get(b['reference'],b['reference']))+'.')
    text.append(f"The run used Python {env['python'].split()[0]}, NumPy {env['versions']['numpy']}, SciPy {env['versions']['scipy']} and a {escape(env['machine'])} {escape(env['platform'].split('-')[0])} environment. The host exposed the processor string ``{escape(env['processor_model'] or 'not recorded')}'' and {env['visible_logical_cpus']} logical CPUs; this identifies a virtualized host, not dedicated bare-metal core allocation. These timings are separate from any later Windows native-validation run.")
    dyn=st.get('native',st.get('dynamics'))
    if dyn:
        mx=dyn['maxima'];text.append(f"Across {dyn['attempts']} state/support witnesses, the reduced dynamics passed {dyn['passed']} consistency checks against the independent physical-point KKT formulation, with maximum normalized acceleration discrepancy "+value_tex(sci(mx['relative_acceleration_difference']))+'. The ideal support charts have ranks 6, 9 and 12 and mobilities 12, 9 and 6 for two, three and four supporting feet, respectively. These are local fixed-anchor dynamics probes, not a contact-switching or locomotion-stability experiment.')
        if dyn['native']['status']=='completed':text.append('Native Pinocchio '+escape(dyn['native']['version'])+' was also executed; the maximum normalized native acceleration discrepancy was '+value_tex(sci(mx['native_acceleration_relative']))+'.')
        else:text.append('Native Pinocchio evaluation for this new Go2 benchmark was not executed in this run; the separately provided native stage must be run before adding a native-agreement claim.')
        text.append(f"A separate {cfg['curvature_states']}-configuration, four-speed-scale ablation evaluates $\\ddot q=N\\ddot q_a+c$ with and without the velocity-dependent curvature contribution. All {dyn['curvature_passed']}/{dyn['curvature_attempts']} complete-mapping cases satisfied the physical acceleration-constraint gate. The omitted contribution is an intentional component ablation, not a correctly formulated external dynamics baseline.")
    text.append(r'''For normalized discrepancies,
\[
\delta(X,X_{\mathrm{ref}})=\frac{\|X-X_{\mathrm{ref}}\|_{\max}}
{\max(1,\|X_{\mathrm{ref}}\|_{\max})},\qquad
\|X\|_{\max}=\max_{i,j}|X_{ij}|.
\]
For vectors the maximum-component definition is used. This is a numerical, coordinate-dependent normalization, not a unit-invariant physical percentage. Evaluation counts include fresh combined residual/Jacobian calls for analytical derivatives and every fresh residual call, including finite-difference perturbations, for numerical derivatives; they are not iteration counts.''')
    text.append(r'The results identify implementation and interface benefits within the tested Go2 class. Global and modular analytical TRF alternatives are retained with the same generated predictor information; no universal corrector-speed advantage is claimed. Shared physical parameters limit the dynamics conclusions to formulation consistency. The experiments do not establish arbitrary-robot generality, global convergence, stable unilateral contact, hardware accuracy, or superiority over URDF+/generalized\_rbda, which was not benchmarked.')
    return '\n\n'.join(text)+'\n'
