"""Generate result narratives and publication tables directly from raw summaries."""
from __future__ import annotations
import csv,html,json
from pathlib import Path

LABELS={
'PACDM_original_chart':'PACDM, original free-platform chart',
'PACDM_compiled_tree':'PACDM, physical-graph compiled tree',
'PACDM_compiled_no_predictor':'Compiled PACDM, predictor removed',
'TRF_compiled_analytic_predictor':'TRF, same compiled equations + predictor',
'TRF_analytic_dense':'Generated analytical Jacobian + dense TRF',
'TRF_FD3_dense':'3-point finite differences + dense TRF',
'TRF_analytic_sparse':'Generated analytical Jacobian + sparse TRF',
'TRF_FD3_colored':'Sparsity-colored 3-point differences + sparse TRF'}


def fmt(v,dec=3):
    if v is None:return 'not available'
    if isinstance(v,float):
        if v and (abs(v)<1e-3 or abs(v)>=1e5):return f'{v:.3e}'
        return f'{v:.{dec}f}'
    return str(v)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(fmt(v) for v in r)+' |' for r in rows])


def tex_table(headers,rows,caption,label):
    def esc(x):
        return fmt(x).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&')
    lines=['\\begin{table*}[t]','\\centering','\\small',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
           '\\begin{tabular}{'+'l'*len(headers)+'}','\\hline',' & '.join(esc(x) for x in headers)+' \\\\','\\hline']
    lines+=[' & '.join(esc(x) for x in r)+' \\\\' for r in rows]
    lines+=['\\hline','\\end{tabular}','\\end{table*}','']
    return '\n'.join(lines)


def build_report(out,summary):
    out=Path(out);parts=[];tables=[];benefits=[]
    parts += ['# Stewart CMG/PACDM — framework-benefit experiment',
        f'**Execution status:** {summary["status"]}. **Profile:** {summary["profile"]}.',
        '**Provenance:** These measurements were produced by this package. The graph compiler is in `src/compiler.py`; the original PACDM core remains unchanged under `prior_benchmark/original/vendor/`.',
        '**Scope:** Stewart only. Physical-graph lowering, numerical assembly, analytic derivatives, and sampled rigid-body dynamics. No new closed-loop/contact physics rollout or hardware experiment. The prior benchmark and its unfavorable results remain intact in `prior_benchmark/`.',
        '## 1. Physical graph to constraint/model pipeline',
        'The input contains 21 physical bodies including ground, 25 physical joints, six declared actuators, joint attachments and inertias. Its graph has five fundamental cycles. The compiler keeps scalar/fixed joints first and includes the lexically first admissible spherical joint in the tree. The remaining five spherical joints become closures. A spherical tree joint is represented by three rotation-chart coordinates and two massless frames. No spanning-tree labels, cut selection, closure functions or Jacobian functions are supplied in the input.',
        'The original six-cut model is also mechanically valid: its extra free-platform coordinate chain is a computational representation, not six physical joints. Six cuts in that representation must not be confused with six independent cycles of the physical mechanism.',
        'This compiler supports the tested fixed/scalar/spherical-joint class and spherical chords. General CAD import, optimal tree selection and arbitrary-joint lowering are outside this scope. Independent actuators, geometry, inertias and seeds are given as input. Topological cycle count is not substituted for numerical constraint rank.']
    headers=['Quantity','Original representation','Graph-compiled representation']
    rows=[['Tree coordinates',24,21],['Free closure-orientation coordinates',18,15],['Total augmented coordinates',42,36],
          ['Dependent coordinates / augmented residual rows',36,30],['Physical point residual rows',18,15],
          ['Independent actuator coordinates',6,6],['Massless chart frames',5,2]]
    parts.append(table(headers,rows));tables.append(tex_table(headers,rows,'Equivalent Stewart representations; counts are not independent physical cycle counts.','tab:stewart_dimensions'))
    benefits += [dict(benefit='Smaller generated assembly representation',comparison='Original valid free-platform chart',
        evidence='42 -> 36 augmented coordinates; 36 -> 30 dependent coordinates/residual rows',boundary='Different equivalent parameterizations; not a comparison with every possible hand-optimized model')]
    if 'pipeline' in summary:
        s=summary['pipeline'];parts += [f'**Pipeline witnesses:** {s["passed"]}/{s["attempts"]} passed across {s["models"]} compiled model/tree combinations. Median compilation time was {fmt(s["median_compile_ms"])} ms.',
            'The full profile uses 12 predefined Stewart data/representation variants, all six possible spherical tree choices, and five witness configurations per model. These are generate-and-verify probes with known feasible witnesses, not cold-start convergence trials. Checks cover all six physical spherical joints, body poses, off-manifold derivatives, task mappings, reduced inertia, potential energy and generated sparsity. No timing-based tree selection is performed.',
            'Geometry/inertia variants, body-frame reexpression, record reordering and actuator-order changes modify data, not the compiler or PACDM source. This demonstrates executable automation/reuse, not measured human engineering hours.']
        benefits.append(dict(benefit='Automatic model generation and data-only adaptation',comparison='Manually selected input tree/cuts and mechanism-specific derivative coding',
            evidence=f'{s["models"]} compiled models; {s["passed"]}/{s["attempts"]} checks',boundary='Structural input/workflow comparison, not a timed human-user study or native URDF+ measurement'))
    if 'warm' in summary:
        s=summary['warm'];parts += ['## 2. Continuation: equivalent representations and predictor ablation',
          'Same original 22-second commanded-length route; each method uses its own previous accepted state, never the target solution. Three repeats are used in the full profile. Times include assembly and output mapping, exclude independent physical checking, and retain hold samples. Method order is randomized per sample. The compiled and original coordinates/bounds are different parameterizations; physical outputs and accepted targets are checked independently.']
        headers=['Method','Accepted / attempts','Median ms','P95 ms','Max physical gap (m)']
        order=['PACDM_original_chart','PACDM_compiled_tree','PACDM_compiled_no_predictor','TRF_compiled_analytic_predictor']
        by={r['method']:r for r in s['summary']};rows=[]
        for k in order:
            r=by[k];rows.append([LABELS[k],f'{r["success"]}/{r["attempts"]}',r['total_ms_median'],r['total_ms_p95'],r['gap_m_max']])
        parts.append(table(headers,rows));tables.append(tex_table(headers,rows,'Stewart continuation: assembly plus mapping; repeated samples are not independent robot experiments.','tab:stewart_continuation'))
        original=by['PACDM_original_chart']['total_ms_median'];compiled=by['PACDM_compiled_tree']['total_ms_median'];nop=by['PACDM_compiled_no_predictor']['total_ms_median'];trf=by['TRF_compiled_analytic_predictor']['total_ms_median']
        if compiled<original:
            text=f'The graph-compiled representation reduced measured median PACDM assembly-plus-mapping time by {100*(1-compiled/original):.1f}% ({original/compiled:.2f}x speed ratio) relative to the original representation.'
            benefits.append(dict(benefit='Measured representation-level continuation benefit',comparison='Same PACDM core with original free-platform chart',evidence=text,boundary='Local implementation timing; includes parameterization/conditioning effects, not complexity or universal speed superiority'))
            parts.append('**Observed benefit:** '+text)
        else:parts.append('The smaller representation did not reduce median continuation time in this run; the dimensional benefit remains separate from runtime.')
        if compiled<nop:
            text=f'Removing the predictor increased median time from {fmt(compiled)} to {fmt(nop)} ms; enabling it reduced median time by {100*(1-compiled/nop):.1f}% in this implementation.'
            parts.append('**Predictor ablation:** '+text);benefits.append(dict(benefit='Predictor reduces continuation work',comparison='Same compiled PACDM with predictor removed',evidence=text,boundary='Component ablation, not novelty or superiority over other predictors'))
        parts.append(f'**Strong baseline retained:** TRF given the same generated equations, analytical Jacobian and predictor had a median of {fmt(trf)} ms. The relative speed of PACDM and this strong corrector depends on the condition. Solver tolerances are original PACDM tolerances versus TRF 1e-11; common physical acceptance is imposed. Full, moving-only and per-repeat summaries are all saved.')
    if 'derivatives' in summary:
        s=summary['derivatives'];parts += ['## 3. Benefit of generated analytical derivatives',
          'This is a matched derivative-availability experiment, NOT PACDM-versus-all-optimization algorithms. Each pair uses the same bounded SciPy TRF corrector, equations, initial prediction, tolerances and linear solver. Dense and sparsity-colored three-point finite-difference baselines are both included. Finite-difference residual evaluation does not calculate and discard an analytical Jacobian. The colored baseline receives graph-derived structural sparsity, making it stronger than an unassisted black-box workflow. Times below are corrector-only; shared upstream prediction and downstream mapping/checking are excluded equally.']
        by={r['method']:r for r in s['summary']};headers=['Derivative route','Accepted / attempts','Median ms','P95 ms','Median oracle evaluations'];rows=[]
        for k in ['TRF_analytic_dense','TRF_FD3_dense','TRF_analytic_sparse','TRF_FD3_colored']:
            r=by[k];rows.append([LABELS[k],f'{r["success"]}/{r["attempts"]}',r['corrector_ms_median'],r['corrector_ms_p95'],r['oracle_evaluations_median']])
        parts.append(table(headers,rows));tables.append(tex_table(headers,rows,'Matched derivative-availability comparisons on the same compiled Stewart equations.','tab:stewart_derivatives'))
        for a,b,title in [('TRF_analytic_dense','TRF_FD3_dense','dense finite differences'),('TRF_analytic_sparse','TRF_FD3_colored','sparsity-colored finite differences')]:
            va=by[a]['corrector_ms_median'];vb=by[b]['corrector_ms_median']
            if va<vb:
                text=f'Against {title}, generated analytical derivatives reduced median corrector time by {100*(1-va/vb):.1f}% ({vb/va:.2f}x speed ratio).'
                parts.append('**Observed benefit:** '+text);benefits.append(dict(benefit='Generated analytic-derivative availability',comparison=title,evidence=text,boundary='Not unique to PACDM; an equally correct analytic/automatic-differentiation backend can share this benefit'))
        parts.append('The advantage is that the graph pipeline supplies verified derivative information without mechanism-specific Jacobian functions; it is not evidence that analytical differentiation is novel or that another graph/automatic-differentiation framework cannot do the same.')
    dyn=summary.get('dynamics',summary.get('native'))
    if dyn:
        parts+=['## 4. Consistent mappings, dynamics and acceleration curvature',
            f'**Dynamics witnesses:** {dyn["passed"]}/{dyn["attempts"]} passed. The compiled PACDM route is compared both with a KKT solve in the compiled coordinates and with the original free-platform model through physical body velocities and accelerations. Both NumPy models use the same declared physical inertias; this is formulation/coordinate consistency, not independent hardware identification.']
        headers=['Quantity','Maximum'];rows=[[k.replace('_',' '),v] for k,v in dyn['maxima'].items()]
        parts.append(table(headers,rows));tables.append(tex_table(headers,rows,'Sampled Stewart dynamics consistency; body acceleration norms contain mixed translational/angular SI components.','tab:stewart_dynamics'))
        parts+=[f'**Curvature ablation:** {dyn["curvature_full_passed"]}/{dyn["curvature_ablations"]} full-curvature cases satisfy the physical acceleration constraint. The intentionally omitted-curvature residuals, at 0.5x, 1x, 2x and 4x prescribed speed, are retained in `curvature_ablation.csv`. This demonstrates the role of the acceleration interface, not an advantage over a correctly formulated KKT method.',
            '**Native Pinocchio status:** `'+json.dumps(dyn['native'])+'`.']
        benefits.append(dict(benefit='Generated dynamics/motion interface remains consistent',comparison='Independent physical-point KKT and original coordinate representation',evidence=f'{dyn["passed"]}/{dyn["attempts"]} states passed',boundary='Correctness/integration benefit (not a speed or hardware-accuracy comparison)'))
    if 'verification' in summary:
        s=summary['verification'];parts+=['## 5. Verification layer',f'{s["passed"]}/{s["tests"]} designed checks passed, including 13 explicit invalid-model/invalid-chart cases and one valid edge-direction reversal. These are unit/precondition tests, not a real-world fault-detection rate.']
    parts+=['## 6. Scope and boundaries',
        '**Supported focus:** physical-graph-to-model automation within the supported Stewart class; reduced representation size; measured representation/predictor benefits where observed; generated derivative availability versus finite-difference workflows; and verified motion/dynamics consistency.',
        '**Not established:** unique novelty of standard spanning trees or analytical differentiation; superiority over all good solvers; a native URDF+/generalized_rbda comparison; module-level speedup (Stewart produces one coupled candidate module); arbitrary topology/robot generality; global convergence; hardware accuracy; or real-time operation.',
        'URDF+ arXiv v1 already describes automatic grouping and constraint Jacobians. It requires tree joints and loop joints to be specified. The new input contract here removes that authored tree/cut choice for the supported graph class, but URDF+ can represent the resulting smaller tree too. No inability or runtime disadvantage of URDF+ is inferred.',
        '## 7. Reproduction',
        '```bash\npython -m pip install -r requirements.txt\npython run_stewart.py --profile full --native off --out results_local\n```',
        'Use a fresh output folder. For native Pinocchio after installing it, run `python run_stewart.py --profile full --stages native --native required --out results_native`. The native integration code is included; a `not_run` status means it was not executed in this environment. The previous full benchmark is reproducible separately with `python prior_benchmark/run_comparison.py --profile full --native off --out prior_reproduction`.',
        'All source hashes, input model hashes, protocol, environment, raw cases, failures, generated models, CSV tables and saved states are included. Timings vary by hardware. No statistical significance or independent repeated-robot success rate is asserted.',
        '## References and evidence locations',
        '- Raw evidence: CSV/JSON/NPZ files adjacent to this report; generated physical graphs, topology plans and compiled models under `generated/`.',
        '- Unchanged algorithm source: `prior_benchmark/original/vendor/pacdm_original.py`.',
        '- Original comparison and limitations: `prior_benchmark/results_here/REPORT.md`.',
        '- URDF+ paper, arXiv v1: https://arxiv.org/html/2411.19753v1 (Sections III-A, III-C and IV).',
        '- SciPy 1.17.0 least_squares: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html',
        '- Native Pinocchio v3.8.0 reference: https://github.com/stack-of-tasks/pinocchio/blob/v3.8.0/examples/simulation-closed-kinematic-chains.py',
        '- URDF+ associated library, NOT benchmarked: https://github.com/ROAM-Lab-ND/generalized_rbda']
    md='\n\n'.join(parts)+'\n';(out/'REPORT.md').write_text(md,encoding='utf-8')
    (out/'tables.tex').write_text('% Requires a two-column document for table*; no special table packages needed.\n\n'+'\n'.join(tables),encoding='utf-8')
    with (out/'benefits.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['benefit','comparison','evidence','boundary']);writer.writeheader();writer.writerows(benefits)
    (out/'benefits.json').write_text(json.dumps(benefits,indent=2)+'\n',encoding='utf-8')
    # No optional markdown dependency: produce a readable, printable HTML view.
    body=[];in_table=False;in_code=False
    for line in md.splitlines():
        if line.startswith('```'):
            if in_table:body.append('</tbody></table>');in_table=False
            body.append('</pre>' if in_code else '<pre>');in_code=not in_code;continue
        if in_code:body.append(html.escape(line)+'\n');continue
        if line.startswith('|'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if all(set(c)<=set('-: ') for c in cells):continue
            if not in_table:body.append('<table><tbody>');in_table=True
            body.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>');continue
        if in_table:body.append('</tbody></table>');in_table=False
        if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.strip():body.append('<p>'+html.escape(line)+'</p>')
    if in_table:body.append('</tbody></table>')
    style='body{font:15px/1.55 system-ui,Arial;max-width:1150px;margin:40px auto;padding:0 25px;color:#172b3a}h1{font-size:28px}h2{margin-top:35px;color:#174f63}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td{border:1px solid #cbd5df;padding:9px;vertical-align:top}tr:first-child{background:#e8f0f4;font-weight:bold}pre{background:#eff3f6;padding:15px;white-space:pre-wrap}p{overflow-wrap:anywhere}@media print{body{margin:0;font-size:11px}h2{break-after:avoid}table{font-size:10px}}'
    (out/'REPORT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Stewart framework benefits</title><style>'+style+'</style></head><body>'+'\n'.join(body)+'</body></html>',encoding='utf-8')
