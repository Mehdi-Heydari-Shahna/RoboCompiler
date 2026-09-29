"""Build figures and a self-contained HTML report from measured results."""
from pathlib import Path
import base64,csv,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contact_audit import load_trace
from validate_contact_task import CASES
ROOT=Path(__file__).resolve().parent

def build():
    a=load_trace('landing_nominal');r=json.loads((ROOT/'results/landing_nominal.json').read_text())
    validation=json.loads((ROOT/'results/complete_validation_v22.json').read_text())
    audit=json.loads((ROOT/'results/landing_nominal_audit.json').read_text())
    runs=[json.loads((ROOT/'results'/f'{n}.json').read_text()) for n,_,_ in CASES]
    figdir=ROOT/'results/figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
    t=a['time'];h=a['history'];stride=max(1,round(.0025/r['timestep_s']));hh=h[::stride]
    fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    axs[0,0].plot(t,a['q'][:,2],color='#174f78');axs[0,0].set(ylabel='Pelvis height (m)',title='Landing, 12 cm crouch and recovery')
    axs[0,1].plot(t,1000*a['q'][:,1],label='Lateral pelvis position',color='#174f78');axs[0,1].set(ylabel='Lateral position (mm)',title='Weight transfer and push response')
    for j,label,color in [(0,'Left','#bd6b06'),(1,'Right','#147fb3')]:axs[1,0].plot(t,a['foot_force'][:,j,2],label=label,color=color)
    axs[1,0].axhline(r['mass_kg']*9.81/2,ls='--',color='gray',label='Half body weight');axs[1,0].legend();axs[1,0].set(ylabel='Foot normal force (N)',title='Actual unilateral contact forces')
    axs[1,1].plot(hh[:,0],hh[:,19],color='#b66208');axs[1,1].axhline(5000,ls='--',color='#b83a32',label='Source force bound');axs[1,1].legend();axs[1,1].set(ylabel='Maximum |motor force| (N)',title='Twelve finite-response drives')
    axs[2,0].plot(hh[:,0],hh[:,3],label='Motor ports');axs[2,0].plot(hh[:,0],hh[:,4],label='Viscous loss');axs[2,0].plot(hh[:,0],hh[:,6],label='Contact');axs[2,0].set(ylabel='Mechanical power (W)',title='Signed power flow');axs[2,0].legend()
    for rr in runs[:2]:
        ar=load_trace(rr['name']);axs[2,1].plot(ar['time'],1000*ar['ledger'],label=f"dt = {rr['timestep_s']*1e6:g} microseconds")
    axs[2,1].set(ylabel='Energy residual (mJ)',title='Native-step energy ledger and refinement');axs[2,1].legend()
    for ax in axs.flat:
        ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.axvspan(6.7,6.95,color='#e76253',alpha=.15)
    fig.savefig(figdir/'contact_task_evidence.png');fig.savefig(figdir/'contact_task_evidence.svg');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    labels=['yaw','hip 2','hip 3','length','ankle 4','ankle 5']
    for side in range(2):
        for j,label in enumerate(labels):axs[0,side].plot(t,a['act'][:,side*6+j],label=label,lw=1)
        axs[0,side].set(title=('Left' if side==0 else 'Right')+' actuator forces',ylabel='Force (N)',xlabel='Time (s)');axs[0,side].legend(ncol=3,fontsize=8)
    x=np.arange(12);axs[1,0].bar(x-.18,a['motor_positive_work'][-1],.36,label='Delivered work');axs[1,0].bar(x+.18,-a['motor_negative_work'][-1],.36,label='Absorbed work');axs[1,0].set_xticks(x,[s+' '+l for s in ['L','R'] for l in labels],rotation=60,ha='right');axs[1,0].set(ylabel='Mechanical work (J)',title='Separate motoring and braking');axs[1,0].legend()
    axs[1,1].plot(hh[:,0],hh[:,15]*1e6,label='Point gap (micrometres)');axs[1,1].plot(hh[:,0],hh[:,16]*1e3,label='Universal dot error × 1000');axs[1,1].set(title='Closed-loop constraint errors',xlabel='Time (s)');axs[1,1].legend()
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(figdir/'actuators_and_closures.png');plt.close(fig)
    fields=['name','timestep_s','maximum_motor_force_N','maximum_ground_normal_N','maximum_penetration_m','maximum_tilt_deg','final_tilt_deg','maximum_final_base_speed_m_s','maximum_universal_dot','maximum_energy_ledger_error_J']
    with (ROOT/'results/tables/trial_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:rr[k] for k in fields} for rr in runs)
    def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    def img(p):return '<img alt="Measured simulation results" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
    trialtable=table(['Trial','Peak drive (N)','Peak ground (N)','Final tilt (deg)','Max energy residual (J)'],[[rr['name'],f"{rr['maximum_motor_force_N']:.1f}",f"{rr['maximum_ground_normal_N']:.1f}",f"{rr['final_tilt_deg']:.3f}",f"{rr['maximum_energy_ledger_error_J']:.5f}"] for rr in runs])
    energytable=table(['Mechanical ledger component','Signed work (J)'],[[k,f'{v:+.6f}'] for k,v in r['work_J'].items()]+[['Change in stored mechanical energy',f"{r['energy_change_J']:+.6f}"],['Final ledger residual',f"{r['final_energy_ledger_error_J']:+.6f}"]])
    audittable=table(['Independent check','Maximum error'],[[k,f'{v:.6g}'] for k,v in audit['errors'].items()])
    gatetable=table(['Suite / gate','Result','Value','Limit'],[[g.get('suite','')+'/'+g['name'],'PASS' if g['passed'] else 'FAIL',f"{g['value']:.6g}" if 'value'in g else '',f"{g['limit']:.6g}" if 'limit'in g else ''] for g in validation['gates']])
    Wplus=float(a['motor_positive_work'][-1].sum());Wminus=float(a['motor_negative_work'][-1].sum())
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Kangaroo v22 — complete contact benchmark</title>
<style>body{{font:17px/1.55 system-ui,sans-serif;color:#17283a;max-width:1160px;margin:32px auto;padding:0 24px;background:#f4f7fa}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:38px}}.card{{padding:22px;background:white;border:1px solid #d9e3ed;border-radius:12px}}.pass{{color:#087454;font-weight:700}}.note{{border-left:5px solid #bd7b14;padding:12px 18px;background:#fff4dd}}img{{width:100%;height:auto;background:white;margin:18px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;background:white}}th,td{{border-bottom:1px solid #d9e3ed;padding:9px;text-align:left}}th{{background:#e8f0f6}}code{{background:#e5edf4;padding:2px 5px}}details{{margin:24px 0}}a{{color:#175c94}}</style>
<h1>Kangaroo v22<br>Whole-body contact, actuation, forces and energy</h1>
<p class="pass">{validation['status']} — {validation['gates_passed']} / {validation['gates_total']} required gates passed.</p>
<div class="card"><b>What this benchmark demonstrates.</b> The Kangaroo prototype model lands from a 5 cm release, crouches by 12 cm, shifts its weight to both sides while yawing, rises, and recovers from a 50 N peak sideways torso load. The pelvis is free. Control enters twelve physical prismatic force ports. Native MuJoCo contact supports the robot; no foot weld or post-initialization state projection is used.</div>
<p class="note"><b>Model assumptions:</b> the benchmark uses the v20 <em>published-cut reconstruction</em>, not an author-verified hardware model. Drive response, controller gains, floor friction and contact compliance are assumptions; electrical motor, battery, thermal and physical pin-load calibration are outside its scope.</p>
<h2>Model and workflow</h2><p>78 rigid bodies, 42.214867608 kg, 76 internal scalar coordinates, 24 loop cuts and a floating pelvis. The prototype includes a fixed torso and two complete legs; the source model has no articulated arms or head. Six linear force ports per leg retain the source ±5000 N bounds, joint position limits, armature and viscous damping.</p>
<p>The unchanged PACDM solver assembles and continues the closed-chain configuration in a 140-coordinate graph containing 64 massless closure-chart coordinates. The task path has 401 accepted samples with all 144 residual rows checked and rank 128. Native integration then evolves all physical tree coordinates, contacts and twelve force-activation states. PACDM is the reference and closure-mapping layer; MuJoCo is the trajectory/contact integrator; independent NumPy and Pinocchio dynamics check the mechanics.</p>
<h2>Measured task behavior</h2>{img(figdir/'contact_task_evidence.png')}{trialtable}
<p>The harder trial releases from 8 cm and applies an 80 N peak push. Separate trials halve floor friction from 0.8 to 0.4 and double actuator response time from 2 ms to 4 ms. These are deterministic parameter checks, not a statistical robustness guarantee. The red chart band is the disturbance interval, 6.70–6.95 s. Its smooth sine-squared profile has impulse 6.25 N·s nominally and 10 N·s in the harder trial.</p>
<h2>All twelve actuators and closed loops</h2>{img(figdir/'actuators_and_closures.png')}
<p>Force commands update at 1 kHz, pass through a 1.5 MN/s command slew bound and an exact first-order force response, and remain within source force bounds. Mechanical power uses actual force times actual prismatic speed. Positive work is {Wplus:.6f} J; absorbed work is {-Wminus:.6f} J. Their difference is net motor work, not battery consumption.</p>
<h2>Complete mechanical energy ledger</h2>{energytable}
<p>Stored energy sums every rigid body's translational and rotational kinetic energy, gravitational potential and source armature kinetic energy. Native-step trapezoidal integration accounts separately for motor work, passive viscous loss, loop-constraint work, contact work, active-limit work and the known torso disturbance. Neither loop nor contact work is silently set to zero. Constraint work is solver-dependent; the residual measures integration consistency, not hardware realism or a passivity certificate.</p>
<p>The initial 10 ms contact setting produced positive net contact work. The included parameter-development record shows this limitation. The final 3 ms setting reduced penetration and produced dissipative net contact work in the landing sensitivity test. Time refinement keeps contact and loop parameters fixed. The electrical-energy CSV is only an efficiency-scenario calculation; it excludes idle consumption, electronics, heating and battery losses.</p>
<h2>Independent force and body checks</h2>{audittable}
<p>All 78 individual body energies are exported. Per-body CSV/NPZ records include actuator force-pair wrenches, external contact wrenches and native tree-parent reaction wrenches, expressed in world axes about each body's COM. Full generalized reactions are also saved. Redundant closed-loop constraint multipliers and tree reactions depend on the chosen representation and regularization; these are not uniquely identified physical bearing or pin loads.</p>
<p>Force-to-generalized-force mapping is checked independently using equal-and-opposite actuator force pairs and contact wrenches. Source and Pinocchio mass matrices, bias forces and energies are compared at logged task states. The source/native equation residual is scaled before cancellation; millinewton-level differences can arise from the source-to-native principal-inertia serialization in very stiff internal modes.</p>
<h2>Negative controls and retained baseline</h2><p>Disabling floor contact must cause a fall. Disabling loop constraints must produce a large loop gap. Disabling the drives must fail to maintain standing height. These are expected diagnostic failures, not successful task trials. The 136 accepted v20 mechanics/motion gates were rerun and retained, including URDF+RoboIR round trips, three gravity vectors, reduced/KKT agreement and free-body conservation.</p>
<h2>Reproduce and inspect</h2><p>Run <code>python verify_package.py</code> to check the files without simulation dependencies. In the prepared Python environment, use <code>python run_kangaroo_v22.py</code> for the entire benchmark, or <code>python run_kangaroo_v22.py --reuse-results</code> to recheck included results. Add <code>--visuals</code> to rebuild the report and MP4. The standalone MP4 replays the actual logged states; variable playback speeds are displayed.</p>
<details><summary><b>All {validation['gates_total']} required gates</b></summary>{gatetable}</details>
<h2>Sources and model limitations</h2><p>Original source: <a href="https://github.com/hucebot/mujoco_kangaroo_sim2sim">hucebot/mujoco_kangaroo_sim2sim</a>, pinned commit <code>c020b68f3690930f5228d9f301b59ad9ab405e9a</code>. The reconstruction hypothesis is recorded in the CMG metadata; source assets and license are preserved.</p>
<p>The <a href="https://arxiv.org/html/2312.04161v2">Kangaroo constrained-dynamics paper</a> motivates squat/landing mechanics and the parallel leg topology. Implementation references: <a href="https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#actuator-general">MuJoCo 3.3.7 force activation</a> and <a href="https://mujoco.readthedocs.io/en/3.3.7/computation/index.html">MuJoCo computation and contact model</a>. This release does not establish walking, autonomous terrain negotiation, powered jump takeoff, full-body self-collision avoidance, or hardware performance.</p></html>'''
    (ROOT/'Kangaroo_v22_report.html').write_text(doc,encoding='utf-8')
    summary=f"""# Kangaroo v22 measured results\n\n{validation['status']}: {validation['gates_passed']}/{validation['gates_total']} required gates.\n\nNominal task: 5 cm release, 12 cm crouch, lateral weight transfer and yaw, rise, 50 N peak side push, recovery.\n\n- Peak motor force: {r['maximum_motor_force_N']:.3f} N (source bound 5000 N).\n- Maximum point gap: {r['maximum_loop_gap_m']:.6g} m.\n- Maximum universal dot error: {r['maximum_universal_dot']:.6g}.\n- Final tilt: {r['final_tilt_deg']:.6f} degrees.\n- Maximum energy ledger residual: {r['maximum_energy_ledger_error_J']:.6f} J.\n- Positive / absorbed mechanical motor work: {Wplus:.6f} / {-Wminus:.6f} J.\n\nThe self-contained HTML report contains all gate results, comparisons, energy terms and qualifications. Raw numerical data are in results/. The model is an explicitly documented reconstruction, not hardware-validated.\n"""
    (ROOT/'RESULTS.md').write_text(summary,encoding='utf-8')
    print('Figures, tables, and self-contained report built',flush=True)

if __name__=='__main__':build()
