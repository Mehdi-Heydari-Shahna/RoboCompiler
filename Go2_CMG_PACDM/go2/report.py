"""Self-contained report and scientific figures from the saved Go2 evidence."""
from __future__ import annotations
import base64
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .task import PHASES,HURDLES

TEAL='#078f98'; AMBER='#df9b33'; INK='#193348'; MUTED='#657d90'
COLORS=['#079da5','#d49c2b','#637fc4','#bb6c9d']
LEGS=['FL','FR','RL','RR']; LIMITS=np.tile([23.7,23.7,45.43],4)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold',
                     'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#d2dce5','xtick.color':MUTED,'ytick.color':MUTED,
                     'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#e6edf1','grid.linewidth':.7})

def _read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else ({} if default is None else default)

def _embed(path):
    return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')

def _num(value):
    if isinstance(value,(float,np.floating)):
        if not np.isfinite(value):return str(value)
        return f'{value:.6g}'
    if isinstance(value,(list,dict)):return json.dumps(value)
    return str(value)

def _windows(log):
    active=np.linalg.norm(log['push'][:,:3],axis=1)>1e-8
    edges=np.diff(np.r_[False,active,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    t=log['time'];dt=np.median(np.diff(t))
    return [(float(t[a]),float(t[b-1]+dt)) for a,b in zip(starts,ends)]

def _time_axis(ax,t,log):
    ax.set_xlim(t[0],t[-1]);ax.set_xlabel('Time [s]');ax.grid(axis='y')
    for a,b,label in PHASES[1:]:ax.axvline(a,color='#cbd8e0',ls=':',lw=.8,zorder=0)
    for a,b in _windows(log):ax.axvspan(a,b,color=AMBER,alpha=.15,lw=0,zorder=0)

def _save(fig,path):
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig);return path

def motion_plot(root,log):
    t=log['time'];q=log['q'];ref=log['q_ref']
    fig,axes=plt.subplots(2,2,figsize=(13.6,8.1),layout='constrained')
    ax=axes[0,0]
    ax.plot(ref[:,0],ref[:,1],ls='--',lw=1.5,color=AMBER,label='PACDM body reference')
    ax.plot(q[:,0],q[:,1],lw=1.6,color=TEAL,label='Actual base')
    ax.scatter(q[0,0],q[0,1],c=TEAL,s=38,marker='o',zorder=4,label='Start')
    ax.scatter(ref[-1,0],ref[-1,1],c=INK,s=65,marker='x',zorder=4,label='Dock goal')
    for x,h in HURDLES:ax.axvspan(x-.025,x+.025,color=AMBER,alpha=.16)
    scene=root/'results/nominal.xml'
    if scene.exists():
        gate=ET.parse(scene).getroot().find(".//geom[@name='low_gate']")
        if gate is not None:
            gx=float(gate.get('pos').split()[0]);half=float(gate.get('size').split()[0])
            ax.axvspan(gx-half,gx+half,color=TEAL,alpha=.12,label='Low gate')
    ax.set(title='Course trajectory in the horizontal plane',xlabel='World x [m]',ylabel='World y [m]')
    ax.grid();ax.legend(frameon=False,fontsize=9,loc='upper left');ax.set_aspect('equal',adjustable='datalim')
    ax=axes[0,1]
    ax.plot(t,ref[:,2]*1000,color=AMBER,ls='--',lw=1.3,label='Reference')
    ax.plot(t,q[:,2]*1000,color=TEAL,lw=1.3,label='Actual base')
    ax.set(title='Height changes while walking',ylabel='Base origin height [mm]');ax.legend(frameon=False,fontsize=9)
    _time_axis(ax,t,log)
    ax=axes[1,0]
    error=np.linalg.norm(q[:,:3]-ref[:,:3],axis=1)*1000
    ax.plot(t,error,color=TEAL,lw=1.3,label='3D position error')
    ax.set(title='Tracking the prescribed base motion',ylabel='Base position error [mm]');_time_axis(ax,t,log)
    ax=axes[1,1]
    for i,(label,color) in enumerate(zip(['Yaw','Pitch','Roll'],COLORS[:3])):
        ax.plot(t,np.rad2deg(q[:,3+i]),color=color,lw=1.2,label=label)
        ax.plot(t,np.rad2deg(ref[:,3+i]),color=color,lw=.7,ls='--',alpha=.6)
    ax.set(title='Floating-base orientation',ylabel='ZYX chart angle [deg]');ax.legend(frameon=False,ncol=3,fontsize=9)
    _time_axis(ax,t,log)
    fig.suptitle('UNITREE GO2  /  MEASURED MOTION',fontsize=18,weight='bold')
    return _save(fig,root/'results/motion.png')

def contact_plot(root,log):
    t=log['time'];fig,axes=plt.subplots(2,2,figsize=(13.6,8.3),layout='constrained')
    ax=axes[0,0]
    for i,leg in enumerate(LEGS):ax.plot(t,log.get('support_force',log['normal_force'])[:,i],color=COLORS[i],lw=.85,label=leg)
    ax.set(title='Native ground-contact reaction',ylabel='Per-foot world-vertical reaction [N]');ax.legend(frameon=False,ncol=4,fontsize=9)
    _time_axis(ax,t,log)
    ax=axes[0,1]
    ratios=(np.abs(log['torque'])/LIMITS).reshape(-1,4,3).max(axis=2)
    for i,leg in enumerate(LEGS):ax.plot(t,100*ratios[:,i],color=COLORS[i],lw=.9,label=leg)
    ax.axhline(100,color='#a55454',ls='--',lw=1,label='Source limit')
    ax.set(title='Applied motor effort relative to limits',ylabel='Largest joint |torque| / limit per leg [%]')
    ax.legend(frameon=False,ncol=5,fontsize=8,loc='upper left',bbox_to_anchor=(0,.93));_time_axis(ax,t,log)
    ax=axes[1,0]
    dt=float(np.median(np.diff(t)))
    for i,leg in enumerate(LEGS):
        for mask,offset,height,color in [(log['stance'][:,i].astype(bool),-.20,.17,'#c1ccd5'),(log.get('support_force',log['normal_force'])[:,i]>2,.02,.17,COLORS[i])]:
            edges=np.diff(np.r_[False,mask,False].astype(int))
            spans=[(float(t[a]),float(t[b-1]+dt-t[a])) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
            ax.broken_barh(spans,(i+offset,height),facecolors=color,linewidth=0)
    ax.set(yticks=range(4),yticklabels=LEGS,ylim=(-.6,3.6));ax.set_title('Planned stance and measured support',pad=25)
    ax.invert_yaxis();_time_axis(ax,t,log)
    ax.text(.01,1.01,'Grey: planned stance   ·   Color: measured force > 2 N',transform=ax.transAxes,color=MUTED,fontsize=8)
    ax=axes[1,1]
    for i,(label,color) in enumerate(zip(['World x','World y','World z'],COLORS[:3])):ax.plot(t,log['push'][:,i],color=color,lw=1.3,label=label)
    ax.set(title='Declared disturbance applied to the base',ylabel='External force [N]');ax.legend(frameon=False,ncol=3,fontsize=9)
    _time_axis(ax,t,log)
    fig.suptitle('UNITREE GO2  /  CONTACT, ACTUATION AND DISTURBANCE',fontsize=18,weight='bold')
    return _save(fig,root/'results/contact.png')

def _cases(validation,metrics):
    data=validation.get('cases',{})
    items=data.items() if isinstance(data,dict) else [(str(i),v) for i,v in enumerate(data)]
    out=[]
    for name,value in items:
        if not isinstance(value,dict):continue
        value=value.get('metrics',value)
        if not isinstance(value,dict):continue
        if 'final_position_error_m' in value:out.append((str(value.get('name',name)),value))
    return out or [('nominal',metrics)]

def cases_plot(root,cases):
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.7),layout='constrained')
    names=[name.replace('_','\n') for name,_ in cases]
    colors=[TEAL if value.get('completed',True) else AMBER for _,value in cases]
    for ax,(key,scale,title,ylabel) in zip(axes,[('final_position_error_m',1000,'Final base position','3D goal error [mm]'),('min_base_height_m',1000,'Minimum base height','Base origin height [mm]'),('peak_torque_limit_fraction',100,'Peak commanded torque','Largest |torque| / source limit [%]')]):
        values=[float(v.get(key,np.nan))*scale for _,v in cases]
        positions=np.arange(len(values));finite=[x for x in values if np.isfinite(x)]
        logarithmic=key=='final_position_error_m' and finite and min(finite)>0 and max(finite)/min(finite)>50
        if logarithmic:
            ax.scatter(positions,values,c=colors,s=48,zorder=3)
            ax.set_xticks(positions,names);ax.set_xlim(-.6,len(values)-.4);ax.set_yscale('log')
            ylabel='3D goal error [mm, logarithmic scale]'
        else:ax.bar(positions,values,color=colors,width=.65);ax.set_xticks(positions,names)
        ax.set(title=title,ylabel=ylabel);ax.grid(axis='y');ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=8)
        if key=='peak_torque_limit_fraction':ax.axhline(100,color='#a55454',ls='--',lw=.9)
        for position,value in zip(positions,values):
            if np.isfinite(value):ax.annotate(f'{value:.2f}',(position,value),xytext=(0,7 if logarithmic else 4),textcoords='offset points',ha='center',fontsize=8,color=INK)
        ax.margins(y=.17)
    fig.suptitle('SIMULATION CASES  /  TEAL: REACHED DURATION · AMBER: EARLY STOP',fontsize=16,weight='bold')
    return _save(fig,root/'results/cases.png')

def _check_rows(checks):
    rows=[];items=checks.items() if isinstance(checks,dict) else enumerate(checks)
    for name,record in items:
        if isinstance(record,dict):
            passed=record.get('passed',record.get('pass'));label=str(record.get('name',name)).replace('_',' ')
            value=_num(record.get('value',record.get('measured','')));unit=str(record.get('unit',''))
            limit=record.get('limit',record.get('tolerance'));annotation='' if limit is None else f" ({record.get('relation','≤')} {_num(limit)})"
            text=f'{value} {unit}{annotation}'
        else:passed=record if isinstance(record,bool) else None;label=str(name);text=_num(record)
        state='PASS' if passed is True else 'FAIL' if passed is False else 'RECORDED';color='ok' if passed is True else 'fail' if passed is False else ''
        rows.append(f'<tr><th>{html.escape(label)}</th><td><span class="tag {color}">{state}</span> {html.escape(text)}</td></tr>')
    return ''.join(rows)

def _case_table(cases):
    rows=[]
    for name,v in cases:
        row=[html.escape(name.replace('_',' ')),'Yes' if v.get('completed') else 'No',f"{v.get('final_position_error_m',np.nan)*1000:.3f} mm",f"{np.rad2deg(v.get('max_tilt_rad',np.nan)):.3f}°",f"{v.get('peak_torque_limit_fraction',np.nan)*100:.2f}%",str(v.get('unexpected_contact_instances','—'))]
        rows.append('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>')
    return '<table><thead><tr><th>Case</th><th>Completed</th><th>Final 3D error</th><th>Peak tilt norm</th><th>Motor limit usage</th><th>Unintended contacts</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'

def build_report(root):
    root=Path(root);results=root/'results'
    with np.load(results/'nominal.npz',allow_pickle=False) as saved:log={k:saved[k] for k in saved.files}
    validation=_read(results/'validation.json');metrics=_read(results/'nominal.json');reference=_read(results/'reference.json');provenance=_read(root/'upstream/PROVENANCE.json')
    cases=_cases(validation,metrics);plots=[motion_plot(root,log),contact_plot(root,log),cases_plot(root,cases)]
    status=validation.get('passed');status_text='Saved validation: PASS' if status is True else 'Saved validation: FAIL' if status is False else 'Validation not yet recorded'
    passed=validation.get('passed_count',validation.get('passed_checks'));count=validation.get('check_count',validation.get('total_checks'))
    if passed is not None and count is not None:status_text+=f' · {passed}/{count} checks'
    status_html=f'<span class="tag {"ok" if status is True else "fail" if status is False else ""}">{status_text}</span>'
    cards=[('Final base 3D error',metrics.get('final_position_error_m',np.nan)*1000,'mm'),('Body tracking RMS',metrics.get('rms_body_error_m',np.nan)*1000,'mm'),('Peak motor limit usage',metrics.get('peak_torque_limit_fraction',np.nan)*100,'%'),('Duration',float(log['time'][-1]),'s')]
    card_html=''.join(f'<div class="metric"><span>{name}</span><strong>{value:.3f}<small>{unit}</small></strong></div>' for name,value,unit in cards)
    phase_html=''.join(f'<article><span>{a:g}–{b:g} s</span><b>{html.escape(name)}</b></article>' for a,b,name in PHASES)
    poster=results/'poster.png';poster_html=f'<img class="poster" src="{_embed(poster)}" alt="Recorded Unitree Go2 dynamics with measured foot contact and motor torque dashboard.">' if poster.exists() else ''
    ref_rows=''.join(f'<tr><th>{html.escape(name)}</th><td>{html.escape(_num(reference.get(key,"not recorded")))}</td></tr>' for key,name in [('samples','Assembled reference samples'),('physical_coordinates','Physical chart coordinates'),('virtual_coordinates','Virtual foot target coordinates'),('constraint_rank','Foot task rank'),('max_residual_inf','Largest position-constraint component [m]'),('max_tangent_residual','Largest J N component [declared SI chart]'),('min_rcond','Smallest passive solve reciprocal condition estimate'),('fallback_count','Continuation fallback count')])
    event_text=', '.join(f'{a:.2f}–{b:.2f} s' for a,b in _windows(log)) or 'none'
    commit=str(provenance.get('commit','not recorded'))
    template='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CMG / PACDM — Unitree Go2 contact course</title><style>
:root{--ink:#193348;--muted:#657d90;--teal:#008e99}*{box-sizing:border-box}body{margin:0;background:#f1f5f8;color:var(--ink);font:16px/1.62 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1170px;margin:auto;padding:48px 30px 64px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.15em;color:var(--teal);text-transform:uppercase}h1{font-size:clamp(31px,5vw,53px);line-height:1.07;letter-spacing:-.035em;margin:13px 0 19px}h2{font-size:24px;letter-spacing:-.02em;margin:0 0 17px}p{margin:0 0 17px}.lead{max-width:940px;font-size:18px;color:var(--muted)}.tag{display:inline-block;font-size:11px;font-weight:800;border-radius:6px;background:#e4ecf2;padding:4px 9px}.ok{background:#d9f1e6;color:#216444}.fail{background:#f9dfde;color:#a32a29}.poster{display:block;width:100%;margin:26px 0 0;border-radius:13px;box-shadow:0 12px 38px #14283c20}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:26px 0}.metric{background:white;border:1px solid #dfe8ef;border-radius:12px;padding:22px}.metric span{display:block;font-size:11px;color:var(--muted);font-weight:750;text-transform:uppercase}.metric strong{display:block;font-size:30px;margin-top:7px;letter-spacing:-.02em}.metric small{font-size:12px;color:var(--muted);margin-left:6px}.section{background:white;border:1px solid #dfe8ef;border-radius:14px;padding:30px;margin:22px 0}.phases{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.phases article{border-top:3px solid #58bfc5;padding-top:10px}.phases span{font-size:12px;font-weight:750;color:var(--teal)}.phases b{display:block;font-size:15px}.small{font-size:13px;color:var(--muted)}.performance{display:block;width:100%}.scope{display:grid;grid-template-columns:1fr 1fr;gap:24px}.scope article{border-left:3px solid #d3e7ec;padding-left:18px}.scope b{display:block;font-size:16px;margin-bottom:6px}.scope p{font-size:14px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 11px;border-bottom:1px solid #e4ecf2;text-align:left;vertical-align:top}th{font-weight:500;color:var(--muted);overflow-wrap:anywhere}td{font-variant-numeric:tabular-nums}thead th{font-size:11px;font-weight:750;text-transform:uppercase}.tablewrap{overflow-x:auto}details{border-top:1px solid #e2eaf0;padding-top:14px;margin-top:21px}summary{cursor:pointer;font-weight:650;color:var(--teal)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:17px;border-radius:8px;font-size:12px}.footer{font-size:12px;color:var(--muted)}a{color:var(--teal)}code{font-size:.86em}@media(max-width:760px){main{padding:28px 16px}.metrics,.phases{grid-template-columns:repeat(2,1fr)}.section{padding:20px}.metric{padding:16px}.scope{grid-template-columns:1fr}table{font-size:11px}th,td{padding:8px 5px}}@media print{body{background:white}main{padding:0}.section{break-inside:avoid}details{display:none}}
</style></head><body><main><div class="eyebrow">CMG / PACDM · floating-base locomotion benchmark</div>
<h1>CMG-based PACDM for Unitree Go2</h1>
<p class="lead">A twelve-motor quadruped follows a prescribed contact course: diagonal foot exchanges, low-rail clearance, turning and crouching under a low gate, external disturbances and a final precision stance. Its six base coordinates evolve through rigid-body dynamics and ground reactions.</p>
{{STATUS}}{{POSTER}}<div class="metrics">{{CARDS}}</div>
<p class="small">Metrics above are from the saved nominal run. Final error is the Euclidean error of the base origin in three dimensions. Motor usage is the largest commanded absolute torque divided by that motor's source effort limit.</p>
<section class="section"><h2>The recorded mission</h2><div class="phases">{{PHASES}}</div><p class="small" style="margin-top:22px">Logged disturbance windows: {{EVENTS}}. Amber intervals in the time plots mark applied external force. Each phase follows a prescribed route and support schedule.</p></section>
<section class="section"><h2>Measured motion and tracking</h2><img class="performance" src="{{MOTION}}" alt="Actual and reference horizontal paths, base height, 3D position error and ZYX orientation angles."><p class="small">The body reference and foot targets are task choices. PACDM solves the dependent leg coordinates. The measured motion is the result of motor torques, gravity and native MuJoCo contact.</p></section>
<section class="section"><h2>Contact changes and torque limits</h2><img class="performance" src="{{CONTACT}}" alt="Per-foot world-vertical support forces, motor limit fractions, planned stance versus measured support and applied disturbance forces."><p class="small">Measured support uses a 2 N threshold for visualization. The ground model has unilateral, frictional, compliant contacts; planned stance need not coincide exactly with physical contact onset. The plotted support force is the world-vertical component of native ground reaction, summed per foot. The separate normal_force log records contact-frame normal magnitudes, which can also include forces against rail sides.</p></section>
<section class="section"><h2>CMG / PACDM implementation</h2><div class="scope"><article><b>An underactuated floating base</b><p>The physical chart has eighteen coordinates: six base coordinates and twelve motor joints. Only twelve joint motors are commanded. The native MuJoCo model retains its free joint; the six scalar chart joints in the independent CMG / Pinocchio representation describe coordinates, not fictitious base actuators.</p></article><article><b>Contact modes across a branched tree</b><p>The Go2 mechanism is a branched kinematic tree. Stance relations create changing robot–environment constraints. This broadens coverage to floating-base hybrid contact; it does not turn the leg mechanism into a permanent mechanical closed chain.</p></article><article><b>PACDM reference assembly</b><p>Twelve virtual foot-target coordinates augment the eighteen physical coordinates. Four position closures contribute twelve equations. PACDM treats the base and foot targets as eighteen independent coordinates and assembles twelve dependent motor coordinates, with an analytic tangent map used to form velocity references.</p></article><article><b>Independent mechanics and control</b><p>Pinocchio evaluates mass and bias terms from the CMG tree. A constrained whole-body controller balances the six unactuated base equations and predicts support forces while respecting motor and friction bounds. MuJoCo determines actual contact forces and motion.</p></article></div></section>
<section class="section"><h2>Task-manifold evidence</h2><table><tbody>{{REFERENCE}}</tbody></table><p class="small" style="margin-top:18px">The foot-task residual is a position difference in metres. Tangent-map checks use the declared mixed SI chart. Reference closures are ideal task relations; compliant or rolling physical contact is evaluated separately in the native simulation. For generic independent point contacts, two-, three- and four-foot stance graphs have twelve, nine and six instantaneous physical degrees of freedom, respectively.</p></section>
<section class="section"><h2>Simulation cases and acceptance checks</h2><img class="performance" src="{{CASES_PLOT}}" alt="Final 3D error, minimum base height and peak motor limit usage across simulation cases."><div class="tablewrap">{{CASES}}</div><p class="small" style="margin-top:18px">Completion records whether a case reached its requested duration. Automated pass labels refer to the separately declared acceptance checks. An intentionally failing control can be accepted only when its associated check detects the expected failure.</p>{{ABLATION}}<details><summary>Individual acceptance checks</summary><table><tbody>{{ROWS}}</tbody></table></details><details><summary>Exact machine-readable validation record</summary><pre>{{JSON}}</pre></details></section>
<section class="section"><h2>Reproducibility and scope</h2><p>The source is the open MuJoCo Menagerie Unitree Go2 model. The package preserves upstream assets, licence, file hashes and commit <code>{{COMMIT}}</code>, alongside an explicit scalar CMG model, the PACDM implementation, the controller, reference, test cases and recorded states.</p><p>The benchmark overrides the upstream rubber-compression settings at the feet with <code>solref="0.008 1"</code> and <code>solimp="0.95 0.99 0.001"</code>; the original source XML remains unchanged. Native foot contact retains <code>condim="6"</code>, including torsional and rolling friction. The controller predicts three force components per foot and does not model these contact moments, so its force allocation approximates the native contact law.</p><p>The base uses an XYZ translation and ZYX Euler-angle chart. This chart becomes singular at pitch ±90°; the benchmark motion and validation stay within a regular local chart. MuJoCo's native quaternion state is converted with the corresponding velocity and acceleration transformations for independent mechanics checks.</p><p>Pinocchio validates and supplies rigid-body mechanics. These comparisons do not claim that Pinocchio reproduces the MuJoCo contact trajectory. The contact properties, disturbance sizes, route and support timing are benchmark parameters rather than identified hardware behavior. This is a prescribed simulation course, not autonomous terrain perception or a general locomotion policy.</p><p>No physical Go2 or hardware execution has been validated. The motor torques and controller are for this numerical benchmark. Reported success applies to the simulated cases and declared thresholds.</p><p class="small">The video replays saved native states without advancing physics. Its dashboard uses measured contact and commanded torque logs; the cyan ground trail is derived from recorded base positions. All three scientific figures and the validation record are embedded for offline viewing.</p></section>
<p class="footer">Numerical report generated from the recorded simulation and validation results. Model source: <a href="https://github.com/google-deepmind/mujoco_menagerie/tree/{{COMMIT}}/unitree_go2">MuJoCo Menagerie / Unitree Go2</a>.</p></main></body></html>'''
    replacement={'STATUS':status_html,'POSTER':poster_html,'CARDS':card_html,'PHASES':phase_html,'EVENTS':html.escape(event_text),
                 'MOTION':_embed(plots[0]),'CONTACT':_embed(plots[1]),'CASES_PLOT':_embed(plots[2]),'REFERENCE':ref_rows,
                 'CASES':_case_table(cases),'ABLATION':'<p class="small">'+html.escape(str(validation['ablation_scope']))+'</p>' if validation.get('ablation_scope') else '',
                 'ROWS':_check_rows(validation.get('checks',{})),'JSON':html.escape(json.dumps(validation,indent=2)),'COMMIT':html.escape(commit)}
    document=re.sub(r'\{\{([A-Z_]+)\}\}',lambda m:replacement[m.group(1)],template)
    output=results/'report.html';output.write_text(document,encoding='utf-8');return output
