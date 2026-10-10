#!/usr/bin/env python3
"""Generate four supplementary figures from prepared repository results.

Usage: python figures/make_figures.py [--data-dir DIR] [--output-dir DIR]
First run figures/prepare_plot_data.py. No physics engine is required for
plotting. See the generated data/provenance.json for each plotted quantity.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BLUE = '#17649A'
ORANGE = '#C97125'
GREEN = '#22806B'
PURPLE = '#775799'
INK = '#263238'
COLORS = [BLUE, ORANGE, GREEN, '#AA4C70', PURPLE, '#657C35']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#8B949A',
    'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.linewidth': .65, 'lines.linewidth': 1.25,
    'grid.color': '#D9E0E3', 'grid.linewidth': .5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300,
})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).resolve().parents[1]/'results/supplement_figures'
    p.add_argument('--data-dir', type=Path, default=default_out/'data',
                   help='Directory containing plot_data.npz and metrics.json.')
    p.add_argument('--output-dir', type=Path, default=default_out,
                   help='Destination for four vector PDFs and four PNGs.')
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.data_dir/'plot_data.npz')
    m = json.loads((args.data_dir/'metrics.json').read_text())
    n, r = m['nominal'], m['refined']
    t, tr = d['contact_t'], d['ref_t']

    def finish(fig, name):
        fig.savefig(out/f'{name}.pdf', bbox_inches='tight', pad_inches=.035,
                    metadata={'Title': name.replace('_', ' '),
                              'Subject': 'Recorded Kangaroo CMG/PACDM simulation evidence'})
        fig.savefig(out/f'{name}.png', bbox_inches='tight', pad_inches=.035)
        plt.close(fig)

    def panel(ax, label, title):
        ax.set_title(f'({label})  {title}', loc='left', pad=8, fontweight='medium')
        ax.grid(True, axis='y', zorder=0)
        ax.set_axisbelow(True)

    def time_axis(ax, push=True, end=10):
        ax.set_xlim(0, end)
        ax.set_xlabel('Time (s)')
        if push:
            ax.axvspan(6.70, 6.95, color=PURPLE, alpha=.13, linewidth=0, zorder=0)

    # Figure 1: reference CoM is independently reconstructed from CMG inertias
    # and all 401 compatible reference configurations; it is not the pelvis target.
    fig, a = plt.subplots(2, 2, figsize=(7.15, 4.45), layout='constrained')
    ax = a[0, 0]
    ax.plot(t, d['contact_com'][:, 2], color=BLUE, label='MuJoCo')
    ax.plot(tr, d['ref_com'][:, 2], color=ORANGE, ls='--', label='CMG reference')
    panel(ax, 'a', 'Vertical center-of-mass motion')
    ax.set_ylabel('CoM height (m)')
    ax.legend(loc='lower right', frameon=False)
    ax.annotate('5 cm release', xy=(0, d['contact_com'][0, 2]), xytext=(.9, .79),
                fontsize=7, arrowprops={'arrowstyle': '-', 'lw': .6, 'color': INK})
    ax = a[0, 1]
    ax.plot(t, d['contact_com'][:, 1]*1e3, color=BLUE)
    ax.plot(tr, d['ref_com'][:, 1]*1e3, color=ORANGE, ls='--')
    panel(ax, 'b', 'Lateral center-of-mass motion')
    ax.set_ylabel('CoM lateral position (mm)')
    ax.text(.98, .93, 'Shaded interval: 50 N push', ha='right', va='top',
            transform=ax.transAxes, fontsize=7, color=PURPLE)
    ax = a[1, 0]
    ax.plot(t, d['feet_Fz'][:, 0], color=BLUE, label='Left foot')
    ax.plot(t, d['feet_Fz'][:, 1], color=ORANGE, ls='--', label='Right foot')
    panel(ax, 'c', 'Native foot–ground contact forces')
    ax.set_ylabel('Normal force per foot (N)')
    ax.legend(loc='upper right', frameon=False)
    ax.text(.98, .58, f"Touchdown: {n['touchdown_s']:.3f} s\nFinal total support: {n['mean_final_ground_normal_N']:.2f} N",
            transform=ax.transAxes, ha='right', va='top', fontsize=7)
    ax = a[1, 1]
    ax.plot(t, d['contact_tilt_deg'], color=BLUE)
    panel(ax, 'd', 'Pelvis tilt and recovery')
    ax.set_ylabel('Tilt (deg)')
    ax.text(.03, .93, f"Final tilt: {n['final_tilt_deg']:.5f}°\nFinal position error: {n['final_position_error_m']*1e3:.4f} mm",
            transform=ax.transAxes, ha='left', va='top', fontsize=7)
    for ax in a.flat:
        time_axis(ax)
    finish(fig, 'Kangaroo_Task_Response')

    # Figure 2: use the actual port limits. Full-rate maxima are distinct from
    # downsampled histories and explicitly marked as such.
    fig, a = plt.subplots(2, 2, figsize=(7.15, 4.45), layout='constrained')
    for col, start in enumerate((0, 6)):
        ax = a[0, col]
        for k in range(6):
            i = start+k
            ax.plot(t, np.abs(d['motor_force'][:, i])/d['motor_bounds'][i],
                    color=COLORS[k], label=m['motor_labels'][i], lw=1.1,
                    ls='--' if k == 3 else '-')
        ax.axhline(1, color='#657078', ls=':', lw=.9)
        ax.set_ylim(0, 1.06)
        panel(ax, 'ab'[col], f"{'Left' if col == 0 else 'Right'}-leg actuator utilization")
        ax.set_ylabel(r'$|f_i| / f_{i,\max}$')
        ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, .94),
                  frameon=False, columnspacing=.9, handlelength=1.8)
        ax.text(.02, .955, 'Port limit', transform=ax.transAxes, fontsize=6.5, va='top')
    ax = a[1, 0]
    ax.semilogy(t, np.maximum(d['point_gap_m']*1e6, 1e-10), color=BLUE,
                label='5 ms output samples')
    full_gap = n['maximum_loop_gap_m']*1e6
    ax.axhline(full_gap, color=ORANGE, ls='--', lw=1)
    ax.text(.98, .98, f'Physics-step maximum: {full_gap:.3f} µm',
            transform=ax.transAxes, ha='right', va='top', color=ORANGE, fontsize=7)
    ax.set_ylim(1e-6, 1)
    panel(ax, 'c', 'Point-closure compatibility')
    ax.set_ylabel('Maximum point gap (µm)')
    ax = a[1, 1]
    ax.plot(t, d['universal_dot']*1e3, color=BLUE)
    full_dot = n['maximum_universal_dot']*1e3
    ax.axhline(full_dot, color=ORANGE, ls='--', lw=1)
    ax.text(.98, .95, f'Physics-step maximum: {full_dot:.3f} × 10⁻³',
            transform=ax.transAxes, ha='right', va='top', color=ORANGE, fontsize=7)
    ax.set_ylim(0, full_dot*1.32)
    panel(ax, 'd', 'Universal-axis compatibility')
    ax.set_ylabel(r'Maximum $|\mathbf{a}^{\mathsf{T}}\mathbf{b}|$ ($10^{-3}$)')
    for ax in a.flat:
        time_axis(ax)
    finish(fig, 'Kangaroo_Actuation_Closure')

    # Figure 3: work arrays were accumulated by the simulator at the physics
    # step; no quadrature of 5 ms power samples is used.
    fig, a = plt.subplots(2, 2, figsize=(7.15, 4.55), layout='constrained')
    ax = a[0, 0]
    ax.plot(t, d['motor_positive_work'].sum(axis=1), color=BLUE, label='Positive work')
    ax.plot(t, d['motor_absorbed_work'].sum(axis=1), color=ORANGE, ls='--', label='Absorbed work')
    ax.plot(t, d['motor_net_work'].sum(axis=1), color=GREEN, label='Net work')
    panel(ax, 'a', 'Cumulative actuator work')
    ax.set_ylabel('Mechanical work (J)')
    ax.legend(loc='upper left', frameon=False)
    time_axis(ax)
    ax = a[0, 1]
    keys = ['motor', 'passive', 'loop', 'contact', 'limit', 'disturbance']
    vals = [n['work_J'][k] for k in keys] + [n['energy_change_J']]
    labels = ['Actuator', 'Passive', 'Loop', 'Contact', 'Joint limit', 'Push', 'ΔE']
    y = np.arange(len(vals))
    ax.barh(y, vals, height=.62, color=[BLUE if v >= 0 else ORANGE for v in vals[:-1]]+[GREEN], zorder=2)
    ax.axvline(0, color='#7E898F', lw=.7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-33, 10.5)
    for yi, v in zip(y, vals):
        ax.text(v + (.6 if v >= 0 else -.6), yi, f'{v:+.3f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=6.6)
    panel(ax, 'b', 'Terminal mechanical-energy budget')
    ax.grid(False)
    ax.set_xlabel('Signed work / energy change (J)')
    ax = a[1, 0]
    residuals = np.array([n['maximum_energy_ledger_error_J'], r['maximum_energy_ledger_error_J']])*1e3
    ax.bar([0, 1], residuals, width=.5, color=[BLUE, GREEN], zorder=2)
    for k, v in enumerate(residuals):
        ax.text(k, v+1.3, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_xticks([0, 1], ['25 µs', '12.5 µs'])
    ax.set_ylim(0, 51)
    panel(ax, 'c', 'Energy-ledger step refinement')
    ax.set_ylabel(r'$\max_t |\Delta E-\sum_j W_j|$ (mJ)')
    ax.set_xlabel('Physics step')
    ax = a[1, 1]
    ax.semilogy(t, np.maximum(d['native_relative_residual'], 1e-18), color=BLUE, lw=.9)
    panel(ax, 'd', 'Native generalized-force balance')
    ax.set_ylabel('Relative equation residual')
    ax.text(.98, .93, f"Maximum: {m['computed']['native_relative_residual_max']:.2e}",
            transform=ax.transAxes, ha='right', va='top', fontsize=7)
    ax.set_ylim(1e-17, 2e-12)
    time_axis(ax)
    finish(fig, 'Kangaroo_Energy_Consistency')

    # Figure 4: all 40001 native samples, identical PD gains in both trials.
    fig, a = plt.subplots(1, 3, figsize=(7.15, 2.6), layout='constrained',
                          gridspec_kw={'width_ratios': [1, 1.45, 1]})
    tf = d['fixed_t']
    ax = a[0]
    for key, c, label in [('pd', ORANGE, 'PD only'), ('ff', BLUE, 'FF + PD')]:
        err = np.sqrt(np.mean(d[f'fixed_error_{key}_m']**2, axis=1))*1e6
        ax.semilogy(tf, np.where(err > 1e-6, err, np.nan), color=c, label=label, lw=1)
    panel(ax, 'a', 'Motor tracking')
    ax.set_ylabel('Instantaneous 12-motor RMS (µm)')
    ax.legend(frameon=False, loc='center right')
    ax.set_ylim(.005, 1e4)
    time_axis(ax, push=False, end=4)
    ax = a[1]
    x = np.arange(12)
    ff = np.array(m['fixed']['per_port_rmse_ff_m'])*1e6
    pd = np.array(m['fixed']['per_port_rmse_pd_m'])*1e6
    ax.bar(x-.18, pd, .36, color=ORANGE, label='PD only', zorder=2)
    ax.bar(x+.18, ff, .36, color=BLUE, label='FF + PD', zorder=2)
    ax.set_yscale('log')
    ax.set_ylim(.01, 2e4)
    ax.set_xticks(x, [s.replace('-length', 'ℓ') for s in m['motor_labels']], rotation=60, ha='right')
    ax.set_ylabel('Temporal RMS error (µm)')
    panel(ax, 'b', 'Per-port error')
    ax.text(.5, .96, f"Aggregate reduction: {m['fixed']['rmse_reduction_ratio']:.0f}×",
            transform=ax.transAxes, ha='center', va='top', fontsize=7)
    ax = a[2]
    for key, c in [('pd', ORANGE), ('ff', BLUE)]:
        ax.plot(tf, np.max(np.abs(d[f'fixed_force_{key}_N']), axis=1), color=c, lw=1)
    panel(ax, 'c', 'Actuator demand')
    ax.set_ylabel('Maximum absolute force (N)')
    ax.set_ylim(0, 290)
    ax.text(.97, .05, f"Peak PD: {m['fixed']['peak_force_pd_N']:.2f} N\nPeak FF + PD: {m['fixed']['peak_force_ff_N']:.2f} N",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.7)
    time_axis(ax, push=False, end=4)
    finish(fig, 'Kangaroo_Feedforward_Comparison')
    print('Saved four vector PDFs and four 300-dpi PNGs to', out)


if __name__ == '__main__':
    main()
