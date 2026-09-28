#!/usr/bin/env python
"""Format the runner's measured tables for a two-column paper (no new data).
Usage: python format_paper_tables.py --results results_here --out presentation
"""
from pathlib import Path
import argparse,json
from src.reporting import escape,value_tex

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results_here');p.add_argument('--out',default='presentation');a=p.parse_args()
    source=Path(a.results);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tables=json.loads((source/'tables.json').read_text())
    headers={
        'dimensions':['Quantity','Original','Compiled'],
        'compiler':['Generated plan','Plans','Checks/plan','Accepted/checks','Rank/chart dim.'],
        'continuation':['Method','Accepted/attempts','Median (ms)','P95 (ms)','Max. gap (m)'],
        'incremental':['Changed feet','Global PACDM','Modular PACDM','No reuse','Global TRF','Modular TRF'],
        'derivatives':['Derivative route','Accepted/attempts','Median (ms)','P95 (ms)','Model evals.'],
        'support_dynamics':['Supporting feet','Accepted/states','Rank/mobility','Norm. accel. error','Point residual'],
        'dynamics_consistency':['Quantity','Maximum','Limit'],
        'curvature':['Speed scale','State/support cases','Full-term max.','Omitted-term max.']}
    outtext=['% Go2 measured tables. Standard LaTeX only; designed for two-column table* floats.',
             '% Original numerical records remain in results_here/TABLES.tex and CSV/JSON.',
             '% Frame widths and final pagination must be checked with the journal class.']
    for t in tables:
        key=t['key'];rows=t['rows'];notes=t['note']
        if key=='dimensions':notes+=' The task-chart dimension is 18 (six prescribed base plus twelve target coordinates), not the number of physical actuators.'
        if key=='compiler':notes+=' Task-chart dimension includes virtual targets; ideal-support chart dimension is remaining physical mobility.'
        if key=='incremental':notes+=' All displayed times are milliseconds; every method accepted 144/144 attempts in each condition in the recorded full run. The original-evaluator comparison is retained in the raw/Excel tables.'
        if key=='support_dynamics':notes+=' Point residuals are in $\\mathrm{m\\,s^{-2}}$.'
        if key=='curvature':notes=notes.replace('12 configurations across','12 state/support witnesses across')
        if key in ('dimensions','dynamics_consistency'):spec='p{0.58\\textwidth}rr'
        elif key in ('continuation','derivatives'):spec='p{0.39\\textwidth}rrrr'
        elif key=='compiler':spec='p{0.34\\textwidth}rrrr'
        elif key=='incremental':spec='rrrrrr'
        elif key=='support_dynamics':spec='rrrrr'
        else:spec='rrrr'
        # All units are explicit in the note, keeping the column headings short.
        note=escape(notes).replace(escape('$\\mathrm{m\\,s^{-2}}$'),'$\\mathrm{m\\,s^{-2}}$')
        outtext += [r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
            r'\caption{'+escape(t['title'])+'.}',r'\label{tab:go2_'+key+'}',r'\begin{tabular}{'+spec+'}',r'\hline',
            ' & '.join(escape(x)for x in headers[key])+r' \\',r'\hline']
        outtext += [' & '.join(value_tex(x)for x in row)+r' \\'for row in rows]
        outtext += [r'\hline',r'\end{tabular}',r'\par\smallskip',r'\parbox{0.98\linewidth}{\footnotesize '+note+'}',r'\end{table*}','']
    target=out/'Go2_Framework_Benefits_Tables.tex';target.write_text('\n'.join(outtext)+'\n',encoding='utf-8');print(target)
if __name__=='__main__': main()
