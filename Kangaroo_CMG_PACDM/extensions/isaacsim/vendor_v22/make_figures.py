"""Scientific figures from recorded, physical motor-driven native simulation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def run():
    a=np.load(ROOT/'results/motion_dt_5e-05.npz');b=np.load(ROOT/'results/standing_2.5e-05.npz');stride=20;t=a['time'][::stride];h=a['history'][::stride];hs=b['history'][::20];tt=hs[:,0]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    fig,ax=plt.subplots(3,2,figsize=(13,11),layout='constrained');names=['Yaw','Hip 2','Hip 3','Length','Ankle 4','Ankle 5']
    for side in range(2):
        aa=ax[0,side]
        for k,name in enumerate(names):aa.plot(t,h[:,9+side*6+k],lw=1.3,label=name)
        aa.set(title=('Left' if side==0 else 'Right')+' physical motor forces',xlabel='Time (s)',ylabel='Force (N)');aa.legend(ncol=3,fontsize=8)
    for k,label in [(2,'Motor power'),(3,'Viscous loss'),(4,'Loop constraint power')]:ax[1,0].plot(t,h[:,k],label=label,lw=1.2)
    ax[1,0].set(title='Mechanical power ledger',xlabel='Time (s)',ylabel='Power (W)');ax[1,0].legend(fontsize=8)
    ax[1,1].plot(t,(h[:,:2].sum(axis=1)-h[0,:2].sum()),label='Change in mechanical energy');ax[1,1].plot(t,(h[:,:2].sum(axis=1)-h[0,:2].sum())-a['balance'][::stride],'--',label='Integrated total mechanical work');ax[1,1].set(title='Energy and work agree',xlabel='Time (s)',ylabel='Energy (J)');ax[1,1].legend(fontsize=8)
    for dt in [.0002,.0001,.00005]:
        x=np.load(ROOT/'results'/f'motion_dt_{dt:g}.npz');k=max(1,len(x['time'])//1200);ax[2,0].plot(x['time'][::k],x['balance'][::k]*1e3,label=f'{dt*1e6:g} microseconds')
    ax[2,0].set(title='Energy error decreases with timestep',xlabel='Time (s)',ylabel='Energy ledger residual (mJ)');ax[2,0].legend(fontsize=8)
    ax[2,1].plot(tt,hs[:,8],label='Measured vertical ground force');ax[2,1].axhline(414.127851235,color='black',ls='--',label='Full robot weight');ax[2,1].set(title='Floating-base standing: measured contact support',xlabel='Time (s)',ylabel='Force (N)');ax[2,1].legend(fontsize=8)
    fig.suptitle('Kangaroo whole-body PACDM: 12 physical linear actuators',fontsize=18,fontweight='bold');fig.supxlabel('Explicit published-cut reconstruction. Fixed-pelvis motion (top and left); floating-base standing (bottom right).',fontsize=10)
    fig.savefig(ROOT/'results/evidence_figure.png',dpi=180);fig.savefig(ROOT/'results/evidence_figure.svg');plt.close(fig)
if __name__=='__main__':run()
