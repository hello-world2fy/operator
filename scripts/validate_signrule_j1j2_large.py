"""Large J1-J2 construction, thermal-rank and exact fixed-basis sign controls.

This does not expand a failed small-cluster fingerprint into a dense blind grid.
All eigenvalues are obtained in conserved-Sz blocks; no dense Gibbs matrix is used.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import eigvalsh
from scipy.special import logsumexp
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.geometry import chain
from operator_walshness.signrule.models import heisenberg
from operator_walshness.signrule.temperature import thermal_rank,select_low_temperature
from operator_walshness.signrule.diagnostics import support_balance
from operator_walshness.signrule.workflow import utc


def sector_spectrum(h,n):
    states=np.arange(1<<n,dtype=np.uint64);charge=np.bitwise_count(states);values=[];sectors=[]
    for k in range(n+1):
        ids=np.flatnonzero(charge==k);e=eigvalsh(h[ids][:,ids].toarray(),check_finite=False)
        values.extend(e);sectors.append({'n_down':k,'dimension':len(ids),'lowest_energy':float(e[0])})
    return np.sort(values),sectors


def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,choices=[10,12],required=True);args=p.parse_args()
    n=args.n;out=ROOT/'results/signrule_round3/independent_validation/j1j2_large'/f'n{n}';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for j2 in [0.,.3,.5,1.]:
        dest=out/f'p{j2:g}'
        if (dest/'summary.json').exists():rows.append(json.loads((dest/'summary.json').read_text()));continue
        h=heisenberg(chain(n),j2).tocsr();states=np.arange(1<<n,dtype=np.uint64);charge=np.bitwise_count(states)
        a,b=h.nonzero();assert np.all(charge[a]==charge[b])
        assert np.max(abs((h-h.T).data),initial=0.)<1e-13
        # Explicit all-up energy checks S=sigma/2 for both bond ranges.
        assert abs(h[0,0]-((n-1)+j2*(n-2))/4)<1e-12
        absolute=h.copy();absolute.data=-abs(absolute.data);absolute.setdiag(h.diagonal())
        energies,sectors=sector_spectrum(h,n);abs_energies,_=sector_spectrum(absolute,n)
        assert energies[0]>=abs_energies[0]-1e-10
        graph=support_balance(-h,1e-12,witness_limit=40);graph.pop('gauge')
        thermal=[];selection=select_low_temperature(energies)
        for beta in sorted(set([.25,1.,2.,4.,selection['selected']['beta']])):
            logz=float(logsumexp(-beta*energies));logza=float(logsumexp(-beta*abs_energies))
            gamma=(logza-logz)/(beta*n)
            assert gamma>=-1e-12
            thermal.append({**thermal_rank(energies,beta),'log_Z':logz,'log_Z_abs':logza,'gamma_fixed':gamma,
                'average_worldline_sign_exact':float(np.exp(logz-logza))})
        row={'created_utc':utc(),'n':n,'J2':j2,'boundary':'open','units':'S=sigma/2','sectors':sectors,
             'full_H_support_balance':graph,'thermal':thermal,'adaptive_temperature':selection,
             'scope':'exact full-spectrum scalars and full Hamiltonian support; no native large blind W scan',
             'diagonal_gauge_gamma_invariant':True,'QMC_samples_used':False,
             'phase2_gate':'no independently validated small-size J1-J2 non-Marshall fingerprint; larger blind scan remains conditional'}
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'spectra.npz',energies=energies,absolute_energies=abs_energies)
        atomic_json(dest/'summary.json',row);rows.append(row);print('J1J2_LARGE',n,j2,flush=True)
    atomic_json(out/'summary.json',{'cases':len(rows),'n':n,'all_sector_and_H_checks_passed':True,'native_large_blind_grid':False})


if __name__=='__main__':main()
