"""L=10/12 constrained t-J: true paths, sparse propagator columns and observables.

Column diagnostics explicitly do not certify the full thermal support. Complete
Hamiltonian graph checks are separate. No qubit Walsh metric or full qutrit array.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.geometry import chain
from operator_walshness.signrule.models import tj_hamiltonian,tj_marshall_gauge
from operator_walshness.signrule.diagnostics import path_phase_samples,support_balance,score_phase,paired_improvement_ci
from operator_walshness.signrule.physical_diagnostics import tj_binary_encoding,constrained_linear_fit
from operator_walshness.signrule.matrix_free import (verified_ground,propagator_columns,sampled_column_entries,
    stochastic_thermal_diagonal)
from operator_walshness.signrule.workflow import utc


def thermal_observables(squared,digits,seed):
    probes=squared.shape[1];norm=squared.sum(0)
    hole=(digits==0).astype(float);sz=((digits==1).astype(float)-(digits==2))/2
    raw_features=(hole[:,:,None]*sz[:,None,:]).reshape(len(digits),-1)
    fh=hole.T@squared;fs=sz.T@squared;fhs=raw_features.T@squared
    h=fh.sum(1)/norm.sum();s=fs.sum(1)/norm.sum();hs=(fhs.sum(1)/norm.sum()).reshape(digits.shape[1],-1)
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(200):
        ids=rng.integers(probes,size=probes);z=norm[ids].sum()
        bh=fh[:,ids].sum(1)/z;bs=fs[:,ids].sum(1)/z
        boot.append(fhs[:,ids].sum(1).reshape(digits.shape[1],-1)/z-np.outer(bh,bs))
    return {'hole_density':h,'spin_z':s,'hole_spin_connected':hs-np.outer(h,s),
        'hole_spin_connected_ci95':np.quantile(boot,[.025,.975],axis=0),
        'bootstrap_unit':'Rademacher trace probe; ratio-estimator uncertainty','probes':probes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=int,required=True);p.add_argument('--columns',type=int,default=64);p.add_argument('--probes',type=int,default=64)
    args=p.parse_args();geometries=[chain(10),chain(10,True),chain(12),chain(12,True)]
    g=geometries[args.geometry];out=ROOT/'results/signrule_round3/physical_E_large'/f'chain_{g.n}_{g.boundary}'
    for t in [1.,2.,3.]:
        base=out/f't{t:g}';base.mkdir(parents=True,exist_ok=True)
        h,digits,fock,hops=tj_hamiltonian(g,t=t);sigma,*_=tj_hamiltonian(g,t=t,sigma=True)
        gauge=tj_marshall_gauge(g,digits);codes=tj_binary_encoding(digits)
        hole=np.argmax(digits==0,axis=1)
        left_down=np.sum((digits==2)&(np.arange(g.n)[None,:]<hole[:,None]),axis=1)
        endpoint_reference=gauge*(-1.)**left_down
        if not (base/'path_summary.json').exists():
            rows,records=path_phase_samples(h,sigma,hops,gauge,samples=5000,seed=20261028)
            assert sum(r['phase_string_violations']+r['sigma_nontrivial'] for r in rows)==0
            graphs={}
            for model,ham in [('tJ',h),('sigma_tJ',sigma)]:
                b=support_balance(-ham,1e-12,witness_limit=32);b.pop('gauge');graphs[model]=b
            atomic_json(base/'path_summary.json',{'rows':rows,'complete_H_support_balance':graphs,
                'sampling':'unweighted off-diagonal event random walks conditioned on closure; not a QMC distribution',
                'phase_string_identity_violations':0,'fixed_holes':1,'fixed_n_down':int((digits[0]==2).sum()),
                'sign_path_not_endpoint':True})
            atomic_json(base/'closed_paths.json',records)
        systems={};ground_summary={}
        for model,ham in [('tJ',h),('sigma_tJ',sigma)]:
            e,v,check=verified_ground(ham);systems[model]=(ham,e[0]);ground=e-e[0]<1e-8
            probability=np.mean(abs(v[:,ground])**2,axis=1)
            ground_summary[model]={'eigensolver':check,'hole_density':probability@(digits==0),
                'spin_hole_raw':((digits==0).T*probability)@(((digits==1).astype(float)-(digits==2))/2)}
        atomic_json(base/'ground_observables.json',ground_summary)
        if g.boundary=='periodic' and not (base/'flux_response.json').exists():
            response={}
            for model,is_sigma in [('tJ',False),('sigma_tJ',True)]:
                energies=[];checks=[]
                for flux in [0.,-.1,.1,np.pi]:
                    hf,*_=tj_hamiltonian(g,t=t,sigma=is_sigma,flux=flux)
                    e,_,check=verified_ground(hf);energies.append(float(e[0]));checks.append(check)
                response[model]={'flux':[0.,-.1,.1,float(np.pi)],'ground_energies':energies,'solver_checks':checks,
                    'finite_difference_curvature':float((energies[1]+energies[2]-2*energies[0])/.01),
                    'pi_flux_energy_change':energies[3]-energies[0],
                    'level_crossing_caveat':True}
            atomic_json(base/'flux_response.json',response)
        columns=np.sort(np.random.default_rng(20261028).choice(len(digits),min(args.columns,len(digits)),replace=False))
        for beta in [0.,.25,.5,1.,2.,4.]:
            point=base/f'beta{beta:g}'
            if (point/'summary.json').exists():continue
            result={};arrays={'sampled_columns':columns,'sector_digits':digits,'fock_states':fock}
            for model,(ham,e0) in systems.items():
                kernel=propagator_columns(ham,beta,columns,e0);arrays[model+'_shifted_propagator_columns']=kernel
                diagonal,trace,squared=stochastic_thermal_diagonal(ham,beta,e0,probes=args.probes)
                physical=thermal_observables(squared,digits,20261029)
                arrays[model+'_thermal_diagonal_estimate']=diagonal
                scores={}
                for cutoff in [1e-8,1e-12]:
                    entries=sampled_column_entries(kernel,columns,cutoff)
                    linear,mask=constrained_linear_fit(entries,codes);lower,_=constrained_linear_fit(entries,codes,'test')
                    reference=endpoint_reference if model=='tJ' else gauge
                    scores[f'{cutoff:g}']={'linear_mask':mask,'linear_test':score_phase(entries,linear),
                        'test_linear_lower_bound':score_phase(entries,lower),'known_endpoint_reference_test':score_phase(entries,reference),
                        'reference_versus_test_linear':paired_improvement_ci(entries,lower,reference),
                        'sampled_offdiagonal_entries':len(entries['row']),
                        'coverage':entries['coverage'],'relative_threshold_reference':entries['relative_threshold_reference']}
                column_norms=np.sum(abs(kernel)**2,axis=0)
                purity_estimate=float(len(digits)*column_norms.mean()/trace['trace_shifted_estimate']**2)
                result[model]={'phase_scores':scores,'trace_estimate':trace,'thermal_observables':physical,
                    'purity_estimate':purity_estimate,'purity_estimate_is_not_used_for_discovery_certification':True,
                    'purity_note':'ratio of independent sampled-column Tr(K^2) and random-probe Tr(K)^2; not exact and not clipped',
                    'propagator_selected_square_Hermiticity_max':float(abs(kernel[columns]-kernel[columns].conj().T).max()),
                    'normalization':'K=exp(-beta*(H-E0 I)); overall scalar cancels in phase metrics',
                    'known_endpoint_reference':'Marshall times parity of down spins left of hole' if model=='tJ' else 'hole-compatible Marshall',
                    'reference_scope':'posthoc known formula, not learned by W; periodic validity checked numerically, not assumed'}
            point.mkdir(parents=True,exist_ok=True);np.savez_compressed(point/'propagator_and_observables.npz',**arrays)
            atomic_json(point/'summary.json',{'created_utc':utc(),'n':g.n,'geometry':g.metadata(),'t_over_J':t,'beta':beta,
                'paired_models':result,'sector_dimension':len(digits),'qubit_W_used':False,
                'no_full_qutrit_dense_operator':True,'full_thermal_support_certificate':False,
                'purpose':'larger-geometry physical validation; not a new blind optimization or sign-discovery claim'})
            print('E_LARGE',g.n,g.boundary,t,beta,flush=True)


if __name__=='__main__':main()
