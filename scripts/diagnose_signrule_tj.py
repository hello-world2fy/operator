"""Endpoint, closed-path, spin-hole and flux controls in the same Fock encoding."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.geometry import chain,square
from operator_walshness.signrule.models import tj_hamiltonian,tj_marshall_gauge,block_eigensystem,block_gibbs
from operator_walshness.signrule.workflow import utc
from operator_walshness.signrule.diagnostics import (phase_entries,score_phase,support_balance,
    path_phase_samples,paired_improvement_ci)
from operator_walshness.signrule.physical_diagnostics import (tj_binary_encoding,constrained_linear_fit,
    constrained_features,fit_constrained_boolean,spin_hole_correlations,constrained_operator_entropy,qutrit_full_spectrum)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',type=int);args=parser.parse_args()
    out=ROOT/'results/signrule_round3'
    for geometry_index,g in enumerate([square(3,2),chain(8),chain(8,True)]):
        if args.geometry is not None and args.geometry!=geometry_index:continue
        geometry=g.name+'_'+'x'.join(map(str,g.shape))+'_'+g.boundary
        for t in [1.,2.,3.]:
            h,digits,fock,hops=tj_hamiltonian(g,t=t);sigma,*_=tj_hamiltonian(g,t=t,sigma=True)
            gauge=tj_marshall_gauge(g,digits);codes=tj_binary_encoding(digits)
            features,labels=constrained_features(digits,g.edges,2)
            pairdir=out/'physical_E'/geometry/f'n{g.n}'/f't{t:g}'
            pairdir.mkdir(parents=True,exist_ok=True)
            if not (pairdir/'path_summary.json').exists():
                rows,records=path_phase_samples(h,sigma,hops,gauge,samples=5000,seed=20261028)
                hm=h.toarray()*gauge[:,None]*gauge[None,:]
                sm=sigma.toarray()*gauge[:,None]*gauge[None,:]
                graph={}
                for name,operator in [('tJ',hm),('sigma_tJ',sm)]:
                    b=support_balance(-operator,1e-12,witness_limit=100);b.pop('gauge');graph[name]=b
                violations=sum(r['phase_string_violations']+r['sigma_nontrivial'] for r in rows)
                if violations:raise ArithmeticError('phase-string path identity violated')
                # Wilson intervals for descriptive closed-loop negative fractions.
                for row in rows:
                    n=row['closed'];x=row['negative']
                    if n:
                        ph=x/n;z=1.96;center=(ph+z*z/(2*n))/(1+z*z/n)
                        half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n)
                        row['negative_fraction']=ph;row['wilson95']=[center-half,center+half]
                atomic_json(pairdir/'path_summary.json',{'created_utc':utc(),'rows':rows,'complete_H_support_balance':graph,
                    'sampling':'uniform off-diagonal event random walks conditioned on closure; not QMC weights',
                    'sign_not_endpoint':True,'all_path_identity_violations':violations})
                atomic_json(pairdir/'closed_paths.json',records)
            if g.boundary=='periodic' and not (pairdir/'flux_response.json').exists():
                response={}
                for model,is_sigma in [('tJ',False),('sigma_tJ',True)]:
                    energies=[]
                    for flux in [0.,-.1,.1,np.pi]:
                        hf,*_=tj_hamiltonian(g,t=t,sigma=is_sigma,flux=flux)
                        # Exact small-sector diagonalization avoids a Krylov start
                        # accidentally restricted to an excited symmetry sector.
                        e=eigh(hf.toarray(),eigvals_only=True,subset_by_index=[0,1],check_finite=False)
                        energies.append(float(e.min()))
                    response[model]={'flux':[0.,-.1,.1,float(np.pi)],'ground_energies':energies,
                        'solver':'dense exact Hermitian sector eigensolver',
                        'finite_difference_curvature':(energies[1]+energies[2]-2*energies[0])/.01,
                        'pi_flux_energy_change':energies[3]-energies[0],
                        'caution':'finite-sector ground energy may have level crossings; curvature is descriptive'}
                atomic_json(pairdir/'flux_response.json',response)
            systems={'tJ':block_eigensystem(h),'sigma_tJ':block_eigensystem(sigma)}
            for beta in [0.,.25,.5,1.,2.,4.]:
                dest=pairdir/f'beta{beta:g}'
                if (dest/'summary.json').exists():continue
                # A source point for each model must be frozen before phase labels are scored.
                source_points={model:list((out/model/geometry/f'n{g.n}'/f't{t:g}'/f'beta{beta:g}').glob('alpha*/seed*/metrics.json')) for model in systems}
                if any(not points for points in source_points.values()):continue
                results={};arrays={};correlations={}
                for model,blocks in systems.items():
                    rho,thermal=block_gibbs(blocks,beta,len(digits));thresholds={}
                    for cutoff in [1e-8,1e-12]:
                        tag=f'{cutoff:g}';entries=phase_entries(rho,cutoff)
                        linear,mask=constrained_linear_fit(entries,codes);best_test,best_mask=constrained_linear_fit(entries,codes,'test')
                        quadratic,fit=fit_constrained_boolean(entries,features,max_gates=2*g.n)
                        balance=support_balance(rho,cutoff,witness_limit=24);balance.pop('gauge')
                        thresholds[tag]={'linear_coefficient_mask':mask,'linear_test':score_phase(entries,linear),
                            'test_linear_lower_bound':score_phase(entries,best_test),'quadratic_fit':fit,
                            'quadratic_versus_linear':paired_improvement_ci(entries,linear,quadratic),
                            'quadratic_versus_test_linear_lower_bound':paired_improvement_ci(entries,best_test,quadratic),
                            'complete_endpoint_support_balance':balance,
                            'Marshall_reference_test':score_phase(entries,gauge)}
                        for key in ['row','col','phase','weight','test']:arrays[f'{model}_{tag}_{key}']=entries[key]
                        arrays[f'{model}_{tag}_linear_phase']=linear;arrays[f'{model}_{tag}_quadratic_phase']=quadratic
                    physical=spin_hole_correlations(rho,digits);correlations[model]=physical
                    for key,value in physical.items():arrays[f'{model}_{key}']=value
                    osee=constrained_operator_entropy(rho,digits)
                    if osee.get('computed'):arrays[f'{model}_Schmidt_weights']=osee.pop('Schmidt_weights')
                    if g.n<=6:arrays[f'{model}_Z3_degree_spectrum']=qutrit_full_spectrum(rho,digits)
                    results[model]={'thermal':{k:v for k,v in thermal.items() if k not in ('eigenvalues','energies')},
                        'phase_scores':thresholds,'operator_entanglement':osee,'reliable_entry_masks':'sign_diagnostics.npz',
                        'posthoc_gate_features':labels,'qubit_W_used':False,
                        'interpretation':'endpoint fits are supervised posthoc; do not identify them with a path sign or with blind SU2 recovery'}
                difference=correlations['tJ']['hole_spin_connected']-correlations['sigma_tJ']['hole_spin_connected']
                dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'sign_diagnostics.npz',**arrays,sector_digits=digits,fock_states=fock)
                atomic_json(dest/'summary.json',{'created_utc':utc(),'beta':beta,'t_over_J':t,'geometry':g.metadata(),
                    'paired_models':results,'paired_spin_hole_connected_Frobenius_difference':float(np.linalg.norm(difference)),
                    'phase_string_identity_is_path_statement':True,'source_points':[str(p.relative_to(out)) for ps in source_points.values() for p in ps]})
                print('E_PHYSICS',geometry,t,beta,flush=True)


if __name__=='__main__':main()
