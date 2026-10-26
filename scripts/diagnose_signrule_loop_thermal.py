"""Exact thermal TC/DS sector pairing and post-freeze phase diagnostics."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import brentq
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import (rotate_operator,directions_to_walsh_basis,fixed_basis_spectrum,
    walshness_from_spectrum,x_basis_matrix,normalized_expectations)
from operator_walshness.observables import degree_correlations_from_x_matrix
from operator_walshness.signrule.geometry import honeycomb
from operator_walshness.signrule.loop_topology import closed_sector_labels
from operator_walshness.signrule.models import loop_projectors,commuting_gibbs,block_eigensystem
from operator_walshness.signrule.blind import circuit_phase
from operator_walshness.signrule.diagnostics import (phase_entries,best_linear_phase,score_phase,
    paired_improvement_ci,support_balance,fit_boolean_phase)
from operator_walshness.signrule.workflow import utc


def clean_balance(operator,cutoff):
    result=support_balance(operator,cutoff,witness_limit=24);result.pop('gauge');return result


def sector_comparison(hc,labels,counts,out):
    dest=out/'physical_C/thermal_pair';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return
    systems={};rows=[]
    for model,semion in [('toric_code',False),('double_semion',True)]:
        h,terms=loop_projectors(hc,semion)
        energies=np.sort(np.concatenate([e for _,e,_ in block_eigensystem(h)]))
        systems[model]=(h,terms,energies)
    for beta in [0.,.25,.5,1.,2.,4.]:
        pair={};matrices={}
        for model,(h,terms,energies) in systems.items():
            rho=commuting_gibbs(terms,beta,sparse_output=True);matrices[model]=rho
            p=rho.diagonal().real;sector=[]
            for label in range(4):
                ids=np.flatnonzero(labels==label);block=rho[ids][:,ids];mass=float(p[ids].sum())
                phase=(-1.)**counts[ids] if model=='double_semion' else np.ones(len(ids))
                hblock=h[ids][:,ids];ground_residual=float(np.linalg.norm(hblock@phase+(len(hc['stars'])+len(hc['faces']))*phase))
                e=phase_entries(block)
                sector.append({'homology':[label%2,label//2],'dimension':len(ids),'trace_weight':mass,
                    'conditional_purity':float(np.vdot(block.data,block.data).real/mass**2),
                    'contractible_loop_reference_ground_residual':ground_residual,
                    'loop_reference_validated_in_this_sector':ground_residual<1e-10,
                    'loop_reference_all_entries':score_phase(e,phase,'all'),
                    'interpretation':'loop reference evaluated only in closed-loop support; sector validity checked independently'})
            pair[model]={'purity':float(np.vdot(rho.data,rho.data).real),
                'vertex_defect_weight':float(p[labels<0].sum()),'closed_sector_weights':sector,
                'ground_degeneracy':int(np.sum(abs(energies-energies[0])<1e-9))}
        target=pair['toric_code']['purity'];ds_e=systems['double_semion'][2]
        def purity(b):
            p=np.exp(-b*(ds_e-ds_e[0]));p/=p.sum();return float(p@p)
        matched_beta=0. if beta==0 else brentq(lambda b:purity(b)-target,0.,64.,xtol=1e-12)
        ds_match=commuting_gibbs(systems['double_semion'][1],matched_beta,sparse_output=True)
        moddiff=abs(matrices['toric_code'])-abs(matrices['double_semion'])
        matched_diff=abs(matrices['toric_code'])-abs(ds_match)
        rows.append({'beta':beta,'models':pair,'same_beta_modulus_Frobenius':float(np.linalg.norm(moddiff.data)),
            'DS_beta_matching_TC_purity':matched_beta,'matched_purity_residual':abs(purity(matched_beta)-target),
            'same_purity_modulus_Frobenius':float(np.linalg.norm(matched_diff.data))})
    atomic_json(dest/'summary.json',{'created_utc':utc(),'geometry':hc,'rows':rows,
        'spectrum_max_difference':float(abs(systems['toric_code'][2]-systems['double_semion'][2]).max()),
        'oracle_scope':'closed loops only; never applied to vertex-defect configurations',
        'not_QMC_severity':True,'physical_sector_analysis_not_blind_discovery':True})
    np.savez_compressed(dest/'sector_labels.npz',homology_labels=labels,contractible_loop_counts=counts,
                        **{model+'_energies':value[2] for model,value in systems.items()})


def score_model(model,hc,labels,counts,out):
    h,terms=loop_projectors(hc,model=='double_semion');n=hc['n']
    edges=sorted({tuple(sorted((a,b))) for star in hc['stars'] for a in star for b in star if a<b})
    supports=[(i,) for i in range(n)]+edges+[tuple(star) for star in hc['stars']]
    for beta in [0.,.25,.5,1.,2.,4.]:
        base=out/model/'honeycomb_2x2'/f'n{n}'/'full_Gibbs'/f'beta{beta:g}'
        ready=[p.parent for p in base.glob('alpha*/seed*/metrics.json') if not (p.parent/'posthoc.json').exists()]
        if not ready:continue
        for p in ready:assert json.loads((p/'blind_frozen.json').read_text())['known_label_access'] is False
        rho=commuting_gibbs(terms,beta,sparse_output=True);shared=base/'posthoc_base';shared.mkdir(parents=True,exist_ok=True)
        arrays={};references={};cache=shared/'reference_scores.json'
        if cache.exists():
            references=json.loads(cache.read_text())['thresholds']
            with np.load(shared/'sign_diagnostics.npz') as saved:arrays={k:saved[k] for k in saved.files}
        else:
            for cutoff in [1e-8,1e-12]:
                tag=f'{cutoff:g}';entries=phase_entries(rho,cutoff)
                linear,mask=best_linear_phase(entries,n);testbest,testmask=best_linear_phase(entries,n,'test')
                reference={'linear_mask':mask,'linear_test':score_phase(entries,linear),'test_linear_mask':testmask,
                    'test_linear_lower_bound':score_phase(entries,testbest),'complete_support_balance':clean_balance(rho,cutoff),
                    'retained_offdiag_mass':entries['retained_offdiag_mass'],'total_mass':entries['total_mass']}
                for level in [2,3]:
                    if not len(entries['row']):continue
                    phase,fit=fit_boolean_phase(entries,n,[s for s in supports if len(s)<=level],max_gates=n,
                        starts=16,flips=2000,seed=20261027)
                    reference[f'fit_L{level}']={'fit':fit,'test':score_phase(entries,phase),
                        'versus_test_linear_lower_bound':paired_improvement_ci(entries,testbest,phase)}
                    arrays[f'{tag}_fit_L{level}']=phase
                for key in ['row','col','phase','weight','test']:arrays[f'{tag}_{key}']=entries[key]
                arrays[f'{tag}_linear']=linear;arrays[f'{tag}_test_linear']=testbest;references[tag]=reference
            np.savez_compressed(shared/'sign_diagnostics.npz',**arrays)
            atomic_json(cache,{'thresholds':references,'known_loop_oracle_scope':'closed vacuum sector only',
                'fit_starts':16,'fit_flips':2000,'scored_utc':utc()})
        dense=rho.toarray();fixed_corr=degree_correlations_from_x_matrix(x_basis_matrix(dense))
        for point in ready:
            m=json.loads((point/'metrics.json').read_text());results=json.loads((point/'basis_or_circuit.json').read_text());scores={}
            for label,result in results.items():
                if not label.startswith('circuit'):continue
                phase=circuit_phase(n,result);scores[label]={}
                for cutoff in [1e-8,1e-12]:
                    tag=f'{cutoff:g}';entries={k:arrays[f'{tag}_{k}'] for k in ['row','col','phase','weight','test']}
                    scores[label][tag]={'test':score_phase(entries,phase),
                        'versus_best_linear':paired_improvement_ci(entries,arrays[f'{tag}_linear'],phase),
                        'versus_test_oracle_linear_lower_bound':paired_improvement_ci(entries,arrays[f'{tag}_test_linear'],phase)}
            spectral_path=point/'walsh_spectrum.npz'
            with np.load(spectral_path) as saved:spectra={key:saved[key] for key in saved.files}
            spectra.update(fixed_corr)
            if 'local_SU2' in results and 'local_SU2' not in spectra:
                rotated=rotate_operator(dense,directions_to_walsh_basis(np.asarray(results['local_SU2']['axes_best'])))
                spectra['local_SU2']=fixed_basis_spectrum(rotated)[0]
                residual=abs(walshness_from_spectrum(spectra['local_SU2'],m['alpha'])-results['local_SU2']['W_best'])
                if residual>1e-8:raise ArithmeticError(f'sparse SU2 vs independent dense FWHT {residual}')
            else:residual=0.
            temporary=point/f'walsh_spectrum.{os.getpid()}.tmp.npz';np.savez_compressed(temporary,**spectra);temporary.replace(spectral_path)
            positive=[k for k,v in scores.items() if all(x['versus_test_oracle_linear_lower_bound']['ci95'] is not None
                and x['versus_test_oracle_linear_lower_bound']['ci95'][0]>1e-6 for x in v.values())
                and any(on and len(s)>1 for s,on in zip(results[k]['gate_supports'],results[k]['coefficients_best']))]
            atomic_json(point/'posthoc.json',{'scored_utc':utc(),'circuit_scores':scores,
                'preliminary_non_Marshall_candidates':positive,'independent_validation':'pending',
                'classification':'preliminary_sign_fingerprint' if positive else 'no_verified_non_Marshall_recovery',
                'operator_specific':False,'shared_reference':str(shared.relative_to(out)),
                'local_SU2_dense_validation_residual':residual,'local_magnetization':normalized_expectations(dense)[0],
                'holdout_scope':'entry decoding only; W optimization used the entire operator',
                'labels_not_applied_after_SU2_rotation':True})
            print('THERMAL_C_SCORED',model,beta,m['alpha'],m['seed'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    args=p.parse_args();out=ROOT/'results/signrule_round3';hc=honeycomb(2,2)
    labels,counts,cosets=closed_sector_labels(hc)
    if args.worker==0:sector_comparison(hc,labels,counts,out)
    for i,model in enumerate(['toric_code','double_semion']):
        if i%args.workers==args.worker:score_model(model,hc,labels,counts,out)


if __name__=='__main__':main()
