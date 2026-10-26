"""Read frozen Walsh candidates and only then attach physical sign references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from ..round2 import atomic_json
from .workflow import build_qubit_job,point_directory,utc
from .models import (block_eigensystem,block_gibbs,boolean_phase,levin_gu)
from .geometry import local_gate_supports
from .blind import circuit_phase
from .diagnostics import (phase_entries,score_phase,best_linear_phase,fit_boolean_phase,
    paired_improvement_ci,support_balance,strange_correlator)


def posthoc_oracle(job,g):
    name=job['model'];n=job['n']
    if name=='bipartite' or (name=='j1j2' and job.get('parameter',0)==0):
        return boolean_phase(n,[(i,) for i in range(n) if i%2==0]),'Marshall'
    if name=='cluster':return boolean_phase(n,g.edges),'graph_CZ'
    if name=='levin_gu':return levin_gu(g,0)[1],'domain_wall_loops_at_exact_point'
    if name in ('paramagnet','random_positive'):return np.ones(1<<n),'positive'
    if name in ('random_linear','random_quadratic'):
        rng=np.random.default_rng(job.get('builder_seed',1927))
        supports=local_gate_supports(g,1 if name=='random_linear' else 2)
        return boolean_phase(n,[s for s in supports if rng.random()<.5]),'generating_scramble_held_out'
    return None,None


def score_qubit_job(job,out,*,fit_starts=16,fit_flips=2000):
    g,h=build_qubit_job(job);blocks=block_eigensystem(h);n=job['n']
    scored=0
    for beta in job['betas']:
        points=[point_directory(out,job,beta,alpha,seed) for alpha in job['alphas'] for seed in job['seeds']]
        ready=[p for p in points if (p/'metrics.json').exists() and not (p/'posthoc.json').exists()]
        if not ready:continue
        # All files used for this batch MUST already be frozen before opening an oracle.
        for point in ready:
            frozen=json.loads((point/'blind_frozen.json').read_text())
            digest=hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()
            if digest!=frozen['basis_or_circuit_sha256']:raise ValueError('frozen candidate was modified')
        rho,thermal=block_gibbs(blocks,beta,1<<n)
        oracle,oracle_name=posthoc_oracle(job,g)
        shared=points[0].parents[1]/'posthoc_base'
        shared.mkdir(parents=True,exist_ok=True)
        thresholds={};arrays={};fits={}
        cache_ok=False
        if (shared/'reference_scores.json').exists() and (shared/'sign_diagnostics.npz').exists():
            cached=json.loads((shared/'reference_scores.json').read_text())
            cache_ok=(cached.get('fit_starts',16)==fit_starts and cached.get('fit_flips',2000)==fit_flips
                      and all('linear_test_oracle_lower_bound' in cached.get('thresholds',{}).get(f'{cutoff:g}',{})
                              for cutoff in [1e-8,1e-12]))
            if cache_ok:
                with np.load(shared/'sign_diagnostics.npz') as saved:arrays={key:saved[key] for key in saved.files}
                thresholds=cached['thresholds'];fits=cached['fits']
        for cutoff in ([] if cache_ok else [1e-8,1e-12]):
            tag=f'{cutoff:g}';entries=phase_entries(rho,cutoff)
            linear,mask=best_linear_phase(entries,n)
            test_oracle,oracle_mask=best_linear_phase(entries,n,split='test')
            thresholds[tag]={'linear_coefficient_mask':mask,'linear_train':score_phase(entries,linear,'train'),
                             'linear_test':score_phase(entries,linear),'mask_threshold':entries['threshold'],
                             'linear_test_oracle_lower_bound':score_phase(entries,test_oracle),
                             'linear_test_oracle_mask':oracle_mask,
                             'retained_offdiag_mass':entries['retained_offdiag_mass'],
                             'total_mass':entries['total_mass'],'diagonal_mass':entries['diagonal_mass']}
            for k in ('row','col','phase','weight','test'):arrays[f'{tag}_{k}']=entries[k]
            arrays[f'{tag}_linear_phase']=linear
            arrays[f'{tag}_linear_test_oracle_phase']=test_oracle
            if beta>0 and len(entries['row']):
                balance=support_balance(rho,cutoff,witness_limit=20);balance.pop('gauge')
                thresholds[tag]['support_balance']=balance
                # Same number of active terms, starts, and proposals as blind circuits.
                for level in [2,3]:
                    if level==3 and not g.triangles:continue
                    phase,fit=fit_boolean_phase(entries,n,local_gate_supports(g,level),
                        starts=fit_starts,flips=fit_flips,seed=20261027,max_gates=n)
                    thresholds[tag][f'fit_L{level}']={'train':score_phase(entries,phase,'train'),
                        'test':score_phase(entries,phase),'versus_linear':paired_improvement_ci(entries,linear,phase),
                        'active_terms':int(np.sum(fit['coefficients'])),'max_gates':n}
                    fits[f'{tag}_L{level}']=fit;arrays[f'{tag}_fit_L{level}_phase']=phase
            if oracle is not None:
                thresholds[tag]['oracle']={'name':oracle_name,'test':score_phase(entries,oracle),
                    'versus_linear':paired_improvement_ci(entries,linear,oracle)}
        if not cache_ok:
            if oracle is not None:arrays['oracle_phase']=oracle
            np.savez_compressed(shared/'sign_diagnostics.npz',**arrays)
            atomic_json(shared/'reference_scores.json',{'scored_utc':utc(),'thresholds':thresholds,'fits':fits,
                'fit_starts':fit_starts,'fit_flips':fit_flips,
                'representation':'original computational encoding; no SU2-rotated loop labels',
                'thermal_purity':thermal['purity'],'beta':beta})
        if job['model']=='levin_gu' and job.get('parameter',0)==0:
            diag=strange_correlator(oracle/np.sqrt(len(oracle)),n)
            np.savez_compressed(shared/'strange_correlator.npz',**diag)
        for point in ready:
            candidates=json.loads((point/'basis_or_circuit.json').read_text())
            scores={}
            for label,result in candidates.items():
                if not label.startswith('circuit'):continue
                phase=circuit_phase(n,result);score={}
                for cutoff in [1e-8,1e-12]:
                    tag=f'{cutoff:g}';entries={k:arrays[f'{tag}_{k}'] for k in ['row','col','phase','weight','test']}
                    score[tag]={'test':score_phase(entries,phase),
                        'versus_best_linear':paired_improvement_ci(entries,arrays[f'{tag}_linear_phase'],phase),
                        'versus_test_oracle_linear_lower_bound':paired_improvement_ci(entries,arrays[f'{tag}_linear_test_oracle_phase'],phase),
                        'gate_count':result['gate_count'],'circuit_depth':result['circuit_depth']}
                scores[label]=score
            # A positive entry-level fingerprint is preliminary until an independent
            # temperature AND geometry validate the FROZEN coefficients.
            positive=[]
            for label,score in scores.items():
                has_nonlinear=any(on and len(support)>1 for support,on in zip(candidates[label]['gate_supports'],candidates[label]['coefficients_best']))
                if has_nonlinear and all(v['versus_test_oracle_linear_lower_bound']['ci95'] is not None and v['versus_test_oracle_linear_lower_bound']['ci95'][0]>1e-6
                       for v in score.values()):positive.append(label)
            atomic_json(point/'posthoc.json',{'scored_utc':utc(),'shared_reference':str(shared.relative_to(out)),
                'circuit_scores':scores,'preliminary_non_Marshall_candidates':positive,
                'score_version':2,
                'independent_validation':'pending','classification':('known_control_recovery' if job['track']=='A' else 'preliminary_sign_fingerprint') if positive else 'no_verified_non_Marshall_recovery',
                'holdout_scope':'entries excluded from supervised phase fitting, but W optimization sees the full operator; independent beta/geometry is mandatory',
                'local_SU2_oracle_comparison':'not applied: non-diagonal rotations mix the original configuration labels',
                'operator_specific':'not certified until independent validation even when purity < 0.95'})
            scored+=1
    return scored
