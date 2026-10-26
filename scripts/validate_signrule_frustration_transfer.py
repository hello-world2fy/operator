"""Independent validation of frozen boundary-anchored phase circuits; no fitting."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.geometry import chain
from operator_walshness.signrule.models import frustrated,block_eigensystem,block_gibbs
from operator_walshness.signrule.blind import circuit_phase,circuit_depth
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase,score_phase,paired_improvement_ci,support_balance
from operator_walshness.signrule.workflow import utc


def boundary_transfer(result,source_n,target_n):
    """Keep each local support's distance to its closest open boundary (ties left)."""
    supports=[]
    for support in result['gate_supports']:
        shift=target_n-source_n if min(support)+max(support)>source_n-1 else 0
        supports.append(tuple(i+shift for i in support))
    assert len(set(supports))==len(supports)
    return {**result,'gate_supports':supports,'circuit_depth':circuit_depth(supports,result['coefficients_best']),
        'transfer_rule':'nearest_open_boundary_anchor; tie_left; no_new_gates',
        'target_label_or_W_access':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extend-n12',action='store_true');args=parser.parse_args()
    sizes=[12] if args.extend_n12 else [6,8,10];parameters=[.4] if args.extend_n12 else [.35,.4,.45]
    out=ROOT/'results/signrule_round3';dest=out/'independent_validation'/('frustration_n12_transfer' if args.extend_n12 else 'frustration_transfer');dest.mkdir(parents=True,exist_ok=True)
    source=out/'controlled_frustration/chain_open/n6/p0.4/beta4/alpha2'
    candidates={}
    for seed in [20261027,20261028,20261029]:
        p=source/f'seed{seed}'
        assert json.loads((p/'blind_frozen.json').read_text())['known_label_access'] is False
        candidates[seed]=json.loads((p/'basis_or_circuit.json').read_text())['circuit_L2']
    transfers={n:{seed:boundary_transfer(c,6,n) for seed,c in candidates.items()} for n in sizes}
    if not (dest/'registration.json').exists():
        atomic_json(dest/'registration.json',{'registered_utc':utc(),'source':'n6 lambda0.4 beta4 alpha2 all3seeds circuit_L2',
            'target_n':sizes,'target_beta':[3.,5.],'target_lambda':parameters,
            'all_transfer_rules_frozen_before_target_construction':True,'transfers':transfers})
    rows=[]
    for n in sizes:
        for lam in parameters:
            g=chain(n);h=frustrated(g,lam);blocks=block_eigensystem(h)
            for beta in [3.,5.]:
                point=dest/f'n{n}'/f'lambda{lam:g}'/f'beta{beta:g}'
                if (point/'summary.json').exists():rows.extend(json.loads((point/'summary.json').read_text())['rows']);continue
                rho,thermal=block_gibbs(blocks,beta,1<<n);result=[];arrays={};comparisons={}
                for cutoff in [1e-8,1e-12]:
                    tag=f'{cutoff:g}';entries=phase_entries(rho,cutoff)
                    linear,mask=best_linear_phase(entries,n);lower,testmask=best_linear_phase(entries,n,'test')
                    for key in ['row','col','phase','weight','test']:arrays[f'{tag}_{key}']=entries[key]
                    balance=support_balance(rho,cutoff,witness_limit=12);balance.pop('gauge')
                    for seed,candidate in transfers[n].items():
                        phase=circuit_phase(n,candidate)
                        key=(len(entries['row']),np.packbits(phase<0).tobytes())
                        if key not in comparisons:comparisons[key]=paired_improvement_ci(entries,lower,phase)
                        comparison=comparisons[key]
                        result.append({'n':n,'lambda':lam,'beta':beta,'alpha_source':2.,'seed':seed,'cutoff':cutoff,
                            'purity':thermal['purity'],'test':score_phase(entries,phase),'test_best_linear':score_phase(entries,lower),
                            'trained_linear_test':score_phase(entries,linear),'versus_test_linear_lower_bound':comparison,
                            'complete_support_balance':balance,'gate_count':candidate['gate_count'],
                            'positive_lower_confidence_bound':comparison['ci95'] is not None and comparison['ci95'][0]>1e-6})
                point.mkdir(parents=True,exist_ok=True);np.savez_compressed(point/'sign_diagnostics.npz',**arrays)
                atomic_json(point/'summary.json',{'rows':result,'target_optimized':False,'created_utc':utc()});rows.extend(result)
                print('F_TRANSFER',n,lam,beta,flush=True)
    primary=[r for r in rows if r['lambda']==.4];perturbed=[r for r in rows if r['lambda']!=.4]
    atomic_json(dest/'summary.json',{'created_utc':utc(),'target_rows':len(rows),
        'all_temperature_size_seed_threshold_targets_pass':all(r['positive_lower_confidence_bound'] for r in primary),
        'all_perturbed_targets_pass':all(r['positive_lower_confidence_bound'] for r in perturbed) if perturbed else None,
        'minimum_primary_improvement':min(r['versus_test_linear_lower_bound']['improvement'] for r in primary),
        'minimum_primary_ci95_low':min(r['versus_test_linear_lower_bound']['ci95'][0] for r in primary),
        'minimum_perturbed_improvement':min((r['versus_test_linear_lower_bound']['improvement'] for r in perturbed),default=None),
        'passed_mixed_primary_targets':sum(r['positive_lower_confidence_bound'] and r['purity']<.95 for r in primary),
        'classification':'independent_boundary_phase_rule_validation; not a known topological/SPT/path rule',
        'MLP_gate':'requires robust interpretation and null-control review in addition to these tests',
        'no_target_optimization':True})


if __name__=='__main__':main()
