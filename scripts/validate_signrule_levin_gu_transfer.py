"""Predict unseen temperatures and n=12 geometry using frozen n=9 circuits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.geometry import triangular
from operator_walshness.signrule.models import levin_gu,paramagnet_gibbs
from operator_walshness.signrule.blind import circuit_phase,fwht_torch_flat
from operator_walshness.signrule.transfer import transfer_triangular_circuit,exact_grouped_phase_bootstrap
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase,score_phase
from operator_walshness.signrule.workflow import utc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:4');args=parser.parse_args();torch.set_num_threads(1)
    out=ROOT/'results/signrule_round3';dest=out/'independent_validation'/'Levin_Gu_transfer';dest.mkdir(parents=True,exist_ok=True)
    plan_path=dest/'frozen_transfer_plan.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())
    else:
        targets={};sources=[]
        for seed in [20261027,20261028,20261029]:
            source=out/'levin_gu/3x3_periodic/n9/p0/beta0.5/alpha1'/f'seed{seed}'
            frozen=json.loads((source/'blind_frozen.json').read_text());payload=(source/'basis_or_circuit.json').read_bytes()
            assert hashlib.sha256(payload).hexdigest()==frozen['basis_or_circuit_sha256']
            results=json.loads(payload);sources.append(str(source.relative_to(out)))
            for shape in [(3,3),(3,4)]:
                name=f'{shape[0]}x{shape[1]}'
                for label,result in results.items():
                    if not label.startswith('circuit'):continue
                    targets[f'{name}_{seed}_{label}']=transfer_triangular_circuit(result,(3,3),shape)
        plan={'frozen_utc':utc(),'source_points':sources,'targets':targets,'target_beta':[.75,1.5],
            'target_shapes':[[3,3],[3,4]],'no_target_phase_or_objective_in_transfer':True,
            'alpha':1.,'geometry_extension':'repeat anchored gate coefficients modulo source cell; retain target-local faces/bonds',
            'gate_cap_rule':'floor(source cap times target/source sites), fixed geometric ordering if trimming needed',
            'success_rule':'beyond best test-set linear lower bound at both thresholds, both unseen beta, n12, and all 3 source seeds',
            'subtlety':'g=0 is exactly solvable; even successful partial fingerprints are not evidence for a novel phase'}
        atomic_json(plan_path,plan)
    rows=[]
    for shape in plan['target_shapes']:
        g=triangular(*shape);_,oracle,_=levin_gu(g)
        n=g.n;degrees=np.bitwise_count(np.arange(1<<n,dtype=np.uint64)).astype(float)
        weight=torch.tensor(np.exp(degrees[:,None]+degrees[None,:]-2*n),device=args.device)
        for beta in plan['target_beta']:
            target=dest/f'{shape[0]}x{shape[1]}'/f'beta{beta:g}';target.mkdir(parents=True,exist_ok=True)
            if (target/'summary.json').exists():rows+=json.loads((target/'summary.json').read_text())['scores'];continue
            rho=paramagnet_gibbs(n,beta,oracle);purity=float(np.vdot(rho,rho).real)
            op=torch.tensor(rho,dtype=torch.float64,device=args.device)
            def W(phase):
                ph=torch.tensor(phase,dtype=torch.float64,device=args.device)
                x=fwht_torch_flat((op*ph[:,None]*ph[None,:]).reshape(1,-1)).reshape(1<<n,1<<n)
                return float(torch.log((abs(x)**2*weight).sum()/purity)+2*n)
            fixed=W(np.ones(1<<n));scores=[];arrays={}
            for cutoff in [1e-8,1e-12]:
                entries=phase_entries(rho,cutoff);linear,mask=best_linear_phase(entries,n,'test')
                tag=f'{cutoff:g}';arrays[f'{tag}_linear_phase']=linear
                for key in ['row','col','phase','weight','test']:arrays[f'{tag}_{key}']=entries[key]
                for name,result in plan['targets'].items():
                    if not name.startswith(f'{shape[0]}x{shape[1]}_'):continue
                    phase=circuit_phase(n,result);row={'candidate':name,'shape':shape,'beta':beta,'purity':purity,
                        'cutoff':cutoff,'phase_score':score_phase(entries,phase),'linear_lower_bound':score_phase(entries,linear),
                        'beyond_linear':exact_grouped_phase_bootstrap(entries,linear,phase),
                        'gate_count':result['gate_count'],'gate_depth':result['circuit_depth'],'W_fixed':fixed,'W_transferred':W(phase),
                        'target_optimization_performed':False,'gatewise_Z2_symmetric':result['gatewise_Z2_symmetric']}
                    scores.append(row)
            np.savez_compressed(target/'sign_diagnostics.npz',**arrays)
            atomic_json(target/'summary.json',{'created_utc':utc(),'source_plan':'../../frozen_transfer_plan.json','scores':scores,
                'operator':'exact mixed Gibbs at solvable point','purity':purity,'new_geometry':n==12,'new_temperature':True})
            rows+=scores;print('TRANSFER',shape,beta,'done',flush=True)
    candidate_names=['circuit_L1','circuit_L2','circuit_L3','circuit_symmetric'];certificates={}
    for label in candidate_names:
        selected=[row for row in rows if row['candidate'].endswith('_'+label)]
        certificate=bool(selected) and all(row['beyond_linear']['ci95'] is not None and row['beyond_linear']['ci95'][0]>1e-6 for row in selected)
        certificates[label]={'passed_all_target_seeds_temperatures_sizes_thresholds':certificate,
            'rows':len(selected),'minimum_improvement':min((row['beyond_linear']['improvement'] for row in selected),default=None),
            'minimum_ci95_low':min((row['beyond_linear']['ci95'][0] for row in selected if row['beyond_linear']['ci95']),default=None)}
    atomic_json(dest/'summary.json',{'created_utc':utc(),'certificates':certificates,
        'screening_scope':'frozen circuits at g0, n9 beta0.5 alpha1, transferred to beta0.75/1.5 and n12',
        'not_a_global_nonexistence_claim':True,'needs_perturbation_validation_before_MLP_gate':True})


if __name__=='__main__':main()
