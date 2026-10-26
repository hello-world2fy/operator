"""Validate the native n8 two-boundary candidate against saved independent operators."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc
from operator_walshness.signrule.blind import circuit_phase,circuit_depth
from operator_walshness.signrule.diagnostics import best_linear_phase,score_phase,paired_improvement_ci


def main():
    out=ROOT/'results/signrule_round3';dest=out/'independent_validation/frustration_double_boundary';dest.mkdir(parents=True,exist_ok=True)
    sources=[];selected=None
    for seed in [20261027,20261028,20261029]:
        p=out/'controlled_frustration/chain_open/n8/p0.4/beta4/alpha2'/f'seed{seed}'
        assert json.loads((p/'blind_frozen.json').read_text())['known_label_access'] is False
        result=json.loads((p/'basis_or_circuit.json').read_text())['circuit_L2']
        active=[s for s,c in zip(result['gate_supports'],result['coefficients_best']) if c]
        assert sorted(active)==[[1,2],[5,6]];sources.append(str(p.relative_to(out)));selected=result
    plan={}
    for n in [6,8,10,12]:
        supports=[(1,2),(n-3,n-2)]
        plan[n]={**selected,'gate_supports':supports,'coefficients_best':[1,1],'gate_count':2,
                 'circuit_depth':circuit_depth(supports,[1,1]),'transfer_rule':'active source gates anchored by distance to nearest open boundary',
                 'target_label_or_W_access':False}
    if not (dest/'registration.json').exists():atomic_json(dest/'registration.json',{'registered_utc':utc(),
        'source_points':sources,'identical_source_rules_deduplicated':True,'transfers':plan,
        'target_operators':'previously generated independent n6/8/10/12 validation operators; these targets were not used to optimize the native n8 source'})
    all_rows=[]
    targets=list((out/'independent_validation/frustration_transfer').glob('n*/lambda*/beta*/summary.json'))
    targets+=list((out/'independent_validation/frustration_n12_transfer').glob('n*/lambda*/beta*/summary.json'))
    for path in sorted(targets):
        reference=json.loads(path.read_text())['rows'][0];n=reference['n'];lam=reference['lambda'];beta=reference['beta']
        result_path=dest/f'n{n}'/f'lambda{lam:g}'/f'beta{beta:g}'/'summary.json'
        if result_path.exists():all_rows.extend(json.loads(result_path.read_text())['rows']);continue
        phase=circuit_phase(n,plan[n]);rows=[];cache={}
        with np.load(path.parent/'sign_diagnostics.npz') as a:
            for cutoff in [1e-8,1e-12]:
                tag=f'{cutoff:g}';entries={k:a[f'{tag}_{k}'] for k in ['row','col','phase','weight','test']}
                lower,_=best_linear_phase(entries,n,'test');key=len(entries['row'])
                if key not in cache:cache[key]=paired_improvement_ci(entries,lower,phase)
                delta=cache[key];rows.append({'n':n,'lambda':lam,'beta':beta,'cutoff':cutoff,'purity':reference['purity'],
                    'test':score_phase(entries,phase),'test_best_linear':score_phase(entries,lower),'versus_test_linear':delta,
                    'passed':delta['ci95'][0]>1e-6,'target_optimized':False,'source_seeds_share_identical_rule':3})
        atomic_json(result_path,{'rows':rows,'created_utc':utc()});all_rows+=rows
        print('DOUBLE_BOUNDARY',n,lam,beta,flush=True)
    primary=[r for r in all_rows if r['lambda']==.4];perturbed=[r for r in all_rows if r['lambda']!=.4]
    atomic_json(dest/'summary.json',{'created_utc':utc(),'rows':len(all_rows),'primary_rows':len(primary),
        'all_primary_pass':all(r['passed'] for r in primary),'all_perturbed_pass':all(r['passed'] for r in perturbed),
        'min_primary_improvement':min(r['versus_test_linear']['improvement'] for r in primary),
        'min_primary_ci95_low':min(r['versus_test_linear']['ci95'][0] for r in primary),
        'min_perturbed_improvement':min(r['versus_test_linear']['improvement'] for r in perturbed),
        'interpretation':'two-boundary version of the same bounded local CZ correction; not a bulk/topological rule or proof of computational benefit'})


if __name__=='__main__':main()
