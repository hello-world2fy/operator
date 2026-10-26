"""Temperature-screen every distinct preliminary LG candidate, before size expansion."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc
from operator_walshness.signrule.geometry import triangular
from operator_walshness.signrule.models import levin_gu,paramagnet_gibbs
from operator_walshness.signrule.blind import circuit_phase
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase,score_phase
from operator_walshness.signrule.transfer import exact_grouped_phase_bootstrap


def main():
    out=ROOT/'results/signrule_round3';dest=out/'independent_validation/Levin_Gu_all_candidates';dest.mkdir(parents=True,exist_ok=True)
    plan_path=dest/'frozen_plan.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())
    else:
        unique={}
        for p in sorted((out/'levin_gu').glob('*/n9/p0/*/alpha*/seed*/posthoc.json')):
            scored=json.loads(p.read_text());results=json.loads((p.parent/'basis_or_circuit.json').read_text())
            for family in scored.get('preliminary_non_Marshall_candidates',[]):
                result=results[family];key=(family,tuple(result['coefficients_best']))
                if key not in unique:unique[key]={'family':family,'circuit':result,'source_points':[]}
                unique[key]['source_points'].append(str(p.parent.relative_to(out)))
        candidates=[{'candidate_id':i,**v} for i,v in enumerate(unique.values())]
        plan={'registered_utc':utc(),'candidates':candidates,'target_beta':[.75,1.5],
              'selection':'all distinct preliminary positive circuit coefficient vectors, deduplicated within family',
              'rule':'require positive CI lower bound at both new temperatures and both cutoffs before further size/perturbation work',
              'target_optimized':False}
        atomic_json(plan_path,plan)
    _,oracle,_=levin_gu(triangular(3,3));rows=[]
    for beta in plan['target_beta']:
        rho=paramagnet_gibbs(9,beta,oracle)
        for cutoff in [1e-8,1e-12]:
            entries=phase_entries(rho,cutoff);linear,_=best_linear_phase(entries,9,'test')
            for candidate in plan['candidates']:
                phase=circuit_phase(9,candidate['circuit']);delta=exact_grouped_phase_bootstrap(entries,linear,phase)
                rows.append({'candidate_id':candidate['candidate_id'],'family':candidate['family'],'beta':beta,'cutoff':cutoff,
                    'purity':float(np.vdot(rho,rho).real),'test':score_phase(entries,phase),'versus_test_linear':delta,
                    'passed':delta['ci95'][0]>1e-6,'no_target_fitting':True})
    survivors=[c['candidate_id'] for c in plan['candidates'] if all(r['passed'] for r in rows if r['candidate_id']==c['candidate_id'])]
    atomic_json(dest/'summary.json',{'created_utc':utc(),'distinct_preliminary_candidates':len(plan['candidates']),
        'source_candidate_occurrences':sum(len(c['source_points']) for c in plan['candidates']),
        'temperature_validation_rows':len(rows),'survivors_needing_size_and_perturbation_tests':survivors,'rows':rows,
        'interpretation':'each rejected frozen candidate fails at least one independent-temperature/cutoff test; this is not a global nonexistence claim'})
    print(json.dumps({'candidates':len(plan['candidates']),'survivors':survivors}))


if __name__=='__main__':main()
