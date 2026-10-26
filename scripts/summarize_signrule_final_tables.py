"""Aggregate feasible spectral tails and physical evidence without changing frozen candidates."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from operator_walshness.walsh import tail_degree
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc


def main():
    out=ROOT/'results/signrule_round3';tables=out/'tables';tails=[];baselines=[];thermal=[]
    paths=list(out.glob('*/*/n*/*/*/alpha*/seed*/metrics.json'))
    for p in paths:
        m=json.loads(p.read_text());point=p.parent;relative=str(point.relative_to(out));base={k:m.get(k) for k in ['track','model','n','beta','alpha','seed','purity']}
        base.update(path=relative,metric='Z3_support_degree' if m['track']=='E' else 'qubit_W')
        spectrum=point/'walsh_spectrum.npz'
        if spectrum.exists():
            with np.load(spectrum) as a:
                for name in a.files:
                    if name.startswith('degree_'):continue
                    v=a[name];k=np.arange(len(v))
                    tails.append({**base,'representation':name,'degree_mean':float(v@k),'degree_variance':float(v@(k*k)-(v@k)**2),
                        'D95':tail_degree(v,.05),'D99':tail_degree(v,.01),'D999':tail_degree(v,.001),'mass_above_n':float(v[m['n']+1:].sum())})
        b=json.loads((point/'representation_baselines.json').read_text())
        scalars={key:value for key,value in b.items() if isinstance(value,(float,int)) and not isinstance(value,bool)}
        reference=b.get('theory_or_fitted_reference')
        if reference:scalars.update(W_reference=reference['W'],reference_name=reference['name'])
        baselines.append({**base,**scalars})
        if (point/'thermal_spectrum.npz').exists() and m['alpha']==1 and m['seed']==20261027:
            with np.load(point/'thermal_spectrum.npz') as a:e=a['energies'];prob=a['eigenvalues']
            ground=abs(e-e.min())<1e-9
            thermal.append({**base,'R_eff':1/m['purity'],'energy':float(prob@e),'ground_degeneracy':int(ground.sum()),
                'ground_manifold_weight':float(prob[ground].sum()),'gap':float((e[e>e.min()+1e-9]-e.min()).min(initial=np.inf)),
                'pure_or_mixed':'mixed_operator_candidate' if m['beta']>0 and m['purity']<.95 else 'identity_or_near_pure'})
    pd.DataFrame(tails).to_csv(tables/'degree_tail_summary.csv',index=False)
    pd.DataFrame(baselines).to_csv(tables/'representation_baselines.csv',index=False)
    pd.DataFrame(thermal).to_csv(tables/'thermal_rank_summary.csv',index=False)
    gamma=[]
    for p in out.glob('*/*/n*/*/*/physics_F/matched_real_optimization.json'):
        d=json.loads(p.read_text());parts=p.relative_to(out).parts
        for name,value in d['comparisons'].items():gamma.append({'path':str(p.relative_to(out)),'model':parts[0],'n':int(parts[2][1:]),
            'parameter':float(parts[3][1:]),'beta':float(parts[4][4:]),'objective':name,'gamma':value['gamma']['gamma'],'W_alpha1':value['W_alpha1'],
            'starts':d['starts'],'eval_cap_per_start':d['eval_cap_per_start']})
    pd.DataFrame(gamma).to_csv(tables/'F_matched_real_QMC.csv',index=False)
    he=[]
    for p in (out/'physical_B/projector_HE').glob('p*/*/summary.json'):
        d=json.loads(p.read_text())
        for cutoff,score in d['phase_scores'].items():
            for name,fit in score['fits'].items():he.append({'J2':d['J2'],'object':d['object'],'cutoff':cutoff,'template':name,
                'phase_loss':fit['test']['circular_loss'],'phase_locking':fit['test']['phase_locking'],
                'parameters':fit['parameter_count'],'structure_factor_K':d['structure_factor_K'],'OSEE':d['OSEE_fixed']})
    pd.DataFrame(he).to_csv(tables/'B_projector_HE.csv',index=False)
    d=json.loads((out/'independent_validation/Levin_Gu_all_candidates/summary.json').read_text())
    lg=[{k:v for k,v in r.items() if not isinstance(v,dict)}|{'phase_loss':r['test']['circular_loss'],
        'improvement':r['versus_test_linear']['improvement'],'ci95_low':r['versus_test_linear']['ci95'][0],
        'ci95_high':r['versus_test_linear']['ci95'][1]} for r in d['rows']]
    pd.DataFrame(lg).to_csv(tables/'D_all_candidates_validation.csv',index=False)
    double=[]
    for p in (out/'independent_validation/frustration_double_boundary').glob('n*/lambda*/beta*/summary.json'):
        for r in json.loads(p.read_text())['rows']:
            double.append({k:v for k,v in r.items() if not isinstance(v,dict)}|{'phase_loss':r['test']['circular_loss'],
                'linear_phase_loss':r['test_best_linear']['circular_loss'],'improvement':r['versus_test_linear']['improvement'],
                'ci95_low':r['versus_test_linear']['ci95'][0],'ci95_high':r['versus_test_linear']['ci95'][1]})
    pd.DataFrame(double).to_csv(tables/'F_double_boundary_validation.csv',index=False)
    atomic_json(out/'final_table_inventory.json',{'updated_utc':utc(),'primary_points':len(paths),'spectral_tail_rows':len(tails),
        'baseline_rows':len(baselines),'distinct_thermal_spectra':len(thermal),'matched_real_QMC_rows':len(gamma),
        'HE_projector_template_rows':len(he),'LG_all_candidate_rows':len(lg),'double_boundary_rows':len(double)})
    print(json.dumps({'points':len(paths),'tail_rows':len(tails),'baseline_rows':len(baselines),'QMC_rows':len(gamma)}))


if __name__=='__main__':main()
