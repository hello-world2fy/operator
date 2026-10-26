"""Summarize independently validated signals, physical controls and conditional MLPs."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc


def main():
    out=ROOT/'results/signrule_round3';tables=out/'tables';figures=out/'figures';tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    summary={'updated_utc':utc(),'status':'in_progress_not_final'}
    for label,path in [('F_independent','independent_validation/frustration_transfer/summary.json'),
                       ('F_independent_n12','independent_validation/frustration_n12_transfer/summary.json'),
                       ('D_independent','independent_validation/Levin_Gu_transfer/summary.json'),
                       ('learning_gate','registration/phase3_gate.json')]:
        p=out/path
        if p.exists():summary[label]=json.loads(p.read_text())
    rows=[]
    for p in (out/'independent_validation').glob('frustration*transfer/n*/lambda*/beta*/summary.json'):
        for row in json.loads(p.read_text())['rows']:
            rows.append({k:v for k,v in row.items() if not isinstance(v,dict)}|{
                'phase_loss':row['test']['circular_loss'],'linear_phase_loss':row['test_best_linear']['circular_loss'],
                'improvement':row['versus_test_linear_lower_bound']['improvement'],
                'ci95_low':row['versus_test_linear_lower_bound']['ci95'][0],
                'ci95_high':row['versus_test_linear_lower_bound']['ci95'][1]})
    if rows:pd.DataFrame(rows).to_csv(tables/'F_independent_transfer.csv',index=False)
    projectors=[]
    for p in (out/'physical_B/projector_blind').glob('p*/*/alpha*/seed*/transfer.json'):
        d=json.loads(p.read_text());parts=p.relative_to(out/'physical_B/projector_blind').parts
        for cutoff,values in d['phase_scores'].items():
            for family,score in values['candidates'].items():
                c=score['versus_test_linear_lower_bound']
                projectors.append({'J2':float(parts[0][1:]),'object':parts[1],'alpha':float(parts[2][5:]),'seed':int(parts[3][4:]),
                    'family':family,'cutoff':cutoff,'target_n':12,'phase_loss':score['test']['circular_loss'],
                    'improvement':c['improvement'],'ci95_low':c['ci95'][0],'ci95_high':c['ci95'][1]})
    if projectors:
        frame=pd.DataFrame(projectors);frame.to_csv(tables/'B_projector_transfer.csv',index=False)
        summary['B_projector_validation']={'frozen_source_projectors':len(projectors)//6,
            'positive_individual_target_intervals':int((frame.ci95_low>0).sum()),
            'all_target_intervals_positive':bool((frame.ci95_low>0).all()),'pure_state_only':True,
            'finite_sector_degeneracy_caveat':True}
    p=out/'physical_C/thermal_pair/summary.json'
    if p.exists():
        data=json.loads(p.read_text());summary['C_thermal_pair']={'spectrum_max_difference':data['spectrum_max_difference'],
            'max_same_beta_modulus_Frobenius':max(x['same_beta_modulus_Frobenius'] for x in data['rows']),
            'max_matched_purity_residual':max(x['matched_purity_residual'] for x in data['rows']),
            'scope':'only the tested projected commuting-projector 2x2 honeycomb realization'}
        paired=[]
        for row in data['rows']:
            for model,m in row['models'].items():
                for sector in m['closed_sector_weights']:
                    paired.append({'beta':row['beta'],'model':model,'purity':m['purity'],'vertex_defect_weight':m['vertex_defect_weight'],
                        'homology':str(sector['homology']),'sector_weight':sector['trace_weight'],'sector_purity':sector['conditional_purity'],
                        'loop_reference_ground_residual':sector['contractible_loop_reference_ground_residual'],
                        'loop_reference_validated':sector['loop_reference_validated_in_this_sector']})
        pd.DataFrame(paired).to_csv(tables/'C_thermal_sector_pairs.csv',index=False)
    path_rows=[]
    for root in ['physical_E','physical_E_large']:
        for p in (out/root).glob('**/path_summary.json'):
            d=json.loads(p.read_text());graphs=d['complete_H_support_balance']
            for row in d['rows']:
                path_rows.append({'path':str(p.parent.relative_to(out)),**row,
                    'tJ_full_H_balanced':graphs['tJ']['balanced'],'sigma_tJ_full_H_balanced':graphs['sigma_tJ']['balanced']})
    if path_rows:
        pd.DataFrame(path_rows).to_csv(tables/'E_actual_closed_paths.csv',index=False)
        summary['E_path_validation']={'rows':len(path_rows),'phase_string_identity_violations':sum(x['phase_string_violations'] for x in path_rows),
            'sigma_path_violations':sum(x['sigma_nontrivial'] for x in path_rows),
            'large_geometry_thermal_diagnostics':len(list((out/'physical_E_large').glob('*/t*/beta*/summary.json'))),
            'large_propagator_coverage':'64 exact propagator columns; full H support checked separately; thermal diagonal estimated with 64 probes'}
    null_rows=[]
    for model in ['controlled_frustration','levin_gu']:
        for p in (out/'negative_controls'/model).glob('*/draw*/metrics.json'):
            m=json.loads(p.read_text());score=m['phase_scores']['1e-08']['versus_test_linear_lower_bound']
            null_rows.append({k:m[k] for k in ['model','n','family','draw','beta','alpha','purity','W_fixed','W_local','W_circuit']}|
                {'phase_improvement':score['improvement'],'phase_ci95_low':score['ci95'][0],'phase_ci95_high':score['ci95'][1]})
    if null_rows:
        frame=pd.DataFrame(null_rows);frame.to_csv(tables/'same_spectrum_nulls.csv',index=False)
        summary['same_spectrum_nulls']=[{'model':model,'family':family,'draws':len(subset),
            'positive_phase_intervals':int((subset.phase_ci95_low>0).sum()),'max_phase_improvement':float(subset.phase_improvement.max())}
            for (model,family),subset in frame.groupby(['model','family'])]
    summary['C_random_phase_controls']=len(list((out/'negative_controls/loop_phase_scramble').glob('n*/draw*/metrics.json')))
    learning=[]
    for p in (out/'learning').glob('n*/beta*/alpha*/w*/metrics.json'):
        m=json.loads(p.read_text())
        for row in m['rows']:learning.append({k:m[k] for k in ['n','beta','alpha','width','depth']}|row)
    if learning:
        frame=pd.DataFrame(learning);frame.to_csv(tables/'learning_metrics.csv',index=False)
        grouped=frame.groupby(['n','beta','alpha','width','depth','representation','step']).agg(
            median_error=('full_Frobenius_relative_squared','median'),mean_error=('full_Frobenius_relative_squared','mean'),
            std_error=('full_Frobenius_relative_squared','std'),median_heldout=('heldout_Frobenius_relative_squared','median'),
            median_phase_loss=('offdiagonal_weighted_phase_loss','median'),effective_parameters=('effective_parameters','first'),
            FLOPs=('estimated_training_FLOPs','first')).reset_index()
        grouped.to_csv(tables/'learning_seed_summary.csv',index=False)
        summary['learning']={'completed_architecture_batches':len(learning)//144,'planned_architecture_batches':36,
            'completed_independent_networks':len(learning)//3,'planned_independent_networks':1728,
            'phase_floor_checked':True,'Hermitian_partners_share_holdout_split':True,
            'interpretation':'test a bounded phase correction; oracle lookup storage and nonzero learned-phase error floor are reported'}
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for axis,n in zip(axes,[6,8]):
            subset=grouped[(grouped.n==n)&(grouped.alpha==2)&(grouped.step==40000)]
            for representation,data in subset.groupby('representation'):
                axis.scatter(data.effective_parameters,data.median_error,s=22,label=representation)
            axis.set(xscale='log',yscale='log',xlabel='Effective network parameters',ylabel='Median full relative Frobenius error squared',title=f'n={n}, alpha=2, 40k steps')
            if len(subset):axis.legend(fontsize=6)
        fig.savefig(figures/'learning_error_vs_parameters.png',dpi=160);fig.savefig(figures/'learning_error_vs_parameters.pdf');plt.close(fig)
    atomic_json(out/'supplementary_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['F_independent','D_independent','learning_gate']},indent=2))


if __name__=='__main__':main()
