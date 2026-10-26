"""Paired-seed comparisons at fixed architecture/steps and observed resource thresholds."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc


def paired_statistics(reference,candidate,*,draws=10000):
    reference=np.asarray(reference);candidate=np.asarray(candidate)
    assert reference.shape==candidate.shape and reference.ndim==1 and len(reference)==8
    indices=np.random.default_rng(1927).integers(0,len(reference),(draws,len(reference)))
    difference=reference-candidate;difference_draws=difference[indices].mean(1)
    ratio=np.log10(np.maximum(candidate,1e-30)/np.maximum(reference,1e-30))
    ratio_draws=ratio[indices].mean(1);ci=np.quantile(ratio_draws,[.025,.975])
    return {'paired_seed_count':len(reference),'mean_reference_minus_candidate':float(difference.mean()),
            'improvement_ci95_low':float(np.quantile(difference_draws,.025)),
            'improvement_ci95_high':float(np.quantile(difference_draws,.975)),
            'geometric_candidate_over_reference':float(10**ratio.mean()),
            'ratio_ci95_low':float(10**ci[0]),'ratio_ci95_high':float(10**ci[1]),
            'candidate_wins':int((candidate<reference).sum())}


def main():
    out=ROOT/'results/signrule_round3';tables=out/'tables';rows=[]
    for p in (out/'learning').glob('n*/beta*/alpha*/w*/metrics.json'):
        data=json.loads(p.read_text())
        for row in data['rows']:
            rows.append({k:data[k] for k in ['n','beta','alpha','width','depth']}|row|
                        {'allocated_parameters':data['training']['allocated_parameters_per_model']})
    frame=pd.DataFrame(rows);assert len(frame)==5184
    pairs=[('linear_phase_amplitude','learned_CZ_phase_amplitude'),
           ('plain_original','learned_CZ_phase_amplitude'),
           ('plain_local_SU2','learned_CZ_phase_amplitude'),
           ('plain_original','plain_frozen_CZ_basis'),
           ('plain_original','plain_local_SU2'),
           ('oracle_element_phase_amplitude','learned_CZ_phase_amplitude')]
    metrics=['full_Frobenius_relative_squared','heldout_Frobenius_relative_squared','offdiagonal_weighted_phase_loss']
    keys=['n','beta','alpha','width','depth','step'];comparisons=[]
    for values,group in frame.groupby(keys):
        header=dict(zip(keys,values));assert group.allocated_parameters.nunique()==1 and group.estimated_training_FLOPs.nunique()==1
        for ref,cand in pairs:
            a=group[group.representation==ref].sort_values('seed_index');b=group[group.representation==cand].sort_values('seed_index')
            assert list(a.seed_index)==list(b.seed_index)==list(range(8))
            for metric in metrics:
                comparisons.append({**header,'reference':ref,'candidate':cand,'metric':metric,
                    'allocated_parameters':int(a.allocated_parameters.iloc[0]),'estimated_training_FLOPs':int(a.estimated_training_FLOPs.iloc[0]),
                    'reference_effective_parameters':int(a.effective_parameters.iloc[0]),'candidate_effective_parameters':int(b.effective_parameters.iloc[0]),
                    **paired_statistics(a[metric].values,b[metric].values)})
    paired=pd.DataFrame(comparisons);paired.to_csv(tables/'learning_paired_comparisons.csv',index=False)
    # Each threshold is checked only at recorded checkpoints; failed/censored
    # architectures remain explicit. No interpolated first-passage time claim.
    thresholds=[.1,.03,.01,.003,.001];resources=[]
    for values,group in frame.groupby(['n','beta','alpha','representation','seed_index']):
        header=dict(zip(['n','beta','alpha','representation','seed_index'],values))
        for threshold in thresholds:
            accepted=group[group.full_Frobenius_relative_squared<=threshold]
            row={**header,'target_full_error_squared':threshold,'achieved_in_recorded_grid':len(accepted)>0}
            for resource in ['allocated_parameters','effective_parameters','estimated_training_FLOPs']:
                if len(accepted):
                    best=accepted.sort_values([resource,'full_Frobenius_relative_squared']).iloc[0]
                    row['min_'+resource]=int(best[resource]);row[resource+'_w_d_step']=f'{int(best.width)}/{int(best.depth)}/{int(best.step)}'
                else:row['min_'+resource]=None;row[resource+'_w_d_step']=None
            resources.append(row)
    resource=pd.DataFrame(resources);resource.to_csv(tables/'learning_observed_resources.csv',index=False)
    counts=resource.groupby(['n','beta','alpha','representation','target_full_error_squared']).agg(
        successful_seeds=('achieved_in_recorded_grid','sum'),median_min_parameters=('min_effective_parameters','median'),
        median_min_FLOPs=('min_estimated_training_FLOPs','median')).reset_index()
    counts.to_csv(tables/'learning_resource_summary.csv',index=False)
    final=paired[(paired.step==40000)&(paired.metric==metrics[0])]
    effects=[]
    for (n,ref,cand),group in final.groupby(['n','reference','candidate']):
        unique=group if 'plain_local_SU2' in (ref,cand) else group[group.alpha==2]
        effects.append({'n':int(n),'reference':ref,'candidate':cand,'architecture_alpha_comparisons':len(group),
            'distinct_representation_conditions':len(unique),
            'distinct_candidate_lower_error_ci_excludes_equal':int((unique.ratio_ci95_high<1).sum()),
            'distinct_reference_lower_error_ci_excludes_equal':int((unique.ratio_ci95_low>1).sum()),
            'candidate_lower_error_ci_excludes_equal':int((group.ratio_ci95_high<1).sum()),
            'reference_lower_error_ci_excludes_equal':int((group.ratio_ci95_low>1).sum()),
            'min_geometric_ratio':float(group.geometric_candidate_over_reference.min()),
            'max_geometric_ratio':float(group.geometric_candidate_over_reference.max())})
    selected=paired[(paired.step==40000)&(paired.alpha==2)&(paired.metric==metrics[0])&
                    (paired.candidate=='learned_CZ_phase_amplitude')&paired.reference.isin(['linear_phase_amplitude','plain_local_SU2'])]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,n in zip(axes,[6,8]):
        for offset,(ref,group) in enumerate(selected[selected.n==n].groupby('reference')):
            group=group.sort_values(['width','depth']);x=np.arange(len(group))+.1*offset
            value=group.geometric_candidate_over_reference.to_numpy();low=group.ratio_ci95_low.to_numpy();high=group.ratio_ci95_high.to_numpy()
            ax.errorbar(x,value,yerr=np.maximum(0,np.array([value-low,high-value])),fmt='o',capsize=3,label=ref)
        ax.set(yscale='log',xticks=np.arange(6)+.05,xticklabels=['64/1','64/2','64/3','128/1','128/2','128/3'],
               xlabel='Width / depth',ylabel='Learned CZ / comparator error',title=f'n={n}, alpha=2, 40k steps')
        ax.axhline(1,color='gray',lw=.8);ax.legend(fontsize=7)
    fig.savefig(out/'figures/learning_paired_error_ratios.png',dpi=160);fig.savefig(out/'figures/learning_paired_error_ratios.pdf');plt.close(fig)
    summary={'updated_utc':utc(),'networks':1728,'checkpoint_rows':len(frame),'paired_comparison_rows':len(paired),
             'pairing':'same 8 initialization indices, initial layer weights and minibatch schedule across representations',
             'repeated_alpha_caveat':'alpha changes only the plain local-SU2 basis in these learning inputs. Other representations repeat identical training conditions across alpha; count 768 distinct representation/architecture/seed conditions within 1728 training instances, not 1728 independent statistical samples.',
             'matched_resource':'same n,width,depth and step: identical allocated parameters and estimated dense-kernel FLOPs',
             'effective_parameter_caveat':'positive-amplitude models leave one allocated output head unused; effective count is lower by width+1',
             'FLOPs_scope':'training matmul and Adam estimate; excludes data preparation, W optimization, oracle construction and activation costs',
             'phase_storage':'learned CZ needs two site indices; linear rule needs n coefficients; per-entry oracle needs d^2 phase scalars. Training caches for all three use d^2 entries.',
             'uncertainty':'paired seed bootstrap, 10000 draws; describes optimizer variability for fixed operators, not independent physical samples or a multiplicity-corrected discovery test',
             'resource_scope':'minimum observed resource among the recorded architectures/checkpoints; censored failures retained, not a global optimum or exact first hitting time',
             'effects_at_40k':effects,'conclusion':'bounded phase correction is tested against both linear phase and plain local-SU2 MLP; physical phase gain alone does not establish computational advantage'}
    atomic_json(out/'learning/comparison_summary.json',summary)
    print(json.dumps({'effects_at_40k':effects,'paired_rows':len(paired)},indent=2))


if __name__=='__main__':main()
