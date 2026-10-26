"""Conditional small-system learning after independently validated non-linear phase gain."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import rotate_operator,directions_to_walsh_basis
from operator_walshness.signrule.geometry import chain
from operator_walshness.signrule.models import frustrated,block_eigensystem,block_gibbs
from operator_walshness.signrule.blind import circuit_phase
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase
from operator_walshness.signrule.learning import train_phase_batched,positive_phase_floor
from operator_walshness.signrule.workflow import utc
from validate_signrule_frustration_transfer import boundary_transfer


def register_gate(out):
    validation=json.loads((out/'independent_validation/frustration_transfer/summary.json').read_text())
    assert validation['all_temperature_size_seed_threshold_targets_pass']
    assert validation['passed_mixed_primary_targets']==36
    nulls=[json.loads(p.read_text()) for p in (out/'negative_controls/controlled_frustration').glob('*/draw*/metrics.json')]
    assert len(nulls)==60
    positive=sum(x['phase_scores']['1e-08']['versus_test_linear_lower_bound']['ci95'][0]>0 for x in nulls)
    assert positive==0
    gate={'registered_utc':utc(),'status':'enabled_for_bounded_controlled_frustration_application',
        'evidence':validation,'null_draws':60,'nulls_with_positive_ci95_lower_bound':positive,
        'scope':'lambda0.4 open-chain boundary CZ correction; not a topological, SPT or phase-string discovery',
        'coupling_caveat':'not all lambda0.35/0.45 targets pass; no claim across the full parameter interval',
        'statistical_caveat':'20 draws per family and finite-entry bootstrap are descriptive; not a corrected discovery p-value',
        'source_target':{'n':6,'beta':4.},'independent_learning_target':{'n':8,'beta':3.},
        'alpha':[.5,1.,2.],'width':[64,128],'depth':[1,2,3],'initializations':8,'checkpoints':[2000,10000,40000],
        'objective':'test compression benefit or failure of a frozen non-linear phase correction'}
    path=out/'registration/phase3_gate.json'
    if not path.exists():atomic_json(path,gate)


def prepare(out,n,beta,alpha):
    h=frustrated(chain(n),.4);rho,thermal=block_gibbs(block_eigensystem(h),beta,1<<n)
    source=out/'controlled_frustration/chain_open/n6/p0.4/beta4/alpha2/seed20261027'
    learned=json.loads((source/'basis_or_circuit.json').read_text())['circuit_L2']
    transferred=boundary_transfer(learned,6,n);phase=circuit_phase(n,transferred);pair=phase[:,None]*phase[None,:]
    local_path=out/'controlled_frustration/chain_open'/f'n{n}'/'p0.4/beta4'/f'alpha{alpha:g}'/'seed20261027/basis_or_circuit.json'
    local=json.loads(local_path.read_text())['local_SU2'];rotated=rotate_operator(rho,directions_to_walsh_basis(local['axes_best']))
    # Comparator is trained on the same prior phase-decoding split, then frozen.
    linear,_=best_linear_phase(phase_entries(rho),n);linear_pair=linear[:,None]*linear[None,:]
    oracle=np.ones_like(rho,dtype=complex);nonzero=abs(rho)>1e-15;oracle[nonzero]=rho[nonzero]/abs(rho[nonzero])
    names=['plain_original','plain_local_SU2','plain_frozen_CZ_basis','linear_phase_amplitude','learned_CZ_phase_amplitude','oracle_element_phase_amplitude']
    operators=[rho,rotated,rho*pair,rho,rho,rho]
    factors=[np.ones_like(rho),np.ones_like(rho),np.ones_like(rho),linear_pair,pair,oracle]
    amplitude=[False,False,False,True,True,True]
    targets=np.array([(o*(1<<n)/np.linalg.norm(o)).ravel() for o in operators])
    factors=np.array([p.ravel() for p in factors]);floors=[positive_phase_floor(t,p) if a else 0. for t,p,a in zip(targets,factors,amplitude)]
    metadata={'names':names,'amplitude_modes':amplitude,'phase_error_floors':floors,'n':n,'beta':beta,'alpha':alpha,
        'purity':thermal['purity'],'learned_phase':transferred,'phase_source':str(source.relative_to(out)),
        'local_basis_source':str(local_path.relative_to(out)),'local_basis_was_optimized_at_beta':4.,
        'representation_scope':'known sign formulas are not applied in the non-diagonal SU2 basis',
        'oracle_cost':'oracle is a full per-entry phase lookup upper bound, not a learned compact rule',
        'learned_CZ_rule_storage':'one two-site diagonal gate plus its geometry; arrays are evaluation caches only',
        'task':'fixed-operator compression; W and frozen basis selection may access the full source operator'}
    return targets,factors,amplitude,metadata


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();out=ROOT/'results/signrule_round3';register_gate(out);jobs=[]
    for n,beta in [(6,4.),(8,3.)]:
        for alpha in [.5,1.,2.]:
            for width in [64,128]:
                for depth in [1,2,3]:jobs.append((n,beta,alpha,width,depth))
    for index,(n,beta,alpha,width,depth) in enumerate(jobs):
        if index%args.workers!=args.worker:continue
        point=out/'learning'/f'n{n}'/f'beta{beta:g}'/f'alpha{alpha:g}'/f'w{width}_d{depth}'
        if (point/'metrics.json').exists():continue
        point.mkdir(parents=True,exist_ok=True);targets,phases,modes,metadata=prepare(out,n,beta,alpha)
        atomic_json(point/'metadata.json',metadata)
        def progress(step,values,seconds):
            atomic_json(point/'progress.json',{'step':step,'median_full_error':float(np.median(values[:,0])),'seconds':seconds})
            print('LEARNING_CHECKPOINT',n,beta,alpha,width,depth,step,float(np.median(values[:,0])),flush=True)
        result=train_phase_batched(targets,phases,modes,n,width,depth,device=args.device,progress=progress)
        np.savez_compressed(point/'final_weights.npz',**result.pop('model_state'))
        np.savez_compressed(point/'learning_curve.npz',steps=result.pop('curve_steps'),loss=result.pop('curve_loss'))
        rows=[]
        for step,evaluation in result.pop('evaluations').items():
            for mode,name in enumerate(metadata['names']):
                for seed in range(8):
                    full,held,phase=evaluation[mode*8+seed]
                    floor=metadata['phase_error_floors'][mode]
                    if full<floor-1e-5:raise ArithmeticError('positive-amplitude phase floor violation')
                    rows.append({'representation':name,'seed_index':seed,'step':step,'full_Frobenius_relative_squared':float(full),
                        'heldout_Frobenius_relative_squared':float(held),'offdiagonal_weighted_phase_loss':float(phase),
                        'irreducible_positive_phase_floor':floor,'effective_parameters':result['effective_parameters_per_model'][mode],
                        'estimated_training_FLOPs':step*result['estimated_training_flops_per_step']})
        atomic_json(point/'metrics.json',{'rows':rows,'training':result,'width':width,'depth':depth,**metadata})
        print('LEARNING_DONE',n,beta,alpha,width,depth,flush=True)


if __name__=='__main__':main()
