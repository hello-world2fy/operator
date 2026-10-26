"""n=10/12 control identities and frozen-rule generalization, without target fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import fixed_basis_spectrum,walshness_from_spectrum,identity_walshness
from operator_walshness.signrule.workflow import build_qubit_job,utc
from operator_walshness.signrule.models import block_eigensystem,block_gibbs,paramagnet_gibbs
from operator_walshness.signrule.scoring import posthoc_oracle
from operator_walshness.signrule.blind import circuit_phase,circuit_depth


def periodic_rule_extension(result,source_n,target_n):
    learned={tuple(s):int(v) for s,v in zip(result['gate_supports'],result['coefficients_best'])}
    singles=[learned.get((i,),0) for i in range(source_n)]
    bonds=[learned.get((i,i+1),0) for i in range(source_n-1)]
    period=next(p for p in range(1,source_n+1) if all(all(x==seq[i%p] for i,x in enumerate(seq)) for seq in [singles,bonds]))
    supports=[(i,) for i in range(target_n)]+[(i,i+1) for i in range(target_n-1)]
    coeff=[singles[i%period] for i in range(target_n)]+[bonds[i%period] if i%period<len(bonds) else 0 for i in range(target_n-1)]
    active=np.flatnonzero(coeff);untrimmed=len(active)
    for i in active[target_n:]:coeff[i]=0
    return {**result,'gate_supports':supports,'coefficients_best':coeff,'gate_count':sum(coeff),
        'max_gates':target_n,'max_depth':target_n,'circuit_depth':circuit_depth(supports,coeff),
        'source_inferred_spatial_period':period,'untrimmed_gate_count':untrimmed,
        'rule':'smallest joint period of frozen one-site and bond coefficients; missing coefficients zero; fixed-order gate cap',
        'target_label_or_objective_access':False}


def full_phase_score(rho,phase,cutoff):
    diagonal=np.diag(rho);scale=abs(rho).max();total=good_mass=wrong_mass=0.;count=wrong=0
    for start in range(0,len(rho),128):
        stop=min(start+128,len(rho));values=rho[start:stop];mag=abs(values)
        mask=mag>max(cutoff*scale,np.finfo(float).tiny)
        mask[np.arange(stop-start),np.arange(start,stop)]=False
        prediction=phase[start:stop,None]*phase[None,:].conj()
        loss=abs(np.divide(values,mag,out=np.zeros_like(values,dtype=complex),where=mag>0)-prediction)**2/4
        weight=mag**2;good_mass+=float(weight[mask].sum());wrong_mass+=float((weight*loss)[mask].sum())
        count+=int(mask.sum());wrong+=int((mask&(loss>1e-12)).sum())
    return {'weighted_phase_loss':wrong_mass/good_mass if good_mass else None,
            'reliable_directed_offdiagonal_entries':count,'unweighted_phase_agreement':1-wrong/count if count else None,
            'full_operator_entries_checked':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,choices=[10,12],required=True);args=parser.parse_args()
    n=args.n;out=ROOT/'results/signrule_round3';dest=out/'independent_validation/large_known_controls'/f'n{n}';dest.mkdir(parents=True,exist_ok=True)
    models=['bipartite','cluster','paramagnet','random_linear','random_quadratic','random_positive'];frozen={}
    for model in models:
        size=6 if model=='random_positive' else 8
        for seed in [20261027,20261028,20261029]:
            source=out/model/'chain_open'/f'n{size}'/'p0/beta1/alpha1'/f'seed{seed}'
            assert json.loads((source/'blind_frozen.json').read_text())['known_label_access'] is False
            candidate=json.loads((source/'basis_or_circuit.json').read_text())['circuit_L2']
            frozen.setdefault(model,{})[seed]=periodic_rule_extension(candidate,size,n)
    atomic_json(dest/'transfer_registration.json',{'registered_utc':utc(),'source_beta':1.,'source_alpha':1.,'frozen_transfers':frozen,
        'target_optimization':False,'validation_scope':'known exact control identities and source-only phase-rule extension, not a native large blind grid'})
    rows=[]
    for model in models:
        job={'model':model,'n':n,'track':'A'};g,h=build_qubit_job(job);oracle,name=posthoc_oracle(job,g)
        analytic=model in ['cluster','paramagnet','random_linear','random_quadratic']
        blocks=None if analytic else block_eigensystem(h)
        for beta in [0.,.25,.5,1.,2.,4.]:
            point=dest/model/f'beta{beta:g}'
            if (point/'summary.json').exists():rows.append(json.loads((point/'summary.json').read_text()));continue
            if analytic:
                rho=paramagnet_gibbs(n,beta,oracle);thermal={'purity':float(np.vdot(rho,rho).real)}
            else:rho,thermal=block_gibbs(blocks,beta,1<<n)
            positive=rho*oracle[:,None]*oracle[None,:]
            fixed,check=fixed_basis_spectrum(rho);reference,_=fixed_basis_spectrum(positive)
            if check['mass_relative_error']>1e-10:raise ArithmeticError('large-control FWHT normalization')
            if np.min(positive)<-1e-11:raise ArithmeticError('known control phase does not make Gibbs entries nonnegative')
            scores={};cache={}
            for seed,transfer in frozen[model].items():
                phase=circuit_phase(n,transfer);key=np.packbits(phase<0).tobytes()
                if key not in cache:cache[key]={str(c):full_phase_score(rho,phase,c) for c in [1e-8,1e-12]}
                scores[seed]=cache[key]
            values={str(a):{'W_fixed':walshness_from_spectrum(fixed,a),'W_theory_reference':walshness_from_spectrum(reference,a)} for a in [.5,1.,2.]}
            if beta==0:
                for a in [.5,1.,2.]:assert abs(values[str(a)]['W_fixed']-identity_walshness(n,a))<1e-10
            point.mkdir(parents=True,exist_ok=True);np.savez_compressed(point/'walsh_spectrum.npz',fixed=fixed,theory_reference=reference)
            row={'n':n,'model':model,'beta':beta,'purity':thermal['purity'],'W':values,'frozen_rule_scores':scores,
                'oracle_name':name,'oracle_full_matrix_negative_min':float(min(0.,positive.min())),
                'trace_residual':float(abs(np.trace(rho)-1)),'Hermiticity_max':float(abs(rho-rho.conj().T).max()),
                'target_optimized':False,'native_large_grid_not_claimed':True,'full_Gibbs_operator':True}
            atomic_json(point/'summary.json',row);rows.append(row);print('LARGE_CONTROL',n,model,beta,flush=True)
    atomic_json(dest/'summary.json',{'created_utc':utc(),'n':n,'models':models,'beta_points':len(rows),
        'all_exact_oracle_checks_passed':True,'all_identity_checks_passed':True,'target_optimized':False,
        'interpretation':'frozen rule extension can fail for newly drawn random scrambles; known identities do not establish blind universal recovery'})


if __name__=='__main__':main()
