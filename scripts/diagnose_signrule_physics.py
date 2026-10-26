"""Independent B/D/F observables for already frozen qubit experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import eigh
from operator_walshness.round2 import atomic_json,optimize_real_basis,real_unitaries
from operator_walshness.walsh import rotate_operator,directions_to_walsh_basis,walshness,state_walshness
from operator_walshness.signrule.workflow import build_qubit_job,point_directory,utc
from operator_walshness.signrule.models import block_eigensystem,block_gibbs,levin_gu,paramagnet_gibbs
from operator_walshness.signrule.huse_elser import (reference_features,trial_wavefunction,fit_circular_phase,
    structure_factor_K,axis_texture)
from operator_walshness.signrule.diagnostics import (phase_entries,score_phase,best_linear_phase,
    paired_improvement_ci,support_balance,strange_correlator)
from operator_walshness.signrule.physical_diagnostics import worldline_phase_severity


def clean_balance(operator,cutoff):
    result=support_balance(operator,cutoff,witness_limit=24);result.pop('gauge');return result


def diagnose_B(job,out):
    g,h=build_qubit_job(job);blocks=block_eigensystem(h);sz,sub,cubic,reference=reference_features(g)
    features={'linear_U1':sz,'three_sublattice_U1':sub,'linear_plus_HE_three_spin':np.column_stack([sz,cubic])}
    for beta in job['betas']:
        ready=[point_directory(out,job,beta,a,s) for a in job['alphas'] for s in job['seeds']
               if (point_directory(out,job,beta,a,s)/'metrics.json').exists()]
        if not ready:continue
        dest=ready[0].parents[1]/'physics_B'
        rho,thermal=block_gibbs(blocks,beta,1<<g.n)
        if not (dest/'summary.json').exists():
            scores={};arrays={}
            for cutoff in [1e-8,1e-12]:
                entries=phase_entries(rho,cutoff);linear,mask=best_linear_phase(entries,g.n)
                fits={};predictions={}
                for name,feature in features.items():
                    phase,fit=fit_circular_phase(entries,feature,starts=16,maxiter=300)
                    fits[name]=fit;predictions[name]=phase;arrays[f'{cutoff:g}_{name}_phase']=phase
                scores[f'{cutoff:g}']={'fits':fits,'Boolean_linear_mask':mask,'Boolean_linear_test':score_phase(entries,linear),
                    'HE_versus_U1_linear':paired_improvement_ci(entries,predictions['linear_U1'],predictions['linear_plus_HE_three_spin']),
                    'HE_versus_Boolean_linear':paired_improvement_ci(entries,linear,predictions['linear_plus_HE_three_spin'])}
            structure,correlations=structure_factor_K(rho,g);arrays['spin_correlation']=correlations
            dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'phase_fits.npz',**arrays)
            atomic_json(dest/'summary.json',{'created_utc':utc(),'beta':beta,'J2':job.get('parameter',0),'n':g.n,
                'structure_factor_K':structure,'purity':thermal['purity'],'gap':thermal['gap'],
                'ground_degeneracy':thermal['ground_degeneracy'],'ground_manifold_weight':thermal['ground_manifold_weight'],
                'HE_reference':reference,'phase_scores':scores,
                'interpretation':'posthoc variational template fits; neither an exact HE oracle nor evidence of blind recovery by itself'})
        for point in ready:
            path=point/'basis_texture.json'
            if path.exists():continue
            results=json.loads((point/'basis_or_circuit.json').read_text())
            if 'local_SU2' not in results:continue
            candidate=results['local_SU2'];texture=axis_texture(candidate['axes_best'],g)
            texture['restart_texture']=[axis_texture(a,g)['sublattice_axis_dot_products'] for a in candidate['axes_restarts']]
            atomic_json(path,texture)
    # Finite-cluster physical ground-state validation within fixed Sz, including
    # degeneracy labels; no claim that a real vector in a degenerate sector is unique.
    base=point_directory(out,job,0,1,job['seeds'][0]).parents[2]/'projector_references'
    if not (base/'summary.json').exists():
        target=g.n//2;indices=np.flatnonzero(np.bitwise_count(np.arange(1<<g.n,dtype=np.uint64))==target)
        e,v=eigh(h[indices][:,indices].toarray(),check_finite=False)
        first_excited=np.flatnonzero(e>e[0]+1e-8)[0]
        rows=[];arrays={'sector_energies':e,'sector_configs':indices}
        for label,k in [('ground',0),('selected_excited',int(first_excited))]:
            psi=np.zeros(1<<g.n);psi[indices]=v[:,k]
            arrays[label+'_state']=psi
            rows.append({'object':label,'energy':float(e[k]),'sector':'fixed_n_down','n_down':target,
                'degeneracy_in_sector':int(np.sum(abs(e-e[k])<1e-8)),
                'residual':float(np.linalg.norm(h@psi-e[k]*psi)),
                'W_projector_fixed':{str(a):2*state_walshness(psi,a) for a in job['alphas']}})
        for real in [False,True]:
            trial,meta=trial_wavefunction(g,real_part=real,n_down=target)
            label='HE_real_trial' if real else 'HE_complex_trial';arrays[label]=trial
            for name,state in [(label,trial),(label+'_same_moduli',abs(trial))]:
                rows.append({'object':name,'energy':float(np.vdot(state,h@state).real),
                    'W_projector_fixed':{str(a):2*state_walshness(state,a) for a in job['alphas']},'pure_carryover':True})
        base.mkdir(parents=True,exist_ok=True);np.savez_compressed(base/'states.npz',**arrays)
        atomic_json(base/'summary.json',{'objects':rows,'HE_reference':reference,'phase_rule_not_exact':True,
            'blind_ground_projector_optimization':'pending independent stage'})


def diagnose_D(job,out):
    g,h=build_qubit_job(job);blocks=block_eigensystem(h);_,oracle,_=levin_gu(g)
    flip=np.arange(1<<g.n)^((1<<g.n)-1)
    commutator=h[flip][:,flip]-h
    symmetry_residual=float(abs(commutator.data).max(initial=0.))
    e0=min(e[0] for _,e,_ in blocks);ground_vectors=[]
    for ids,e,v in blocks:
        for k in np.flatnonzero(abs(e-e0)<1e-8):
            psi=np.zeros(1<<g.n,dtype=v.dtype);psi[ids]=v[:,k];ground_vectors.append(psi)
    for beta in job['betas']:
        ready=[point_directory(out,job,beta,a,s) for a in job['alphas'] for s in job['seeds']
               if (point_directory(out,job,beta,a,s)/'metrics.json').exists()]
        if not ready:continue
        dest=ready[0].parents[1]/'physics_D'
        if (dest/'summary.json').exists():continue
        rho,thermal=block_gibbs(blocks,beta,1<<g.n);control=paramagnet_gibbs(g.n,beta)
        diagonals={};summary=[]
        for k,psi in enumerate(ground_vectors):
            strange=strange_correlator(psi,g.n)
            if strange['defined']:
                correlation=np.asarray(strange['correlation']);diagonals[f'strange_ground{k}']=correlation
                off=correlation[~np.eye(g.n,dtype=bool)]
                summary.append({'ground_index':k,'defined':True,'reference_overlap_real':float(strange['overlap'].real),
                    'reference_overlap_imag':float(strange['overlap'].imag),'mean_offdiagonal_real':float(off.real.mean()),
                    'max_offdiagonal_modulus':float(abs(off).max())})
            else:summary.append({'ground_index':k,'defined':False,'reason':strange['reason']})
        diag=abs(rho)-abs(control);exact=job.get('parameter',0)==0
        record={'beta':beta,'g':job.get('parameter',0),'global_Z2_commutator_max':symmetry_residual,
            'purity':thermal['purity'],'ground_manifold_weight':thermal['ground_manifold_weight'],
            'strange_correlator':summary,'ground_degeneracy':len(ground_vectors),
            'trivial_reference':{'strange_offdiagonal':0.,'thermal_purity':float(np.vdot(control,control).real)},
            'exact_solvable_point':exact,'equal_element_moduli_residual':float(abs(diag).max()) if exact else None,
            'gatewise_symmetric_local_SU2':{'family':'each U_i commutes with X_i, i.e. Rx(theta)',
                'optimized_W_equals_fixed_W_exactly':True,'reason':'the Fourier weight local factor is a function of X_i'},
            'circuit_symmetry':'each local complement-paired diagonal gate commutes with product X on its own support; not a claim about all symmetric circuits',
            'SPT_interpretation':'finite-size independent strange-correlator diagnostic, not a W-defined phase boundary'}
        if exact and record['equal_element_moduli_residual']>1e-11:raise ArithmeticError('Levin-Gu exact thermal modulus control')
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'strange_correlator.npz',**diagonals)
        atomic_json(dest/'summary.json',record)


def diagnose_F(job,out):
    g,h=build_qubit_job(job);blocks=block_eigensystem(h);dense=h.toarray();energies=np.sort(np.concatenate([e for _,e,_ in blocks]))
    for beta in [b for b in job['betas'] if b>0]:
        ready=[point_directory(out,job,beta,a,s) for a in job['alphas'] for s in job['seeds']
               if (point_directory(out,job,beta,a,s)/'metrics.json').exists()]
        if not ready:continue
        dest=ready[0].parents[1]/'physics_F';dest.mkdir(parents=True,exist_ok=True)
        rho,thermal=block_gibbs(blocks,beta,1<<g.n)
        if not (dest/'summary.json').exists():
            fixed=worldline_phase_severity(dense,beta,g.n,energies)
            graph={str(c):{'H_expansion_support':clean_balance(-dense,c),'Gibbs_support':clean_balance(rho,c)} for c in [1e-8,1e-12]}
            atomic_json(dest/'summary.json',{'beta':beta,'fixed_basis':fixed,'support':graph,
                'definitions_separated':True,'large_scale_QMC_sampling':False})
        for point in ready:
            if (point/'worldline_phase.json').exists():continue
            candidates=json.loads((point/'basis_or_circuit.json').read_text());candidate=candidates['local_SU2']
            rotated=rotate_operator(dense,directions_to_walsh_basis(np.asarray(candidate['axes_best'])))
            phase=worldline_phase_severity(rotated,beta,g.n,energies)
            atomic_json(point/'worldline_phase.json',{'W_optimized_basis':phase,
                'all_diagonal_Z_CZ_CCZ_gauges':worldline_phase_severity(dense,beta,g.n,energies),
                'diagonal_gauge_invariance':'H_abs depends on entry magnitudes; every endpoint diagonal gauge leaves gamma invariant',
                'Gibbs_sign_error_is_different_quantity':True})
        # A matched real-basis control compares W and sign objectives with the
        # same 24 generic starts and 150 evaluations, separately from SU2 results.
        if beta in [.25,1.,2.,4.] and not (dest/'matched_real_optimization.json').exists():
            rng=np.random.default_rng(20261027)
            initial=[(np.zeros(g.n),np.zeros(g.n,dtype=int))]
            initial += [(rng.uniform(-np.pi,np.pi,g.n),rng.integers(0,2,g.n)) for _ in range(23)]
            sign=optimize_real_basis(dense,'sign',seed=20261027,starts=24,max_evals=150,beta=beta,eigenvalues=energies,initial=initial)
            wr=optimize_real_basis(rho,'walsh',seed=20261027,starts=24,max_evals=150,alpha=1,initial=initial)
            comparisons={}
            for name,result in [('sign_objective',sign),('W_objective',wr)]:
                unitary=real_unitaries(result['theta'],result['flips'])
                comparisons[name]={'optimization':result,'gamma':worldline_phase_severity(rotate_operator(dense,unitary),beta,g.n,energies),
                    'W_alpha1':walshness(rotate_operator(rho,unitary),1)}
            atomic_json(dest/'matched_real_optimization.json',{'initial_theta':[t for t,f in initial],
                'initial_flips':[f for t,f in initial],'comparisons':comparisons,'starts':24,'eval_cap_per_start':150,
                'blind_discovery_claim':False,'role':'posthoc matched-resource QMC sign control'})


def main():
    p=argparse.ArgumentParser();p.add_argument('track',choices=['B','D','F']);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--job',type=int,help='Optional original registration job index')
    args=p.parse_args();out=ROOT/'results/signrule_round3';jobs=json.loads((out/'registration/jobs.json').read_text())
    selected=[j for i,j in enumerate(jobs) if j['track']==args.track and (args.job is None or args.job==i)]
    for i,job in enumerate(selected):
        if i%args.workers!=args.worker:continue
        {'B':diagnose_B,'D':diagnose_D,'F':diagnose_F}[args.track](job,out)
        print('PHYSICS',args.track,job['job_id'],flush=True)


if __name__=='__main__':main()
