"""Exact full-Hilbert n=12 TC/DS Gibbs scan using sparse moment contractions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import fixed_basis_spectrum,walshness_from_spectrum,identity_walshness
from operator_walshness.signrule.geometry import honeycomb
from operator_walshness.signrule.models import loop_projectors,commuting_gibbs,block_eigensystem
from operator_walshness.signrule.sparse_blind import optimize_sparse_local,optimize_sparse_circuit
from operator_walshness.signrule.physical_diagnostics import sparse_operator_entropy
from operator_walshness.signrule.blind import circuit_phase
from operator_walshness.signrule.workflow import utc


def thermal_jobs():
    return [{'model':'double_semion' if ds else 'toric_code','beta':beta,'alpha':alpha,'seed':seed}
            for ds in [False,True] for beta in [0.,.25,.5,1.,2.,4.]
            for alpha in [.5,1.,2.] for seed in [20261027,20261028,20261029]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.set_num_threads(1);out=ROOT/'results/signrule_round3';hc=honeycomb(2,2);n=12
    status=out/'jobs'/f'C_thermal_worker{args.worker}.json'
    atomic_json(status,{'status':'running','pid':os.getpid(),'device':args.device,'started_utc':utc()})
    edges=sorted({tuple(sorted((a,b))) for star in hc['stars'] for a in star for b in star if a<b})
    supports=[(i,) for i in range(n)]+edges+[tuple(star) for star in hc['stars']]
    systems={};cache={}
    try:
        for index,job in enumerate(thermal_jobs()):
            if index%args.workers!=args.worker:continue
            model,beta,alpha,seed=job['model'],job['beta'],job['alpha'],job['seed']
            point=out/model/'honeycomb_2x2'/f'n{n}'/'full_Gibbs'/f'beta{beta:g}'/f'alpha{alpha:g}'/f'seed{seed}'
            if (point/'metrics.json').exists():continue
            point.mkdir(parents=True,exist_ok=True)
            if model not in systems:
                h,terms=loop_projectors(hc,model=='double_semion');blocks=block_eigensystem(h)
                energies=np.sort(np.concatenate([e for _,e,_ in blocks]));systems[model]=(h,terms,energies)
            h,terms,energies=systems[model]
            if (model,beta) not in cache:
                rho=commuting_gibbs(terms,beta,sparse_output=True)
                dense=rho.toarray();fixed,check=fixed_basis_spectrum(dense)
                if check['mass_relative_error']>1e-10:raise ArithmeticError('FWHT mass error')
                osee=sparse_operator_entropy(rho,n)
                cache={(model,beta):(rho,fixed,osee)}
                del dense
            rho,fixed,osee=cache[model,beta]
            probabilities=np.exp(-beta*(energies-energies.min()));probabilities/=probabilities.sum()
            purity=float(np.vdot(rho.data,rho.data).real);wf=walshness_from_spectrum(fixed,alpha)
            if (point/'blind_frozen.json').exists():results=json.loads((point/'basis_or_circuit.json').read_text())
            elif beta==0:
                if abs(wf-identity_walshness(n,alpha))>1e-10:raise ArithmeticError('identity check')
                results={'identity_control':{'W_best':wf,'no_discovery':True}}
            else:
                results={'local_SU2':optimize_sparse_local(rho,alpha,starts=24,steps=150,polish=20,seed=seed,device=args.device,chunk=4)}
                for level in [1,2,3]:
                    results[f'circuit_L{level}']=optimize_sparse_circuit(rho,alpha,[s for s in supports if len(s)<=level],
                        starts=16,flips=2000,seed=seed,max_gates=n,max_depth=n,device=args.device)
            atomic_json(point/'metadata.json',{'model':model,'geometry':hc,'object':'full_Hilbert_Gibbs','beta':beta,'alpha':alpha,'seed':seed,
                'Hamiltonian_convention':'-sum Q_v -sum B_p; adjacent vertex projectors included in both models',
                'thermal_sector':'all vertex and topology sectors; no sector projection',
                'optimization_known_labels_available':False,'density_is_exact':True,'sparse_optimization_is_not_truncation':True})
            atomic_json(point/'basis_or_circuit.json',results)
            atomic_json(point/'blind_frozen.json',{'frozen_utc':utc(),'known_label_access':False,'objective':'Operator_Walshness_only',
                'basis_or_circuit_sha256':hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()})
            np.savez_compressed(point/'thermal_spectrum.npz',eigenvalues=probabilities,energies=energies)
            spectra={'fixed':fixed};rows=[]
            for label,result in results.items():
                if label.startswith('circuit'):
                    phase=circuit_phase(n,result)
                    gauged=rho.multiply(phase[:,None]).multiply(phase[None,:]).toarray()
                    spectra[label]=fixed_basis_spectrum(gauged)[0]
                    if abs(walshness_from_spectrum(spectra[label],alpha)-result['W_best'])>1e-8:raise ArithmeticError('sparse circuit vs FWHT mismatch')
                    del gauged
                rows.append({'representation':label,'W':result['W_best'],'delta_W':wf-result['W_best'],'W_per_2n':result['W_best']/(2*n)})
            np.savez_compressed(point/'walsh_spectrum.npz',**spectra)
            np.savez_compressed(point/'operator_schmidt.npz',weights=osee['Schmidt_weights'])
            gap=float(energies[np.flatnonzero(energies>energies[0]+1e-9)[0]]-energies[0]);ground=abs(energies-energies[0])<1e-9
            atomic_json(point/'metrics.json',{'track':'C','model':model,'n':n,'beta':beta,'alpha':alpha,'seed':seed,'W_fixed':wf,
                'representations':rows,'purity':purity,'R_eff':1/purity,'beta_gap':beta*gap,
                'ground_degeneracy':int(ground.sum()),'ground_manifold_weight':float(probabilities[ground].sum()),
                'OSEE_fixed':osee['OSEE'],'energy':float(probabilities@energies),'density_nonzeros':rho.nnz,
                'mixed_operator_gate':bool(beta>0 and purity<.95),'operator_specific':False,
                'classification':'exact_mixed_Gibbs_pending_posthoc_and_independent_validation',
                'trace_residual':float(abs(rho.diagonal().sum()-1))})
            print('FROZEN C_THERMAL',model,beta,alpha,seed,flush=True)
    except Exception as error:
        import traceback
        atomic_json(status,{'status':'failed','error':str(error),'traceback':traceback.format_exc(),'pid':os.getpid()});raise
    atomic_json(status,{'status':'blind_complete','finished_utc':utc()})


if __name__=='__main__':main()
