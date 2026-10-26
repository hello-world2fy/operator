"""Selected physical-point Haar nulls, using matched local/circuit budgets."""
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
from operator_walshness.walsh import fixed_basis_spectrum,walshness_from_spectrum,rotate_operator,directions_to_walsh_basis
from operator_walshness.signrule.geometry import chain,triangular,local_gate_supports
from operator_walshness.signrule.models import frustrated,levin_gu,block_eigensystem,block_gibbs
from operator_walshness.signrule.null_controls import same_spectrum_null
from operator_walshness.signrule.blind import optimize_local,optimize_circuit,circuit_phase
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase,score_phase,paired_improvement_ci
from operator_walshness.signrule.workflow import utc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(1);out=ROOT/'results/signrule_round3/negative_controls';counter=0
    for model in ['controlled_frustration','levin_gu']:
        if model=='controlled_frustration':
            g=chain(6);h=frustrated(g,.4);beta=4.;alpha=2.;symmetric=False;level=2
            permutation=np.array([int(f'{s:06b}'[::-1],2) for s in range(64)])
        else:
            g=triangular(3,3);h=levin_gu(g,0)[0];beta=.5;alpha=1.;symmetric=True;level=3
            permutation=np.arange(512)^511
        rho,thermal=block_gibbs(block_eigensystem(h),beta,1<<g.n)
        for family in ['complex_Haar','real_orthogonal_Haar','sector_Haar']:
            for draw in range(20):
                index=counter;counter+=1
                if index%args.workers!=args.worker:continue
                point=out/model/family/f'draw{draw:02d}'
                if (point/'metrics.json').exists():continue
                point.mkdir(parents=True,exist_ok=True);seed=202610270+draw
                random_rho,checks=same_spectrum_null(rho,family,seed,permutation)
                if (point/'blind_frozen.json').exists():results=json.loads((point/'basis_or_circuit.json').read_text())
                else:
                    results={'local_SU2':optimize_local(random_rho,alpha,starts=24,steps=150,polish=30,seed=20261027,device=args.device),
                        'circuit_selected':optimize_circuit(random_rho,alpha,local_gate_supports(g,level),starts=16,flips=2000,
                            seed=20261027,max_gates=g.n,max_depth=g.n,symmetric=symmetric,device=args.device)}
                    atomic_json(point/'basis_or_circuit.json',results)
                    atomic_json(point/'blind_frozen.json',{'frozen_utc':utc(),'known_label_access':False,
                        'basis_or_circuit_sha256':hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()})
                fixed=fixed_basis_spectrum(random_rho)[0];spectra={'fixed':fixed};scores={};candidate=circuit_phase(g.n,results['circuit_selected'])
                for label,result in results.items():
                    rotated=rotate_operator(random_rho,directions_to_walsh_basis(result['axes_best'])) if label=='local_SU2' else random_rho*candidate[:,None]*candidate[None,:]
                    spectra[label]=fixed_basis_spectrum(rotated)[0]
                    assert abs(walshness_from_spectrum(spectra[label],alpha)-result['W_best'])<1e-8
                for cutoff in [1e-8,1e-12]:
                    entries=phase_entries(random_rho,cutoff);linear,mask=best_linear_phase(entries,g.n);lower,_=best_linear_phase(entries,g.n,'test')
                    scores[f'{cutoff:g}']={'linear_test':score_phase(entries,linear),'test_best_linear':score_phase(entries,lower),
                        'circuit_test':score_phase(entries,candidate),'versus_test_linear_lower_bound':paired_improvement_ci(entries,lower,candidate)}
                np.savez_compressed(point/'walsh_spectrum.npz',**spectra)
                atomic_json(point/'metrics.json',{'created_utc':utc(),'model':model,'n':g.n,'family':family,'draw':draw,'seed':seed,
                    'beta':beta,'alpha':alpha,'purity':thermal['purity'],'W_fixed':walshness_from_spectrum(fixed,alpha),
                    'W_local':results['local_SU2']['W_best'],'W_circuit':results['circuit_selected']['W_best'],
                    'circuit_family':'gatewise_Z2_symmetric' if symmetric else 'Z_CZ','phase_scores':scores,'numerical_checks':checks,
                    'preserved_sector':'global_spin_flip' if model=='levin_gu' else 'spatial_reflection',
                    'sector_is_preserved_only_for_sector_Haar':True,
                    'source_parameters':'g0 beta0.5 alpha1' if model=='levin_gu' else 'lambda0.4 beta4 alpha2',
                    'not_physical_discovery':True})
                print('NULL',model,family,draw,flush=True)


if __name__=='__main__':main()
