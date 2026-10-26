"""Blind finite-Sz projectors and frozen-circuit transfer to an open 12-site cluster."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.linalg import eigh
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import state_degree_spectrum,walshness_from_spectrum,directions_to_walsh_basis
from operator_walshness.signrule.geometry import triangular,local_gate_supports
from operator_walshness.signrule.models import heisenberg
from operator_walshness.signrule.blind import optimize_pure_local,optimize_circuit,circuit_phase,rotate_pure
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase,score_phase,paired_improvement_ci
from operator_walshness.signrule.transfer import transfer_triangular_circuit
from operator_walshness.signrule.workflow import utc


def eigenstates(g,j2):
    h=heisenberg(g,j2);ids=np.flatnonzero(np.bitwise_count(np.arange(1<<g.n,dtype=np.uint64))==g.n//2)
    e,v=eigh(h[ids][:,ids].toarray(),check_finite=False)
    k=int(np.flatnonzero(e>e[0]+1e-8)[0]);out={}
    for label,index in [('ground',0),('selected_excited',k)]:
        psi=np.zeros(1<<g.n);psi[ids]=v[:,index]
        out[label]=(psi,{'energy':float(e[index]),'sector':'fixed_n_down','n_down':g.n//2,
            'sector_degeneracy':int(np.sum(abs(e-e[index])<1e-8)),
            'eigenvector_residual':float(np.linalg.norm(h@psi-e[index]*psi)),
            'selection':'first real eigh vector in the specified energy eigenspace; not a unique symmetry-selected state'})
    return out


def score(psi,candidates,n):
    ids=np.flatnonzero(abs(psi)>1e-15);small=np.outer(psi[ids],psi[ids].conj());scores={};arrays={}
    for cutoff in [1e-8,1e-12]:
        tag=f'{cutoff:g}';entries=phase_entries(small,cutoff)
        entries['row']=ids[entries['row']];entries['col']=ids[entries['col']]
        linear,mask=best_linear_phase(entries,n);lower,lower_mask=best_linear_phase(entries,n,'test')
        scores[tag]={'linear_mask':mask,'linear_test':score_phase(entries,linear),
                     'test_linear_lower_bound':score_phase(entries,lower),'candidates':{}}
        for key in ['row','col','phase','weight','test']:arrays[f'{tag}_{key}']=entries[key]
        for label,result in candidates.items():
            if not label.startswith('circuit'):continue
            phase=circuit_phase(n,result)
            scores[tag]['candidates'][label]={'test':score_phase(entries,phase),
                'versus_linear':paired_improvement_ci(entries,linear,phase),
                'versus_test_linear_lower_bound':paired_improvement_ci(entries,lower,phase)}
    return scores,arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(1);out=ROOT/'results/signrule_round3/physical_B/projector_blind'
    for index,j2 in enumerate([0.,.1,.2]):
        if index%args.workers!=args.worker:continue
        source_g=triangular(3,3);target_g=triangular(3,4,False)
        sources=eigenstates(source_g,j2);targets=eigenstates(target_g,j2)
        for label,(psi,metadata) in sources.items():
            target_psi,target_metadata=targets[label]
            for alpha in [.5,1.,2.]:
                for seed in [20261027,20261028,20261029]:
                    point=out/f'p{j2:g}'/label/f'alpha{alpha:g}'/f'seed{seed}'
                    if (point/'transfer.json').exists():continue
                    point.mkdir(parents=True,exist_ok=True)
                    if (point/'blind_frozen.json').exists():results=json.loads((point/'basis_or_circuit.json').read_text())
                    else:
                        results={'local_SU2':optimize_pure_local(psi,alpha,starts=24,steps=150,seed=seed,device=args.device)}
                        for level in [1,2,3]:
                            results[f'circuit_L{level}']=optimize_circuit(psi,alpha,local_gate_supports(source_g,level),
                                starts=16,flips=2000,seed=seed,max_gates=9,max_depth=9,device=args.device,pure=True)
                        atomic_json(point/'basis_or_circuit.json',results)
                        atomic_json(point/'blind_frozen.json',{'frozen_utc':utc(),'known_label_access':False,
                            'objective':'Operator_Walshness_only',
                            'basis_or_circuit_sha256':hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()})
                    fixed=np.convolve(state_degree_spectrum(psi),state_degree_spectrum(psi));spectra={'fixed':fixed}
                    for family,result in results.items():
                        rotated=rotate_pure(psi,directions_to_walsh_basis(result['axes_best'])) if family=='local_SU2' else psi*circuit_phase(9,result)
                        pure=state_degree_spectrum(rotated);spectra[family]=np.convolve(pure,pure)
                        assert abs(walshness_from_spectrum(spectra[family],alpha)-result['W_best'])<1e-8
                    np.savez_compressed(point/'walsh_spectrum.npz',**spectra)
                    scores,arrays=score(psi,results,9);np.savez_compressed(point/'sign_diagnostics.npz',**arrays)
                    atomic_json(point/'metrics.json',{'track':'B','object':label,'pure_carryover':True,'operator_specific':False,
                        'n':9,'J2':j2,'alpha':alpha,'seed':seed,'source_geometry':source_g.metadata(),
                        'source_state':metadata,'W_fixed':walshness_from_spectrum(fixed,alpha),'phase_scores':scores})
                    transferred={key:transfer_triangular_circuit(value,[3,3],[3,4],target_periodic=False)
                                 for key,value in results.items() if key.startswith('circuit')}
                    transfer_scores,transfer_arrays=score(target_psi,transferred,12)
                    np.savez_compressed(point/'target_sign_diagnostics.npz',**transfer_arrays)
                    atomic_json(point/'transfer.json',{'created_utc':utc(),'target_geometry':target_g.metadata(),
                        'target_state':target_metadata,'frozen_source_circuits':transferred,'phase_scores':transfer_scores,
                        'target_optimized':False,'pure_state_carryover':True,'operator_specific':False,
                        'claim':'independent geometry test of a specified transfer rule; no temperature validation for these projectors'})
                    print('B_PROJECTOR',j2,label,alpha,seed,flush=True)


if __name__=='__main__':main()
