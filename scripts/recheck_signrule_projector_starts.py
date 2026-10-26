"""Preserve frozen projector scans and add a full-budget alpha0-initialized local check."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc
from operator_walshness.signrule.geometry import triangular
from operator_walshness.signrule.blind import optimize_pure_local,rotate_pure
from operator_walshness.signrule.huse_elser import axis_texture
from operator_walshness.walsh import directions_to_walsh_basis,state_degree_spectrum,walshness_from_spectrum
from run_signrule_triangular_projectors import eigenstates


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(1);out=ROOT/'results/signrule_round3';g=triangular(3,3);index=0
    for j2 in [0.,.1,.2]:
        for label,(psi,state) in eigenstates(g,j2).items():
            for alpha in [.5,1.,2.]:
                for seed in [20261027,20261028,20261029]:
                    current=index;index+=1
                    if current%args.workers!=args.worker:continue
                    source=out/'physical_B/projector_blind'/f'p{j2:g}'/label/f'alpha{alpha:g}'/f'seed{seed}'
                    dest=out/'physical_B/projector_alpha0_recheck'/f'p{j2:g}'/label/f'alpha{alpha:g}'/f'seed{seed}'
                    if (dest/'metrics.json').exists():continue
                    dest.mkdir(parents=True,exist_ok=True)
                    result=optimize_pure_local(psi,alpha,starts=24,steps=150,seed=seed,device=args.device)
                    atomic_json(dest/'basis_or_circuit.json',{'local_SU2':result})
                    atomic_json(dest/'blind_frozen.json',{'known_label_access':False,'objective':'Operator_Walshness_only','frozen_utc':utc(),
                        'basis_or_circuit_sha256':hashlib.sha256((dest/'basis_or_circuit.json').read_bytes()).hexdigest()})
                    old=json.loads((source/'basis_or_circuit.json').read_text())['local_SU2']
                    rotated=rotate_pure(psi,directions_to_walsh_basis(result['axes_best']));p=state_degree_spectrum(rotated);p=np.convolve(p,p)
                    assert abs(walshness_from_spectrum(p,alpha)-result['W_best'])<1e-9
                    np.savez_compressed(dest/'walsh_spectrum.npz',local_SU2=p)
                    atomic_json(dest/'metrics.json',{'created_utc':utc(),'J2':j2,'object':label,'n':9,'alpha':alpha,'seed':seed,
                        'state':state,'pure_state_carryover':True,'operator_specific':False,'W':result['W_best'],
                        'previous_24_start_W':old['W_best'],'source_point':str(source.relative_to(out)),
                        'correction':'original pure local runner had uniform/random starts but omitted a distinct nonzero alpha0 direction; this replacement 24-start budget includes it',
                        'budgets_not_pooled':True,'phase_circuits_unchanged':True,'basis_texture':axis_texture(result['axes_best'],g)})
                    print('PURE_ALPHA0',j2,label,alpha,seed,flush=True)


if __name__=='__main__':main()
