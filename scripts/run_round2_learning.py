"""R1 first wave. Fixed-X means Walsh axis X (U=I); fixed-Z means axis Z (U=H)."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from operator_walshness.round2 import *
from operator_walshness.round2_learning import train_batched
from operator_walshness.walsh import intrinsic_alpha0, fixed_x_directions, tail_degree,walshness_from_directions
from operator_walshness.exact import diagonalize
from scripts.run_round2_cpu import basis_record


def prepare_targets(out,n,beta,target_kind,e,v):
    rho,p=gibbs_density(e,v,beta)
    if target_kind=='normalized_sqrt_rho_beta':
        op=(v*np.sqrt(p))@v.conj().T
        effective_beta=beta/2
    else:op=rho;effective_beta=beta
    best,diagnostic=basis_record('mixed_field_ising',n,{'g':1.05,'h':.5},effective_beta,op,out)
    nearby=effective_beta/2
    nearbyop,_=gibbs_density(e,v,nearby)
    transferred,transfer_diag=basis_record('mixed_field_ising',n,{'g':1.05,'h':.5},nearby,nearbyop,out)
    _,alpha0=intrinsic_alpha0(op)
    rng=np.random.default_rng(20261026+n);random=rng.normal(size=(n,3));random/=np.linalg.norm(random,axis=1,keepdims=True)
    axes={'fixed_X':fixed_x_directions(n),'fixed_Z':np.tile([0.,0.,1.],(n,1)),
          'random_local':random,'best_numerical_W':best,
          'nearby_beta_transferred':transferred,'alpha0_basis':alpha0}
    targets=[];rows=[];arrays={}
    for basis,axis in axes.items():
        rotated=op.astype(complex) if basis=='fixed_X' else rotate_operator(op,directions_to_walsh_basis(axis))
        spectrum,qa=fixed_basis_spectrum(rotated)
        w=walshness_from_spectrum(spectrum,1.)
        disagreement=abs(w-walshness_from_directions(op,axis,1.))
        if disagreement>1e-8:raise AssertionError('MLP basis mismatch')
        vector=rotated.reshape(-1)*len(op)/np.linalg.norm(rotated)
        targets.append(np.stack([vector.real,vector.imag],axis=-1).astype(np.float32))
        row={'model':'mixed_field_ising','n':n,'g':1.05,'h':.5,'boundary':'open','beta':beta,
             'effective_beta':effective_beta,'alpha':1.,'basis':basis,'target':target_kind,'W':w,
             'D_eps_squared_1e-2':tail_degree(spectrum,1e-2),
             'D_eps_squared_1e-3':tail_degree(spectrum,1e-3),'D_eps_squared_1e-4':tail_degree(spectrum,1e-4),
             'moment_disagreement':disagreement,'purity':float(p@p),'probability_defect':qa['probability_defect']}
        if basis=='best_numerical_W':row.update(diagnostic)
        if basis=='nearby_beta_transferred':row.update(transfer_diag);row['transfer_effective_beta']=nearby
        rows.append(row);arrays[basis+'_P']=spectrum;arrays[basis+'_axes']=axis
    key=f'n{n}_b{beta:g}_{target_kind}'
    if not (Path(out)/'R1_targets'/f'{key}.json').exists():
        save_result(out,'R1_targets',key,rows,**arrays)
    return np.stack(targets),rows,arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:7')
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--hours',type=float,default=2.3)
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--no-graph',action='store_true',default=True)
    p.add_argument('--extended',action='store_true',help='selected beta1,width64,depth2 convergence check')
    args=p.parse_args();torch.set_num_threads(1);torch.cuda.set_device(args.device)
    initialize_run(args.out)
    stage='R1_extended' if args.extended else 'R1'
    start=time.time();deadline=start+args.hours*3600;completed=0;failures=[];planned=12 if args.extended else 68
    for n in (8,10,12):
        e,v,_=diagonalize('mixed_field_ising',n,'open',{'g':1.05,'h':.5})
        for beta in ([1.] if args.extended or n==12 else [.5,1.]):
            for target_kind in ('rho_beta','normalized_sqrt_rho_beta'):
                targets,target_rows,spectra=prepare_targets(args.out,n,beta,target_kind,e,v)
                arches=[(width,depth,activation) for width in (32,64) for depth in (1,2) for activation in ('square','tanh')]
                if n==12 or args.extended:arches=[(64,2,'square'),(64,2,'tanh')]
                # Common FLOP budget equals 2000 updates of width32/depth1.
                base_params=2*n*32+32+32*2+2
                common_flops=args.steps*(6*256*(2*n*32+32*2)+12*base_params)
                for width,depth,activation in arches:
                    key=f'n{n}_b{beta:g}_{target_kind}_{activation}_d{depth}_w{width}'
                    existing=Path(args.out)/stage/f'{key}.json'
                    if existing.exists():
                        stored=json.loads(existing.read_text())
                        if max(row['steps'] for row in stored)!=args.steps:
                            raise ValueError(f'{existing}: stored training budget differs; use a new output directory')
                        completed+=1;continue
                    if time.time()>deadline:
                        atomic_json(Path(args.out)/f'{stage}_status.json',{'status':'partial','reason':'runtime cap','completed_batches':completed,'planned_batches':planned});return
                    print(f'R1 start {key}',flush=True)
                    try:
                        result=train_batched(targets,n,width,depth,activation,seeds=8,steps=args.steps,
                            device=args.device,graph=not args.no_graph,matched_flops=None if args.extended else common_flops,
                            checkpoints=[2000,10000,40000] if args.extended else None,
                            progress=lambda step,error,seconds:print(f'R1 {key} step={step} median_error={error:.5g} seconds={seconds:.1f}',flush=True))
                        rows=[]
                        for step,errors in result['evaluations'].items():
                            for bi,target_row in enumerate(target_rows):
                                for seed in range(8):
                                    err=errors[bi*8+seed]
                                    pdegree=spectra[target_row['basis']+'_P']
                                    lower=float(pdegree[min(len(pdegree),2**depth+1):].sum()) if activation=='square' else None
                                    rows.append({**target_row,'activation':activation,'depth':depth,'width':width,'seed':seed,
                                      'steps':step,'parameters':result['params_per_model'],
                                      'train_FLOPs_estimate':step*result['flops_per_step'],
                                      'matched_FLOPs_checkpoint':step==result['matched_step'],
                                      'full_relative_frobenius_squared':float(err[0]),'full_relative_frobenius':float(np.sqrt(err[0])),
                                      'train_relative_frobenius_squared':float(err[1]),'held_out_relative_frobenius_squared':float(err[2]),
                                      'square_architecture_spectral_error_floor':lower,
                                      'runtime_batch_seconds':result['runtime_seconds'],'batch_models':48,
                                      'batch_size':256,'learning_rate':.001,'held_out_seed':19473,'held_out_fraction':.1,
                                      'sampling':'uniform_without_heldout; same entries for all bases and initialization seeds',
                                      'FLOPs_convention':'6*batch*sum_linear_products + 12*parameters per update; nonlinearities excluded',
                                      'cuda_graph':result['graph']})
                        arrays={'curve_steps':result['curve_steps'],'curve_minibatch_loss':result['curve_loss']}
                        arrays.update({'model_'+k:v for k,v in result['model_state'].items()})
                        save_result(args.out,stage,key,rows,**arrays);completed+=1
                        print(f'R1 done {key} {result["runtime_seconds"]:.1f}s',flush=True)
                    except Exception:
                        import traceback
                        failures.append({'key':key,'traceback':traceback.format_exc()})
                        atomic_json(Path(args.out)/f'{stage}_errors.json',failures)
                        raise
                    if args.benchmark:
                        atomic_json(Path(args.out)/'R1_benchmark.json',{'seconds':result['runtime_seconds'],
                                    'estimated_68_batches_minutes':result['runtime_seconds']*68/60,
                                    'steps':args.steps,'models':48});return
                del targets
                torch.cuda.empty_cache()
    atomic_json(Path(args.out)/f'{stage}_status.json',{'status':'complete','completed_batches':completed,
                'planned_batches':planned,'elapsed_seconds':time.time()-start,'failed':failures})


if __name__=='__main__':main()
