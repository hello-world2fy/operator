"""TC/DS wavefunction and t-J constrained-Hilbert scans with separate metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import state_degree_spectrum,walshness_from_spectrum,directions_to_walsh_basis
from operator_walshness.signrule.geometry import honeycomb,chain,square,Geometry
from operator_walshness.signrule.models import (loop_projectors,ideal_loop_state,tj_hamiltonian,
    tj_marshall_gauge,block_eigensystem,block_gibbs)
from operator_walshness.signrule.blind import (optimize_pure_local,optimize_circuit,circuit_phase,
    rotate_pure,optimize_qutrit_local,qutrit_moment)
from operator_walshness.signrule.workflow import utc


def freeze(point,results,metadata):
    point.mkdir(parents=True,exist_ok=True)
    atomic_json(point/'metadata.json',metadata)
    atomic_json(point/'basis_or_circuit.json',results)
    atomic_json(point/'blind_frozen.json',{'frozen_utc':utc(),'known_label_access':False,
        'objective':'Operator_Walshness_only',
        'basis_or_circuit_sha256':hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()})


def scan_loops(out,device):
    for shape in [(2,2),(2,3)]:
        hc=honeycomb(*shape);n=hc['n']
        # Edge-qubit gates can meet only on a honeycomb vertex. No face-wide loop gates.
        edges=sorted({tuple(sorted((a,b))) for star in hc['stars'] for a in star for b in star if a<b})
        supports=[(i,) for i in range(n)]+edges+[tuple(star) for star in hc['stars']]
        for semion in [False,True]:
            model='double_semion' if semion else 'toric_code'
            psi,states=ideal_loop_state(hc,semion)
            h,terms=loop_projectors(hc,semion)
            residual=float(np.linalg.norm(h@psi+len(terms)*psi))
            if residual>1e-12:raise ArithmeticError('loop state not a ground state')
            for alpha in [.5,1.,2.]:
                for seed in [20261027,20261028,20261029]:
                    point=out/model/f'honeycomb_{shape[0]}x{shape[1]}'/f'n{n}'/'vacuum_sector'/'pure'/f'alpha{alpha:g}'/f'seed{seed}'
                    if (point/'metrics.json').exists():continue
                    if (point/'blind_frozen.json').exists():
                        results=json.loads((point/'basis_or_circuit.json').read_text())
                    else:
                        results={'local_SU2':optimize_pure_local(psi,alpha,starts=24,steps=200,seed=seed,device=device)}
                        for level in [1,2,3]:
                            selected=[s for s in supports if len(s)<=level]
                            results[f'circuit_L{level}']=optimize_circuit(psi,alpha,selected,starts=16,flips=2000,
                                seed=seed,max_gates=n,max_depth=n,device=device,pure=True)
                        freeze(point,results,{'model':model,'geometry':hc,'alpha':alpha,'seed':seed,
                            'sector':'vacuum plaquette-span; matched TC/DS support','object':'pure_state_projector',
                            'dense_operator_constructed':False,'residual':residual})
                    fixed=state_degree_spectrum(psi);spectra={'fixed_operator':np.convolve(fixed,fixed)}
                    for label,result in results.items():
                        transformed=rotate_pure(psi,directions_to_walsh_basis(np.array(result['axes_best']))) if label=='local_SU2' else psi*circuit_phase(n,result)
                        p=state_degree_spectrum(transformed);spectra[label]=np.convolve(p,p)
                        if abs(walshness_from_spectrum(spectra[label],alpha)-result['W_best'])>1e-8:raise ArithmeticError('pure W check')
                    np.savez_compressed(point/'walsh_spectrum.npz',**spectra)
                    np.savez_compressed(point/'supported_wavefunction.npz',configurations=states,amplitudes=psi[states])
                    atomic_json(point/'metrics.json',{'track':'C','model':model,'n':n,'alpha':alpha,'seed':seed,
                        'purity':1.,'operator_specific':False,'classification':'pure_state_carryover_pending_posthoc',
                        'W_fixed':walshness_from_spectrum(spectra['fixed_operator'],alpha),
                        'representations':{label:r['W_best'] for label,r in results.items()},'ground_residual':residual})
                    print('FROZEN C',model,shape,alpha,seed,flush=True)


def scan_tj(out,device,geometries=None):
    # Full planned sizes remain registered for adaptive continuation; start with
    # the smallest ladder and the first open/periodic chains, all t/J and beta.
    for g in (geometries if geometries is not None else [square(3,2),chain(8),chain(8,True)]):
        n=g.n;geometry=g.name+'_'+'x'.join(map(str,g.shape))+'_'+g.boundary
        for t in [1.,2.,3.]:
            for sigma in [False,True]:
                model='sigma_tJ' if sigma else 'tJ'
                h,digits,fock,hops=tj_hamiltonian(g,t=t,sigma=sigma)
                blocks=block_eigensystem(h)
                for beta in [0.,.25,.5,1.,2.,4.]:
                    rho,thermal=block_gibbs(blocks,beta,len(digits))
                    for alpha in [.5,1.,2.]:
                        fixed=np.log(qutrit_moment(rho,digits,alpha))/alpha
                        for seed in [20261027,20261028,20261029]:
                            point=out/model/geometry/f'n{n}'/f't{t:g}'/f'beta{beta:g}'/f'alpha{alpha:g}'/f'seed{seed}'
                            if (point/'metrics.json').exists():continue
                            if (point/'blind_frozen.json').exists():results=json.loads((point/'basis_or_circuit.json').read_text())
                            else:
                                results={'local_charge_preserving':optimize_qutrit_local(rho,digits,alpha,
                                    starts=24,steps=150,seed=seed,device=device)}
                                freeze(point,results,{'model':model,'geometry':g.metadata(),'t_over_J':t,'beta':beta,
                                    'alpha':alpha,'seed':seed,'sector':'one hole; n_down=(n-1)//2',
                                    'local_hilbert':['hole','up','down'],'metric':'Z3_frequency_support_degree',
                                    'basis_order':'canonical interleaved Fock','full_qutrit_dense_operator':False})
                            np.savez_compressed(point/'thermal_spectrum.npz',eigenvalues=thermal['eigenvalues'],energies=thermal['energies'])
                            atomic_json(point/'metrics.json',{'track':'E','model':model,'n':n,'t_over_J':t,'beta':beta,
                                'alpha':alpha,'seed':seed,'W_Z3_fixed':fixed,'W_Z3_optimized':results['local_charge_preserving']['W_best'],
                                'purity':thermal['purity'],'R_eff':1/thermal['purity'],'beta_gap':beta*thermal['gap'],
                                'sector_dimension':len(digits),'qubit_W_comparison_allowed':False,
                                'classification':'pending_independent_path_and_endpoint_scoring'})
                            print('FROZEN E',model,geometry,t,beta,alpha,seed,flush=True)
                # Only now open physical path diagnostics, after all operator candidates freeze.
                from operator_walshness.signrule.diagnostics import path_phase_samples,support_balance
                physical,_,_,physical_hops=tj_hamiltonian(g,t=t,sigma=False)
                off,*_=tj_hamiltonian(g,t=t,sigma=True)
                gauge=tj_marshall_gauge(g,digits)
                rows,records=path_phase_samples(physical,off,physical_hops,gauge,samples=5000,seed=20261027)
                base=out/model/geometry/f'n{n}'/f't{t:g}'/'path_diagnostics'
                atomic_json(base/'summary.json',{'rows':rows,'path_not_endpoint':True,'QMC_average_sign':False})
                atomic_json(base/'closed_paths.json',records)


def main():
    p=argparse.ArgumentParser();p.add_argument('track',choices=['C','E'])
    p.add_argument('--out',type=Path,default=ROOT/'results/signrule_round3');p.add_argument('--device',default='cuda:0')
    p.add_argument('--E-geometry',choices=['initial','ladder2x4'],default='initial')
    args=p.parse_args();torch.set_num_threads(1)
    suffix='_ladder2x4' if args.track=='E' and args.E_geometry=='ladder2x4' else ''
    status=args.out/'jobs'/f'special_{args.track}{suffix}.json'
    atomic_json(status,{'status':'running','pid':os.getpid(),'device':args.device,'started_utc':utc()})
    try:
        if args.track=='C':scan_loops(args.out,args.device)
        else:scan_tj(args.out,args.device,[square(4,2)] if args.E_geometry=='ladder2x4' else None)
    except Exception as error:
        import traceback
        atomic_json(status,{'status':'failed','error':str(error),'traceback':traceback.format_exc()});raise
    atomic_json(status,{'status':'initial_special_scan_complete','finished_utc':utc(),
        'remaining':'thermal TC/DS and posthoc scoring' if args.track=='C' else 'larger geometry and path/endpoint/observable validation'})


if __name__=='__main__':main()
