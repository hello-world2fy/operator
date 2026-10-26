"""Random Z2 phase scrambles on the same TC/DS support and amplitudes."""
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
from operator_walshness.walsh import state_degree_spectrum,walshness_from_spectrum,directions_to_walsh_basis
from operator_walshness.signrule.geometry import honeycomb,vacuum_loop_sector
from operator_walshness.signrule.blind import optimize_pure_local,optimize_circuit,circuit_phase,rotate_pure
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase,score_phase,paired_improvement_ci
from operator_walshness.signrule.workflow import utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(1);out=ROOT/'results/signrule_round3/negative_controls/loop_phase_scramble'
    counter=0
    for shape in [(2,2),(2,3)]:
        hc=honeycomb(*shape);n=hc['n'];ids=vacuum_loop_sector(hc)
        edges=sorted({tuple(sorted((a,b))) for star in hc['stars'] for a in star for b in star if a<b})
        supports=[(i,) for i in range(n)]+edges+[tuple(star) for star in hc['stars']]
        for draw in range(20):
            index=counter;counter+=1
            if index%args.workers!=args.worker:continue
            point=out/f'n{n}'/f'draw{draw:02d}'
            if (point/'metrics.json').exists():continue
            point.mkdir(parents=True,exist_ok=True);seed=202610280+draw
            signs=np.random.default_rng(seed).choice([-1.,1.],len(ids));signs*=signs[0]
            psi=np.zeros(1<<n);psi[ids]=signs/np.sqrt(len(ids))
            if (point/'blind_frozen.json').exists():results=json.loads((point/'basis_or_circuit.json').read_text())
            else:
                results={'local_SU2':optimize_pure_local(psi,1.,starts=24,steps=200,seed=20261027,device=args.device)}
                for level in [1,2,3]:
                    results[f'circuit_L{level}']=optimize_circuit(psi,1.,[s for s in supports if len(s)<=level],
                        starts=16,flips=2000,seed=20261027,max_gates=n,max_depth=n,pure=True,device=args.device)
                atomic_json(point/'basis_or_circuit.json',results)
                atomic_json(point/'blind_frozen.json',{'frozen_utc':utc(),'known_label_access':False,
                    'basis_or_circuit_sha256':hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()})
            state_spectrum=state_degree_spectrum(psi);spectra={'fixed':np.convolve(state_spectrum,state_spectrum)}
            for label,result in results.items():
                transformed=rotate_pure(psi,directions_to_walsh_basis(result['axes_best'])) if label=='local_SU2' else psi*circuit_phase(n,result)
                spectrum=state_degree_spectrum(transformed);spectra[label]=np.convolve(spectrum,spectrum)
                assert abs(walshness_from_spectrum(spectra[label],1.)-result['W_best'])<1e-8
            entries=phase_entries(np.outer(psi[ids],psi[ids]));entries['row']=ids[entries['row']];entries['col']=ids[entries['col']]
            linear,mask=best_linear_phase(entries,n);lower,_=best_linear_phase(entries,n,'test');scores={}
            for label,result in results.items():
                if not label.startswith('circuit'):continue
                phase=circuit_phase(n,result);scores[label]={'test':score_phase(entries,phase),
                    'versus_test_linear_lower_bound':paired_improvement_ci(entries,lower,phase)}
            np.savez_compressed(point/'walsh_spectrum.npz',**spectra)
            np.savez_compressed(point/'supported_wavefunction.npz',configurations=ids,amplitudes=psi[ids])
            atomic_json(point/'metrics.json',{'created_utc':utc(),'n':n,'draw':draw,'generator_seed':seed,
                'alpha':1.,'purity':1.,'support_size':len(ids),'W_fixed':walshness_from_spectrum(spectra['fixed'],1.),
                'representations':{k:v['W_best'] for k,v in results.items()},'scores':scores,
                'trained_linear_test':score_phase(entries,linear),'test_linear_lower_bound':score_phase(entries,lower),
                'matched_amplitudes':'TC and DS vacuum-sector wavefunctions; positive-amplitude control is the existing toric-code scan',
                'cutoff_sensitivity':'all supported amplitudes equal magnitude; 1e-8 and 1e-12 masks identical',
                'pure_state_only':True,'no_dense_operator_constructed':True,'not_physical_discovery':True})
            print('LOOP_NULL',n,draw,flush=True)


if __name__=='__main__':main()
