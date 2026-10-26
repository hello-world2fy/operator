"""Post-freeze loop decoding on matched support, without dense projectors."""
import json
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.diagnostics import (phase_entries,best_linear_phase,score_phase,paired_improvement_ci)
from operator_walshness.signrule.blind import circuit_phase
from operator_walshness.signrule.workflow import utc


def main():
    out=ROOT/'results/signrule_round3';count=0
    for model in ['toric_code','double_semion']:
        for metrics in (out/model).glob('*/n*/vacuum_sector/pure/alpha*/seed*/metrics.json'):
            point=metrics.parent
            if (point/'posthoc.json').exists():continue
            frozen=json.loads((point/'blind_frozen.json').read_text())
            if hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()!=frozen['basis_or_circuit_sha256']:
                raise ValueError('candidate changed after freeze')
            m=json.loads(metrics.read_text());n=m['n'];data=np.load(point/'supported_wavefunction.npz')
            configs=data['configurations'];psi=data['amplitudes']
            # This tiny matrix is only 8x8 or 32x32, not a 2^n projector.
            tiny=psi[:,None]*psi[None,:].conj();entries=phase_entries(tiny)
            entries['row']=configs[entries['row']];entries['col']=configs[entries['col']]
            linear,mask=best_linear_phase(entries,n);oraclelinear,oracle_mask=best_linear_phase(entries,n,'test')
            baseline={'trained_linear_mask':mask,'trained_linear_test':score_phase(entries,linear),
                      'test_best_linear_mask':oracle_mask,'test_linear_lower_bound':score_phase(entries,oraclelinear)}
            results=json.loads((point/'basis_or_circuit.json').read_text());scores={}
            for label,result in results.items():
                if not label.startswith('circuit'):continue
                phase=circuit_phase(n,result)
                scores[label]={'test':score_phase(entries,phase),
                    'versus_linear':paired_improvement_ci(entries,linear,phase),
                    'versus_test_linear_lower_bound':paired_improvement_ci(entries,oraclelinear,phase),
                    'gate_count':result['gate_count'],'gate_depth':result['circuit_depth']}
            atomic_json(point/'posthoc.json',{'scored_utc':utc(),'baseline':baseline,'circuit_scores':scores,
                'pure_state_carryover':True,'operator_specific':False,'support_count':len(configs),
                'claim':'preliminary matched-sector sign comparison; independent size transfer still required',
                'holdout_scope':'phase decoding holdout, W sees full wavefunction',
                'cutoff_sensitivity':'all nonzero supported amplitudes have equal modulus; 1e-8 and 1e-12 masks coincide',
                'local_SU2_labels':'not compared to original loop oracle after non-diagonal rotation'})
            count+=1
    print('loop candidates scored',count)


if __name__=='__main__':main()
