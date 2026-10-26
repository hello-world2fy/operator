"""Artifact checks with explicit incomplete coverage; optional derived-spectrum repair."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import walshness_from_spectrum
from operator_walshness.signrule.workflow import utc,point_directory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repair-spectrum-order',action='store_true')
    parser.add_argument('--verify-freeze-digests',action='store_true')
    args=parser.parse_args();out=ROOT/'results/signrule_round3'
    errors=[];repairs=[];checked=0;freeze_checks=0
    for metrics_path in out.glob('*/*/n*/*/*/alpha*/seed*/metrics.json'):
        point=metrics_path.parent;m=json.loads(metrics_path.read_text());checked+=1
        try:
            results=json.loads((point/'basis_or_circuit.json').read_text())
            frozen=json.loads((point/'blind_frozen.json').read_text())
            assert frozen['known_label_access'] is False
            if args.verify_freeze_digests:
                assert hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest()==frozen['basis_or_circuit_sha256']
                freeze_checks+=1
            for label,result in results.items():
                if result.get('no_discovery'):continue
                assert result['objective']=='Operator_Walshness_only'
                if 'W_restarts' in result:
                    assert abs(result['W_best']-min(result['W_restarts']))<1e-7
                if label.startswith('circuit'):
                    assert result['gate_count']<=result['max_gates']
                    assert result['circuit_depth']<=result['max_depth']
                    assert result['evaluations']==len(result['W_restarts'])*(result['flips_per_start']+1)
            spectrum_path=point/'thermal_spectrum.npz'
            if spectrum_path.exists():
                with np.load(spectrum_path) as a:
                    probabilities=a['eigenvalues'].copy();energies=a['energies'].copy()
                beta=m['beta'];expected=np.exp(-beta*(energies-energies.min()));expected/=expected.sum()
                assert abs(probabilities.sum()-1)<1e-9
                assert probabilities.min()>=-1e-14
                assert abs(np.sum(probabilities**2)-m['purity'])<1e-9
                if not np.allclose(probabilities,expected,atol=1e-12,rtol=1e-10):
                    if not args.repair_spectrum_order:raise AssertionError('thermal spectrum probability ordering')
                    assert np.allclose(np.sort(probabilities)[::-1],expected,atol=1e-12,rtol=1e-10)
                    temp=point/f'thermal_spectrum.{os.getpid()}.tmp.npz'
                    np.savez_compressed(temp,eigenvalues=expected,energies=energies);temp.replace(spectrum_path)
                    repairs.append(str(spectrum_path.relative_to(out)))
            if (point/'walsh_spectrum.npz').exists():
                with np.load(point/'walsh_spectrum.npz') as a:
                    for label in a.files:
                        if label in ('degree_mean','degree_correlation','degree_second'):continue
                        p=a[label];assert p.min()>=-1e-14 and abs(p.sum()-1)<1e-9
                        if label in results:assert abs(walshness_from_spectrum(p,m['alpha'])-results[label]['W_best'])<1e-7
            if m['track']=='E':assert m['qubit_W_comparison_allowed'] is False
            if m['track']=='C':assert m['operator_specific'] is False
        except Exception as error:errors.append({'point':str(point.relative_to(out)),'error':str(error)})
    jobs=json.loads((out/'registration/jobs.json').read_text());expected=ready=0
    missing=[]
    for job in jobs:
        for beta in job['betas']:
            for alpha in job['alphas']:
                for seed in job['seeds']:
                    expected+=1;p=point_directory(out,job,beta,alpha,seed)
                    if (p/'metrics.json').exists():ready+=1
                    else:missing.append(str(p.relative_to(out)))
    result={'checked_utc':utc(),'status':'failed' if errors else 'checked_artifacts_pass_scope_still_incomplete',
        'artifacts_checked':checked,'freeze_digests_checked':freeze_checks,'errors':errors,
        'derived_spectra_repaired_this_pass':repairs,'ABDF_expected':expected,'ABDF_complete':ready,
        'ABDF_missing':len(missing),'C_initial_pure_expected':36,'E_initial_expected':972,
        'full_objective_complete':False,
        'pending_requirements':'registration/progress.json; a passing artifact check does not close the research goal'}
    atomic_json(out/'acceptance_progress.json',result)
    if repairs:
        with (out/'registration/spectrum_repairs.jsonl').open('a') as f:
            f.write(json.dumps({'utc':utc(),'paths':repairs})+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('derived_spectra_repaired_this_pass','errors')},indent=2))
    print('errors',len(errors),'repairs',len(repairs))
    if errors:print(json.dumps(errors[:10],indent=2));sys.exit(1)


if __name__=='__main__':main()
