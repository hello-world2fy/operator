"""Acceptance audit on saved artifacts; exits nonzero on missing/invalid results."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from operator_walshness.round2 import atomic_json
from scripts.analyze_round2 import read_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    checks={};failures=[]
    def check(name,value,passed):
        checks[name]={'value':value,'passed':bool(passed)}
        if not passed:failures.append(name)
    for stage,expected in [('R1',68),('R2',14),('R3',72),('R4',18),('R1_extended',12)]:
        files=list((out/stage).glob('*.json'))
        check(stage+'_completed_files',len(files),len(files)==expected)
        status=out/f'{stage}_status.json'
        state=json.loads(status.read_text())['status'] if status.exists() else 'missing'
        check(stage+'_status',state,state=='complete')
    r1=read_rows(out,'R1');ext=read_rows(out,'R1_extended');r2=read_rows(out,'R2');r3=read_rows(out,'R3');r4=read_rows(out,'R4')
    if not r1.empty:
        check('MLP_2000_independent_networks',int((r1.steps==2000).sum()),(r1.steps==2000).sum()==3264)
        check('MLP_basis_moment_error',float(r1.moment_disagreement.max()),r1.moment_disagreement.max()<1e-8)
        all_learning=pd.concat([r1,ext],ignore_index=True)
        finite=np.isfinite(all_learning.full_relative_frobenius_squared).all()
        check('MLP_finite_error',bool(finite),finite)
        slack=all_learning.full_relative_frobenius_squared-all_learning.square_architecture_spectral_error_floor
        check('square_spectral_floor_min_slack',float(slack.min()),slack.min()>-2e-6)
        counts=all_learning.groupby(['source_file','basis','steps']).seed.nunique()
        check('MLP_initialization_seeds_min',int(counts.min()),counts.min()==8)
        keys=['n','beta','target','basis','seed','activation','depth','width','steps']
        repeated=ext[ext.steps==2000].merge(r1[r1.steps==2000],on=keys,suffixes=('_extended','_original'))
        if len(repeated):
            diff=float(abs(repeated.full_relative_frobenius_squared_extended-repeated.full_relative_frobenius_squared_original).max())
            check('independent_2000_step_reproduction',diff,diff<1e-6)
    if not r2.empty:
        check('thermal_identity_abs_error',float(r2.thermal_identity_error.max()),r2.thermal_identity_error.max()<1e-8)
        bound=(r2.I_W_bits-2*r2.osee_bits).max()
        check('I_W_le_2OSEE_max_violation',float(bound),bound<1e-10)
        cov=float(r2.inter_dimer_max_cov.max());check('inter_dimer_cov_max',cov,cov<1e-10)
    if not r3.empty:
        check('random_null_draws',int(r3.groupby('source_file').draws.first().sum()),r3.groupby('source_file').draws.first().sum()==7200)
        check('random_spectrum_purity_error',float(r3.max_purity_error.max()),r3.max_purity_error.max()<1e-10)
    if not r4.empty:
        check('QMC_physical_points',r4.groupby(['n','boundary','beta','j2']).ngroups,r4.groupby(['n','boundary','beta','j2']).ngroups==87)
        control=r4[(r4.j2==0)&(r4.basis=='Marshall')]
        error=float(abs(control.log_average_sign).max());check('Marshall_sign_one_log_error',error,error<1e-9)
        error=float(r4.gauge_logsign_error.max());check('gauge_logsign_error',error,error<1e-9)
        error=float(r4.logZ_rotation_error.max());check('logZ_independent_diagonalization_error',error,error<1e-9)
        check('average_sign_upper_bound',float(r4.log_average_sign.max()),r4.log_average_sign.max()<1e-9)
    spectra=0;min_p=0.;max_defect=0.;high_t=0.;high_t_sign=0.
    for stage in ('R1_targets','R2','R3','R4'):
        for path in (out/stage).glob('*.npz'):
            with np.load(path) as z:
                for key in z.files:
                    if key.endswith('_P') or '_P_' in key or key in ('physical_P','null_P'):
                        a=z[key]
                        if not np.all(np.isfinite(a)):failures.append(str(path)+':'+key+':nonfinite')
                        min_p=min(min_p,float(a.min()));max_defect=max(max_defect,float(abs(a.sum(-1)-1).max()));spectra+=int(np.prod(a.shape[:-1])) if a.ndim>1 else 1
                if 'high_temperature_W_delta_minus_W_H' in z:
                    high_t=max(high_t,float(abs(z['high_temperature_W_delta_minus_W_H']).max()))
                    high_t_sign=max(high_t_sign,float(abs(z['high_temperature_logsign'])))
    check('probability_histograms',spectra,spectra>7200)
    check('minimum_probability',min_p,min_p>=-1e-12)
    check('maximum_probability_defect',max_defect,max_defect<1e-10)
    check('beta_1e-5_centered_W_minus_H_max',high_t,high_t<1e-3)
    check('beta_1e-5_logsign_abs_max',high_t_sign,high_t_sign<1e-9)
    acceptance={'status':'passed' if not failures else 'failed','checks':checks,'failures':failures,
                'note':'finite-alpha optimization is budget-limited; acceptance does not certify global minima or a thermodynamic scaling law'}
    atomic_json(out/'acceptance.json',acceptance)
    print(json.dumps(acceptance,indent=2))
    if failures:sys.exit(1)


if __name__=='__main__':main()
