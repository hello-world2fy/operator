"""Independent extra sampling for the flagged n6, beta1 complex-Haar mean."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from operator_walshness.round2 import *
from scripts.run_round2_cpu import prepare_r3


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--draws',type=int,default=5000);parser.add_argument('--seed',type=int,default=703194)
    args=parser.parse_args();out=Path(args.out);key=f'independent_{args.draws}_draws'
    if (out/'R3_validation'/f'{key}.json').exists():return
    z=np.load(prepare_r3(out,6));rho,p=gibbs_density(z['e'],z['v'],1.);purity=float(p@p)
    rng=np.random.default_rng(args.seed);alphas=np.array([.5,1.,2.]);values=[]
    for _ in range(args.draws):
        u=haar_unitary(64,rng);randomized=(u*p)@u.conj().T;spec,_=fixed_basis_spectrum(randomized)
        values.append([np.exp(a*walshness_from_spectrum(spec,a)) for a in alphas])
    values=np.array(values);rows=[]
    for ai,a in enumerate(alphas):
        exact=haar_mean_moment(6,purity,a);mean=float(values[:,ai].mean());sem=float(values[:,ai].std(ddof=1)/np.sqrt(args.draws))
        rows.append({'n':6,'beta':1.,'alpha':float(a),'draws':args.draws,'seed':args.seed,'G_exact':exact,
                     'G_mean':mean,'G_sem':sem,'zscore':(mean-exact)/sem,
                     'reason':'independent post-hoc precision check for largest nondegenerate deviation among initial 100-draw groups'})
    save_result(out,'R3_validation',key,rows,G_samples=values,alphas=alphas)


if __name__=='__main__':main()
