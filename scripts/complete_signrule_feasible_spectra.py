"""Exact optimized n6 qutrit spectra and common observables for B pure projectors."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.geometry import square,triangular
from operator_walshness.signrule.models import tj_hamiltonian,block_eigensystem,block_gibbs
from operator_walshness.signrule.physical_diagnostics import qutrit_full_spectrum
from operator_walshness.signrule.blind import rotate_pure
from operator_walshness.walsh import walshness_from_spectrum,fwht,state_walshness,directions_to_walsh_basis
from run_signrule_triangular_projectors import eigenstates


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['all','E','B'],default='all');args=parser.parse_args()
    out=ROOT/'results/signrule_round3';g=square(3,2);geometry='square_3x2_open';done=0
    for model in ([] if args.group=='B' else ['tJ','sigma_tJ']):
        for t in [1.,2.,3.]:
            h,digits,*_=tj_hamiltonian(g,t=t,sigma=model=='sigma_tJ');blocks=block_eigensystem(h)
            for base in (out/model/geometry/'n6'/f't{t:g}').glob('beta*'):
                beta=float(base.name[4:]);rho,_=block_gibbs(blocks,beta,len(digits));fixed=qutrit_full_spectrum(rho,digits)
                for p in base.glob('alpha*/seed*/metrics.json'):
                    dest=p.parent/'walsh_spectrum.npz'
                    if dest.exists():continue
                    m=json.loads(p.read_text());local=json.loads((p.parent/'basis_or_circuit.json').read_text())['local_charge_preserving']
                    unitary=[]
                    for q in np.asarray(local['quaternions_best']):
                        q=q/np.linalg.norm(q);a=q[0]+1j*q[1];b=q[2]+1j*q[3]
                        u=np.eye(3,dtype=complex);u[1:,1:]=[[a,b],[-b.conjugate(),a.conjugate()]];unitary.append(u)
                    spec=qutrit_full_spectrum(rho,digits,unitary)
                    residual=abs(walshness_from_spectrum(spec,m['alpha'])-m['W_Z3_optimized'])
                    if residual>1e-8:raise ArithmeticError(f'Z3 rotated spectrum mismatch {residual}')
                    np.savez_compressed(dest,Z3_fixed=fixed,local_charge_preserving=spec);done+=1
            print('Z3_SPECTRA',model,t,done,flush=True)
    for j2 in ([] if args.group=='E' else [0.,.1,.2]):
        for label,(psi,state) in eigenstates(triangular(3,3),j2).items():
            he=out/'physical_B/projector_HE'/f'p{j2:g}'/label
            with np.load(he/'phase_fits.npz') as a:reference=a['1e-08_linear_plus_HE_three_spin_phase']
            q=fwht(psi)/np.sqrt(len(psi));prob=abs(q)**2
            bits=((np.arange(len(psi))[:,None]>>np.arange(9))&1).astype(float);mean=prob@bits;second=(bits.T*prob)@bits
            atomic_json(he/'degree_moments.json',{'degree_mean':2*mean,'degree_correlation':2*(second-np.outer(mean,mean)),
                'degree_second':2*second+2*np.outer(mean,mean),'projector_factorization_used':True})
            for p in (out/'physical_B/projector_alpha0_recheck'/f'p{j2:g}'/label).glob('alpha*/seed*/metrics.json'):
                m=json.loads(p.read_text());result=json.loads((p.parent/'basis_or_circuit.json').read_text())['local_SU2'];alpha=m['alpha']
                row={'W_fixed':2*state_walshness(psi,alpha),'W_random':2*state_walshness(rotate_pure(psi,directions_to_walsh_basis(result['initial_axes'][2])),alpha),
                    'W_at_alpha0_axes':2*state_walshness(rotate_pure(psi,directions_to_walsh_basis(result['alpha0_axes'])),alpha),
                    'alpha0_W':result['alpha0_W'],'W_HE_fitted_reference':2*state_walshness(psi*reference.conj(),alpha),
                    'HE_reference_is_posthoc_variational_not_exact':True,'purity':1.,'rank':1,'energy':state['energy']}
                atomic_json(p.parent/'representation_baselines.json',row)
    print('FEASIBLE_SPECTRA_DONE',done,flush=True)


if __name__=='__main__':main()
