"""Add random, alpha-zero and posthoc reference baselines to frozen candidates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.optimize import batched_moment
from operator_walshness.walsh import walshness,fwht
from operator_walshness.signrule.geometry import honeycomb,chain,square
from operator_walshness.signrule.models import (block_eigensystem,block_gibbs,commuting_gibbs,loop_projectors,
    tj_hamiltonian,tj_marshall_gauge)
from operator_walshness.signrule.blind import (alpha0_from_hs_rows,supported_pure_moment,
    qutrit_local_vectors,qutrit_batched_moment)
from operator_walshness.signrule.sparse_blind import SmallOffdiagonalMoment
from operator_walshness.signrule.scoring import posthoc_oracle
from operator_walshness.signrule.physical_diagnostics import qutrit_degree_moments
from operator_walshness.signrule.workflow import build_qubit_job,utc


def write_qubit_points(base,rho,device,reference=None,reference_name=None,sparse_input=False):
    points=[p.parent for p in base.glob('alpha*/seed*/metrics.json') if not (p.parent/'representation_baselines.json').exists()]
    if not points:return
    n=int(np.log2(rho.shape[0]));dense=None
    if sparse_input:
        squared=(rho@rho).tocsr();weights=squared.diagonal().real/np.vdot(rho.data,rho.data).real;states=np.arange(1<<n)
        bloch=[]
        for i in range(n):
            z=1.-2*((states>>i)&1);off=np.asarray(squared[states,states^(1<<i)]).ravel()/np.vdot(rho.data,rho.data).real
            bloch.append([off.sum().real,np.sum(1j*z*off).real,weights@z])
        bloch=np.asarray(bloch);length=np.linalg.norm(bloch,axis=1);axes0=np.tile([1.,0.,0.],(n,1));good=length>1e-12;axes0[good]=bloch[good]/length[good,None]
        w0=float(n-length.sum());structure=SmallOffdiagonalMoment(rho,device)
        moment=lambda axes,alpha:structure(axes,alpha)
    else:
        dense=rho;w0,axes0=alpha0_from_hs_rows(rho);op=torch.tensor(rho,dtype=torch.complex128,device=device)
        moment=lambda axes,alpha:batched_moment(op,axes,alpha)
    reference_cache={}
    for point in points:
        m=json.loads((point/'metrics.json').read_text());candidates=json.loads((point/'basis_or_circuit.json').read_text())
        if 'local_SU2' in candidates:random=np.asarray(candidates['local_SU2']['initial_axes'])[2]
        else:
            random=np.random.default_rng(m['seed']).normal(size=(n,3));random/=np.linalg.norm(random,axis=1,keepdims=True)
        axes=torch.tensor(np.array([random,axes0]),dtype=torch.float64,device=device)
        with torch.no_grad():values=(torch.log(moment(axes,m['alpha']))/m['alpha']).cpu().numpy()
        ref=None
        if reference is not None:
            if m['alpha'] not in reference_cache:
                if dense is None:dense=rho.toarray()
                reference_cache[m['alpha']]=walshness(dense*reference[:,None]*reference[None,:].conj(),m['alpha'])
            ref={'name':reference_name,'W':reference_cache[m['alpha']],'role':'posthoc only, never a blind warm start'}
        atomic_json(point/'representation_baselines.json',{'created_utc':utc(),'random_local_axes':random,'W_random':float(values[0]),
            'random_choice':'third saved generic start, or registered seeded Haar axes for identity controls',
            'alpha0_axes':axes0,'alpha0_global_minimum':w0,'W_at_alpha0_axes':float(values[1]),
            'theory_or_fitted_reference':ref,'alpha0_is_a_different_objective':True,
            'purity_invariance':'all transformations are unitary conjugations; density spectrum preserved'})


def qubits(out,device):
    jobs=json.loads((out/'registration/jobs.json').read_text())
    for job in jobs:
        g,h=build_qubit_job(job);blocks=block_eigensystem(h);reference,name=posthoc_oracle(job,g)
        geometry=('x'.join(map(str,job['shape'])) if 'shape' in job else 'chain')+'_'+job.get('boundary','periodic' if 'shape' in job else 'open')
        for beta in job['betas']:
            base=out/job['model']/geometry/f'n{g.n}'/f'p{job.get("parameter",0):g}'/f'beta{beta:g}'
            if not any(not (p.parent/'representation_baselines.json').exists() for p in base.glob('alpha*/seed*/metrics.json')):continue
            rho,_=block_gibbs(blocks,beta,1<<g.n)
            ref=reference;label=name
            fitted=base/'physics_B/phase_fits.npz'
            if job['track']=='B' and fitted.exists():
                with np.load(fitted) as data:ref=data['1e-08_linear_plus_HE_three_spin_phase']
                label='heldout-evaluated Huse-Elser continuous phase fit; not an exact oracle'
            write_qubit_points(base,rho,device,ref,label)
        print('BASELINES',job['job_id'],flush=True)


def loops(out,device):
    for model in ['toric_code','double_semion']:
        for p in (out/model).glob('*/n*/vacuum_sector/pure/alpha*/seed*/metrics.json'):
            point=p.parent
            if (point/'representation_baselines.json').exists():continue
            m=json.loads(p.read_text());n=m['n'];alpha=m['alpha'];result=json.loads((point/'basis_or_circuit.json').read_text())['local_SU2']
            with np.load(point/'supported_wavefunction.npz') as data:ids=data['configurations'];amp=data['amplitudes']
            psi=np.zeros(1<<n,complex);psi[ids]=amp;prob=abs(psi)**2;states=np.arange(1<<n);bloch=[]
            for i in range(n):
                z=1.-2*((states>>i)&1);off=psi.conj()*psi[states^(1<<i)]
                bloch.append([off.sum().real,np.sum(-1j*z*off).real,prob@z])
            bloch=np.asarray(bloch);length=np.linalg.norm(bloch,axis=1);axes0=np.tile([1.,0.,0.],(n,1));good=length>1e-12;axes0[good]=bloch[good]/length[good,None]
            random=np.asarray(result['initial_axes'])[1];axes=torch.tensor(np.array([random,axes0]),dtype=torch.float64,device=device)
            with torch.no_grad():values=(2*torch.log(supported_pure_moment(torch.tensor(amp,dtype=torch.complex128,device=device),torch.tensor(ids,device=device),axes,alpha))/alpha).cpu().numpy()
            weights=abs(fwht(psi))**2/len(psi);bits=((states[None,:]>>np.arange(n)[:,None])&1).astype(float);mean=bits@weights;second=(bits*weights)@bits.T
            singular=np.linalg.svd(psi.reshape(1<<(n//2),-1),compute_uv=False);w=singular**2;positive=w>1e-15
            refpsi=abs(psi);ref_w=2*np.log(np.sum(abs(fwht(refpsi))**2/len(psi)*np.exp(alpha*np.bitwise_count(states.astype(np.uint64)))))/alpha
            atomic_json(point/'representation_baselines.json',{'created_utc':utc(),'W_random':float(values[0]),'random_local_axes':random,
                'alpha0_axes':axes0,'alpha0_global_minimum':float(n-length.sum()),'W_at_alpha0_axes':float(values[1]),
                'theory_reference':{'name':'positive equal-modulus state in the same closed vacuum sector','W':ref_w},
                'local_magnetization':bloch,'OSEE_fixed':float(-2*w[positive]@np.log2(w[positive])),
                'degree_mean':2*mean,'degree_correlation':2*(second-np.outer(mean,mean)),
                'degree_second':2*second+2*np.outer(mean,mean),'no_dense_projector_constructed':True})
        hc=honeycomb(2,2);_,terms=loop_projectors(hc,model=='double_semion')
        for beta in [0.,.25,.5,1.,2.,4.]:
            base=out/model/'honeycomb_2x2/n12/full_Gibbs'/f'beta{beta:g}'
            rho=commuting_gibbs(terms,beta,sparse_output=True)
            # A global loop formula is undefined in vertex-defect sectors.
            write_qubit_points(base,rho,device,sparse_input=True)
        print('BASELINES',model,flush=True)


def qutrits(out,device):
    for g in [square(3,2),chain(8),chain(8,True),square(4,2)]:
        geometry=g.name+'_'+'x'.join(map(str,g.shape))+'_'+g.boundary
        for t in [1.,2.,3.]:
            for model,sigma in [('tJ',False),('sigma_tJ',True)]:
                h,digits,*_=tj_hamiltonian(g,t=t,sigma=sigma);blocks=block_eigensystem(h);gauge=tj_marshall_gauge(g,digits)
                for beta in [0.,.25,.5,1.,2.,4.]:
                    base=out/model/geometry/f'n{g.n}'/f't{t:g}'/f'beta{beta:g}'
                    ready=[p.parent for p in base.glob('alpha*/seed*/metrics.json') if not (p.parent/'representation_baselines.json').exists()]
                    if not ready:continue
                    rho,thermal=block_gibbs(blocks,beta,len(digits));weights=np.sum(abs(rho)**2,axis=1)/thermal['purity']
                    local_prob=np.array([[np.sum(weights[digits[:,i]==s]) for s in range(3)] for i in range(g.n)])
                    zero=np.zeros((g.n,4));zero[:,0]=1/np.sqrt(2);zero[:,2]=np.where(local_prob[:,1]>=local_prob[:,2],1.,-1.)/np.sqrt(2)
                    w0=float(2*g.n-2*np.sum(local_prob[:,0]/3+2*np.maximum(local_prob[:,1],local_prob[:,2])/3))
                    op=torch.tensor(rho,dtype=torch.complex128,device=device);configs=torch.tensor(digits,dtype=torch.int64,device=device)
                    phase_op=op*torch.tensor(gauge[:,None]*gauge[None,:],device=device);identity_raw=torch.zeros((1,g.n,4),dtype=torch.float64,device=device);identity_raw[:,:,0]=1
                    reference_cache={};shared=base/'common_observables';shared.mkdir(parents=True,exist_ok=True)
                    if not (shared/'degree_moments.npz').exists():np.savez_compressed(shared/'degree_moments.npz',**qutrit_degree_moments(rho,digits,device))
                    for point in ready:
                        m=json.loads((point/'metrics.json').read_text());candidate=json.loads((point/'basis_or_circuit.json').read_text())['local_charge_preserving'];random=np.asarray(candidate['initial_quaternions'])[1]
                        raw=torch.tensor(np.array([random,zero]),dtype=torch.float64,device=device)
                        with torch.no_grad():
                            value=(torch.log(qutrit_batched_moment(op,configs,m['alpha'],qutrit_local_vectors(raw)))/m['alpha']).cpu().numpy()
                            if m['alpha'] not in reference_cache:
                                reference_cache[m['alpha']]=float(torch.log(qutrit_batched_moment(phase_op,configs,m['alpha'],qutrit_local_vectors(identity_raw)))[0]/m['alpha'])
                        atomic_json(point/'representation_baselines.json',{'created_utc':utc(),'metric':'Z3_support_degree','W_Z3_random':float(value[0]),
                            'random_quaternions':random,'alpha0_quaternions':zero,'alpha0_W_Z3':w0,'W_Z3_at_alpha0_basis':float(value[1]),
                            'Marshall_endpoint_reference_W_Z3':reference_cache[m['alpha']],
                            'path_phase_is_not_a_general_endpoint_oracle':True,
                            'degree_moments':str((shared/'degree_moments.npz').relative_to(out)),
                            'alpha0_scope':'fixed one-hole/Sz sector implies diagonal one-site HS marginals; spin block aligns its larger eigenvalue'})
                print('BASELINES_E',model,geometry,t,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('group',choices=['qubits','loops','qutrits']);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(1);out=ROOT/'results/signrule_round3'
    {'qubits':qubits,'loops':loops,'qutrits':qutrits}[args.group](out,args.device)


if __name__=='__main__':main()
