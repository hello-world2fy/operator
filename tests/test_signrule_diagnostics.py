import ast
from pathlib import Path
import numpy as np
import pytest

from operator_walshness.walsh import walshness,identity_walshness,fwht,state_walshness
from operator_walshness.signrule.blind import (fwht_torch_flat,optimize_circuit,circuit_phase,
    gate_features,qutrit_moment,optimize_local)
from operator_walshness.signrule.diagnostics import (phase_entries,score_phase,best_linear_phase,
    support_balance,path_phase_samples)
from operator_walshness.signrule.models import (boolean_phase,paramagnet_gibbs,tj_hamiltonian,
    tj_marshall_gauge)
from operator_walshness.signrule.geometry import chain,square,local_gate_supports
import torch


def test_gpu_style_transform_matches_reference_and_identity():
    rng=np.random.default_rng(33);op=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
    actual=fwht_torch_flat(torch.tensor(op.reshape(1,-1))).numpy().reshape(8,8)
    assert np.allclose(actual,fwht(fwht(op,0),1)/8)
    for alpha in (.5,1,2):assert walshness(np.eye(8),alpha)==pytest.approx(identity_walshness(3,alpha))


def test_exact_linear_fit_and_hermitian_pair_holdout():
    d=boolean_phase(6,[(0,),(3,),(5,)])
    rho=paramagnet_gibbs(6,.7,d);entries=phase_entries(rho)
    predicted,mask=best_linear_phase(entries,6)
    assert mask==(1<<0)+(1<<3)+(1<<5)
    assert score_phase(entries,predicted)['circular_loss']<1e-24
    assert np.all(entries['row']<entries['col'])
    e2=phase_entries(rho,1e-12)
    assert np.array_equal(entries['test'],e2['test'])


def test_support_balance_finds_negative_square_without_any_triangle():
    graph=np.zeros((4,4))
    for a,b in [(0,1),(1,2),(2,3),(3,0)]:graph[a,b]=graph[b,a]=1
    graph[0,1]=graph[1,0]=-1
    result=support_balance(graph)
    assert not result['balanced']
    assert result['support_edges_checked']==4
    assert result['sampled_negative_cycle_lengths']=={4:1}
    graph[2,3]=graph[3,2]=-1
    assert support_balance(graph)['balanced']


def test_complex_phase_cycle_is_not_coerced_to_real_sign():
    phase=np.exp(1j*np.array([.1,.7,2.5]));a=phase[:,None]*phase[None,:].conj()
    assert support_balance(a)['balanced']
    a[0,1]*=np.exp(.4j);a[1,0]=a[0,1].conjugate()
    assert not support_balance(a)['balanced']


def test_individual_symmetric_gate_truth_tables():
    supports=local_gate_supports(square(3,2))
    features,used=gate_features(6,supports,True)
    for feature,support in zip(features,used):
        mask=sum(1<<i for i in support)
        assert np.array_equal(feature,feature[np.arange(64)^mask])
    # Z0 Z1 commutes globally with X0 X1 but each Z is forbidden separately.
    features,used=gate_features(2,[(0,),(1,)],True)
    assert len(used)==0


def test_blind_circuit_objective_and_freeze_reconstruction():
    n=4;phase=boolean_phase(n,chain(n).edges);rho=paramagnet_gibbs(n,.6,phase)
    result=optimize_circuit(rho,1,local_gate_supports(chain(n)),starts=4,flips=60,seed=5)
    found=circuit_phase(n,result)
    assert result['W_best']==pytest.approx(walshness(rho*found[:,None]*found[None,:],1),abs=1e-11)
    psi=np.array([1,1j,2,3j])/np.sqrt(15)
    pure=optimize_circuit(psi,.5,[(0,),(1,),(0,1)],starts=2,flips=4,pure=True)
    p=circuit_phase(2,pure)
    assert pure['W_best']==pytest.approx(2*state_walshness(psi*p,.5),abs=1e-11)
    assert walshness(np.outer(psi,psi.conj()),.5)==pytest.approx(2*state_walshness(psi,.5))


def test_qutrit_kernel_matches_full_fourier_with_physical_rotation():
    g=chain(3);h,digits,_,_=tj_hamiltonian(g)
    rng=np.random.default_rng(8);a=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape)
    rho=a@a.conj().T;rho/=np.trace(rho)
    ids=digits@3**np.arange(3);embedded=np.zeros((27,27),complex);embedded[np.ix_(ids,ids)]=rho
    f=np.exp(2j*np.pi*np.outer(np.arange(3),np.arange(3))/3)/np.sqrt(3)
    f3=np.kron(f,np.kron(f,f));unitaries=[]
    for i in range(3):
        theta=.3+i*.2;u=np.eye(3,dtype=complex);u[0,0]=np.exp(.4j*i)
        u[1:,1:]=[[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]];unitaries.append(u)
    full_u=np.kron(unitaries[2],np.kron(unitaries[1],unitaries[0]))
    rotated=full_u.conj().T@embedded@full_u;q=f3@rotated@f3.conj().T
    hw=np.array([sum(x//(3**i)%3!=0 for i in range(3)) for x in range(27)])
    for alpha in (.5,1,2):
        explicit=np.sum(abs(q)**2*np.exp(alpha*(hw[:,None]+hw[None,:])))/np.vdot(rho,rho).real
        assert qutrit_moment(rho,digits,alpha,unitaries)==pytest.approx(explicit,rel=1e-12)


def test_closed_paths_use_actual_hop_spin_events():
    g=square(3,2);h,digits,_,hops=tj_hamiltonian(g);off,*_=tj_hamiltonian(g,sigma=True)
    gauge=tj_marshall_gauge(g,digits)
    rows,records=path_phase_samples(h,off,hops,gauge,samples=500,lengths=(4,8,12),seed=9)
    assert sum(r['closed'] for r in rows)>0
    assert all(r['phase_string_violations']==0 and r['sigma_nontrivial']==0 for r in rows)
    assert all(r['trajectory'][0]==r['trajectory'][-1] for r in records)


def test_blind_optimizer_import_boundary():
    path=Path(__file__).resolve().parents[1]/'operator_walshness/signrule/blind.py'
    tree=ast.parse(path.read_text())
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any('diagnostic' in name or 'models' in name or 'geometry' in name for name in imports)
    import inspect
    assert 'oracle' not in inspect.signature(optimize_circuit).parameters
    assert 'oracle' not in inspect.signature(optimize_local).parameters


def test_pure_and_qutrit_optimized_candidates_reproduce_scores():
    from operator_walshness.signrule.blind import (optimize_pure_local,rotate_pure,
        optimize_qutrit_local,qutrit_local_vectors,qutrit_batched_moment)
    from operator_walshness.walsh import directions_to_walsh_basis
    rng=np.random.default_rng(4);psi=rng.normal(size=16)+1j*rng.normal(size=16);psi/=np.linalg.norm(psi)
    result=optimize_pure_local(psi,1,starts=3,steps=5)
    rotated=rotate_pure(psi,directions_to_walsh_basis(result['axes_best']))
    assert result['W_best']==pytest.approx(2*state_walshness(rotated,1),abs=1e-10)
    h,digits,*_=tj_hamiltonian(chain(4));rho=np.eye(h.shape[0])/h.shape[0]
    result=optimize_qutrit_local(rho,digits,1,starts=3,steps=5)
    raw=torch.tensor(result['quaternions_best'][None,:,:]);op=torch.tensor(rho,dtype=torch.complex128)
    moment=qutrit_batched_moment(op,torch.tensor(digits.astype(np.int64)),1,qutrit_local_vectors(raw))
    assert result['W_best']==pytest.approx(float(torch.log(moment)[0]),abs=1e-10)
