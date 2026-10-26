"""Independent small-system admission tests for the round-3 physics builders."""
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm

from operator_walshness.signrule.geometry import (triangular,chain,square,honeycomb,
    domain_wall_signs,vacuum_loop_sector)
from operator_walshness.signrule.models import (heisenberg,paramagnet,cluster,boolean_phase,
    conjugate_diagonal,levin_gu,levin_gu_local,loop_projectors,ideal_loop_state,
    commuting_gibbs,paramagnet_gibbs,block_eigensystem,block_gibbs,tj_hamiltonian,
    tj_marshall_gauge,fermion_action)


def maxabs(a):
    return abs(a.data).max(initial=0.) if sparse.issparse(a) else abs(a).max(initial=0.)


def test_triangular_periodic_geometry_and_spin_units():
    g=triangular(3,3)
    assert len(g.edges)==27 and len(g.triangles)==18
    assert all(g.sublattice[i]!=g.sublattice[j] for i,j in g.edges)
    assert len(g.next_edges)==27  # image multiplicity must not be dropped
    assert heisenberg(g)[0,0]==pytest.approx(27/4)
    pair=heisenberg(chain(2)).toarray()
    assert np.allclose(np.linalg.eigvalsh(pair),[-.75,.25,.25,.25])
    with pytest.raises(ValueError):triangular(2,3)


@pytest.mark.parametrize('periodic',[False,True])
def test_cluster_independent_pauli_and_gauge_constructions(periodic):
    g=chain(6,periodic);d=boolean_phase(g.n,g.edges)
    h=cluster(g)
    assert maxabs(h-conjugate_diagonal(paramagnet(g.n),d))<1e-14
    rho=paramagnet_gibbs(g.n,.7,d)
    exact=expm(-.7*h.toarray());exact/=np.trace(exact)
    assert np.allclose(rho,exact,atol=1e-13)
    p=block_gibbs(block_eigensystem(h),.7,1<<g.n)[0]
    assert np.allclose(p,rho,atol=1e-13)


def test_thermal_eigenvalues_and_energies_have_matching_order():
    h=heisenberg(chain(6),.3)
    rho,info=block_gibbs(block_eigensystem(h),1.3,64)
    weights=np.exp(-1.3*(info['energies']-info['energies'][0]));weights/=weights.sum()
    assert np.allclose(info['eigenvalues'],weights,atol=1e-14)
    assert info['energy']==pytest.approx(weights@info['energies'])


@pytest.mark.parametrize('shape',[(3,3),(3,4)])
def test_levin_gu_closed_walls_local_formula_and_Z2(shape):
    g=triangular(*shape);h,d,counts=levin_gu(g)
    assert np.array_equal(d,d[::-1])
    assert maxabs(h-levin_gu_local(g))==0
    assert maxabs(h-h.T)==0
    assert np.array_equal(counts,counts[::-1])
    psi=d/np.sqrt(len(d))
    assert np.linalg.norm(h@psi+g.n*psi)<1e-12
    # Conjugated X stabilizers commute; check all pairs on random vectors.
    from operator_walshness.signrule.models import flip_term
    terms=[conjugate_diagonal(flip_term(g.n,[i]),d) for i in range(g.n)]
    for i in range(g.n):
        for j in range(i):assert maxabs(terms[i]@terms[j]-terms[j]@terms[i])==0


@pytest.mark.parametrize('shape',[(2,2),(2,3)])
@pytest.mark.parametrize('semion',[False,True])
def test_loop_geometry_projectors_and_matched_vacuum_sector(shape,semion):
    hc=honeycomb(*shape);h,terms=loop_projectors(hc,semion)
    for q in terms:
        assert maxabs(q-q.T)==0
        assert maxabs(q@q-q)==0
    for i,a in enumerate(terms):
        for b in terms[:i]:assert maxabs(a@b-b@a)==0
    psi,states=ideal_loop_state(hc,semion)
    assert len(states)==2**(shape[0]*shape[1]-1)
    assert np.linalg.norm(h@psi+len(terms)*psi)<1e-12
    assert np.vdot(psi,psi)==pytest.approx(1)
    if shape==(2,2):
        blocks=block_eigensystem(h)
        energies=np.concatenate([e for _,e,_ in blocks])
        assert np.count_nonzero(abs(energies+len(terms))<1e-10)==4
        rho=commuting_gibbs(terms,.7)
        other,info=block_gibbs(blocks,.7,len(psi))
        assert np.allclose(rho,other,atol=1e-13)
        assert np.trace(rho)==pytest.approx(1)
        assert np.vdot(rho,rho).real==pytest.approx(info['purity'])
    else:
        with pytest.raises(ValueError):commuting_gibbs(terms,.7)


@pytest.mark.parametrize('g',[chain(6),chain(6,True),square(3,2),square(3,3)])
def test_tj_fock_sign_sector_and_phase_string(g):
    h,digits,fock,hops=tj_hamiltonian(g,t=2)
    hs,ds,_,_=tj_hamiltonian(g,t=2,sigma=True)
    assert np.array_equal(digits,ds)
    assert maxabs(h-h.T)==0
    assert np.all((digits==0).sum(1)==1)
    assert np.all((digits==2).sum(1)==(g.n-1)//2)
    # An independent full spin-exchange Fock operation, including all four signs.
    idx={int(m):i for i,m in enumerate(fock)}
    for col,m in enumerate(fock):
        for a,b in g.edges:
            if digits[col,a]==2 and digits[col,b]==1:
                target,sign=fermion_action(int(m),[(2*b,False),(2*b+1,True),
                                                  (2*a+1,False),(2*a,True)])
                assert sign==1 and h[idx[target],col]==pytest.approx(.5)
    gauge=tj_marshall_gauge(g,digits)
    hm=conjugate_diagonal(h,gauge);hsm=conjugate_diagonal(hs,gauge)
    for col,row,down,_,_ in hops:
        assert -hm[row,col]==pytest.approx(2*(-1)**down)
        assert -hsm[row,col]==pytest.approx(2)


def test_tj_flux_is_hermitian_and_periodic():
    g=chain(6,True)
    h,*_=tj_hamiltonian(g,flux=.31)
    assert maxabs(h-h.conj().T)<1e-14
    h2,*_=tj_hamiltonian(g,flux=.31+2*np.pi)
    assert maxabs(h-h2)<1e-14
