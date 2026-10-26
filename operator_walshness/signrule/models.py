"""Hamiltonians and independently accessible post-hoc oracle data.

Spin S=sigma/2. All spin bases use bit 0=up, 1=down and little-endian sites.
t-J Fock orbitals are ordered (0 up,0 down,1 up,1 down,...); its basis
is the canonical ordered creation-operator basis, never an ungraded tensor proxy.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

from .geometry import domain_wall_signs, vacuum_loop_sector, loop_count


def flip_term(n, sites, values=1.):
    states=np.arange(1<<n,dtype=np.int64)
    mask=sum(1<<i for i in sites)
    values=np.broadcast_to(values,states.shape)
    return sparse.coo_matrix((values,(states^mask,states)),shape=(1<<n,)*2).tocsr()


def heisenberg(g,j2=0.,selection_field=0.):
    n=g.n;states=np.arange(1<<n,dtype=np.int64)
    diagonal=np.zeros(len(states));h=sparse.csr_matrix((len(states),)*2)
    for bonds,coupling in [(g.edges,1.),(g.next_edges or [],j2)]:
        if not coupling:continue
        for i,j in bonds:
            opposite=((states>>i)^(states>>j))&1
            diagonal+=coupling*(1-2*opposite)/4
            h+=flip_term(n,[i,j],coupling*opposite/2)
    if selection_field:
        # A diagnostic only, not in the main zero-field Hamiltonian.
        diagonal-=selection_field*(1-2*(states&1))/2
    return h+sparse.diags(diagonal)


def paramagnet(n):
    return -sum((flip_term(n,[i]) for i in range(n)),start=sparse.csr_matrix((1<<n,)*2))


def boolean_phase(n,supports):
    states=np.arange(1<<n,dtype=np.int64);parity=np.zeros(1<<n,dtype=np.int8)
    for support in supports:
        mask=sum(1<<int(i) for i in support)
        parity^=((states&mask)==mask)
    return 1.-2*parity


def conjugate_diagonal(h,phase):
    d=sparse.diags(phase)
    return (d.conj()@h@d).tocsr()


def cluster(g):
    """Independent Pauli construction of -sum_i X_i product_neighbor Z_j."""
    states=np.arange(1<<g.n,dtype=np.int64);h=sparse.csr_matrix((1<<g.n,)*2)
    neighbors=[[] for _ in range(g.n)]
    for a,b in g.edges:neighbors[a].append(b);neighbors[b].append(a)
    for i in range(g.n):
        values=-np.ones(len(states))
        for j in neighbors[i]:values*=1-2*((states>>j)&1)
        h+=flip_term(g.n,[i],values)
    return h


def levin_gu(g,perturbation=0.):
    """D(-sum X)D - g sum X, D=(-1)^domain-wall-components on torus."""
    phase,counts=domain_wall_signs(g)
    h0=paramagnet(g.n)
    return conjugate_diagonal(h0,phase)+perturbation*h0,phase,counts


def levin_gu_local(g):
    """Independent local formula B_i=-X_i i^(domain walls around i); H=-sum B_i."""
    states=np.arange(1<<g.n,dtype=np.int64);h=sparse.csr_matrix((1<<g.n,)*2)
    for i in range(g.n):
        opposite_edges=[tuple(j for j in t if j!=i) for t in g.triangles if i in t]
        number=sum((((states>>a)^(states>>b))&1 for a,b in opposite_edges),start=np.zeros(len(states),int))
        assert np.all(number%2==0)
        h+=flip_term(g.n,[i],(-1.)**(number//2))
    return h


def frustrated(g,lam):
    states=np.arange(1<<g.n,dtype=np.int64);h=paramagnet(g.n)
    diag=np.zeros(len(states))
    for i in range(g.n):diag-=.15*(1-2*((states>>i)&1))
    for i,j in g.edges:
        diag-=.4*(1-2*(((states>>i)^(states>>j))&1))
        h+=lam*flip_term(g.n,[i,j])
    return h+sparse.diags(diag)


def loop_projectors(hc,semion=False):
    """Full-Hilbert commuting-projector string-net extension, H=-sum Q_v-sum B_p.

    B_p = (1 + X_boundary)/2 P_boundary (TC), or
    B_p = (1 - X_boundary i^N_legs)/2 P_boundary (DS).
    Vertex projectors make DS Hermitian also outside the closed-loop subspace.
    The same extension is used for the paired TC model; thermal excitation
    convention is recorded rather than silently replacing it by unprojected TC.
    """
    n=hc['n'];states=np.arange(1<<n,dtype=np.int64)
    qs=[]
    for star in hc['stars']:
        parity=np.zeros(len(states),int)
        for e in star:parity^=(states>>e)&1
        qs.append(1.-parity)
    terms=[sparse.diags(q,format='csr') for q in qs]
    for face,vertices,legs in zip(hc['faces'],hc['face_vertices'],hc['legs_with_incidence_multiplicity']):
        allowed=np.prod([qs[v] for v in vertices],axis=0)
        if semion:
            nlegs=sum(((states>>e)&1 for e in legs),start=np.zeros(len(states),int))
            assert np.all((nlegs[allowed>0]%2)==0)
            sign=-(-1.)**(nlegs//2)
        else:sign=np.ones(len(states))
        terms.append((sparse.diags(allowed)+flip_term(n,face,allowed*sign))/2)
    return (-sum(terms,start=sparse.csr_matrix((1<<n,)*2))).tocsr(),terms


def ideal_loop_state(hc,semion=False):
    states=vacuum_loop_sector(hc)
    psi=np.zeros(1<<hc['n'])
    for state in states:
        loops=loop_count([e for e in range(hc['n']) if (state>>e)&1],hc['edges'],hc['vertices'])
        psi[state]=(-1.)**loops if semion else 1.
    return psi/np.linalg.norm(psi),states


def commuting_gibbs(terms,beta,*,sparse_output=False):
    """Exact exp(beta sum projectors), sparse throughout. No dense 18-qubit path."""
    d=terms[0].shape[0]
    if d>4096:raise ValueError('dense thermal output forbidden above 12 qubits')
    rho=sparse.eye(d,format='csr')
    # Rescaled factors avoid overflow; normalization cancels the constant.
    for term in terms:rho=np.exp(-beta)*rho+(-np.expm1(-beta))*(term@rho)
    rho/=rho.diagonal().sum()
    return rho if sparse_output else rho.toarray()


def paramagnet_gibbs(n,beta,phase=None):
    rho=np.array([[1.,np.tanh(beta)],[np.tanh(beta),1.]])/2
    result=np.array([[1.]])
    for _ in range(n):result=np.kron(result,rho)
    if phase is not None:result=result*phase[:,None]*phase[None,:]
    return result


def block_eigensystem(h):
    """Exploit exact configuration-support components, preserving all sectors."""
    h=h.tocsr();graph=h.copy();graph.setdiag(0);graph.eliminate_zeros()
    _,labels=connected_components(abs(graph),directed=False)
    blocks=[]
    for label in np.unique(labels):
        indices=np.flatnonzero(labels==label)
        energies,vectors=eigh(h[indices][:,indices].toarray(),check_finite=False)
        blocks.append((indices,energies,vectors))
    return blocks


def block_gibbs(blocks,beta,d):
    if d>6000:raise ValueError('thermal size guard; use matrix-free methods')
    e0=min(float(e[0]) for _,e,_ in blocks)
    total=sum(float(np.exp(-beta*(e-e0)).sum()) for _,e,_ in blocks)
    rho=np.zeros((d,d),dtype=np.result_type(*[v.dtype for _,_,v in blocks]))
    eigenvalues=[];energy=0.
    for indices,e,v in blocks:
        p=np.exp(-beta*(e-e0))/total
        rho[np.ix_(indices,indices)]=(v*p)@v.conj().T
        eigenvalues.extend(p);energy+=float(p@e)
    all_e=np.sort(np.concatenate([e for _,e,_ in blocks]));excited=all_e[all_e>e0+1e-9]
    sorted_probabilities=np.exp(-beta*(all_e-e0))/total
    return rho,{'purity':float(np.sum(np.array(eigenvalues)**2)),'energy':energy,
                'gap':float(excited[0]-e0) if len(excited) else 0.,
                'ground_degeneracy':int(np.sum(abs(all_e-e0)<1e-9)),
                'eigenvalues':sorted_probabilities,'energies':all_e,
                'ground_manifold_weight':float(np.sum(sorted_probabilities[abs(all_e-e0)<1e-9])),
                'sector_sizes':[len(i) for i,_,_ in blocks]}


def tj_basis(n,n_down=None):
    n_down=(n-1)//2 if n_down is None else n_down
    states=[];fock=[]
    for hole in range(n):
        occupied=[i for i in range(n) if i!=hole]
        for down in combinations(occupied,n_down):
            digits=np.ones(n,dtype=np.int8);digits[hole]=0;digits[list(down)]=2
            mask=sum(1<<(2*i+(int(digits[i])-1)) for i in occupied)
            states.append(digits);fock.append(mask)
    return np.array(states),np.array(fock,dtype=np.int64)


def fermion_action(mask,operations):
    """operations in application order: (orbital, creation_bool)."""
    sign=1
    for orbital,creation in operations:
        present=(mask>>orbital)&1
        if bool(present)==bool(creation):return None,0
        sign*=(-1)**((mask&((1<<orbital)-1)).bit_count())
        mask^=1<<orbital
    return mask,sign


def tj_hamiltonian(g,t=1.,J=1.,sigma=False,n_down=None,flux=0.):
    digits,fock=tj_basis(g.n,n_down);index={int(m):i for i,m in enumerate(fock)}
    rows=[];cols=[];values=[];hops=[]
    for col,mask64 in enumerate(fock):
        mask=int(mask64);s=digits[col];diagonal=0.
        for i,j in g.edges:
            if s[i] and s[j]:
                if s[i]!=s[j]:
                    diagonal-=J/2
                    flipped=s.copy();flipped[i],flipped[j]=flipped[j],flipped[i]
                    target=sum(1<<(2*k+int(flipped[k])-1) for k in range(g.n) if flipped[k])
                    rows.append(index[target]);cols.append(col);values.append(J/2)
            for dest,source in [(i,j),(j,i)]:
                if s[dest]!=0 or s[source]==0:continue
                spin=int(s[source])-1
                target,sgn=fermion_action(mask,[(2*source+spin,False),(2*dest+spin,True)])
                amplitude=-t*sgn*((-1)**spin if sigma else 1)
                # Flux on the closing bond of an oriented 1D ring only.
                if flux and {dest,source}=={0,g.n-1}:
                    amplitude*=np.exp(1j*flux*(1 if dest==0 else -1))
                row=index[target];rows.append(row);cols.append(col);values.append(amplitude)
                hops.append((col,row,int(s[source]==2),dest,source))
        rows.append(col);cols.append(col);values.append(diagonal)
    h=sparse.coo_matrix((values,(rows,cols)),shape=(len(digits),)*2).tocsr()
    return h,digits,fock,np.array(hops,dtype=np.int64)


def tj_marshall_gauge(g,digits):
    """Canonical Fock -> hole-compatible Marshall basis, only on bipartite graphs.

    The hole-dependent phase removes the canonical ordering sign on every bond;
    the remaining -H hopping sign is (-1) for a down-spin exchange.
    """
    if g.sublattice is None or any(g.sublattice[a]==g.sublattice[b] for a,b in g.edges):
        raise ValueError('phase-string identity requires a bipartite geometry')
    holes=np.argmax(digits==0,axis=1)
    down_a=((digits==2)&(np.array(g.sublattice)[None,:]==0)).sum(1)
    return (-1.)**(holes+np.array(g.sublattice)[holes]+down_a)
