import numpy as np
from scipy import sparse
from operator_walshness.signrule.diagnostics import phase_entries,score_phase
from operator_walshness.signrule.geometry import honeycomb,loop_count,vacuum_loop_sector
from operator_walshness.signrule.loop_topology import closed_loop_topology,closed_sector_labels
from operator_walshness.signrule.models import loop_projectors


def test_sparse_phase_entries_match_dense_complex():
    rng=np.random.default_rng(71);a=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
    a[a.real<.7]=0;a=a+a.conj().T
    for cutoff in [1e-8,1e-12]:
        dense=phase_entries(a,cutoff);compressed=phase_entries(sparse.csr_matrix(a),cutoff)
        for key in dense:np.testing.assert_allclose(dense[key],compressed[key],atol=1e-12)


def test_windings_and_face_cosets_agree_with_projector_sectors():
    hc=honeycomb(2,2);labels,counts,cosets=closed_sector_labels(hc)
    assert set(labels)=={-1,0,1,2,3}
    assert [np.sum(labels==k) for k in range(4)]==[8]*4
    for state in np.flatnonzero(labels>=0):
        geometry=closed_loop_topology(hc,int(state))
        assert len(geometry['windings'])==loop_count([e for e in range(12) if (state>>e)&1],hc['edges'],hc['vertices'])
    for semion in [False,True]:
        h,_=loop_projectors(hc,semion);a=h.tocoo()
        active=(labels[a.row]>=0)&(labels[a.col]>=0)
        assert np.all(labels[a.row[active]]==labels[a.col[active]])
    ids=vacuum_loop_sector(hc);psi=np.zeros(1<<12);psi[ids]=(-1.)**counts[ids]/np.sqrt(len(ids))
    h,_=loop_projectors(hc,True)
    assert np.linalg.norm(h@psi+len(hc['stars']+hc['faces'])*psi)<1e-12


def test_open_string_has_no_loop_oracle():
    assert closed_loop_topology(honeycomb(2,2),1)['closed'] is False
