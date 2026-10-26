import numpy as np
from operator_walshness.signrule.geometry import triangular,local_gate_supports
from operator_walshness.signrule.models import paramagnet_gibbs,boolean_phase
from operator_walshness.signrule.blind import circuit_phase
from operator_walshness.signrule.transfer import transfer_triangular_circuit,exact_grouped_phase_bootstrap
from operator_walshness.signrule.diagnostics import phase_entries,paired_improvement_ci


def test_transfer_same_lattice_is_identity_and_target_gates_remain_geometric():
    supports=local_gate_supports(triangular(3,3));coef=np.zeros(len(supports),int);coef[[0,5,10,15,40]]=1
    r={'gate_supports':supports,'coefficients_best':coef,'max_gates':9,'max_depth':9,'gatewise_Z2_symmetric':False}
    same=transfer_triangular_circuit(r,(3,3),(3,3))
    assert np.array_equal(circuit_phase(9,r),circuit_phase(9,same))
    bigger=transfer_triangular_circuit(r,(3,3),(3,4));valid=set(local_gate_supports(triangular(3,4)))
    assert all(tuple(s) in valid for s in bigger['gate_supports'])
    assert bigger['gate_count']<=12 and bigger['target_label_or_W_access'] is False


def test_grouped_bootstrap_has_same_mean_and_sampling_law():
    phase=boolean_phase(6,[(0,1),(2,3)]);rho=paramagnet_gibbs(6,.6,phase);entries=phase_entries(rho)
    reference=np.ones(64);candidate=boolean_phase(6,[(0,1)])
    grouped=exact_grouped_phase_bootstrap(entries,reference,candidate,draws=5000)
    ordinary=paired_improvement_ci(entries,reference,candidate,draws=5000)
    assert abs(grouped['improvement']-ordinary['improvement'])<1e-14
    assert np.allclose(grouped['ci95'],ordinary['ci95'],atol=.007)
def test_open_target_catalog_matches_geometry():
    from operator_walshness.signrule.geometry import triangular,local_gate_supports
    from operator_walshness.signrule.transfer import triangular_catalog
    g=triangular(3,4,False)
    assert {x[3] for x in triangular_catalog(3,4,False)}==set(local_gate_supports(g,3))
