import numpy as np
from operator_walshness.signrule.temperature import thermal_rank,select_low_temperature


def test_adaptive_temperature_preserves_degenerate_ground_rank_and_energy_shift():
    e=np.array([0.,0.,.1,.2,.5])
    r=select_low_temperature(e)
    assert r['add_temperature'] and r['selected']['ground_manifold_weight']>=.95
    assert r['selected']['R_eff']>2 and r['selected']['purity']<.5
    assert select_low_temperature(e+123)['selected']['beta']==r['selected']['beta']
    assert abs(thermal_rank(e,1000)['R_eff']-2)<1e-12


def test_already_cold_or_fully_degenerate_system_does_not_expand():
    assert not select_low_temperature([0.,10.])['add_temperature']
    r=select_low_temperature(np.zeros(8))
    assert not r['add_temperature'] and r['selected']['R_eff']==8
