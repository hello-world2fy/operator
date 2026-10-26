"""Energy-only temperature selection; no phase labels or optimized candidates."""
import numpy as np
from scipy.optimize import brentq


def thermal_rank(energies, beta, tolerance=1e-9):
    energies=np.asarray(energies,dtype=float)
    if energies.ndim!=1 or not len(energies) or not np.isfinite(energies).all():
        raise ValueError('finite nonempty energy vector required')
    shifted=energies-energies.min();ground=shifted<tolerance
    weights=np.exp(-beta*shifted);weights/=weights.sum()
    purity=float(weights@weights)
    return {'beta':float(beta),'purity':purity,'R_eff':1/purity,
            'ground_degeneracy':int(ground.sum()),'ground_manifold_weight':float(weights[ground].sum()),
            'gap':float(shifted[~ground].min()) if (~ground).any() else 0.}


def select_low_temperature(energies, initial_beta=4., target_ground_weight=.95):
    if not 0<target_ground_weight<1 or initial_beta<=0:raise ValueError('invalid thermal target')
    source=thermal_rank(energies,initial_beta)
    selected=initial_beta
    if source['ground_manifold_weight']<target_ground_weight:
        high=initial_beta*2
        while thermal_rank(energies,high)['ground_manifold_weight']<target_ground_weight:high*=2
        root=brentq(lambda b:thermal_rank(energies,b)['ground_manifold_weight']-target_ground_weight,initial_beta,high)
        selected=float(np.ceil(root*100)/100)
    target=thermal_rank(energies,selected)
    return {'initial':source,'selected':target,'add_temperature':selected>initial_beta,
            'selection':f'smallest beta reaching {target_ground_weight:.0%} ground-manifold weight, rounded upward to 0.01',
            'target_ground_weight':target_ground_weight,'energy_only_selection':True,
            'degeneracy_note':'ground-manifold weight is not purity; asymptotic effective rank equals ground degeneracy'}
