import numpy as np
import torch
from operator_walshness.round2_learning import BatchedMLP, features_from_indices, split_constants, evaluate


def test_batched_networks_have_paired_bases_and_independent_seeds():
    torch.set_num_threads(1)
    model=BatchedMLP(3,8,2,'tanh',bases=2,seeds=2,device='cpu')
    x=features_from_indices(torch.arange(64),3);pred=model(x)
    torch.testing.assert_close(pred[0],pred[2]);torch.testing.assert_close(pred[1],pred[3])
    assert not torch.allclose(pred[0],pred[1])
    original=model.weights[0][2].clone()
    with torch.no_grad():model.weights[0][0].add_(1.)
    torch.testing.assert_close(model.weights[0][2],original)


def test_split_is_disjoint_and_evaluation_matches_dense_manual():
    n=3;total=2**(2*n);mult,off,inv,hold=split_constants(total)
    permutation=(np.arange(total)*mult+off)&(total-1)
    assert len(np.unique(permutation))==total
    train=((np.arange(hold,total)-off)*inv)&(total-1)
    assert np.all(permutation[train]>=hold)
    model=BatchedMLP(n,8,1,'square',bases=2,seeds=1,device='cpu')
    targets=torch.arange(2*total*2).reshape(2,total,2).float()/100
    errors=evaluate(model,targets,torch.arange(2),n,(mult,off,inv,hold),chunk=7)
    with torch.no_grad():
        pred=model(features_from_indices(torch.arange(total),n))
        manual=(pred-targets).double().square().sum((1,2))/targets.double().square().sum((1,2))
    np.testing.assert_allclose(errors[:,0],manual.numpy(),rtol=1e-6)
