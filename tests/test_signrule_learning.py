import numpy as np
import torch
from operator_walshness.signrule.learning import hermitian_holdout,positive_phase_floor,phase_predictions,train_phase_batched


def test_partner_split_and_positive_phase_error_floor():
    mask=hermitian_holdout(4).reshape(16,16)
    assert np.array_equal(mask,mask.T)
    target=np.array([1.,-2.,3.]);phase=np.ones(3)
    assert positive_phase_floor(target,phase)==4/14
    assert positive_phase_floor(target,np.sign(target))==0
    complex_target=np.array([1j,2.])
    assert positive_phase_floor(complex_target,np.ones(2))==1/5


def test_phase_output_matches_real_and_complex_factors_and_backpropagates():
    raw=torch.zeros((2,3,2),requires_grad=True)
    phases=torch.tensor([[[0.,1.]]*3,[[-1.,0.]]*3]);modes=torch.tensor([True,False])
    pred=phase_predictions(raw,phases,modes)
    assert torch.all(pred[0,:,0]==0) and torch.all(pred[0,:,1]>0)
    pred.square().sum().backward();assert raw.grad[0,:,0].abs().sum()>0
    assert raw.grad[0,:,1].abs().sum()==0


def test_tiny_cpu_training_reports_exact_errors_and_resources():
    target=np.ones((2,16))/4;phase=np.ones_like(target)
    result=train_phase_batched(target,phase,[False,True],2,8,1,seeds=2,steps=3,checkpoints=[1,3],batch_size=8,device='cpu')
    assert result['evaluations'][3].shape==(4,3)
    assert np.isfinite(result['evaluations'][3]).all()
    assert result['effective_parameters_per_model'][0]-result['effective_parameters_per_model'][1]==9
