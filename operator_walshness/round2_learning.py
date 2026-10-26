"""Independent batched MLPs with paired seeds, uniform training and held-out split.

Targets have unit mean squared matrix-element magnitude (d O / ||O||_F),
so a mean square prediction loss is the squared relative Frobenius loss.
Every model has its own parameters and Adam states; batching does not couple
the networks. The production runner uses ordinary eager PyTorch execution.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from torch import nn


def split_constants(total, fraction=.1, seed=19473):
    # An invertible affine permutation of the power-of-two entry space.
    rng=np.random.default_rng(seed)
    multiplier=int(rng.integers(1,total))|1
    offset=int(rng.integers(0,total))
    return multiplier,offset,pow(multiplier,-1,total),int(total*fraction)


def features_from_indices(indices,n):
    shifts=torch.arange(2*n,device=indices.device)
    return 1.-2.*((indices[:,None] >> shifts)&1).float()


class BatchedMLP(nn.Module):
    def __init__(self,n,width,depth,activation,bases,seeds,device):
        super().__init__(); self.activation=activation
        dims=[2*n]+[width]*depth+[2]
        self.weights=nn.ParameterList();self.biases=nn.ParameterList()
        for di,do in zip(dims[:-1],dims[1:]):
            allw=[];allb=[]
            for basis in range(bases):
                for seed in range(seeds):
                    # Same seed and same layer weights across all physical bases.
                    gen=torch.Generator().manual_seed(50000+seed*1000+len(self.weights))
                    allw.append((torch.rand(di,do,generator=gen)*2-1)/math.sqrt(di))
                    allb.append(torch.zeros(1,do))
            self.weights.append(nn.Parameter(torch.stack(allw).to(device)))
            self.biases.append(nn.Parameter(torch.stack(allb).to(device)))
        self.params_per_model=sum(p.numel() for p in self.parameters())//(bases*seeds)
        self.matmul_coeff=sum(di*do for di,do in zip(dims[:-1],dims[1:]))

    def forward(self,features):
        x=features.unsqueeze(0).expand(self.weights[0].shape[0],-1,-1)
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            x=torch.bmm(x,weight)+bias
            if i<len(self.weights)-1:
                x=x.square() if self.activation=='square' else torch.tanh(x)
        return x


@torch.no_grad()
def evaluate(model,targets,basis_index,n,split,chunk=16384):
    total=2**(2*n);mult,offset,_,hold=split
    model_count=len(basis_index)
    numerator=torch.zeros((model_count,3),dtype=torch.float64,device=targets.device)
    denominator=torch.zeros_like(numerator)
    for start in range(0,total,chunk):
        idx=torch.arange(start,min(start+chunk,total),device=targets.device)
        y=targets[basis_index[:,None],idx[None,:],:]
        pred=model(features_from_indices(idx,n))
        err=(pred-y).double().square().sum(-1);norm=y.double().square().sum(-1)
        held=((idx*mult+offset)&(total-1))<hold
        numerator[:,0]+=err.sum(-1);denominator[:,0]+=norm.sum(-1)
        numerator[:,1]+=err[:,~held].sum(-1);denominator[:,1]+=norm[:,~held].sum(-1)
        numerator[:,2]+=err[:,held].sum(-1);denominator[:,2]+=norm[:,held].sum(-1)
    return (numerator/denominator.clamp_min(1e-300)).cpu().numpy()


def train_batched(targets,n,width,depth,activation,*,seeds=8,steps=2000,
                  batch_size=256,device='cuda:7',lr=1e-3,graph=False,
                  matched_flops=None,progress=None,checkpoints=None):
    """Return per-model exact errors at both equal-step and equal-FLOP budgets."""
    if graph:
        raise ValueError('CUDA graph capture is unsupported in this environment; use graph=False')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(4321);torch.cuda.manual_seed_all(4321)
    targets=torch.as_tensor(targets,dtype=torch.float32,device=device)
    bases=targets.shape[0];model_count=bases*seeds;total=2**(2*n)
    model=BatchedMLP(n,width,depth,activation,bases,seeds,device)
    basis_index=torch.arange(bases,device=device).repeat_interleave(seeds)
    split=split_constants(total);mult,offset,inverse,held=split
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,capturable=False,foreach=False)
    flops_step=6*batch_size*model.matmul_coeff+12*model.params_per_model
    matched_step=max(1,min(steps,int(matched_flops//flops_step))) if matched_flops else steps
    milestones=sorted(set([matched_step,steps]+[int(s) for s in (checkpoints or []) if s<=steps]))
    last_loss=None
    def iteration():
        nonlocal last_loss
        rank=torch.randint(held,total,(batch_size,),device=device)
        idx=((rank-offset)*inverse)&(total-1)
        features=features_from_indices(idx,n)
        y=targets[basis_index[:,None],idx[None,:],:]
        optimizer.zero_grad(set_to_none=True)
        pred=model(features)
        loss=(pred-y).square().sum(-1).mean(-1)
        loss.sum().backward();optimizer.step()
        last_loss=loss
    begin=time.perf_counter();done=0;learning_curve=[];evaluations={}
    # Warm-up updates count toward the reported training budget.
    for _ in range(min(3,steps)):
        iteration();done+=1
        if done in milestones: evaluations[done]=evaluate(model,targets,basis_index,n,split)
    while done<steps:
        iteration()
        done+=1
        if done%25==0 or done in milestones:
            loss_np=last_loss.detach().cpu().numpy()
            if not np.all(np.isfinite(loss_np)): raise FloatingPointError('nonfinite MLP loss')
            learning_curve.append((done,loss_np.copy()))
        if done in milestones:
            evaluations[done]=evaluate(model,targets,basis_index,n,split)
            if progress: progress(done,float(np.median(evaluations[done][:,0])),time.perf_counter()-begin)
    torch.cuda.synchronize(device)
    state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    return {'evaluations':evaluations,'matched_step':matched_step,'flops_per_step':flops_step,
            'params_per_model':model.params_per_model,'runtime_seconds':time.perf_counter()-begin,
            'curve_steps':np.array([x[0] for x in learning_curve]),
            'curve_loss':np.stack([x[1] for x in learning_curve]) if learning_curve else np.empty((0,model_count)),
            'model_state':state,'split':split,'graph':graph}
