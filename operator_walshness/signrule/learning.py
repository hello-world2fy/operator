"""Paired operator MLP and positive-amplitude representations of frozen phases."""
from __future__ import annotations
import time
import numpy as np
import torch
from torch.nn import functional as F
from ..round2_learning import BatchedMLP,features_from_indices


def hermitian_holdout(n,seed=19473,fraction=.1):
    d=1<<n;row=np.repeat(np.arange(d),d);col=np.tile(np.arange(d),d)
    a=np.minimum(row,col).astype(np.uint64);b=np.maximum(row,col).astype(np.uint64)
    key=a*np.uint64(0x9E3779B185EBCA87)^b^np.uint64(seed)
    key^=key>>np.uint64(30);key*=np.uint64(0xBF58476D1CE4E5B9)
    key^=key>>np.uint64(27);key*=np.uint64(0x94D049BB133111EB);key^=key>>np.uint64(31)
    return (key%np.uint64(1000000))<int(fraction*1000000)


def positive_phase_floor(target,phase):
    projection=np.maximum((target*np.conj(phase)).real,0.)
    return max(0.,float(np.sum(abs(target)**2-projection**2)/np.sum(abs(target)**2)))


def phase_predictions(raw,phases,amplitude):
    magnitude=F.softplus(raw[:,:,0])
    structured=magnitude[:,:,None]*phases
    return torch.where(amplitude[:,None,None],structured,raw)


@torch.no_grad()
def evaluate(model,targets,phases,amplitude,holdout,n,chunk=4096):
    models=len(targets);total=1<<(2*n);d=1<<n
    sums=torch.zeros((models,6),dtype=torch.float64,device=targets.device)
    maximum=torch.linalg.vector_norm(targets,dim=-1).max(1).values
    for start in range(0,total,chunk):
        ids=torch.arange(start,min(start+chunk,total),device=targets.device)
        y=targets[:,ids];pred=phase_predictions(model(features_from_indices(ids,n)),phases[:,ids],amplitude)
        norms=y.double().square().sum(-1);errors=(pred-y).double().square().sum(-1)
        held=holdout[ids]
        sums[:,0]+=errors.sum(1);sums[:,1]+=norms.sum(1)
        sums[:,2]+=errors[:,held].sum(1);sums[:,3]+=norms[:,held].sum(1)
        ymag=torch.linalg.vector_norm(y,dim=-1);pmag=torch.linalg.vector_norm(pred,dim=-1)
        good=((ids//d)!=(ids%d))[None,:]&(ymag>maximum[:,None]*1e-8)
        unit_y=y/ymag[:,:,None].clamp_min(1e-30);unit_p=pred/pmag[:,:,None].clamp_min(1e-30)
        phase_loss=(unit_y-unit_p).double().square().sum(-1)/4
        sums[:,4]+=(norms*phase_loss*good).sum(1);sums[:,5]+=(norms*good).sum(1)
    return torch.stack([sums[:,0]/sums[:,1],sums[:,2]/sums[:,3],sums[:,4]/sums[:,5]],1).cpu().numpy()


def train_phase_batched(targets,phase_factors,amplitude_modes,n,width,depth,*,seeds=8,steps=40000,
                        checkpoints=(2000,10000,40000),batch_size=256,device='cuda:0',progress=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(4321);torch.cuda.manual_seed_all(4321)
    targets=np.asarray(targets);phases=np.asarray(phase_factors);bases=len(targets)
    y=torch.tensor(np.stack([targets.real,targets.imag],-1),dtype=torch.float32,device=device).repeat_interleave(seeds,dim=0)
    factors=torch.tensor(np.stack([phases.real,phases.imag],-1),dtype=torch.float32,device=device).repeat_interleave(seeds,dim=0)
    amplitude=torch.tensor(amplitude_modes,dtype=torch.bool,device=device).repeat_interleave(seeds)
    held=torch.tensor(hermitian_holdout(n),device=device);train_ids=torch.where(~held)[0]
    model=BatchedMLP(n,width,depth,'tanh',bases,seeds,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,capturable=False,foreach=False)
    evaluations={};curve=[];begin=time.monotonic();time_to_threshold={str(t):[None]*(bases*seeds) for t in [.1,.03,.01,.003,.001]}
    for step in range(1,steps+1):
        ids=train_ids[torch.randint(len(train_ids),(batch_size,),device=device)]
        optimizer.zero_grad(set_to_none=True)
        pred=phase_predictions(model(features_from_indices(ids,n)),factors[:,ids],amplitude)
        losses=(pred-y[:,ids]).square().sum(-1).mean(-1)
        losses.sum().backward();optimizer.step()
        if step%250==0:
            values=losses.detach().cpu().numpy()
            if not np.isfinite(values).all():raise FloatingPointError('nonfinite phase MLP loss')
            curve.append((step,values))
        if step in checkpoints or step==steps:
            evaluation=evaluate(model,y,factors,amplitude,held,n);evaluations[step]=evaluation
            for threshold,first in time_to_threshold.items():
                for index,value in enumerate(evaluation[:,0]):
                    if value<=float(threshold) and first[index] is None:first[index]=step
            if progress:progress(step,evaluation,time.monotonic()-begin)
    flops=6*batch_size*model.matmul_coeff+12*model.params_per_model
    return {'evaluations':evaluations,'curve_steps':np.array([x[0] for x in curve]),
        'curve_loss':np.array([x[1] for x in curve]),'model_state':{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()},
        'allocated_parameters_per_model':model.params_per_model,
        'effective_parameters_per_model':[model.params_per_model-(width+1 if a else 0) for a in amplitude_modes],
        'estimated_training_flops_per_step':flops,'runtime_seconds':time.monotonic()-begin,'first_checkpoint_below_error':time_to_threshold,
        'phase_lookup_storage_scalars':[len(p) if a else 0 for p,a in zip(phases,amplitude_modes)],
        'hermitian_partner_holdout':True,'learning_rate':1e-3,'batch_size':batch_size,'seeds':seeds,'activation':'tanh',
        'phase_head_note':'batched kernels allocate two outputs for every model; amplitude modes use only the first output, reported in effective parameters'}
