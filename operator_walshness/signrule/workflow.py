"""Resumable small-system blind scan; freeze files are written before any scoring.

All round-3 output is separate from rounds 1/2. Jobs own disjoint directories.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import eigvalsh
from scipy import sparse
import torch

from ..round2 import atomic_json,schmidt
from ..walsh import (fixed_basis_spectrum,walshness_from_spectrum,rotate_operator,
    directions_to_walsh_basis,x_basis_matrix,normalized_expectations,identity_walshness)
from ..observables import degree_correlations_from_x_matrix
from .geometry import chain,triangular,local_gate_supports
from .models import (heisenberg,cluster,paramagnet,levin_gu,frustrated,boolean_phase,
    conjugate_diagonal,block_eigensystem,block_gibbs)
from .blind import optimize_local,optimize_circuit,circuit_phase

ROOT=Path(__file__).resolve().parents[2]


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def register(out):
    out=Path(out);(out/'registration').mkdir(parents=True,exist_ok=True)
    path=out/'registration'/'provenance.json'
    if path.exists():return
    source_paths=sorted((ROOT/'operator_walshness'/'signrule').glob('*.py'))
    source_hash=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in source_paths)).hexdigest()
    atomic_json(path,{'created_utc':utc(),'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'round3_source_sha256_at_registration':source_hash,'python':sys.version,'platform':platform.platform(),
        'libraries':{name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','pandas','pytest']},
        'conventions':{'S':'sigma/2','qubit_site_order':'little endian','basis':'same U: Udagger O U',
                       'diagonal_circuit':'geometric supports only; no known labels in blind search',
                       'entropy_log':2,'tJ':'canonical interleaved Fock orbitals; fixed one-hole, fixed Sz',
                       'TC_DS':'H=-sum Q_v-sum B_p; B_p includes all adjacent vertex projectors'},
        'output_hash_policy':'one registration source digest; per-freeze digest, not raw-matrix rehash loops',
        'completion_status':'active: not all phases implemented or executed'})


def build_qubit_job(job):
    name=job['model'];n=job['n'];p=job.get('parameter',0.)
    g=triangular(*job['shape'],periodic=job.get('boundary','periodic')=='periodic') if 'shape' in job else chain(n,job.get('boundary','open')=='periodic')
    if name in ('bipartite','j1j2','triangular'):h=heisenberg(g,p)
    elif name=='cluster':h=cluster(g)
    elif name=='paramagnet':h=paramagnet(n)
    elif name=='levin_gu':h,_,_=levin_gu(g,p)
    elif name=='controlled_frustration':h=frustrated(g,p)
    elif name in ('random_linear','random_quadratic','random_positive'):
        rng=np.random.default_rng(job.get('builder_seed',1927))
        if name=='random_positive':
            # A stoquastic real random Hamiltonian produces a positive Gibbs kernel.
            a=rng.uniform(.01,1.,(1<<n,1<<n));h=sparse.csr_matrix(-(a+a.T)/(2**n))
        else:
            supports=local_gate_supports(g,1 if name=='random_linear' else 2)
            chosen=[s for s in supports if rng.random()<.5]
            h=conjugate_diagonal(paramagnet(n),boolean_phase(n,chosen))
    else:raise ValueError(name)
    return g,h.tocsr()


def initial_jobs():
    """Phase-1 small geometries. Larger/independent cases remain explicit pending gates."""
    jobs=[]
    for n in [6,8]:
        for name in ['bipartite','cluster','paramagnet','random_linear','random_quadratic']:
            jobs.append({'track':'A','model':name,'n':n})
    jobs.append({'track':'A','model':'random_positive','n':6})
    for j2 in [0.,.1,.2]:jobs.append({'track':'B','model':'triangular','n':9,'shape':[3,3],'parameter':j2})
    for g in [0.,.2,.5,1.]:jobs.append({'track':'D','model':'levin_gu','n':9,'shape':[3,3],'parameter':g})
    jobs.append({'track':'D','model':'paramagnet','n':9,'shape':[3,3]})
    for n in [6,8]:
        for j2 in [0.,.3,.5,1.]:jobs.append({'track':'F','model':'j1j2','n':n,'parameter':j2})
        for lam in [0.,.05,.1,.2,.4]:jobs.append({'track':'F','model':'controlled_frustration','n':n,'parameter':lam})
    for index,job in enumerate(jobs):
        job.update({'job_id':f'{index:03d}_{job["model"]}_n{job["n"]}_p{job.get("parameter",0):g}',
                    'betas':[0.,.25,.5,1.,2.,4.],'alphas':[.5,1.,2.],
                    'seeds':[20261027,20261028,20261029],
                    'local_starts':24,'local_steps':150,'local_polish':30,
                    'circuit_starts':16,'circuit_flips':2000,'gate_budget_per_site':1})
    return jobs


def point_directory(out,job,beta,alpha,seed):
    geometry=('x'.join(map(str,job['shape'])) if 'shape' in job else 'chain')+'_'+job.get('boundary','periodic' if 'shape' in job else 'open')
    return Path(out)/job['model']/geometry/f'n{job["n"]}'/f'p{job.get("parameter",0):g}'/f'beta{beta:g}'/f'alpha{alpha:g}'/f'seed{seed}'


def run_qubit_job(job,out,device):
    status_path=Path(out)/'jobs'/f'{job["job_id"]}.json'
    atomic_json(status_path,{'status':'running','pid':os.getpid(),'device':device,'started_utc':utc(),'job':job})
    g,h=build_qubit_job(job);blocks=block_eigensystem(h);n=job['n']
    for beta in job['betas']:
        rho,thermal=block_gibbs(blocks,beta,1<<n)
        fixed,spectral_check=fixed_basis_spectrum(rho)
        if spectral_check['mass_relative_error']>1e-10:raise ArithmeticError('FWHT norm defect')
        _,osee=schmidt(rho)
        correlations=degree_correlations_from_x_matrix(x_basis_matrix(rho))
        for alpha in job['alphas']:
            for seed in job['seeds']:
                point=point_directory(out,job,beta,alpha,seed)
                if (point/'metrics.json').exists():continue
                point.mkdir(parents=True,exist_ok=True)
                atomic_json(point/'metadata.json',{'job':job,'geometry':g.metadata(),'beta':beta,'alpha':alpha,
                    'seed':seed,'started_utc':utc(),'operator_representation':'original computational qubit',
                    'optimization_known_labels_available':False,'phase':'blind_stage'})
                wfixed=walshness_from_spectrum(fixed,alpha)
                if (point/'blind_frozen.json').exists():
                    results=json.loads((point/'basis_or_circuit.json').read_text())
                elif beta==0:
                    if abs(wfixed-identity_walshness(n,alpha))>1e-10:raise ArithmeticError('identity failed')
                    results={'identity_control':{'W_best':wfixed,'no_discovery':True}}
                else:
                    results={'local_SU2':optimize_local(rho,alpha,starts=job['local_starts'],
                        steps=job['local_steps'],polish=job['local_polish'],seed=seed,device=device)}
                    for level in [1,2,3]:
                        supports=local_gate_supports(g,level)
                        if level==3 and not g.triangles:continue
                        results[f'circuit_L{level}']=optimize_circuit(rho,alpha,supports,
                            starts=job['circuit_starts'],flips=job['circuit_flips'],seed=seed,
                            max_gates=n*job['gate_budget_per_site'],max_depth=n,device=device)
                    if job['model']=='levin_gu':
                        results['circuit_symmetric']=optimize_circuit(rho,alpha,local_gate_supports(g,3),
                            starts=job['circuit_starts'],flips=job['circuit_flips'],seed=seed,
                            max_gates=n*job['gate_budget_per_site'],max_depth=n,symmetric=True,device=device)
                # The freeze marker is durable before any oracle or held-out scores.
                atomic_json(point/'basis_or_circuit.json',results)
                freeze={'frozen_utc':utc(),'objective':'Operator_Walshness_only','known_label_access':False,
                        'basis_or_circuit_sha256':hashlib.sha256((point/'basis_or_circuit.json').read_bytes()).hexdigest(),
                        'job':job,'beta':beta,'alpha':alpha,'seed':seed}
                atomic_json(point/'blind_frozen.json',freeze)
                spectra={'fixed':fixed};rows=[]
                for label,result in results.items():
                    row={'representation':label,'W':result['W_best'],'delta_W':wfixed-result['W_best'],
                         'W_per_2n':result['W_best']/(2*n)}
                    if label=='local_SU2':
                        rotated=rotate_operator(rho,directions_to_walsh_basis(result['axes_best']))
                        spectra[label]=fixed_basis_spectrum(rotated)[0]
                    elif label.startswith('circuit'):
                        phase=circuit_phase(n,result)
                        spectra[label]=fixed_basis_spectrum(rho*phase[:,None]*phase[None,:])[0]
                    if label in spectra and abs(walshness_from_spectrum(spectra[label],alpha)-result['W_best'])>1e-8:
                        raise ArithmeticError('saved candidate does not reproduce blind objective')
                    rows.append(row)
                np.savez_compressed(point/'walsh_spectrum.npz',**spectra,**correlations)
                np.savez_compressed(point/'thermal_spectrum.npz',eigenvalues=thermal['eigenvalues'],energies=thermal['energies'])
                # Keep thermal dict intact across alphas/seeds (no mutation).
                clean_thermal={k:v for k,v in thermal.items() if k not in ('energies','eigenvalues')}
                purity=float(np.vdot(rho,rho).real)
                atomic_json(point/'metrics.json',{'beta':beta,'alpha':alpha,'seed':seed,'model':job['model'],'n':n,
                    'parameter':job.get('parameter',0),'track':job['track'],'W_fixed':wfixed,'representations':rows,
                    'purity':purity,'R_eff':1/purity,'mixed_operator_gate':bool(beta>0 and purity<.95),
                    'beta_gap':beta*thermal['gap'],'thermal':clean_thermal,'OSEE_fixed':osee,
                    'local_magnetization':normalized_expectations(rho)[0],
                    'Hermiticity_residual':float(np.max(abs(rho-rho.conj().T))),
                    'trace_residual':float(abs(np.trace(rho)-1)),'scoring_status':'pending_posthoc',
                    'discovery_classification':'unscored_no_claim'})
                print(f'FROZEN {job["job_id"]} beta={beta} alpha={alpha} seed={seed}',flush=True)
    atomic_json(status_path,{'status':'blind_complete','finished_utc':utc(),'job':job})


def run_worker(out,worker,workers,device):
    torch.set_num_threads(1)
    jobs=json.loads((Path(out)/'registration'/'jobs.json').read_text())
    for index,job in enumerate(jobs):
        if index%workers!=worker:continue
        status=Path(out)/'jobs'/f'{job["job_id"]}.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='blind_complete':continue
        try:run_qubit_job(job,out,device)
        except Exception as error:
            import traceback
            atomic_json(status,{'status':'failed','error':str(error),'traceback':traceback.format_exc(),'job':job})
            raise
