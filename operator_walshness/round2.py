"""Round-2 exact observables. All entropies use log base two.

QMC signs refer only to the full-configuration continuous-time worldline
expansion specified in the plan. No Monte Carlo sampling is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import datetime
import platform
import subprocess
import sys
import importlib.metadata
import fcntl
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, svdvals
from scipy.special import logsumexp
from scipy.optimize import minimize

from .hamiltonians import add_xx, add_yy, add_zz, Z
from .observables import degree_correlations_from_x_matrix
from .walsh import (fixed_basis_spectrum, walshness_from_spectrum, rotate_operator,
                    directions_to_walsh_basis, align_direction_to_z, gibbs_density,
                    degree_histogram, spectrum_from_masses)

ROOT = Path(__file__).resolve().parents[1]


def initialize_run(out):
    """Create provenance once, including when independent stages start together."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.initialize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'metadata.json').exists():return
        (out/'config').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
        for name in ('operator_walshness_round2_experiment_plan.yaml','round2_first_wave.yaml'):
            (out/'config'/name).write_bytes((ROOT/name).read_bytes())
        try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        except (OSError,subprocess.CalledProcessError):commit='unavailable'
        files=sorted(list((ROOT/'operator_walshness').glob('*.py'))+list((ROOT/'scripts').glob('*round2*.py')))
        source_hash=hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in files)).hexdigest()
        meta={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_commit':commit,
              'python':sys.version,'platform':platform.platform(),'source_sha256':source_hash,
              'config_sha256':hashlib.sha256((ROOT/'round2_first_wave.yaml').read_bytes()).hexdigest(),
              'dtype_physics':'float64 / complex128','dtype_MLP':'float32','seed':20261026,
              'scope':'bounded first wave; see config and per-stage status','entropy_log_base':2}
        for package in ('numpy','scipy','torch','pandas','matplotlib','PyYAML'):
            meta[package]=importlib.metadata.version(package)
        atomic_json(out/'metadata.json',meta)


def jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, default=jsonable, ensure_ascii=False, indent=2))
    temp.replace(path)


def save_result(out, stage, key, rows, **arrays):
    """One immutable completed artifact per task; JSON is its commit marker."""
    base = Path(out)/stage/key
    base.parent.mkdir(parents=True, exist_ok=True)
    # Keys contain decimal physical parameters; with_suffix would truncate them.
    json_path = base.parent / (base.name + '.json')
    npz_path = base.parent / (base.name + '.npz')
    if json_path.exists():
        raise FileExistsError(f'completed result exists: {base}')
    if arrays:
        temp = base.parent / (base.name + f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(temp, **arrays)
        temp.replace(npz_path)
    atomic_json(json_path, rows)


def schmidt(operator, cut=None):
    n = int(np.log2(len(operator)))
    cut = cut or n//2
    da, db = 2**cut, 2**(n-cut)
    reshaped = operator.reshape(db,da,db,da).transpose(0,2,1,3).reshape(db*db,da*da)
    s = svdvals(reshaped, check_finite=False)
    weights = s*s / np.vdot(operator,operator).real
    pos = weights > 0
    entropy = -np.dot(weights[pos],np.log2(weights[pos]))
    return np.sqrt(weights), float(entropy)


def walsh_mutual_information(xmatrix, cut=None):
    n = int(np.log2(len(xmatrix)))
    cut = cut or n//2
    da,db = 2**cut,2**(n-cut)
    p = abs(xmatrix.reshape(db,da,db,da).transpose(0,2,1,3).reshape(db*db,da*da))**2
    p /= p.sum()
    pa,pb = p.sum(0),p.sum(1)
    expected = pb[:,None]*pa[None,:]
    mask = (p>0)&(expected>0)
    return float(np.sum(p[mask]*np.log2(p[mask]/expected[mask])))


def thermal_walsh_identity(rho, rho_2beta, axes):
    """Independent equal-time / imaginary-time traces in physical n_i axes.

    E_equal = Tr(rho(2 beta) P_i P_j); E_tau = Tr(rho(beta) P_i
    rho(beta) P_j)/Tr(rho(beta)^2). The latter equals tau=beta at 2 beta.
    After physically aligning P_i to Z these traces can be evaluated using
    diagonal entries and entrywise magnitudes, without constructing P_i P_j.
    """
    unitaries = [align_direction_to_z(a) for a in axes]
    rotated = rotate_operator(rho,unitaries)
    rotated2 = rotate_operator(rho_2beta,unitaries)
    n,d = len(axes),len(rho)
    z = 1.-2.*((np.arange(d)[None,:] >> np.arange(n)[:,None])&1)
    diagonal = np.diag(rotated2).real
    means = z@diagonal
    equal = (z*diagonal)@z.T-np.outer(means,means)
    imaginary = z@(abs(rotated)**2)@z.T/np.vdot(rho,rho).real-np.outer(means,means)
    return equal,imaginary,rotated


def distance_summary(cov):
    return np.array([np.mean(np.diag(cov,r)) for r in range(1,len(cov))])


def exponential_fit(cov):
    """Descriptive finite-size fit, never an automatic area-law assertion."""
    values = abs(distance_summary(cov))
    r = np.arange(1,len(cov))
    mask = values > max(1e-12,values.max(initial=0)*1e-8)
    if mask.sum()<3: return {'xi_fit':None,'xi_r2':None,'xi_points':int(mask.sum())}
    slope,intercept = np.polyfit(r[mask],np.log(values[mask]),1)
    actual = np.log(values[mask]); fitted = slope*r[mask]+intercept
    denom = np.sum((actual-actual.mean())**2)
    r2 = 1-float(np.sum((actual-fitted)**2))/denom if denom>1e-20 else 0.
    return {'xi_fit':float(-1/slope) if slope < -1e-10 else None,
            'xi_r2':float(r2),'xi_points':int(mask.sum())}


def j1j2_hamiltonian(n, j2, boundary='open'):
    if boundary not in ('open','periodic'): raise ValueError(boundary)
    out = np.zeros((2**n,2**n))
    for distance,coupling in ((1,1.),(2,float(j2))):
        pairs = {(min(i,(i+distance)%n),max(i,(i+distance)%n))
                 for i in range(n if boundary=='periodic' else n-distance)}
        for i,j in pairs:
            for add in (add_xx,add_yy,add_zz): add(out,i,j,coupling/4.)
    return out


def real_unitaries(theta, flips=None):
    if flips is None: flips = np.zeros(len(theta),dtype=int)
    return [np.array([[np.cos(t/2),-np.sin(t/2)],[np.sin(t/2),np.cos(t/2)]])
            @ (Z if flip else np.eye(2)) for t,flip in zip(theta,flips)]


def real_rotate(operator, theta, flips=None):
    rotated = rotate_operator(operator,real_unitaries(theta,flips))
    if np.max(abs(rotated.imag)) > 1e-11: raise ValueError('complex QMC basis')
    return rotated.real


def sign_quenched(h):
    if np.max(abs(np.asarray(h).imag)) > 1e-11: raise ValueError('real H required')
    out = -abs(np.asarray(h).real)
    np.fill_diagonal(out,np.diag(h).real)
    return out


def log_partition(eigenvalues,beta):
    return float(logsumexp(-beta*np.asarray(eigenvalues)))


def sign_metrics(h,beta,reference_eigenvalues=None):
    e = eigh(h,eigvals_only=True,check_finite=False) if reference_eigenvalues is None else reference_eigenvalues
    ea = eigh(sign_quenched(h),eigvals_only=True,check_finite=False)
    logz,logza = log_partition(e,beta),log_partition(ea,beta)
    logsign = logz-logza
    if logsign>1e-9: raise ValueError(f'average sign exceeds one: {logsign}')
    n = int(np.log2(len(h)))
    return {'logZ':logz,'logZ_abs':logza,'log_average_sign':logsign,
            'average_sign':float(np.exp(logsign)),'gamma':float(-logsign/(beta*n)) if beta else 0.}


def eta_positive(h):
    off = np.array(h,copy=True); np.fill_diagonal(off,0)
    return float(np.sum(np.maximum(off,0)**2)/max(np.sum(off**2),1e-300))


def haar_mean_moment(n,purity,alpha,trace=1.):
    """Exact E[G] for complex Haar conjugation; this is NOT exp(alpha E[W])."""
    d = 2**n
    a = (abs(trace)**2-purity/d)/(d*d-1)
    b = (purity-abs(trace)**2/d)/(d*d-1)
    return float((a*(1+np.exp(2*alpha))**n+b*(1+np.exp(alpha))**(2*n))/purity)


def haar_unitary(d,rng,real=False):
    a = rng.normal(size=(d,d))
    if not real: a = a+1j*rng.normal(size=(d,d))
    q,r = np.linalg.qr(a)
    diagonal = np.diag(r)
    phase = np.where(abs(diagonal)>0,diagonal/np.maximum(abs(diagonal),1e-300),1)
    return q*phase


def optimize_real_basis(operator, objective, *, seed=0, starts=4, max_evals=100,
                        beta=1., eigenvalues=None, alpha=1., initial=None):
    """Equal per-start function-evaluation cap for all real-basis objectives.

    Every continuous start is followed by deterministic greedy local Z flips,
    included in the same cap. Diagonal gauges cannot change the sign but can
    change Walshness and eta. All endpoints and budgets are retained.
    """
    n = int(np.log2(len(operator))); rng = np.random.default_rng(seed)
    initial = list(initial or [])
    initial += [(np.zeros(n),np.zeros(n,dtype=int)),
                (np.zeros(n),np.arange(n)%2)]
    while len(initial)<starts:
        initial.append((rng.uniform(-np.pi,np.pi,n),rng.integers(0,2,n)))
    results=[];began=time.perf_counter()
    for theta,flips in initial[:starts]:
        flips = np.array(flips,copy=True); count=0; best=[float('inf'),np.array(theta),flips.copy()]
        def fun(t):
            nonlocal count
            h = real_rotate(operator,t,flips); count+=1
            if objective=='walsh':
                p,_ = fixed_basis_spectrum(h); value=walshness_from_spectrum(p,alpha)
            elif objective=='sign': value=sign_metrics(h,beta,eigenvalues)['gamma']
            elif objective=='eta': value=eta_positive(h)
            else: raise ValueError(objective)
            if value<best[0]: best[:]=[float(value),np.array(t),flips.copy()]
            return value
        # Powell's maxfev includes its initial evaluation. Reserve gauge evaluations.
        minimize(fun,theta,method='Powell',options={'maxfev':max(1,max_evals-2*n),
                                                  'xtol':1e-5,'ftol':1e-9})
        for _ in range(2):
            for i in range(n):
                if count>=max_evals: break
                flips=best[2].copy();flips[i]^=1
                fun(best[1])
        results.append({'value':best[0],'theta':best[1],'flips':best[2],'nfev':count})
    best=min(results,key=lambda r:r['value'])
    return {**best,'restart_values':np.array([r['value'] for r in results]),
            'restart_theta':np.array([r['theta'] for r in results]),
            'restart_flips':np.array([r['flips'] for r in results]),
            'restart_nfev':np.array([r['nfev'] for r in results]),
            'runtime_seconds':time.perf_counter()-began,'certified_global':False,
            'budget_limited':True,'solver_tolerance':{'xtol':1e-5,'ftol':1e-9}}
