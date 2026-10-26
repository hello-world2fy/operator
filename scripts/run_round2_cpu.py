"""Bounded, resumable CPU stages R2/R3/R4; every task writes its own artifacts."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import eigh
from scipy.stats import t as student_t
from operator_walshness.round2 import *
from operator_walshness.exact import diagonalize, matched_beta
from operator_walshness.grid import case_name
from operator_walshness.optimize import optimize_walshness
from operator_walshness.walsh import intrinsic_alpha0, fixed_x_directions, x_basis_matrix, tail_degree
from operator_walshness.hamiltonians import build_hamiltonian


def basis_record(model,n,params,beta,operator,out):
    name=case_name(model,n,'open',params)+f'_b{beta:g}_a1_s0'
    old=ROOT/'results/round1/basis'/f'{name}.npz'
    if old.exists():
        z=np.load(old)
        return z['axes_best'],{'basis_source':str(old.relative_to(ROOT)),
              'restart_values':z['W_restarts'],'restart_spread':float(np.ptp(z['W_restarts'])),
              'gradient_norm':z['gradient_norm']}
    if beta==0:
        return fixed_x_directions(n),{'basis_source':'analytic_identity','restart_spread':0.}
    import torch
    torch.set_num_threads(1)
    fit=optimize_walshness(operator,1.,restarts=8,steps=60,polish=20,seed=20261026)
    return fit['axes_best'],{'basis_source':'new_8_start_optimization',
                           'restart_values':fit['W_restarts'],
                           'restart_spread':float(np.ptp(fit['W_restarts'])),
                           'gradient_norm':fit['gradient_norm']}


def r2_task(task):
    out,model,n,params,betas=task
    key=case_name(model,n,'open',params)
    if (Path(out)/'R2'/f'{key}.json').exists(): return f'R2 skip {key}'
    start=time.perf_counter();evals,evecs,_=diagonalize(model,n,'open',params)
    rows=[];arrays={}
    for beta in betas:
        rho,_=gibbs_density(evals,evecs,beta)
        rho2,_=gibbs_density(evals,evecs,2*beta)
        singular,osee=schmidt(rho)
        best,diag=basis_record(model,n,params,beta,rho,out)
        axes_options={'fixed_X':fixed_x_directions(n),'fixed_Z':np.tile([0.,0.,1.],(n,1)),
                      'best_numerical_W':best}
        for label,axes in axes_options.items():
            equal,imaginary,xmat=thermal_walsh_identity(rho,rho2,axes)
            stats=degree_correlations_from_x_matrix(xmat)
            cov=stats['degree_correlation'];error=float(np.max(abs(cov-(equal+imaginary)/2)))
            if error>1e-8: raise AssertionError(f'thermal identity {error}')
            p=spectrum_from_masses(degree_histogram(xmat,n)[0])
            iw=walsh_mutual_information(xmat)
            if iw>2*osee+1e-8: raise AssertionError('I_W bound')
            cross_signed=[float(cov[:cut,cut:].sum()) for cut in range(1,n)]
            cross_abs=[float(abs(cov[:cut,cut:]).sum()) for cut in range(1,n)]
            tag=f'b{beta:g}_{label}'
            arrays.update({f'{tag}_P':p,f'{tag}_axes':axes,f'{tag}_degree_mean':stats['degree_mean'],
                           f'{tag}_Cij':cov,f'{tag}_thermal_equal_time':equal,
                           f'{tag}_thermal_imaginary_time':imaginary,f'{tag}_schmidt_values':singular,
                           f'{tag}_boundary_signed':np.array(cross_signed),
                           f'{tag}_boundary_abs':np.array(cross_abs)})
            row={'model':model,'n':n,'boundary':'open','params':params,'beta':beta,'alpha':1.,
                 'basis':label,'seed':20261026,'W':walshness_from_spectrum(p,1.),
                 'osee_bits':osee,'I_W_bits':iw,'thermal_identity_error':error,
                 'half_boundary_signed':cross_signed[n//2-1],'half_boundary_abs':cross_abs[n//2-1],
                 'probability_defect':float(abs(p.sum()-1)), **exponential_fit(cov)}
            if label=='best_numerical_W': row.update(diag)
            if model=='disconnected_dimers':
                mask=np.arange(n)[:,None]//2 != np.arange(n)[None,:]//2
                row['inter_dimer_max_cov']=float(abs(cov[mask]).max(initial=0))
            rows.append(row)
    save_result(out,'R2',key,rows,**arrays)
    return f'R2 {key}: {len(rows)} rows, {time.perf_counter()-start:.1f}s'


def prepare_r3(out,n):
    path=Path(out)/'cache'/f'xxz_n{n}.npz'
    if path.exists(): return path
    e,v,sector=diagonalize('xxz',n,'open',{'delta':1.})
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,e=e,v=v,sector=sector)
    return path


def r3_task(task):
    out,n,ensemble,value,null,draws=task
    key=f'xxz_n{n}_{ensemble}_{value:g}_{null}'
    if (Path(out)/'R3'/f'{key}.json').exists(): return f'R3 skip {key}'
    start_time=time.perf_counter(); z=np.load(Path(out)/'cache'/f'xxz_n{n}.npz')
    e,v,sector=z['e'],z['v'],z['sector'];d=2**n
    if ensemble=='gibbs':
        physical,p=gibbs_density(e,v,value);window_width=None;beta_matched=float(value)
    else:
        rank=int(value);left=max(0,min(d-rank,d//2-rank//2));p=np.zeros(d);p[left:left+rank]=1/rank
        physical=(v*p)@v.T;window_width=float(e[left+rank-1]-e[left])
        beta_matched=matched_beta(e,float(e@p))
    purity=float(p@p);alphas=[.5,1.,2.]
    physical_P,_=fixed_basis_spectrum(physical)
    physical_W=np.array([walshness_from_spectrum(physical_P,a) for a in alphas])
    physical_s,physical_osee=schmidt(physical)
    canonical,_=gibbs_density(e,v,beta_matched)
    canonical_p,_=fixed_basis_spectrum(canonical)
    # Sector null retains every block's eigenvalues and weight. Sampling in
    # each block is equivalent in either the physical or energy eigenbasis.
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed);samples_W=[];samples_P=[];samples_s=[];samples_osee=[];purity_errors=[]
    for draw in range(draws):
        if null=='sector_respecting_Haar':
            vectors=np.zeros((d,d),complex)
            for charge in np.unique(sector):
                cols=np.flatnonzero(sector==charge)
                vectors[:,cols]=v[:,cols]@haar_unitary(len(cols),rng)
        else: vectors=haar_unitary(d,rng,real=null=='real_Haar_orthogonal')
        randomized=(vectors*p)@vectors.conj().T
        spectrum,_=fixed_basis_spectrum(randomized)
        singular,osee=schmidt(randomized)
        samples_W.append([walshness_from_spectrum(spectrum,a) for a in alphas])
        samples_P.append(spectrum);samples_s.append(singular);samples_osee.append(osee)
        purity_errors.append(abs(float(np.vdot(randomized,randomized).real)-purity))
    samples_W=np.asarray(samples_W);samples_G=np.exp(samples_W*np.array(alphas))
    rows=[]
    for ai,alpha in enumerate(alphas):
        vals=samples_W[:,ai];se=float(vals.std(ddof=1)/np.sqrt(draws));margin=float(student_t.ppf(.975,draws-1)*se)
        gvals=samples_G[:,ai];gse=float(gvals.std(ddof=1)/np.sqrt(draws))
        exact_G=haar_mean_moment(n,purity,alpha) if null=='complex_Haar_global' else None
        row={'model':'xxz','delta':1.,'n':n,'boundary':'open','ensemble':ensemble,'ensemble_value':value,
             'beta':value if ensemble=='gibbs' else None,'rank':int(value) if ensemble=='microcanonical' else None,
             'energy':float(e@p),'energy_window_width':window_width,'beta_energy_matched':beta_matched,
             'canonical_energy_matched_W':walshness_from_spectrum(canonical_p,alpha),
             'alpha':alpha,'null':null,'draws':draws,'seed':seed,'purity':purity,
             'W_physical':float(physical_W[ai]),'W_null_mean':float(vals.mean()),'W_null_sem':se,
             'W_null_ci_low':float(vals.mean()-margin),'W_null_ci_high':float(vals.mean()+margin),
             'G_null_mean':float(gvals.mean()),'G_null_sem':gse,'G_complex_Haar_exact':exact_G,
             'log_mean_G_over_alpha':float(np.log(gvals.mean())/alpha),
             'mean_log_G_over_alpha':float(vals.mean()),
             'haar_mean_zscore':float((gvals.mean()-exact_G)/gse) if exact_G is not None and gse>1e-12*abs(gvals.mean()) else None,
             'haar_G_relative_error':float(abs(gvals.mean()-exact_G)/exact_G) if exact_G is not None else None,
             'osee_physical_bits':physical_osee,'osee_null_mean_bits':float(np.mean(samples_osee)),
             'max_purity_error':max(purity_errors),'W_best_null_status':'deferred_for_runtime_budget'}
        if ensemble=='gibbs':
            old=ROOT/'results/round1/basis'/f'xxz_n{n}_open_delta1_b{value:g}_a{alpha:g}_s0.npz'
            if old.exists():
                oldz=np.load(old);row['W_best_physical']=float(oldz['W_upper']);row['W_best_source']=str(old.relative_to(ROOT))
        rows.append(row)
    save_result(out,'R3',key,rows,physical_P=physical_P,physical_schmidt=physical_s,
                null_P=np.array(samples_P),null_W=samples_W,null_G=samples_G,
                null_schmidt_values=np.array(samples_s),null_osee=np.array(samples_osee),
                eigenvalues=e,ensemble_probabilities=p,symmetry_sector=sector)
    return f'R3 {key}: {draws} draws, {time.perf_counter()-start_time:.1f}s'


def r4_task(task):
    out,n,j2,boundary,betas=task
    key=f'j1j2_n{n}_{boundary}_j2{j2:g}'
    if (Path(out)/'R4'/f'{key}.json').exists(): return f'R4 skip {key}'
    started=time.perf_counter();h=j1j2_hamiltonian(n,j2,boundary)
    e,v=eigh(h,check_finite=False);hcenter=h-np.trace(h)/len(h)*np.eye(len(h))
    seed=20261026+n+int(j2*10000);rng=np.random.default_rng(seed)
    cap=48 if n==10 else 96
    opts={'seed':seed,'starts':4,'max_evals':cap}
    hfit=optimize_real_basis(hcenter,'walsh',**opts)
    etafit=optimize_real_basis(h,'eta',**opts)
    saved_fits={'H_centered':hfit,'eta':etafit};previous=None
    random_bases=[(f'random_{i}',rng.uniform(-np.pi,np.pi,n),rng.integers(0,2,n)) for i in range(8)]
    rows=[];arrays={};alphas=[.25,.5,1.,2.]
    for beta in betas:
        rho,p=gibbs_density(e,v,beta);centered=rho-np.eye(len(h))/len(h)
        rhofit=optimize_real_basis(rho,'walsh',**opts)
        deltafit=optimize_real_basis(centered,'walsh',**opts)
        signfit=optimize_real_basis(h,'sign',beta=beta,eigenvalues=e,**opts)
        saved_fits[f'rho_b{beta:g}']=rhofit;saved_fits[f'delta_b{beta:g}']=deltafit;saved_fits[f'sign_b{beta:g}']=signfit
        candidates=[('original_computational',np.zeros(n),np.zeros(n,dtype=int)),
                    ('Marshall',np.zeros(n),np.arange(n)%2),*random_bases]
        for name,fit in [('Walsh_rho',rhofit),('Walsh_delta',deltafit),('Walsh_H',hfit),
                         ('direct_sign',signfit),('eta_surrogate',etafit)]:
            candidates.append((name,fit['theta'],fit['flips']))
        if previous is not None: candidates.append(('transferred_high_temperature',*previous))
        if n>=8:
            source=Path(out)/'R4'/f'j1j2_n{n-2}_{boundary}_j2{j2:g}.npz'
            if source.exists():
                with np.load(source) as old:
                    oldtag=f'b{beta:g}_Walsh_rho'
                    if oldtag+'_theta' in old:
                        theta=np.pad(old[oldtag+'_theta'],(1,1),mode='edge')
                        flips=np.pad(old[oldtag+'_flips'],(1,1),mode='edge')
                        candidates.append(('transferred_smaller_size',theta,flips))
        previous=(rhofit['theta'].copy(),rhofit['flips'].copy())
        for name,theta,flips in candidates:
            hu=real_rotate(h,theta,flips);ru=real_rotate(rho,theta,flips)
            cu=ru-np.eye(len(h))/len(h)
            ps=[fixed_basis_spectrum(o)[0] for o in (ru,cu,hu-np.trace(hu)/len(h)*np.eye(len(h)))]
            sign=sign_metrics(hu,beta,e);eta=eta_positive(hu)
            # Independent full eigenspectrum checks on the analytic controls.
            if name in ('original_computational','Marshall'):
                sign['logZ_rotation_error']=abs(log_partition(eigh(hu,eigvals_only=True),beta)-sign['logZ'])
            gauge=rng.integers(0,2,n)
            gu=real_rotate(hu,np.zeros(n),gauge);gr=real_rotate(ru,np.zeros(n),gauge)
            gaugesign=sign_metrics(gu,beta,e)
            gaugeerror=abs(gaugesign['log_average_sign']-sign['log_average_sign'])
            if gaugeerror>1e-9: raise AssertionError('gauge sign invariance')
            if j2==0 and name=='Marshall' and abs(sign['log_average_sign'])>1e-9: raise AssertionError('Marshall sign')
            pg=fixed_basis_spectrum(gr)[0];tag=f'b{beta:g}_{name}'
            arrays.update({tag+'_P_rho':ps[0],tag+'_P_delta':ps[1],tag+'_P_H':ps[2],
                           tag+'_theta':theta,tag+'_flips':flips,tag+'_gauge_flips':gauge})
            for alpha in alphas:
                row={'model':'j1j2','n':n,'j2':j2,'boundary':boundary,'beta':beta,'alpha':alpha,
                     'basis':name,'seed':seed,'purity':float(p@p),'eta_pos':eta,**sign,
                     'W_rho':walshness_from_spectrum(ps[0],alpha),
                     'W_delta':walshness_from_spectrum(ps[1],alpha),
                     'W_H':walshness_from_spectrum(ps[2],alpha),
                     'gauge_logsign_error':gaugeerror,
                     'gauge_W_rho':walshness_from_spectrum(pg,alpha),
                     'search_starts':4,'search_eval_cap_per_start':cap,'search_alpha':1.}
                rows.append(row)
        print(f'R4 progress {key} beta={beta:g} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    # Definition-specific high-temperature check, outside the production beta grid.
    tiny,_=gibbs_density(e,v,1e-5)
    pdelta,_=fixed_basis_spectrum(tiny-np.eye(len(h))/len(h));ph,_=fixed_basis_spectrum(hcenter)
    arrays['high_temperature_W_delta_minus_W_H']=np.array([walshness_from_spectrum(pdelta,a)-walshness_from_spectrum(ph,a) for a in alphas])
    arrays['high_temperature_logsign']=np.array(sign_metrics(h,1e-5,e)['log_average_sign'])
    for name,fit in saved_fits.items():
        for field in ('restart_values','restart_theta','restart_flips','restart_nfev'):
            arrays[f'optimization_{name}_{field}']=fit[field]
    save_result(out,'R4',key,rows,**arrays)
    atomic_json(Path(out)/'R4_diagnostics'/f'{key}.json',saved_fits)
    return f'R4 {key}: {len(rows)} rows, {time.perf_counter()-started:.1f}s'


def stoquastic_control(out):
    rows=[]
    for n in (6,8,10):
        h=build_hamiltonian('mixed_field_ising',n,g=1.05,h=.5)
        e,v=eigh(h,check_finite=False)
        for beta in (.25,.5,1.,2.,4.):
            rho,_=gibbs_density(e,v,beta);p,_=fixed_basis_spectrum(rho)
            rows.append({'model':'mixed_field_ising','n':n,'g':1.05,'h':.5,'beta':beta,
                         'basis':'original_computational','alpha':1.,'W_rho':walshness_from_spectrum(p,1.),
                         **sign_metrics(h,beta,e)})
    path=Path(out)/'R4_controls/stoquastic.json'
    if not path.exists(): atomic_json(path,rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['R2','R3','R4'])
    parser.add_argument('--out',required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--hours',type=float,default=2.5);args=parser.parse_args()
    initialize_run(args.out)
    deadline=time.time()+args.hours*3600;tasks=[]
    if args.stage=='R2':
        cases=[('commuting_zz_ising',{}),('transverse_field_ising',{'g':1.}),
               ('mixed_field_ising',{'g':1.05,'h':.5}),('disconnected_dimers',{})]
        for n in (6,8,10):
            for model,params in cases: tasks.append((args.out,model,n,params,[0,.25,.5,1,2,4,8]))
        for model,params in cases[1:3]: tasks.append((args.out,model,12,params,[.5,1,2,4]))
        worker=r2_task
    elif args.stage=='R3':
        for n in (6,8):
            prepare_r3(args.out,n)
            for ensemble,values in [('gibbs',[.25,.5,1,2,4]),('microcanonical',[1,2,4,8,16,32,64])]:
                for value in values:
                    for null in ('complex_Haar_global','real_Haar_orthogonal','sector_respecting_Haar'):
                        tasks.append((args.out,n,ensemble,value,null,100))
        worker=r3_task
    else:
        for n in (6,8):
            for j2 in (0,.15,.3,.5,.75,1.): tasks.append((args.out,n,j2,'open',[.25,.5,1,2,4]))
        for j2 in (0,.5,1.): tasks.append((args.out,6,j2,'periodic',[.25,.5,1,2,4]))
        for j2 in (0,.5,1.): tasks.append((args.out,10,j2,'open',[.5,1,2,4]))
        worker=r4_task
        stoquastic_control(args.out)
    import multiprocessing as mp
    failures=[];completed=0;pending_tasks=iter(tasks)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        running={}
        for _ in range(args.workers):
            task=next(pending_tasks,None)
            if task: running[pool.submit(worker,task)]=task
        while running:
            future=next(as_completed(running));task=running.pop(future)
            try: print(future.result(),flush=True);completed+=1
            except Exception:
                failures.append({'task':task,'traceback':traceback.format_exc()});print(failures[-1],flush=True)
            if time.time()<deadline:
                task=next(pending_tasks,None)
                if task: running[pool.submit(worker,task)]=task
    remaining=list(pending_tasks)
    atomic_json(Path(args.out)/f'{args.stage}_status.json',{'stage':args.stage,'completed_tasks':completed,
                'planned_tasks':len(tasks),'failed':failures,'not_started':remaining,
                'status':'complete' if not failures and not remaining else 'partial'})
    if failures: sys.exit(1)


if __name__=='__main__': main()
