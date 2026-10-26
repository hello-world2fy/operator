"""Register energy-selected low-temperature points, then run unchanged blind budgets."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc,point_directory,run_qubit_job,build_qubit_job
from operator_walshness.signrule.temperature import select_low_temperature,thermal_rank
from operator_walshness.signrule.geometry import square,chain


def register(out):
    path=out/'registration/adaptive_jobs.json'
    if path.exists():raise FileExistsError('adaptive manifest is already registered')
    jobs=[];decisions=[]
    for job in json.loads((out/'registration/jobs.json').read_text()):
        source=point_directory(out,job,4,1,job['seeds'][0])/'thermal_spectrum.npz'
        with np.load(source) as a:selection=select_low_temperature(a['energies'])
        decisions.append({'source':str(source.relative_to(out)),'track':job['track'],**selection})
        if selection['add_temperature']:
            jobs.append({**job,'kind':'qubit','job_id':'adaptive_'+job['job_id'],
                         'betas':[selection['selected']['beta']],'temperature_selection':selection})
    # C already reaches the target at beta4; record the same decision rule.
    for model in ['toric_code','double_semion']:
        source=out/model/'honeycomb_2x2/n12/full_Gibbs/beta4/alpha1/seed20261027/thermal_spectrum.npz'
        with np.load(source) as a:selection=select_low_temperature(a['energies'])
        decisions.append({'source':str(source.relative_to(out)),'track':'C',**selection})
        if selection['add_temperature']:raise RuntimeError('C requires an explicit additional thermal job')
    for index,g in enumerate([square(3,2),chain(8),chain(8,True),square(4,2)]):
        geometry=g.name+'_'+'x'.join(map(str,g.shape))+'_'+g.boundary
        for t in [1.,2.,3.]:
            pair={};spectra={}
            for model in ['tJ','sigma_tJ']:
                source=out/model/geometry/f'n{g.n}'/f't{t:g}'/'beta4/alpha1/seed20261027/thermal_spectrum.npz'
                with np.load(source) as a:spectra[model]=a['energies'].copy()
                pair[model]=select_low_temperature(spectra[model])
                decisions.append({'source':str(source.relative_to(out)),'track':'E',**pair[model]})
            beta=max(s['selected']['beta'] for s in pair.values())
            if beta>4:
                jobs.append({'kind':'qutrit_pair','track':'E','job_id':f'adaptive_E_geometry{index}_t{t:g}',
                    'geometry_index':index,'t':t,'betas':[beta],
                    'paired_temperature_rule':'max energy-selected beta of the two models; compare at the same temperature',
                    'temperature_selection':pair,'paired_thermal_targets':{k:thermal_rank(e,beta) for k,e in spectra.items()},
                    'local_starts':24,'local_steps':150,'alphas':[.5,1.,2.],'seeds':[20261027,20261028,20261029]})
    atomic_json(out/'registration/adaptive_temperature_decisions.json',{'registered_utc':utc(),'decisions':decisions,
        'selection_inputs':'previous exact energy spectra only; no phase labels, W values or target optimization',
        'purpose':'low-temperature/effective-rank coverage; do not classify near-pure improvement as operator-specific'})
    atomic_json(path,jobs)
    print(json.dumps({'jobs':len(jobs),'new_points':sum(9 if j['kind']=='qubit' else 18 for j in jobs),'decisions':len(decisions)}))


def worker(out,index,workers,device):
    torch.set_num_threads(1)
    jobs=json.loads((out/'registration/adaptive_jobs.json').read_text())
    for i,job in enumerate(jobs):
        if i%workers!=index:continue
        status=out/'jobs'/(job['job_id']+'_posthoc.json')
        if status.exists() and json.loads(status.read_text()).get('status')=='complete':continue
        atomic_json(status,{'status':'running','pid':os.getpid(),'job':job,'started_utc':utc()})
        if job['kind']=='qubit':
            run_qubit_job(job,out,device)
            from operator_walshness.signrule.scoring import score_qubit_job,posthoc_oracle
            score_qubit_job(job,out)
            from diagnose_signrule_physics import diagnose_B,diagnose_D,diagnose_F
            if job['track'] in 'BDF':{'B':diagnose_B,'D':diagnose_D,'F':diagnose_F}[job['track']](job,out)
            from complete_signrule_baselines import write_qubit_points
            from operator_walshness.signrule.models import block_eigensystem,block_gibbs
            g,h=build_qubit_job(job);blocks=block_eigensystem(h);oracle,name=posthoc_oracle(job,g)
            for beta in job['betas']:
                base=point_directory(out,job,beta,1,job['seeds'][0]).parents[1]
                rho,_=block_gibbs(blocks,beta,1<<g.n)
                if job['track']=='B':
                    with np.load(base/'physics_B/phase_fits.npz') as a:oracle=a['1e-08_linear_plus_HE_three_spin_phase']
                    name='posthoc fitted Huse-Elser continuous phase; not an exact oracle'
                write_qubit_points(base,rho,device,oracle,name)
        else:
            from run_signrule_special import scan_tj
            from complete_signrule_baselines import qutrits
            g=[square(3,2),chain(8),chain(8,True),square(4,2)][job['geometry_index']]
            scan_tj(out,device,[g],betas=job['betas'],t_values=[job['t']])
            qutrits(out,device,geometries=[g],t_values=[job['t']],betas=job['betas'])
            subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_signrule_tj.py'),'--geometry',str(job['geometry_index']),
                '--t',str(job['t']),'--beta',str(job['betas'][0])],check=True)
        atomic_json(status,{'status':'complete','finished_utc':utc(),'job':job})
        print('ADAPTIVE_DONE',job['job_id'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','launch','worker'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=8);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();out=ROOT/'results/signrule_round3'
    if args.command=='register':register(out)
    elif args.command=='worker':worker(out,args.worker,args.workers,args.device)
    else:
        manifest=out/'registration/adaptive_workers.json'
        if manifest.exists():
            for prior in json.loads(manifest.read_text()):
                path=Path(f'/proc/{prior["pid"]}/cmdline')
                if path.exists() and b'run_signrule_adaptive.py' in path.read_bytes():raise RuntimeError('adaptive worker still live')
        rows=[]
        for i in range(args.workers):
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--worker',str(i),'--workers',str(args.workers),'--device',f'cuda:{i}']
            with (out/'logs'/f'adaptive_worker{i}.log').open('a') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rows.append({'worker':i,'pid':proc.pid,'command':command})
        atomic_json(manifest,rows);print(json.dumps(rows))


if __name__=='__main__':main()
