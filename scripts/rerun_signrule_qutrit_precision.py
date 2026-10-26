"""Archive legacy mixed-precision kernel results and rerun the original W-only budget."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc
from operator_walshness.signrule.geometry import square,chain
from run_signrule_special import scan_tj
from complete_signrule_baselines import qutrits


def register(out):
    path=out/'registration/qutrit_precision_jobs.json'
    if path.exists():raise FileExistsError(path)
    for item in json.loads((out/'registration/adaptive_workers.json').read_text()):
        p=Path('/proc')/str(item['pid'])/'cmdline'
        if p.exists() and b'run_signrule_adaptive.py' in p.read_bytes():raise RuntimeError('wait for the adaptive writer to finish')
    jobs=[]
    for index,g in enumerate([square(3,2),chain(8),chain(8,True),square(4,2)]):
        geometry=g.name+'_'+'x'.join(map(str,g.shape))+'_'+g.boundary
        for t in [1.,2.,3.]:
            for base in sorted((out/'tJ'/geometry/f'n{g.n}'/f't{t:g}').glob('beta*')):
                beta=float(base.name[4:]);points=[]
                for model in ['tJ','sigma_tJ']:
                    found=list((out/model/geometry/f'n{g.n}'/f't{t:g}'/f'beta{beta:g}').glob('alpha*/seed*/metrics.json'))
                    assert len(found)==9
                    points += [str(p.parent.relative_to(out)) for p in found]
                jobs.append({'geometry_index':index,'geometry':geometry,'t':t,'beta':beta,'points':points})
    atomic_json(path,jobs)
    atomic_json(out/'registration/qutrit_precision_correction.json',{'registered_utc':utc(),'status':'rerunning',
        'issue':'np.exp(alpha) times torch.eye(3) used a float32 identity before addition to a complex128 kernel',
        'observed_independent_FWHT_residual':5.552289739085836e-7,
        'affected':'optimized Z3 W values, gradients and baseline kernel evaluations; Hamiltonian, Gibbs spectrum and fixed-basis NumPy moments unaffected',
        'repair':'explicit float64 identity; repeat the same 24 starts and 150 steps for every affected point, then recompute baselines',
        'old_records':'precision_archive/<original point path>; original freeze files retained byte-for-byte',
        'points':sum(len(j['points']) for j in jobs),'jobs':len(jobs),'known_labels_used':False,
        'phase_path_diagnostics':'same Hamiltonians/operators and encoding; existing independent physical diagnostics remain valid',
        'published_old_snapshot_commit':'2459177e2a5fc0c89c04aa4d87c44ea9162230b6'})
    print(json.dumps({'jobs':len(jobs),'points':sum(len(j['points']) for j in jobs)}))


def worker(out,index,workers,device):
    torch.set_num_threads(1);jobs=json.loads((out/'registration/qutrit_precision_jobs.json').read_text())
    for i,job in enumerate(jobs):
        if i%workers!=index:continue
        status=out/'jobs'/f'qutrit_precision_{i:03d}.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='complete':continue
        atomic_json(status,{'status':'running','pid':os.getpid(),'job':job,'started_utc':utc()})
        for relative in job['points']:
            original=out/relative;archive=out/'precision_archive'/relative
            if not archive.exists():
                archive.parent.mkdir(parents=True,exist_ok=True);original.rename(archive)
        g=[square(3,2),chain(8),chain(8,True),square(4,2)][job['geometry_index']]
        scan_tj(out,device,[g],betas=[job['beta']],t_values=[job['t']])
        qutrits(out,device,geometries=[g],t_values=[job['t']],betas=[job['beta']])
        shifts=[]
        for relative in job['points']:
            current=json.loads((out/relative/'basis_or_circuit.json').read_text())['local_charge_preserving']
            assert current['numerical_convention']=='float64_Z3_kernel_constants'
            old=json.loads((out/'precision_archive'/relative/'basis_or_circuit.json').read_text())['local_charge_preserving']
            shifts.append(current['W_best']-old['W_best'])
        atomic_json(status,{'status':'complete','finished_utc':utc(),'job':job,'optimized_W_changes':shifts,
            'comparison_caveat':'changes combine corrected arithmetic and a full optimizer rerun, not just a score correction'})
        print('QUTRIT_PRECISION_DONE',i,job['geometry'],job['t'],job['beta'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','worker','launch'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=8);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();out=ROOT/'results/signrule_round3'
    if args.command=='register':register(out)
    elif args.command=='worker':worker(out,args.worker,args.workers,args.device)
    else:
        manifest=out/'registration/qutrit_precision_workers.json'
        if manifest.exists():
            for old in json.loads(manifest.read_text()):
                p=Path('/proc')/str(old['pid'])/'cmdline'
                if p.exists() and b'rerun_signrule_qutrit_precision.py' in p.read_bytes():raise RuntimeError('precision worker still live')
        processes=[]
        for i in range(args.workers):
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--worker',str(i),'--workers',str(args.workers),'--device',f'cuda:{i}']
            with (out/'logs'/f'qutrit_precision{i}.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append({'worker':i,'pid':process.pid,'command':cmd})
        atomic_json(manifest,processes);print(json.dumps(processes))


if __name__=='__main__':main()
