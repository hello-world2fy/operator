"""Registered, resumable blind round-3 workers; scoring is a separate command."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_walshness.signrule.workflow import register,initial_jobs,run_worker,run_qubit_job
from operator_walshness.round2 import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['register','worker','launch','smoke'])
    parser.add_argument('--out',type=Path,default=ROOT/'results/signrule_round3')
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();register(args.out)
    manifest=args.out/'registration/jobs.json'
    if args.command=='register':
        if manifest.exists():raise FileExistsError('registered job list already exists')
        atomic_json(manifest,initial_jobs())
    elif args.command=='worker':run_worker(args.out,args.worker,args.workers,args.device)
    elif args.command=='smoke':
        job=initial_jobs()[1]
        job.update({'n':4,'betas':[0.,.5],'alphas':[1.],'seeds':[7],
                    'local_steps':8,'local_polish':3,'local_starts':4,'circuit_starts':4,'circuit_flips':20})
        run_qubit_job(job,args.out,args.device)
    else:
        if not manifest.exists():raise FileNotFoundError('register first')
        launchfile=args.out/'registration/workers.json'
        if launchfile.exists():
            prior=json.loads(launchfile.read_text())
            live=[]
            for item in prior:
                try:
                    command=Path(f'/proc/{item["pid"]}/cmdline').read_bytes()
                    if b'run_signrule_round3.py' in command:live.append(item['pid'])
                except FileNotFoundError:pass
            if live:raise RuntimeError(f'workers already live: {live}; do not duplicate')
        (args.out/'logs').mkdir(exist_ok=True)
        processes=[]
        for worker in range(args.workers):
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--out',str(args.out),
                     '--worker',str(worker),'--workers',str(args.workers),'--device',f'cuda:{worker}']
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (args.out/'logs'/f'worker{worker}.log').open('a') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append({'worker':worker,'pid':process.pid,'command':command})
        atomic_json(launchfile,processes)
        print(json.dumps(processes,indent=2))


if __name__=='__main__':main()
