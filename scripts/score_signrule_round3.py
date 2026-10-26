from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_walshness.signrule.scoring import score_qubit_job


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'results/signrule_round3')
    p.add_argument('--job',type=int)
    args=p.parse_args()
    jobs=json.loads((args.out/'registration/jobs.json').read_text())
    for i,job in enumerate(jobs):
        if args.job is not None and i!=args.job:continue
        print(job['job_id'],'scored',score_qubit_job(job,args.out),flush=True)


if __name__=='__main__':main()
