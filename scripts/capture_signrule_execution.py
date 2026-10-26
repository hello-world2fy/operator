"""One small source snapshot and digest after implementation corrections; no data hashing."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc


def main():
    out=ROOT/'results/signrule_round3';dest=out/'registration/source_v4_acceptance'
    paths=sorted(set(list((ROOT/'operator_walshness/signrule').glob('*.py'))+
                     list((ROOT/'scripts').glob('*signrule*.py'))+list((ROOT/'tests').glob('test_signrule*.py'))))
    digest=hashlib.sha256()
    for p in paths:
        relative=p.relative_to(ROOT);data=p.read_bytes();digest.update(str(relative).encode()+b'\0'+data)
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    atomic_json(out/'registration/execution_revisions.json',{'captured_utc':utc(),
        'base_commit_for_current_uncommitted_corrections':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_snapshot':str(dest.relative_to(out)),'source_bundle_sha256':digest.hexdigest(),'source_files':len(paths),
        'hash_scope':'one source-bundle digest only; no raw-data sweep',
        'cuda_runtime':torch.version.cuda,'gpu_driver_and_models':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).splitlines(),
        'stages':[
            {'stage':'initial blind scan','source':'registration/source_v1','provenance':'registration/provenance.json'},
            {'stage':'exact sparse contraction acceleration','source':'registration/source_v3_small_offdiagonal','validation':'registration/small_offdiagonal_backend_validation.json'},
            {'stage':'derived thermal eigenvalue ordering','validation':'registration/spectrum_repairs.jsonl','physical_operator_changed':False},
            {'stage':'t-J flux solver','validation':'registration/flux_solver_validation.json'},
            {'stage':'pure-projector alpha0 starts','new_results':'physical_B/projector_alpha0_recheck','originals_preserved':True},
            {'stage':'float64 qutrit kernel constants','validation':'registration/qutrit_precision_correction.json','old_results':'precision_archive','repeat_full_budget':True},
            {'stage':'energy-selected low temperatures','manifest':'registration/adaptive_jobs.json','selection':'registration/adaptive_temperature_decisions.json'}]})
    print(json.dumps({'source_files':len(paths),'source_bundle_sha256':digest.hexdigest()}))


if __name__=='__main__':main()
