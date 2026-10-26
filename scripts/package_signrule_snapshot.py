"""Export a small, immutable progress snapshot without rehashing raw results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import html
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OMIT = {
    'initial_axes', 'axes_restarts', 'initial_coefficients',
    'coefficients_restarts', 'initial_quaternions', 'quaternions_restarts',
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as packed:
        for row in rows:
            packed.write((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in OMIT}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--test-report', type=Path)
    parser.add_argument('--final', action='store_true', help='Export the accepted scientific report and corrected results')
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise SystemExit(f'Refusing to overwrite snapshot: {out}')
    source = ROOT / 'results/signrule_round3'
    out.mkdir(parents=True)
    summary = json.loads((source / 'summary.json').read_text())
    if args.final:
        review=json.loads((source/'registration/report_review.json').read_text())
        assert review['status']=='scientific_report_ready_for_publication'
    # The table fixes the point set, even while more GPU jobs finish.
    import csv
    with (source / 'tables/points.csv').open() as f:
        points = list(csv.DictReader(f))
    grouped = defaultdict(list)
    posthoc_count = 0
    for point in points:
        directory = source / point['path']
        metrics = json.loads((directory / 'metrics.json').read_text())
        original = json.loads((directory / 'basis_or_circuit.json').read_text())
        candidates = {name: {k: v for k, v in item.items() if k not in OMIT}
                      for name, item in original.items()}
        frozen = json.loads((directory / 'blind_frozen.json').read_text())
        assert frozen['known_label_access'] is False
        for candidate in candidates.values():
            if 'W_restarts' in candidate:
                assert abs(candidate['W_best'] - min(candidate['W_restarts'])) < 1e-7
        record = {'path': point['path'], 'metrics': metrics, 'candidates': candidates,
                  'metadata': json.loads((directory / 'metadata.json').read_text()),
                  'source_freeze_record': frozen}
        if (directory / 'posthoc.json').exists():
            record['posthoc'] = json.loads((directory / 'posthoc.json').read_text())
            posthoc_count += 1
        if (directory / 'representation_baselines.json').exists():
            record['representation_baselines'] = json.loads((directory / 'representation_baselines.json').read_text())
        if (directory / 'walsh_spectrum.npz').exists():
            with np.load(directory / 'walsh_spectrum.npz') as saved:
                record['degree_spectra_and_correlations']={k:saved[k].tolist() for k in saved.files}
        grouped[metrics['track']].append(record)
    for track, records in sorted(grouped.items()):
        # Keep each file well below typical GitHub large-file limits.
        for index in range(0, len(records), 250):
            write_jsonl(out / f'records/{track}_{index // 250:02d}.jsonl.gz', records[index:index + 250])
    diagnostics = []
    for p in sorted(source.rglob('*.json')):
        relative = p.relative_to(source)
        if any(part.startswith('physics_') for part in relative.parts) or relative.parts[0] in {
            'physical_B', 'physical_C', 'physical_E', 'physical_E_large',
            'independent_validation', 'negative_controls', 'learning'
        } or 'projector_references' in relative.parts:
            diagnostics.append({'path': str(relative), 'data': compact(json.loads(p.read_text()))})
    for index in range(0, len(diagnostics), 100):
        write_jsonl(out / f'diagnostics/part_{index // 100:03d}.jsonl.gz', diagnostics[index:index + 100])
    for p in (source / 'tables').glob('*.csv'):
        target = out / 'tables' / (p.name + '.gz')
        target.parent.mkdir(exist_ok=True)
        with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as packed:
            packed.write(p.read_bytes())
    for name in ['specification.yaml', 'design.txt', 'jobs.json', 'references.json',
                 'flux_solver_validation.json', 'progress.json', 'phase3_gate.json',
                 'small_offdiagonal_backend_validation.json', 'C_thermal_backend_transition.json']:
        p = source / 'registration' / name
        if p.exists():
            target = out / 'registration' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(p, target)
    if args.final:
        for name in ['provenance.json','execution_revisions.json','requirements_audit.json','report_review.json',
                     'adaptive_jobs.json','adaptive_temperature_decisions.json','qutrit_precision_correction.json']:
            shutil.copyfile(source/'registration'/name,out/'registration'/name)
        (out/'learning').mkdir(exist_ok=True)
        shutil.copyfile(source/'learning/comparison_summary.json',out/'learning/comparison_summary.json')
    if args.test_report:
        shutil.copyfile(args.test_report, out / 'registration/publish_tests.xml')
    (out / 'figures').mkdir()
    for p in (source / 'figures').glob('*'):
        if p.is_file() and p.suffix in {'.png', '.pdf'} and p.stat().st_size < 2 * 1024**2:
            shutil.copyfile(p, out / 'figures' / p.name)
    for name in ['supplementary_summary.json', 'acceptance_progress.json']:
        if (source / name).exists():
            shutil.copyfile(source / name, out / name)
    transfer = json.loads((source / 'independent_validation/Levin_Gu_transfer/summary.json').read_text())
    write_json(out / 'Levin_Gu_transfer.json', transfer)
    summary.update({
        'snapshot_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed_first_scan_snapshot_pending_full_scope_acceptance',
        'points_frozen': len(points),
        'points_by_track': dict(Counter(p['track'] for p in points)),
        'points_with_posthoc_in_snapshot': posthoc_count,
        'physical_diagnostic_records': len(diagnostics),
        'independent_Levin_Gu_transfer_passed': False,
        'scope_register_note': 'registration/progress.json records completed coverage and outstanding scope; this export is not final research acceptance.',
        'omitted': ['dense matrices', 'NPZ arrays and reliability masks', 'optimizer initializations and non-best axes/gate coefficients',
                    'raw logs', 'checkpoints', 'duplicate source copies'],
        'freeze_note': 'source_freeze_record refers to the original unabridged candidate JSON, not this compact export. No extra digest sweep was run.',
    })
    write_json(out / 'summary.json', summary)
    (out / 'README.txt').write_text(
        'Operator Walshness round 3: partial result snapshot\n\n'
        'The first blind scan has finished; full research-scope acceptance remains open.\n'
        'All frozen points listed in tables/points.csv.gz are included in records/*.jsonl.gz.\n'
        'Records preserve best axes/circuits, every restart W, budgets, seeds, metadata and available posthoc scores.\n'
        'Use gzip.open(path, "rt") and json.loads(line) to read records; pandas.read_csv reads .csv.gz directly.\n'
        'Large arrays, full initialization histories and checkpoints stay on the local machine.\n'
        'Source freeze hashes refer to original unabridged files; compact records cannot reproduce those bytes.\n'
        'Pure TC/DS, mixed Gibbs and qutrit results are labeled separately.\n'
        'Levin-Gu frozen transfer to unseen temperatures and 12 spins failed the all-target criterion.\n'
        'A bounded lambda=0.4 boundary-CZ correction passed independent temperature/size tests through n=12.\n'
        'It did not pass all nearby-coupling tests; no topological, SPT or phase-string recovery is claimed.\n'
        'The conditional MLP comparison completed 1728 networks; read supplementary_summary.json and learning tables.\n'
        'Diagnostic JSON is in diagnostics/part_*.jsonl.gz, including null controls and learning scalar results.\n'
        'See registration/progress.json for completed coverage and outstanding scope.\n'
        'The corrected source fixes thermal eigenvalue serialization and uses dense exact t-J flux spectra.\n'
        'Derived spectrum ordering was checked/repaired after scan completion; NPZ arrays are not exported.\n\n'
        'Reproduce from repository root (use the YAML budgets and seeds):\n'
        '  python scripts/run_signrule_round3.py --help\n'
        '  python scripts/run_signrule_special.py --help\n'
        '  python scripts/run_signrule_loop_thermal.py --help\n'
        '  python scripts/analyze_signrule_round3.py\n'
        '  python scripts/package_signrule_snapshot.py --out results/NEW_SNAPSHOT\n'
    )
    counts = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k, v in sorted(summary['points_by_track'].items()))
    (out / 'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Operator Walshness 第三轮实验快照</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:32px auto;line-height:1.7}}img{{max-width:100%}}td{{padding:4px 20px;border:1px solid #ddd}}</style>
<h1>第三轮实验：阶段性快照</h1><p>{html.escape(summary['snapshot_utc'])}</p>
<p>首批盲扫描已结束，本次保存 {len(points)} 个已冻结实验点；完整方案的最终验收尚未结束。</p>
<table><tr><td>实验轨道</td><td>已冻结点数</td></tr>{counts}</table>
<p>Levin–Gu 在源格点的弱信号没有通过全部新温度与 12 自旋迁移检验，不能认定为稳定发现。
纯态 TC/DS、混合 Gibbs 算符和 qutrit 指标分别报告。Walshness 降低不等同于恢复符号规则。</p>
<p>受控受挫开链 λ=0.4 的边界 CZ 修正在独立温度与尺寸（到 n=12）上优于最佳一次相位；邻近耦合并非全部通过，不能推广为整个参数区间的结论。条件性学习对照已完成 1728 个网络。</p>
<img src="figures/walsh_gain_vs_purity.png" alt="阶段性 Walshness 改善与纯度">
<p><a href="summary.json">汇总</a> · <a href="README.txt">内容与复现说明</a> ·
<a href="tables/all_metrics.csv.gz">全部指标</a> · <a href="tables/phase_scores.csv.gz">后验评分</a> ·
<a href="Levin_Gu_transfer.json">独立迁移检验</a> · <a href="registration/publish_tests.xml">测试记录</a></p>
<p><a href="supplementary_summary.json">物理与学习补充汇总</a> · <a href="tables/learning_seed_summary.csv.gz">学习对照</a> · <a href="registration/progress.json">范围与未完成项</a></p>
<p>快照包含最优轴/电路、各初值目标值、预算、种子与已有物理诊断。大型矩阵、可靠性掩码、优化轨迹和训练检查点未上传。</p></html>''')
    if args.final:
        summary['status']='completed_staged_experiment_corrected_snapshot'
        summary['scope_register_note']='See registration/requirements_audit.json and report.html for verified evidence and conditional expansion decisions.'
        summary['qutrit_precision_notice']='Supersedes earlier snapshots for optimized qutrit W and baselines; all 1512 points repeated with float64 kernel constants.'
        write_json(out/'summary.json',summary)
        report=(source/'report.html').read_text().replace('registration/final_pytest.xml','registration/publish_tests.xml')
        for p in (source/'tables').glob('*.csv'):report=report.replace('tables/'+p.name,'tables/'+p.name+'.gz')
        (out/'report.html').write_text(report)
        (out/'README.txt').write_text(
            'Operator Walshness round 3: completed staged experiment, corrected compact results\n\n'
            'Read report.html for scientific conclusions, actual coverage and conditional size/gate limits.\n'
            '3798 primary points; 54 triangular pure scans plus 54 alpha0-start checks; 160 random controls.\n'
            '1728 learning training instances represent 768 distinct representation/architecture/seed conditions.\n'
            'All 1512 qutrit optimizations were repeated with float64 kernel constants; older snapshots are superseded for these W values.\n'
            'Records retain best axes/circuits, all restart W values, metadata, budgets, scalar baselines, feasible degree spectra and posthoc scores.\n'
            'Use gzip.open(path, "rt") plus json.loads(line) for records/*.jsonl.gz and diagnostics/*.jsonl.gz.\n'
            'pandas.read_csv reads tables/*.csv.gz directly.\n'
            'Large density arrays, full reliability masks, initialization histories, checkpoints and archived legacy results remain local.\n'
            'Freeze hashes refer to original unabridged candidate files; no full raw-result digest sweep was run.\n'
            'The boundary-CZ correction at lambda0.4 has bounded phase benefits; there is no claimed topological/SPT/path-rule recovery or universal learning advantage.\n'
            'Reproduce with the YAML budgets and scripts; begin with python -m pytest and inspect registration/requirements_audit.json.\n')
    inventory = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size}
                 for p in sorted(out.rglob('*')) if p.is_file()]
    assert all(row['bytes'] < 2 * 1024**2 for row in inventory)
    write_json(out / 'file_sizes.json', {'files': inventory, 'total_bytes': sum(x['bytes'] for x in inventory)})
    print(json.dumps({'out': str(out), 'points': len(points), 'files': len(inventory) + 1,
                      'total_bytes': sum(x['bytes'] for x in inventory),
                      'largest_bytes': max(x['bytes'] for x in inventory)}, indent=2))


if __name__ == '__main__':
    main()
