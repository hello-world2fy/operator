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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--test-report', type=Path)
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise SystemExit(f'Refusing to overwrite snapshot: {out}')
    source = ROOT / 'results/signrule_round3'
    out.mkdir(parents=True)
    summary = json.loads((source / 'summary.json').read_text())
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
        grouped[metrics['track']].append(record)
    for track, records in sorted(grouped.items()):
        # Keep each file well below typical GitHub large-file limits.
        for index in range(0, len(records), 250):
            write_jsonl(out / f'records/{track}_{index // 250:02d}.jsonl.gz', records[index:index + 250])
    diagnostics = []
    for p in sorted(source.rglob('*.json')):
        relative = p.relative_to(source)
        if any(part.startswith('physics_') for part in relative.parts) or relative.parts[0] in {
            'physical_E', 'independent_validation'
        } or 'projector_references' in relative.parts:
            diagnostics.append({'path': str(relative), 'data': json.loads(p.read_text())})
    write_jsonl(out / 'physical_diagnostics.jsonl.gz', diagnostics)
    for p in (source / 'tables').glob('*.csv'):
        target = out / 'tables' / (p.name + '.gz')
        target.parent.mkdir(exist_ok=True)
        with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as packed:
            packed.write(p.read_bytes())
    for name in ['specification.yaml', 'design.txt', 'jobs.json', 'references.json',
                 'flux_solver_validation.json', 'progress.json']:
        p = source / 'registration' / name
        if p.exists():
            target = out / 'registration' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(p, target)
    if args.test_report:
        shutil.copyfile(args.test_report, out / 'registration/publish_tests.xml')
    (out / 'figures').mkdir()
    shutil.copyfile(source / 'figures/walsh_gain_vs_purity.png', out / 'figures/walsh_gain_vs_purity.png')
    transfer = json.loads((source / 'independent_validation/Levin_Gu_transfer/summary.json').read_text())
    write_json(out / 'Levin_Gu_transfer.json', transfer)
    summary.update({
        'snapshot_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'partial_snapshot_experiments_still_running',
        'points_frozen': len(points),
        'points_by_track': dict(Counter(p['track'] for p in points)),
        'points_with_posthoc_in_snapshot': posthoc_count,
        'physical_diagnostic_records': len(diagnostics),
        'independent_Levin_Gu_transfer_passed': False,
        'scope_register_note': 'registration/progress.json is historical; see this snapshot and README.txt for newer diagnostics.',
        'omitted': ['dense matrices', 'NPZ arrays and reliability masks', 'optimizer initializations and non-best axes/gate coefficients',
                    'raw logs', 'checkpoints', 'duplicate source copies'],
        'freeze_note': 'source_freeze_record refers to the original unabridged candidate JSON, not this compact export. No extra digest sweep was run.',
    })
    write_json(out / 'summary.json', summary)
    (out / 'README.txt').write_text(
        'Operator Walshness round 3: partial result snapshot\n\n'
        'Experiments remain in progress. This is not final acceptance or a new-physics claim.\n'
        'All frozen points listed in tables/points.csv.gz are included in records/*.jsonl.gz.\n'
        'Records preserve best axes/circuits, every restart W, budgets, seeds, metadata and available posthoc scores.\n'
        'Use gzip.open(path, "rt") and json.loads(line) to read records; pandas.read_csv reads .csv.gz directly.\n'
        'Large arrays, full initialization histories and checkpoints stay on the local machine.\n'
        'Source freeze hashes refer to original unabridged files; compact records cannot reproduce those bytes.\n'
        'Pure TC/DS, mixed Gibbs and qutrit results are labeled separately.\n'
        'Levin-Gu frozen transfer to unseen temperatures and 12 spins failed the all-target criterion.\n'
        'No verified physical discovery is claimed; the MLP discovery gate remains closed.\n'
        'Physical diagnostics include available B/D/E results, not a completed full-grid evaluation.\n'
        'The historical progress register predates the sparse C Gibbs scan, physics diagnostics and LG transfer.\n'
        'Remaining work includes complete scoring, larger geometries, selected null controls and final acceptance.\n'
        'The corrected source fixes thermal eigenvalue serialization and uses dense exact t-J flux spectra.\n'
        'Thermal NPZ files from early live workers require the documented ordering repair and are not exported.\n\n'
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
<p>实验仍在运行，本次保存 {len(points)} 个已冻结实验点；没有已验收的新物理发现。</p>
<table><tr><td>实验轨道</td><td>已冻结点数</td></tr>{counts}</table>
<p>Levin–Gu 在源格点的弱信号没有通过全部新温度与 12 自旋迁移检验，不能认定为稳定发现。
纯态 TC/DS、混合 Gibbs 算符和 qutrit 指标分别报告。Walshness 降低不等同于恢复符号规则。</p>
<img src="figures/walsh_gain_vs_purity.png" alt="阶段性 Walshness 改善与纯度">
<p><a href="summary.json">汇总</a> · <a href="README.txt">内容与复现说明</a> ·
<a href="tables/all_metrics.csv.gz">全部指标</a> · <a href="tables/phase_scores.csv.gz">后验评分</a> ·
<a href="Levin_Gu_transfer.json">独立迁移检验</a> · <a href="registration/publish_tests.xml">测试记录</a></p>
<p>快照包含最优轴/电路、各初值目标值、预算、种子与已有物理诊断。大型矩阵、可靠性掩码、优化轨迹和训练检查点未上传。</p></html>''')
    inventory = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size}
                 for p in sorted(out.rglob('*')) if p.is_file()]
    assert all(row['bytes'] < 2 * 1024**2 for row in inventory)
    write_json(out / 'file_sizes.json', {'files': inventory, 'total_bytes': sum(x['bytes'] for x in inventory)})
    print(json.dumps({'out': str(out), 'points': len(points), 'files': len(inventory) + 1,
                      'total_bytes': sum(x['bytes'] for x in inventory),
                      'largest_bytes': max(x['bytes'] for x in inventory)}, indent=2))


if __name__ == '__main__':
    main()
