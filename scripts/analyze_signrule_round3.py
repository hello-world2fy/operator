"""Refresh descriptive tables/figures; never promote an unfinished scan to discovery."""
from __future__ import annotations
from collections import Counter
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc


def main():
    out=ROOT/'results/signrule_round3';tables=out/'tables';figures=out/'figures'
    tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    rows=[];points=[];phase_rows=[]
    for path in out.glob('*/*/n*/*/*/alpha*/seed*/metrics.json'):
        m=json.loads(path.read_text());point=path.parent;meta=json.loads((point/'metadata.json').read_text())
        base={k:m.get(k) for k in ['track','model','n','parameter','t_over_J','beta','alpha','seed','purity','R_eff','beta_gap']}
        base['path']=str(point.relative_to(out));base['geometry']=str(point.relative_to(out).parts[1])
        base['object']='pure_projector' if 'vacuum_sector/pure' in str(point) else 'Gibbs'
        base['metric']='Z3_support_degree' if m['track']=='E' else 'qubit_W'
        base['W_fixed']=m.get('W_fixed',m.get('W_Z3_fixed'))
        points.append(base)
        if m['track']=='E':representations={'local_charge_preserving':m['W_Z3_optimized']}
        elif isinstance(m.get('representations'),list):representations={r['representation']:r['W'] for r in m['representations']}
        else:representations=m.get('representations',{})
        candidates=json.loads((point/'basis_or_circuit.json').read_text())
        for label,value in representations.items():
            result=candidates[label];values=np.asarray(result.get('W_restarts',[value]))
            rows.append({**base,'representation':label,'W':value,'delta_W':base['W_fixed']-value,
                'W_per_2n':value/(2*m['n']),'restart_count':len(values),'restart_std':float(values.std()),
                'restart_min':float(values.min()),'restart_max':float(values.max()),
                'gate_count':result.get('gate_count'),'circuit_depth':result.get('circuit_depth'),
                'seconds':result.get('seconds')})
        if (point/'posthoc.json').exists():
            scored=json.loads((point/'posthoc.json').read_text())
            for label,scores in scored.get('circuit_scores',{}).items():
                # C pure support scoring stores one identical threshold mask.
                values=scores.items() if '1e-08' in scores else [('both',scores)]
                for cutoff,score in values:
                    delta=score.get('versus_test_oracle_linear_lower_bound',score.get('versus_test_linear_lower_bound',{}))
                    phase_rows.append({**base,'representation':label,'cutoff':cutoff,
                        'test_phase_loss':score['test']['circular_loss'],
                        'improvement_over_test_best_linear':delta.get('improvement'),
                        'ci95_low':(delta.get('ci95') or [None,None])[0],
                        'ci95_high':(delta.get('ci95') or [None,None])[1],
                        'independently_validated':False})
    df=pd.DataFrame(rows);point_df=pd.DataFrame(points);phase_df=pd.DataFrame(phase_rows)
    df.to_csv(tables/'all_metrics.csv',index=False);point_df.to_csv(tables/'points.csv',index=False)
    phase_df.to_csv(tables/'phase_scores.csv',index=False)
    for track,data in df.groupby('track'):data.to_csv(tables/f'{track}_metrics.csv',index=False)
    if len(df):
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,metric in zip(axes,['qubit_W','Z3_support_degree']):
            subset=df[(df.metric==metric)&(df.object=='Gibbs')&(df.beta>0)]
            for model,data in subset.groupby('model'):
                ax.scatter(data.purity,data.delta_W,s=7,alpha=.4,label=model)
            ax.set(xlabel='Thermal purity',ylabel='W(fixed) - W(candidate)',title=metric+' (preliminary)')
            ax.axhline(0,color='gray',lw=.5)
            if len(subset):ax.legend(fontsize=6,loc='best')
        fig.savefig(figures/'walsh_gain_vs_purity.png',dpi=170);fig.savefig(figures/'walsh_gain_vs_purity.pdf');plt.close(fig)
    status={}
    for p in (out/'jobs').glob('*.json'):
        j=json.loads(p.read_text());status[p.stem]=j.get('status')
    summary={'updated_utc':utc(),'status':'in_progress_not_final','points_frozen':len(points),
        'points_by_track':dict(Counter(p['track'] for p in points)),'representation_rows':len(rows),
        'phase_score_rows':len(phase_rows),'jobs':status,'verified_physical_discoveries':0,
        'interpretation':'No completed cross-temperature/geometry non-Marshall certificate yet.',
        'scope_register':'registration/progress.json'}
    supplemental=out/'supplementary_summary.json'
    if supplemental.exists():
        extra=json.loads(supplemental.read_text());summary['supplementary_results']=extra
        validated=extra.get('F_independent',{}).get('all_temperature_size_seed_threshold_targets_pass',False)
        summary['validated_bounded_non_Marshall_candidate_count']=int(validated)
        if validated:summary['interpretation']='A boundary-CZ correction at lambda0.4 passed independent beta/size tests; coupling range and application benefit remain limited. No established topological/SPT/path-rule recovery is claimed.'
    atomic_json(out/'summary.json',summary)
    progress=json.loads((out/'registration/progress.json').read_text())
    status_table=pd.DataFrame([{'job':k,'status':v} for k,v in status.items()]).to_html(index=False,escape=True)
    count_table=pd.DataFrame(sorted(summary['points_by_track'].items()),columns=['track','frozen points']).to_html(index=False)
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Operator Walshness Round 3 — progress</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:35px auto;line-height:1.6}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:5px 10px}}img{{max-width:100%}}.notice{{padding:18px;background:#fff5d6}}pre{{white-space:pre-wrap}}</style>
<h1>第三轮非 Marshall 符号探索：运行中报告</h1><p>更新时间：{html.escape(summary['updated_utc'])}</p>
<div class="notice">本报告仍是运行进度。受控受挫链 λ=0.4 的边界 CZ 修正在独立温度与尺寸上优于最佳一次规则，但没有通过全部相邻耦合检验；不能扩大为整个参数区间的结论。已知 Cluster 控制仅用于工具校验。</div>
<p>优化输入只有算符、几何和预算。各候选先写入冻结文件，之后才评分。矩阵元留出用于后验相位拟合；Walsh 优化读取完整算符，所以它不能代替独立模型/温度验证。</p>
<p>Qubit Walshness 与 qutrit Z3 指标分别列出。纯态 TC/DS 仅是 pure-state carryover，不计入 operator-specific 结果。对角电路每层级使用相同非零门数和函数评估上限。</p>
{count_table}<img src="figures/walsh_gain_vs_purity.png" alt="Preliminary Walsh gain versus purity">
<p><a href="tables/all_metrics.csv">全部标量</a> · <a href="tables/phase_scores.csv">后验相位评分</a> · <a href="registration/progress.json">完整范围与未完成项</a> · <a href="registration/full_pytest.xml">测试记录</a></p>
<h2>独立验证、物理对照和学习应用</h2>
<p>Levin–Gu 的冻结电路未通过全部新温度与尺寸检验。受控受挫链的有限范围候选已进入条件性 MLP 对照；已学习相位仍有不可消除的误差下界，oracle 则使用完整矩阵元相位查表。普通局域 SU2 基底 MLP 是必须比较的基线。</p>
<p><a href="supplementary_summary.json">补充汇总</a> · <a href="tables/F_independent_transfer.csv">受控受挫链迁移</a> · <a href="tables/B_projector_transfer.csv">三角格纯态迁移</a> · <a href="tables/C_thermal_sector_pairs.csv">TC/DS 热态扇区</a> · <a href="tables/E_actual_closed_paths.csv">t-J 真正闭合路径</a> · <a href="tables/same_spectrum_nulls.csv">固定谱随机对照</a> · <a href="tables/learning_seed_summary.csv">学习结果与种子统计</a></p>
<img src="figures/learning_error_vs_parameters.png" alt="学习误差与有效参数量">
<h2>任务状态</h2>{status_table}<h2>尚需完成</h2><pre>{html.escape(json.dumps(progress,ensure_ascii=False,indent=2))}</pre></html>'''
    (out/'report.html').write_text(body)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
