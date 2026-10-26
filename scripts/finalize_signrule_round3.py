"""Write the reviewed scientific conclusions after numerical acceptance succeeds."""
import html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from operator_walshness.round2 import atomic_json
from operator_walshness.signrule.workflow import utc


def main():
    out=ROOT/'results/signrule_round3';audit=json.loads((out/'registration/requirements_audit.json').read_text())
    assert all(x['status']=='verified' for x in audit['checks']), 'numerical acceptance remains incomplete'
    test_count=next(x['detail']['passed_test_cases'] for x in audit['checks'] if x['id']=='numerical_tests')
    summary=json.loads((out/'summary.json').read_text());assert summary['points_frozen']==3798
    summary['supplementary_results']=json.loads((out/'supplementary_summary.json').read_text())
    single=json.loads((out/'independent_validation/frustration_transfer/summary.json').read_text())
    large=json.loads((out/'independent_validation/frustration_n12_transfer/summary.json').read_text())
    double=json.loads((out/'independent_validation/frustration_double_boundary/summary.json').read_text())
    lg=json.loads((out/'independent_validation/Levin_Gu_all_candidates/summary.json').read_text())
    learning=json.loads((out/'learning/comparison_summary.json').read_text())
    source=out/'controlled_frustration/chain_open/n6/p0.4/beta4/alpha2/seed20261027'
    m=json.loads((source/'metrics.json').read_text());score=json.loads((source/'posthoc.json').read_text())['circuit_scores']['circuit_L2']['1e-08']
    source_loss=score['test']['circular_loss'];gain=score['versus_test_oracle_linear_lower_bound']['improvement']
    effects=[]
    for e in learning['effects_at_40k']:
        if e['candidate']=='learned_CZ_phase_amplitude' and e['reference'] in ['linear_phase_amplitude','plain_local_SU2']:
            effects.append({'n':e['n'],'比较基线':e['reference'],'不同条件数':e['distinct_representation_conditions'],
                'CZ 较好且配对区间不跨相等':e['distinct_candidate_lower_error_ci_excludes_equal'],
                '基线较好且配对区间不跨相等':e['distinct_reference_lower_error_ci_excludes_equal'],
                'CZ / 基线误差比范围':f'{e["min_geometric_ratio"]:.3g}–{e["max_geometric_ratio"]:.3g}'})
    coverage=[
        ['A',621,'n=6/8 完整盲扫描；n=10/12 共72组完整算符控制','Marshall、cluster、正矩阵和随机相位控制通过；控制成功不算新物理发现'],
        ['B',180,'3×3 PBC Gibbs；54个基态/激发态投影；3×4 OBC 冻结迁移','无稳定盲 Huse–Elser 指纹；324个纯态迁移评分中无正下置信界'],
        ['C',144,'n=12 全 Gibbs 108点；n=12/18 匹配扇区纯态36点','无经验证的盲回路规则恢复；40个同模长随机相位控制'],
        ['D',279,'n=9 全 g 网格和低温补充；独立 n=12 冻结迁移','全部28种初步候选未通过独立温度共同标准；逐门对称类别另列'],
        ['E',1512,'6/8站点全部链、环和梯子 qutrit扫描；10/12站点72组传播核诊断','闭合路径恒等式通过；端点拟合与路径相位分别解释；优化值已用 float64 核复算'],
        ['F',1062,'n=6/8全参数扫描；边界CZ迁移到n=12；大尺寸J1–J2谱/图验证','λ=0.4发现有限范围边界修正；相邻耦合和计算收益限制明确']]
    findings={'bounded_boundary_CZ':{'source_n':6,'lambda':.4,'beta':4.,'alpha':2.,'source_purity':m['purity'],
        'source_test_phase_loss':source_loss,'test_best_linear_phase_loss':source_loss+gain,
        'source_phase_improvement':gain,'source_ci95':score['versus_test_oracle_linear_lower_bound']['ci95'],
        'single_boundary_all_primary_targets_pass':single['all_temperature_size_seed_threshold_targets_pass'] and large['all_temperature_size_seed_threshold_targets_pass'],
        'single_boundary_min_primary_gain':min(single['minimum_primary_improvement'],large['minimum_primary_improvement']),
        'double_boundary':double,'nearby_coupling_uniform_claim':False,'bulk_or_topological_rule_claim':False},
        'known_physical_rules':{'Huse_Elser':'not recovered by the tested blind circuits','TC_DS_loop':'not robustly recovered by the tested blind candidates',
            'Levin_Gu':'28 distinct preliminary candidates all fail at least one independent-temperature criterion',
            'tJ_phase_string':'path identity verified independently; not discovered from static endpoint W optimization'},
        'learning':learning,'QMC':'diagonal endpoint gauges leave exact worldline gamma invariant; W optimization is not a universal sign-severity reducer',
        'controlled_model_stoquastic_basis':'on-site Hadamards map H to -sum Z -0.4 sum XX -0.15 sum X +lambda sum ZZ, with nonpositive off-diagonals for every tested lambda; the original-basis boundary phase gain is not an intrinsic sign obstruction'}
    result={**summary,'updated_utc':utc(),'status':'scientific_analysis_complete_pending_corrected_snapshot_publication',
        'classification':{'basis_texture':'diagnostic only','sign_fingerprint':'bounded boundary-CZ correction at lambda0.4',
            'operator_specific':'mixed-state boundary correction survives independent sizes/temperatures; no universal computational benefit',
            'known_topological_SPT_or_path_rule_blind_recovery':False},
        'findings':findings,'conditional_scope_decisions':audit['conditional_scope_decisions'],
        'numerical_acceptance':'registration/requirements_audit.json','scientific_scope':'staged plan with conditional expansion; no claim to exhaustive native large grids or global optima',
        'precision_notice':'older snapshots predate float64 qutrit correction; use this completed run and the corrected final snapshot',
        'coverage_table':coverage}
    atomic_json(out/'final_summary.json',result);atomic_json(out/'summary.json',result)
    count_table=pd.DataFrame(coverage,columns=['轨道','主网格点数','实际覆盖','结论']).to_html(index=False,escape=True)
    effect_table=pd.DataFrame(effects).to_html(index=False,escape=True)
    limitations=''.join(f'<li><b>{html.escape(x["item"])}</b>：{html.escape(x["decision"])}。依据：{html.escape(x["rule"])}。</li>' for x in audit['conditional_scope_decisions'])
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Operator Walshness 第三轮实验报告</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 18px;line-height:1.75;color:#20252b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd2da;padding:8px;text-align:left}}img{{max-width:100%}}.finding{{background:#edf4fc;padding:18px}}.scope{{background:#fff5df;padding:16px}}code{{background:#f2f3f5;padding:2px 4px}}h2{{margin-top:34px}}</style></head><body>
<h1>Operator Walshness 第三轮：非 Marshall 符号探索</h1><p>完成分析时间：{html.escape(result['updated_utc'])}。原方案：<a href="registration/design.txt">设计稿</a> / <a href="registration/specification.yaml">配置</a>。</p>
<div class="finding"><b>结论：</b>在受控受挫开链 λ=0.4 中，盲 Walshness 优化得到边界局域 CZ 修正，留出相位误差优于最佳一次 Boolean 相位，并通过独立温度和 n≤12 尺寸检验。该修正未通过所有邻近耦合检验，也没有优于普通局域 SU2 基底 MLP 的通用计算收益。未得到稳健的 Huse–Elser、拓扑回路、SPT 或路径符号规则的盲恢复证据。</div>
<p>主网格及能谱自适应补充共 <b>3798</b> 点。另有54个三角格纯态投影盲扫描及54个初始化复核、160个随机控制、独立迁移与物理诊断，以及1728次网络训练。全部{test_count}项测试通过。<a href="registration/requirements_audit.json">逐项证据审计</a>链接配置中的全部叶子要求；完整大尺寸盲网格受原方案的条件门槛约束。</p>
<h2>覆盖与结果</h2>{count_table}
<h2>边界相位修正：有限而可复现的正结果</h2>
<p>源点 n=6、λ=0.4、β=4、α=2 的纯度为 <b>{m['purity']:.6f}</b>。24个通用局域初值、每类相位电路16个初值×2000次提案；不输入已知符号公式。三个优化种子得到左右反射等价的单个边界 CZ：作用于距开边界第二条键。以该冻结规则测试，留出加权相位误差从最佳一次规则的 <b>{source_loss+gain:.6f}</b> 降至 <b>{source_loss:.6f}</b>，改善 {gain:.6f}，条目 bootstrap 95% 区间 [{score['versus_test_oracle_linear_lower_bound']['ci95'][0]:.6f}, {score['versus_test_oracle_linear_lower_bound']['ci95'][1]:.6f}]。</p>
<p>只按离边界距离迁移，β=3/5、n=6/8/10/12、两个截断下均通过 λ=0.4 检验；最小改善 {findings['bounded_boundary_CZ']['single_boundary_min_primary_gain']:.6f}。n=8 的三个种子还得到两侧各一个 CZ 的同一规则；其独立检验最小改善 {double['min_primary_improvement']:.6f}、最小下置信界 {double['min_primary_ci95_low']:.6f}。λ=0.35/0.45 并非全部通过，因此结果限定在被检验的边界修正与参数点。</p>
<p>这些 bootstrap 区间描述固定算符留出条目的变化；有限条目存在相关性，扫描还包含多重比较，不能把它当作热力学极限或经过多重比较校正的发现 p 值。完整支持图仍可有负闭环，修正不表示彻底消除了符号阻碍。</p>
<h2>其他物理轨道的负结果及适用范围</h2>
<p><b>B：</b>使用真正的 Huse–Elser 开角三自旋模板，连续 U(1) 拟合分别比较一次、三子格和三体项。已知复变分态的三体模板控制误差约2.5×10⁻¹²；真实本征态和 Gibbs 态不被当成精确 HE oracle。单个实本征矢在简并固定 Sz 空间中的选择并不唯一，纹理或三子格轴相关不构成相位规则证书。</p>
<p><b>C：</b>2×2 honeycomb 的匹配真空支持仅有8个配置，DS相位在该小支持上可等价于一次规则；2×3的32配置支持提供非线性检验，但本轮盲候选没有恢复稳定回路规则。测试的 n=12 投影 commuting Hamiltonian 中 TC/DS 全谱和同温度元素模长一致，纯度匹配残差小于1.5×10⁻¹⁵；这一有限实现的结果不推广到任意热激发模型。回路 oracle 只用于闭合回路支持。</p>
<p><b>D：</b>保持 Z2 的局域门逐门检查；对称单点 Rx 族的 W 不变，对称对角门使用明确受限的互补真值表族。独立 strange correlator 与全局对称检查另列。28种不同初步正候选（34次出现）全部未通过 β=0.75/1.5 的共同标准，选定候选的 n=12 冻结迁移也失败。结论限定于已测试的门族和预算。</p>
<p><b>E：</b>所有模型保持一个空穴和指定 Sz 的原始扇区，使用保空穴的局域 U(1)⊕SU(2) 变换以及 qutrit Z3 频率支持度。96组闭合路径统计均无 phase-string 恒等式违例；路径是未加 QMC 权重的事件随机游走。n=10/12使用64个精确传播核列、64个随机迹探针与完整 Hamiltonian 支持图，未声称覆盖完整热态支持。自旋–空穴关联、基态通量响应和端点拟合提供独立物理对照。</p>
<h2>学习应用与计算量</h2>{effect_table}
<p>表中误差为全矩阵相对 Frobenius 误差平方，40k步，8个配对初始化；同一行比较相同宽度、深度、步数、分配参数量和估计密集核 FLOPs。正幅度模型有一个未使用的输出头，有效参数少 width+1。1728次训练中有768组不同的表示／架构／种子条件：α仅改变局域 SU2 输入基底，其余模型在不同α下重复，统计不把这些重复当作新增独立样本。</p>
<p>n=6 上六个架构的 CZ 相位幅度模型优于一次相位模型；n=8 收益不稳定。局域 SU2 普通 MLP 在两种尺寸的全部被比较条件下误差更低。完整逐条目 oracle 有额外 d² 相位存储，学习到的 CZ 只有局部门信息但仍有不可消除的相位误差下界。报告参数／FLOPs达到误差阈值的<b>已观察最小值</b>和未达标情况，不插值为精确首次到达时间，也不宣称全局最优。</p>
<img src="figures/learning_paired_error_ratios.png" alt="配对种子误差比与区间"><p><a href="tables/learning_paired_comparisons.csv">完整配对比较</a> · <a href="tables/learning_resource_summary.csv">资源与达标率</a> · <a href="learning/comparison_summary.json">统计解释</a></p>
<h2>QMC、温度与数值控制</h2>
<p>对角端点 gauge 保持 H 的元素模长，因此 exact worldline γ 不变。相同24初值、150次评估预算下另比较实基的 W 优化和直接符号优化；不会用 Gibbs 矩阵元误差代替路径平均符号。较低 W 并不普适降低 γ。<a href="tables/F_matched_real_QMC.csv">同预算数据</a>。</p>
<p>受控模型还有一个直接的基底对照：逐点 Hadamard 将 H 变为 −ΣZ−0.4ΣXX−0.15ΣX+λΣZZ，其非对角元非正。该对照在全部五个λ上通过回归验证，γ=0。因此原始 Z 编码中的边界相位收益不构成内禀符号问题的证据，也与局域 SU2 MLP 明显更好的结果相容。</p>
<p>63组能谱统一检查 β=4 的基态流形权重；未达到95%的体系补至能谱确定的温度，向上取整0.01，配对 t-J 模型共用较大的所需β。新增360个点保留原优化预算。简并体系的基态流形权重与纯度分别报告；大β下的纯化不称为新相变或混合算符发现。</p>
<div class="scope"><b>复核修正已保留记录：</b>早期热谱概率排序只影响保存顺序，已修复；t-J 通量使用可靠的基态求解器；三角格纯态补入α→0初值的54次同预算复算改变W小于2×10⁻¹⁵；qutrit 核单位矩阵常数改为float64后1512个相关点均按原预算重算。旧记录在本地 precision_archive 中，早期 GitHub 快照属于开发阶段记录。完整频谱在可行尺寸独立重建并核对，所有主点保留重启分布、基底、掩码和种子。</div>
<h2>条件性范围与复现</h2><ul>{limitations}</ul>
<p>局域优化24初值×150步；普通 qubit 另有至多30步精修，C 热态20步，C 纯态200步。相位层级共用16×2000提案、至多n个门和深度n；链上无三角时无独立三角CCZ层。所有最低值为该预算找到的结果。仅在原始或明确固定的编码中评分已知相位，不向 SU2 旋转后的标签直接套原回路公式。</p>
<p><a href="summary.json">JSON总表</a> · <a href="tables/all_metrics.csv">全部W指标</a> · <a href="tables/phase_scores.csv">相位评分</a> · <a href="tables/degree_tail_summary.csv">频谱与尾部</a> · <a href="tables/thermal_rank_summary.csv">能谱、纯度和有效秩</a> · <a href="registration/final_pytest.xml">{test_count}项测试记录</a> · <a href="registration/execution_revisions.json">源代码与执行版本</a></p>
<p>复现入口：先运行 python -m pytest，再按原 YAML 与 registration 中清单运行 scripts/run_signrule_round3.py、run_signrule_special.py、run_signrule_loop_thermal.py、run_signrule_adaptive.py；物理评分与独立验证使用同目录独立脚本。历史复算仅用于保留纠错轨迹，新运行直接使用修正后的核。训练权重、完整矩阵、可靠性掩码等大型文件保留本地；GitHub提供压缩标量、最优基底、电路和可行频谱。</p>
<img src="figures/walsh_gain_vs_purity.png" alt="各模型W改善与纯度"></body></html>'''
    (out/'report.html').write_text(report)
    atomic_json(out/'registration/report_review.json',{'reviewed_utc':utc(),'status':'scientific_report_ready_for_publication',
        'numerical_audit_checks':len(audit['checks']),'claims':findings['known_physical_rules'],
        'main_positive':'bounded one/two-boundary CZ corrections at lambda0.4; no uniform nearby-coupling or computational-advantage claim',
        'scoped_negative':'tested finite geometries, gate classes, sectors and multistart budgets only',
        'remaining':'publish corrected compact snapshot and verify remote branch'})
    print(json.dumps({'status':result['status'],'points':3798,'report':str(out/'report.html')}))


if __name__=='__main__':main()
