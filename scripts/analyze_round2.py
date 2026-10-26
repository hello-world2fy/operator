"""Build a local, self-contained scientific report from completed first-wave artifacts."""
from __future__ import annotations
import argparse
import html
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator_walshness.round2 import atomic_json

BASES=['fixed_X','fixed_Z','random_local','best_numerical_W','nearby_beta_transferred','alpha0_basis']
COLORS=dict(zip(BASES,plt.get_cmap('tab10').colors))


def read_rows(out,stage):
    rows=[]
    for path in sorted((out/stage).glob('*.json')):
        records=json.loads(path.read_text())
        if isinstance(records,list):
            for row in records: rows.append({**row,'source_file':str(path.relative_to(out))})
    return pd.DataFrame(rows)


def paired_summary(frame,metric,index,reference,candidates):
    pivot=frame.pivot_table(index=index,columns='basis',values=metric,aggfunc='first')
    result=[]
    for basis in candidates:
        if basis not in pivot or reference not in pivot:continue
        paired=pivot[[reference,basis]].dropna();delta=paired[basis]-paired[reference]
        result.append({'basis':basis,'reference':reference,'pairs':len(delta),'mean_delta':float(delta.mean()),
                       'median_delta':float(delta.median()),'fraction_improved':float(np.mean(delta < -1e-8)),
                       'max_worsening':float(delta.max()),'max_improvement':float(delta.min())})
    return pd.DataFrame(result)


def savefig(fig,out,name):
    fig.tight_layout();fig.savefig(out/'figures'/f'{name}.png',dpi=180)
    fig.savefig(out/'figures'/f'{name}.pdf');plt.close(fig)


def regress_cv(r4):
    d=r4[(r4.alpha==1)&(r4.boundary=='open')].copy()
    records=[]
    if d.empty:return pd.DataFrame()
    # No test targets enter centering/scaling or training. Physical covariates
    # are the baseline, so W is not rewarded just for correlating with beta.
    n=d.n.to_numpy();b=d.beta.to_numpy();j=d.j2.to_numpy()
    base=np.stack([n,b,j,n*b,b*j,j*j,b*b,d.purity.to_numpy()],axis=1)
    features={'physical_only':base}
    for column in ('W_rho','W_delta','W_H'):
        features['physical_plus_'+column]=np.column_stack([base,d[column].to_numpy()/(2*n)])
    features['physical_plus_eta']=np.column_stack([base,d.eta_pos.to_numpy()])
    y=d.gamma.to_numpy()
    for group in ('beta','j2','n'):
        for value in sorted(d[group].unique()):
            test=(d[group].to_numpy()==value);train=~test
            if train.sum()<10:continue
            for name,x in features.items():
                mu=x[train].mean(0);scale=x[train].std(0);scale[scale<1e-10]=1.
                z=(x-mu)/scale;z=np.column_stack([np.ones(len(z)),z]);penalty=np.eye(z.shape[1])*.01;penalty[0,0]=0
                coef=np.linalg.solve(z[train].T@z[train]+penalty,z[train].T@y[train]);pred=z[test]@coef
                records.append({'held_out_axis':group,'held_out_value':value,'model':name,'train_rows':int(train.sum()),
                                'test_rows':int(test.sum()),'rmse':float(np.sqrt(np.mean((pred-y[test])**2))),
                                'mae':float(np.mean(abs(pred-y[test])))})
    return pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    (out/'figures').mkdir(exist_ok=True);(out/'tables').mkdir(exist_ok=True)
    frames={stage:read_rows(out,stage) for stage in ('R1','R1_extended','R2','R3','R3_validation','R4','R4_controls')}
    for stage,d in frames.items():
        d.to_csv(out/'tables'/f'{stage}_all.csv',index=False)
    r1,r2,r3,r4=(frames[k] for k in ('R1','R2','R3','R4'))
    checks={}
    if not r1.empty:
        final=r1[r1.steps==2000].copy()
        checks['R1_independent_networks']=int(len(final))
        checks['R1_moment_abs_error_max']=float(r1.moment_disagreement.max())
        checks['R1_square_spectral_floor_min_slack']=float((r1.full_relative_frobenius_squared-r1.square_architecture_spectral_error_floor).min())
        idx=['n','beta','target','activation','depth','width','seed','steps']
        pairs=paired_summary(final,'full_relative_frobenius_squared',idx,'fixed_X',BASES[1:])
        pairs.to_csv(out/'tables/R1_paired_vs_fixed_X.csv',index=False)
        paired=final.pivot_table(index=idx,columns='basis',values='full_relative_frobenius_squared')
        # Uncertainty is over eight paired initialization seeds, within each
        # physical parameter / architecture group (never treating checkpoints as draws).
        stats=[]
        grouping=['n','beta','target','activation','depth','width','steps']
        for keys,group in final.groupby(grouping):
            pivot=group.pivot(index='seed',columns='basis',values='full_relative_frobenius_squared')
            for basis in BASES[1:]:
                delta=pivot[basis]-pivot.fixed_X;m=delta.mean();se=delta.std(ddof=1)/np.sqrt(len(delta))
                ci=student_t.ppf(.975,len(delta)-1)*se
                stats.append(dict(zip(grouping,keys),basis=basis,paired_seeds=len(delta),mean_delta=m,
                                  ci95_low=m-ci,ci95_high=m+ci,reference='fixed_X'))
        pd.DataFrame(stats).to_csv(out/'tables/R1_paired_seed_confidence.csv',index=False)
        # The actual metric at a common approximate training FLOP budget is
        # retained separately; checkpoint rounding differs by at most one update.
        r1[r1.matched_FLOPs_checkpoint==True].to_csv(out/'tables/R1_equal_FLOPs.csv',index=False)
        subset=final[(final.beta==1)&(final.target=='rho_beta')&(final.width==64)&(final.depth==2)&(final.activation=='tanh')]
        fig,axes=plt.subplots(1,3,figsize=(14,4),squeeze=False)
        for ax,n in zip(axes[0],[8,10,12]):
            for basis,group in subset[subset.n==n].groupby('basis'):
                err=group.full_relative_frobenius_squared
                ax.errorbar(group.W.mean(),err.mean(),yerr=student_t.ppf(.975,7)*err.std()/np.sqrt(len(err)),
                            fmt='o',color=COLORS[basis],label=basis,capsize=3)
            ax.set(title=f'n={n}, beta=1, 2-layer tanh, width 64',xlabel='W (alpha=1)',ylabel='Squared relative Frobenius error')
            ax.set_yscale('log');ax.grid(alpha=.2)
        axes[0,-1].legend(fontsize=7)
        savefig(fig,out,'Fig1_W_vs_MLP_error')
        resource_rows=[]
        all_learning=pd.concat([r1,frames['R1_extended']],ignore_index=True)
        for keys,g in all_learning.groupby(['n','beta','target','basis','seed']):
            for tolerance in (.1,.01,.001):
                reached=g[g.full_relative_frobenius_squared<=tolerance]
                resource_rows.append(dict(zip(['n','beta','target','basis','seed'],keys),
                    target_squared_error=tolerance,reached_in_tested_grid=bool(len(reached)),
                    minimum_observed_parameters=float(reached.parameters.min()) if len(reached) else None,
                    minimum_observed_FLOPs=float(reached.train_FLOPs_estimate.min()) if len(reached) else None))
        pd.DataFrame(resource_rows).to_csv(out/'tables/R1_resources_to_error.csv',index=False)
        if not frames['R1_extended'].empty:
            ext=frames['R1_extended'];checks['R1_extended_networks']=int(len(ext[ext.steps==40000]))
            checks['R1_extended_square_floor_min_slack']=float((ext.full_relative_frobenius_squared-ext.square_architecture_spectral_error_floor).min())
            # Independent reruns start at the same seed; 2000-step endpoints
            # must reproduce the first-wave run, up to float32 nondeterminism.
            compare=['n','beta','target','basis','seed','activation','depth','width','steps']
            joined=ext[ext.steps==2000].merge(r1[r1.steps==2000],on=compare,suffixes=('_extended','_original'))
            checks['R1_repeat_2000_step_max_difference']=float(abs(joined.full_relative_frobenius_squared_extended-joined.full_relative_frobenius_squared_original).max())
            paired_summary(ext[ext.steps==40000],'full_relative_frobenius_squared',idx,'fixed_X',BASES[1:]).to_csv(out/'tables/R1_extended_paired_vs_fixed_X.csv',index=False)
            fig,axes=plt.subplots(1,3,figsize=(14,4))
            for ax,n in zip(axes,[8,10,12]):
                selected=ext[(ext.n==n)&(ext.target=='rho_beta')&(ext.activation=='tanh')]
                for basis,g in selected.groupby('basis'):
                    agg=g.groupby('steps').full_relative_frobenius_squared.agg(['mean','sem'])
                    ax.errorbar(agg.index,agg['mean'],yerr=student_t.ppf(.975,7)*agg['sem'],fmt='o-',color=COLORS[basis],label=basis)
                ax.set_xscale('log');ax.set_yscale('log');ax.set(title=f'n={n}, beta=1, tanh width64 depth2',xlabel='Training steps',ylabel='Exact full squared error');ax.grid(alpha=.2)
                ax.set_xticks([2000,10000,40000],['2k','10k','40k'])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            axes[-1].legend(fontsize=7);savefig(fig,out,'Fig1b_selected_training_convergence')
    if not r2.empty:
        checks['R2_rows']=len(r2);checks['R2_thermal_identity_error_max']=float(r2.thermal_identity_error.max())
        checks['R2_MI_minus_2OSEE_max']=float((r2.I_W_bits-2*r2.osee_bits).max())
        checks['R2_inter_dimer_cov_max']=float(r2.get('inter_dimer_max_cov',pd.Series([np.nan])).max())
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        paths=list((out/'R2').glob('mixed_field_ising_n12_*.npz'))
        if paths:
            z=np.load(paths[0])
            for beta in [.5,1,2,4]:
                c=z[f'b{beta:g}_best_numerical_W_Cij'];vals=[np.diag(c,k).mean() for k in range(1,12)]
                axes[0].semilogy(range(1,12),np.maximum(abs(np.array(vals)),1e-16),'o-',label=f'beta={beta:g}')
        axes[0].set(title='Optimized spatial Walsh covariance (n=12)',xlabel='Separation r',ylabel='|mean C_W(r)|');axes[0].legend(fontsize=8)
        for basis,g in r2[(r2.model=='mixed_field_ising')&(r2.beta==1)].groupby('basis'):
            g=g.sort_values('n');axes[1].plot(g.n,g.half_boundary_abs,'o-',label=basis)
        axes[1].set(title='Absolute cross-cut covariance',xlabel='n',ylabel='B_W (half-chain)');axes[1].legend(fontsize=8)
        for model,g in r2.groupby('model'):axes[2].scatter(2*g.osee_bits,g.I_W_bits,s=10,alpha=.6,label=model)
        lim=max(float((2*r2.osee_bits).max()),.1);axes[2].plot([0,lim],[0,lim],'k--',lw=1)
        axes[2].set(title='Classical information bound',xlabel='2 OSEE (bits)',ylabel='I_W (bits)');axes[2].legend(fontsize=6)
        savefig(fig,out,'Fig2_spatial_correlations_and_OSEE')
    if not r3.empty:
        checks['R3_random_draws']=int(r3.groupby('source_file').draws.first().sum())
        checks['R3_purity_error_max']=float(r3.max_purity_error.max())
        osee_intervals=[]
        for path in sorted((out/'R3').glob('*.npz')):
            with np.load(path) as arrays:
                values=arrays['null_osee'];mean=float(values.mean());sem=float(values.std(ddof=1)/np.sqrt(len(values)))
                radius=float(student_t.ppf(.975,len(values)-1)*sem)
                osee_intervals.append({'source_file':str(path.relative_to(out)), 'draws':len(values),
                    'osee_mean_bits':mean,'osee_sem_bits':sem,'ci95_low':mean-radius,'ci95_high':mean+radius})
        pd.DataFrame(osee_intervals).to_csv(out/'tables/R3_OSEE_confidence_intervals.csv',index=False)
        # Exact identity draws have variance zero; roundoff is not statistical noise.
        analytic=r3[r3.null=='complex_Haar_global'].copy()
        analytic['relative_G_error']=abs(analytic.G_null_mean-analytic.G_complex_Haar_exact)/analytic.G_complex_Haar_exact
        valid=analytic.G_null_sem>1e-12*abs(analytic.G_null_mean)
        analytic['zscore_valid']=valid
        analytic['zscore_recomputed']=np.where(valid,(analytic.G_null_mean-analytic.G_complex_Haar_exact)/analytic.G_null_sem,np.nan)
        analytic.to_csv(out/'tables/R3_Haar_formula_validation.csv',index=False)
        checks['R3_Haar_nonzero_variance_max_abs_z']=float(analytic.zscore_recomputed.abs().max())
        checks['R3_Haar_zero_variance_max_relative_error']=float(analytic.loc[~valid,'relative_G_error'].max())
        if not frames['R3_validation'].empty:
            checks['R3_independent_validation_draws']=5000
            checks['R3_validation_max_abs_z']=float(abs(frames['R3_validation'].zscore).max())
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,n in zip(axes[:2],[6,8]):
            physical=r3[(r3.n==n)&(r3.ensemble=='microcanonical')&(r3.alpha==1)]
            phys=physical.drop_duplicates('rank').sort_values('rank');ax.plot(phys['rank'],phys.W_physical,'ko-',label='physical')
            for null,g in physical.groupby('null'):
                g=g.sort_values('rank');ax.errorbar(g['rank'],g.W_null_mean,yerr=(g.W_null_ci_high-g.W_null_ci_low)/2,fmt='o-',label=null)
            ax.set_xscale('log',base=2);ax.set(title=f'Microcanonical rank, n={n}',xlabel='rank',ylabel='W (alpha=1)')
        g=r3[(r3.n==8)&(r3.ensemble=='gibbs')&(r3.alpha==1)]
        for null,h in g.groupby('null'):
            h=h.sort_values('beta');axes[2].errorbar(h.beta,h.W_physical-h.W_null_mean,yerr=(h.W_null_ci_high-h.W_null_ci_low)/2,fmt='o-',label=null)
        axes[2].axhline(0,color='k',lw=1);axes[2].set(title='Physical Gibbs minus matched null, n=8',xlabel='beta',ylabel='Delta W')
        axes[1].legend(fontsize=6);savefig(fig,out,'Fig3_Haar_and_microcanonical')
    if not r4.empty:
        d=r4[r4.alpha==1].copy();checks['R4_physical_points']=int(d.groupby(['n','j2','boundary','beta']).ngroups)
        checks['R4_basis_evaluations']=len(d);checks['R4_gauge_logsign_error_max']=float(d.gauge_logsign_error.max())
        checks['R4_gauge_W_change_max']=float(abs(d.gauge_W_rho-d.W_rho).max())
        control=d[(d.j2==0)&(d.basis=='Marshall')]
        checks['R4_Marshall_logsign_abs_max']=float(abs(control.log_average_sign).max())
        checks['R4_stoquastic_logsign_abs_max']=float(abs(frames['R4_controls'].log_average_sign).max())
        pairs4=paired_summary(d,'gamma',['n','j2','boundary','beta'],'Marshall',
                             ['Walsh_rho','Walsh_delta','Walsh_H','direct_sign','eta_surrogate','transferred_high_temperature','transferred_smaller_size'])
        pairs4.to_csv(out/'tables/R4_paired_vs_Marshall.csv',index=False)
        objective_rows=[]
        for column,basis in [('W_rho','Walsh_rho'),('W_delta','Walsh_delta'),('W_H','Walsh_H')]:
            objective=d.pivot_table(index=['n','j2','boundary','beta'],columns='basis',values=column)
            difference=objective[basis]-objective.original_computational
            objective_rows.append({'objective':column,'basis':basis,'physical_points':len(difference),
                'points_with_W_reduction':int((difference < -1e-8).sum()),'mean_delta_W':float(difference.mean()),
                'max_W_reduction':float(-difference.min())})
        pd.DataFrame(objective_rows).to_csv(out/'tables/R4_optimization_objective_changes.csv',index=False)
        correlations=[]
        for keys,g in d.groupby(['n','j2','boundary','beta']):
            for col in ('W_rho','W_delta','W_H','eta_pos'):
                correlation=float(spearmanr(g[col],g.gamma).statistic) if g.gamma.std()>1e-10 and g[col].std()>1e-10 else None
                correlations.append(dict(zip(['n','j2','boundary','beta'],keys),metric=col,spearman=correlation,bases=len(g)))
        pd.DataFrame(correlations).to_csv(out/'tables/R4_within_point_rank_correlations.csv',index=False)
        regress_cv(r4).to_csv(out/'tables/R4_held_out_prediction.csv',index=False)
        selected=d[(d.n==8)&(d.j2==.5)&(d.beta==2)&(d.boundary=='open')]
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,col in zip(axes,['W_rho','W_delta','W_H']):
            for _,row in selected.iterrows():
                ax.scatter(row[col],row.gamma,s=25,label=row.basis)
                if col=='W_rho':ax.plot([row.W_rho,row.gauge_W_rho],[row.gamma,row.gamma],color='gray',alpha=.45,lw=.8)
            ax.set(title=f'n=8, J2=0.5, beta=2; {col}',xlabel=col,ylabel='gamma = -log(sign)/(beta n)')
        axes[-1].legend(fontsize=5,ncol=2);savefig(fig,out,'Fig4_sign_vs_Walsh_same_basis')
        pivot=d[d.boundary=='open'].pivot_table(index=['n','j2','beta'],columns='basis',values='gamma')
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        gamma_max=max(float(pivot.Marshall.max()),1e-8)
        delta_max=max(float(abs(pivot.Walsh_H-pivot.Marshall).max()),1e-8)
        for col,n in enumerate([6,8,10]):
            if n not in pivot.index.get_level_values('n'):continue
            pn=pivot.loc[n]
            for row,values in enumerate([pn.Marshall,pn.Walsh_H-pn.Marshall]):
                ax=axes[row,col];vals=values.unstack('beta').sort_index()
                im=ax.imshow(vals.to_numpy(),origin='lower',aspect='auto',
                    cmap='viridis' if row==0 else 'coolwarm',vmin=0 if row==0 else -delta_max,
                    vmax=gamma_max if row==0 else delta_max)
                ax.set_xticks(range(len(vals.columns)),[f'{x:g}' for x in vals.columns]);ax.set_yticks(range(len(vals.index)),[f'{x:g}' for x in vals.index])
                title=f'n={n}: Marshall gamma' if row==0 else f'n={n}: Delta gamma, Walsh(H) - Marshall'
                ax.set(title=title,xlabel='beta',ylabel='J2/J1');fig.colorbar(im,ax=ax)
        fig.suptitle(f'Positive Delta gamma means worse sign; max |Delta gamma for Walsh(rho)| = {abs(pivot.Walsh_rho-pivot.Marshall).max():.1e}',fontsize=12)
        savefig(fig,out,'Fig5_frustration_temperature_size')
        pivot.reset_index().to_csv(out/'tables/R4_gamma_by_basis.csv',index=False)
    statuses={}
    for stage in ('R1','R2','R3','R4'):
        path=out/f'{stage}_status.json';statuses[stage]=json.loads(path.read_text()) if path.exists() else {'status':'running'}
    if (out/'R1_extended_status.json').exists():statuses['R1_extended']=json.loads((out/'R1_extended_status.json').read_text())
    atomic_json(out/'summary.json',{'checks':checks,'statuses':statuses})
    build_report(out,frames,checks,statuses)
    print(json.dumps(checks,indent=2))


def table_html(path):
    return pd.read_csv(path).to_html(index=False,float_format=lambda x:f'{x:.5g}',border=0)


def build_report(out,frames,checks,statuses):
    meta=json.loads((out/'metadata.json').read_text());parts=[]
    def add(text):parts.append(text)
    add('<h1>Operator Walshness 第二轮：首批实验报告</h1>')
    add('<p>范围按用户要求收敛到数小时内的首批实验。以下结果对应本次已保存的原始数据；不代表完成原始 YAML 的全部网格。</p>')
    add('<p>阶段状态：'+html.escape('；'.join(f'{k}: {v["status"]}' for k,v in statuses.items()))+'</p>')
    if not frames['R1'].empty:
        d=frames['R1'];focus=d[(d.steps==2000)&(d.beta==1)&(d.target=='rho_beta')&(d.width==64)&(d.depth==2)&(d.activation=='tanh')]
        key_results=focus.groupby(['n','basis']).full_relative_frobenius_squared.mean().unstack()
        columns=[x for x in ('fixed_X','best_numerical_W','alpha0_basis') if x in key_results]
        add('<h2>主要发现</h2><p>在 β=1、两层 tanh、宽度64、2,000步的同预算比较中，平方相对误差如下（8种子均值）：</p>')
        add(key_results[columns].to_html(float_format=lambda x:f'{x:.5g}',border=0))
        add('<p>这些点支持“合适的物理基底能显著降低有限网络的表示与训练难度”。更广的首批网格中 α=0 基底常优于 α=1 数值最优基底，说明最小化某个固定 α 的 W 与最小化实际训练误差仍有区别。</p>')
        ext=frames['R1_extended']
        if not ext.empty:
            focus=ext[(ext.steps==40000)&(ext.target=='rho_beta')&(ext.activation=='tanh')]
            if len(focus):
                end=focus.groupby(['n','basis']).full_relative_frobenius_squared.mean().unstack()
                add('<p>同一选定配置延长到40,000步后的8种子均值（各列训练预算相同）：</p>')
                add(end[[c for c in columns if c in end]].to_html(float_format=lambda x:f'{x:.6g}',border=0))
    if not frames['R3'].empty:
        d=frames['R3'];point=d[(d.n==8)&(d.ensemble=='gibbs')&(d.beta==1)&(d.alpha==1)&(d['null']=='sector_respecting_Haar')]
        if len(point):
            row=point.iloc[0];add(f'<p>同谱、同 Sᶻ 扇区的随机基准也不能解释全部结构：XXZ n=8、β=1 时，物理 W={row.W_physical:.5f}，扇区 Haar 均值 {row.W_null_mean:.5f}（标准误 {row.W_null_sem:.5f}）。</p>')
    if not frames['R4'].empty:
        d=frames['R4'];q=d[d.alpha==1].pivot_table(index=['n','j2','boundary','beta'],columns='basis',values='gamma')
        if 'Walsh_rho' in q:
            delta=q.Walsh_rho-q.Marshall;improved=int((delta < -1e-8).sum())
            add(f'<p>QMC 首批结果为负：在已完成的 {len(delta)} 个物理参数点中，ρ 的 Walsh 最优实基底相对 Marshall 的有效改善点数为 {improved}（判据 Δγ&lt;−10⁻⁸）。本次 ρ 搜索也未找到比原始基底更低的 W，不能把这一结果解释成“ρ 的 W 降低后没有改善”。中心化 Gibbs 与 H 的目标则确实降低，仍未产生有效的 sign 改善。局域 Z gauge 的 W 最大变化为 {checks["R4_gauge_W_change_max"]:.4g}，平均符号保持不变。</p>')
    add('<p>Git 基线：<code>'+meta['git_commit']+'</code>。数值计算为 float64/complex128；MLP 为 float32；所有熵以 bit 为单位。计算基底中的 fixed_basis_spectrum 使用 X 方向 Walsh 轴；本报告 fixed_X 表示 U=I，fixed_Z 表示 U=H。优化基底使用 U_i=align_direction_to_z(n_i) H。</p>')
    add('<h2>范围与可复现性</h2><ul><li>R1：mixed-field Ising，g=1.05、h=0.5；n=8、10，β=0.5、1；6 类基底、8 个初始化、宽度 32/64、1/2 层、square/tanh、ρ 与归一化 √ρ；n=12 仅 β=1、宽度64、2层。每次2,000步。</li><li>R2：4 个模型、n=6/8/10、7 个温度、3 类基底；n=12 增加 TFIM 与 mixed-field Ising 的4个温度。保存所有切分的带符号和绝对值跨边界相关。</li><li>R3：XXZ Δ=1，n=6/8；5 个 Gibbs 温度、7 个微正则 rank；complex Haar、real orthogonal、保持 Sᶻ 扇区的 Haar，每组100样本。</li><li>R4：n=6/8 的开放链全首批 J2、β 网格，n=6 周期控制与 n=10 的 J2=0/0.5/1 验证。各优化目标使用4个初值及相同函数评估次数上限；n=10 使用更小的48次/初值预算。</li></ul>')
    add('<p>配置：<a href="config/round2_first_wave.yaml">实际运行配置</a>；<a href="config/operator_walshness_round2_experiment_plan.yaml">原方案</a>；<a href="metadata.json">运行元数据</a>；<a href="summary.json">机器可读汇总</a>；<a href="acceptance.json">保存数据验收</a>；<a href="qa_tests.xml">单元测试记录</a>；<a href="source/ROUND2_RUNBOOK.md">复现说明</a>。精确的完成范围以各阶段 status 和表格为准。</p>')
    add('<h2>R0 数值校验</h2>')
    add('<p>修复了优化 MLP 的 Hadamard 基底对齐、复数 Gibbs 的虚部丢失、复数空间相关中缺失的共轭；将 α=0 的共享物理基底公式扩展到任意非 Hermitian 算符。旧 MLP 的误差字段明确为平方相对 Frobenius 误差。原始第一轮结果未改写。</p>')
    add(pd.DataFrame([{'check':k,'value':v} for k,v in checks.items()]).to_html(index=False,border=0))
    if not frames['R1'].empty:
        add('<h2>R1 表示难度</h2><img src="figures/Fig1_W_vs_MLP_error.png">')
        add('<p>下表是相同架构、参数点、初始化、2,000步预算下，各基底相对 fixed_X 的配对平方误差变化；负值表示改善。跨参数平均仅为描述性汇总，置信区间在每个参数/架构组内以8个种子计算，见详细 CSV。</p>')
        add(table_html(out/'tables/R1_paired_vs_fixed_X.csv'))
        add('<p>所有基底使用相同的均匀训练样本序列和独立留出集。保存了全部每25步的 mini-batch 学习曲线、模型参数、全矩阵误差和留出误差，并另存相同估算 FLOPs 的检查点。FLOPs 采用线性层前向/反向及 Adam 的明确近似公式，不等同于实测硬件指令数。square 网络的最大 Walsh degree 是 2^depth，对应谱尾给出不可突破的误差下界；tanh 不具有这个硬截断。</p>')
        add('<p>有限 α 的 W 更重视高阶谱尾，α=0 基底可能在实际短预算训练中更好；因此不能由单个 W 值直接推出最容易训练的基底。2,000步结果也不能排除优化尚未收敛的影响。</p>')
        if not frames['R1_extended'].empty:
            add('<h3>选定参数的收敛检查</h3><p>为区分训练预算与表示限制，仅对 n=8/10/12、β=1、宽度64、两层 square/tanh、两种目标、全部6基底和8种子，补做2,000/10,000/40,000步检查。原首批结果独立保留。</p><img src="figures/Fig1b_selected_training_convergence.png">')
            add(table_html(out/'tables/R1_extended_paired_vs_fixed_X.csv'))
    if not frames['R2'].empty:
        add('<h2>R2 空间结构</h2><img src="figures/Fig2_spatial_correlations_and_OSEE.png">')
        add('<p>逐点检验了 Walsh degree 协方差与两项热关联的恒等式，保存热等时关联、虚时间 τ=β 关联、OSEE 和经典互信息。独立二聚体的跨二聚体协方差是零的对照。指数衰减拟合仅作有限尺寸描述，同时报告拟合点数和 R²；当前 n≤12 的数据不足以证明热力学极限面积律。</p>')
    if not frames['R3'].empty:
        add('<h2>R3 随机谱基准</h2><img src="figures/Fig3_Haar_and_microcanonical.png">')
        add('<p>R3 沿用第一轮 XXZ 归一化 H=Σ(XX+YY+ΔZZ)，Δ=1；这与 R4 的 S=σ/2 归一化不同，不应跨模块直接比较同一数值的 β。每次随机化严格保留谱和纯度；扇区 Haar 额外保留各 Sᶻ 扇区的谱及权重。微正则窗口取完整能谱中部，保存窗口宽度、能量和能量匹配的 canonical β。随机置信区间是样本均值的95% Student-t区间；100个样本对重尾生成函数的精度仍有限。</p>')
        add('<p>complex Haar 解析式：E G = [a (1+e^{2α})^n + b (1+e^α)^{2n}]/P，a=(1−P/d)/(d²−1)，b=(P−1/d)/(d²−1)，P=Trρ²，d=2^n。单独记录 E[log G]/α 和 log(E[G])/α。满秩均匀态的方差理论上为0，浮点舍入不能作为统计标准误；分析时对这类点禁用 z-score。</p>')
        if not frames['R3_validation'].empty:
            add('<p>初始100样本中，n=6、β=1 的生成函数均值出现最大约3.88标准误的偏差。另用独立种子增加5,000样本做精度检查（原数据保留）：三个 α 均落在解析均值的约2.3个标准误内，以下给出全部结果，而不把未落入95%区间的点删掉。</p>')
            add(table_html(out/'tables/R3_validation_all.csv'))
        add('<p>物理 Gibbs 的有限 α 数值最优 W 在可用时复用第一轮结果；对每个 Haar 样本再次优化 W 本轮暂缓。微正则/Gibbs 的差异不能解释为 ETH 失效。</p>')
    if not frames['R4'].empty:
        add('<h2>R4 平均符号</h2><img src="figures/Fig4_sign_vs_Walsh_same_basis.png"><img src="figures/Fig5_frustration_temperature_size.png">')
        add('<p>平均符号严格定义为 Tr exp(−βH_U) / Tr exp(−βH_abs)，H_abs 的非对角元为 −|H_U,st|。下表是相对 Marshall 的配对 γ 变化；负值才表示该 worldline 表示中的改善。</p>')
        add(table_html(out/'tables/R4_paired_vs_Marshall.csv'))
        add('<p>同时检查优化是否真的降低了各自的 Walsh 目标，避免把搜索未成功与目标不预测 sign 混为一谈。下表以原始计算基底为目标值参考；H 的结果在不同温度中复用同一 Hamiltonian 基底。</p>')
        add(table_html(out/'tables/R4_optimization_objective_changes.csv'))
        add('<p>本轮保留了未改善和变差的基底。对角 Z gauge 使平均符号不变却可明显改变固定基底 W，排除了将两者简单一一对应的解释。直接优化仅表示给定搜索预算内的最佳结果，未证明全局最优，也不等同于实际 QMC 方差或运行时间改善。</p>')
        add('<p>分别导出固定物理参数点内的秩相关、跨 β/J2/n 的留出预测误差，以及高温/小尺寸基底迁移。留出预测使用仅物理参数的 ridge 基线，再单独加入 Walsh 指标；训练集统计量用于标准化。小规模外推结果只作为诊断。</p>')
    add('<h2>后续优先级与限制</h2><p>依据配对误差和谱尾检查 α=0 与有限 α 的差别。空间关联若要推到面积律，需要更大尺寸及 MPO 的 χ、时间步长收敛。R3 生成函数重尾点需要更多独立样本。QMC 方向仅在有限预算精确结果显示稳定优势后才进入实际采样。原计划的全参数网格、全网格40,000步、n=12 全部 QMC 搜索及 R5 均未作为本轮完成项。随机局域 MLP 基底本轮每个尺寸只抽取一组，8个种子衡量网络初始化不确定性，不代表随机基底总体的不确定性。</p>')
    add('<h2>数据与图</h2><ul>')
    for path in sorted((out/'tables').glob('*.csv')):add(f'<li><a href="tables/{path.name}">{path.name}</a></li>')
    for path in sorted((out/'figures').glob('*.pdf')):add(f'<li><a href="figures/{path.name}">{path.name}</a></li>')
    add('</ul><p>原始 NPZ 分别保存在 R1、R1_targets、R2、R3、R4 目录。初次文件名处理错误的尝试保存在 invalid_filename_attempt，未进入本报告统计。</p>')
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Operator Walshness Round 2</title><style>body{max-width:1150px;margin:40px auto;padding:0 24px;font-family:system-ui,sans-serif;line-height:1.65;color:#17212b}h1,h2{color:#18395a}img{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}td,th{border-bottom:1px solid #dce2e8;padding:7px;text-align:left}a{color:#175eb5}code{background:#eef2f5;padding:3px}</style>'+''.join(parts)+'</html>'
    (out/'report.html').write_text(document)


if __name__=='__main__':main()
