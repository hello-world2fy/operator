"""Evidence-based acceptance of the staged design, with explicit conditional limits.

No raw-data digest sweep. Checks read saved numerical artifacts and the test log.
Large native grids are never inferred from frozen transfers or sampled kernels.
"""
from __future__ import annotations
import json,sys,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from operator_walshness.round2 import atomic_json
from operator_walshness.walsh import walshness_from_spectrum
from operator_walshness.signrule.workflow import point_directory,utc
from operator_walshness.signrule.blind import circuit_depth


def read(path):return json.loads(path.read_text())


def main():
    out=ROOT/'results/signrule_round3';checks=[]
    def check(name,criterion,evidence,fn):
        try:
            detail=fn();checks.append({'id':name,'criterion':criterion,'status':'verified','evidence':evidence,'detail':detail})
        except Exception as e:checks.append({'id':name,'criterion':criterion,'status':'incomplete','evidence':evidence,'detail':f'{type(e).__name__}: {e}'})
    def tests():
        root=ET.parse(out/'registration/final_pytest.xml').getroot();cases=list(root.iter('testcase'))
        assert len(cases)>=108
        assert not list(root.iter('failure')) and not list(root.iter('error'))
        required=['test_random_local_gauge_preserves_spectrum_but_invalidates_raw_sign_labels',
            'test_batched_qutrit_kernel_keeps_exponential_identity_in_float64',
            'test_controlled_frustration_has_an_explicit_local_stoquastic_basis',
            'test_blind_optimizer_import_boundary','test_individual_symmetric_gate_truth_tables',
            'test_closed_paths_use_actual_hop_spin_events','test_adaptive_temperature_preserves_degenerate_ground_rank_and_energy_shift']
        names={c.attrib['name'] for c in cases};assert set(required)<=names
        return {'passed_test_cases':len(cases),'required_regressions':required}
    check('numerical_tests','Independent model, phase, rotation, precision and symmetry regressions pass',
          ['registration/final_pytest.xml','tests/test_signrule_*.py'],tests)
    jobs=read(out/'registration/jobs.json');adaptive=read(out/'registration/adaptive_jobs.json')
    points=list(out.glob('*/*/n*/*/*/alpha*/seed*/metrics.json'))
    counts=Counter(read(p)['track'] for p in points)
    def coverage():
        assert counts=={'A':621,'B':180,'C':144,'D':279,'E':1512,'F':1062},dict(counts)
        for j in jobs+[j for j in adaptive if j['kind']=='qubit']:
            for b in j['betas']:
                for a in j['alphas']:
                    for s in j['seeds']:assert (point_directory(out,j,b,a,s)/'metrics.json').exists()
        statuses=[read(p) for p in (out/'jobs').glob('adaptive*_posthoc.json')]
        assert len(statuses)==28 and all(s['status']=='complete' for s in statuses)
        return {'primary_points':len(points),'by_track':dict(counts),'adaptive_points':360,'adaptive_jobs':28}
    check('small_grid_and_adaptive','All registered small-system alpha/beta/seed grids and energy-selected additions exist',
          ['registration/jobs.json','registration/adaptive_jobs.json','registration/adaptive_temperature_decisions.json'],coverage)
    def artifacts():
        spectra=0;thermal=0;circuits=0;local=0
        for p in points:
            m=read(p);point=p.parent;results=read(point/'basis_or_circuit.json');frozen=read(point/'blind_frozen.json')
            assert frozen['known_label_access'] is False
            assert len(frozen['basis_or_circuit_sha256'])==64
            assert (point/'metadata.json').exists() and (point/'representation_baselines.json').exists()
            for key,result in results.items():
                if result.get('no_discovery'):continue
                assert result['objective']=='Operator_Walshness_only'
                values=np.asarray(result['W_restarts']);assert np.isfinite(values).all()
                assert abs(min(values)-result['W_best'])<1e-8
                if key.startswith('circuit'):
                    circuits+=1;assert len(values)==16 and result['flips_per_start']==2000
                    assert result['evaluations']==16*2001
                    assert sum(result['coefficients_best'])==result['gate_count']<=result['max_gates']
                    assert circuit_depth(result['gate_supports'],result['coefficients_best'])==result['circuit_depth']<=result['max_depth']
                    assert len(result['initial_coefficients'])==len(result['coefficients_restarts'])==16
                else:
                    local+=1;assert len(values)==24
                    vector=result.get('axes_best',result.get('quaternions_best'))
                    assert np.allclose(np.linalg.norm(vector,axis=-1),1,atol=1e-7)
                    assert len(result.get('initial_axes',result.get('initial_quaternions')))==24
            spectrum=point/'thermal_spectrum.npz'
            if spectrum.exists():
                thermal+=1
                with np.load(spectrum) as a:e=a['energies'];w=a['eigenvalues']
                expected=np.exp(-m['beta']*(e-e.min()));expected/=expected.sum()
                assert np.allclose(w,expected,atol=1e-12,rtol=1e-10)
                assert abs(w@w-m['purity'])<1e-9
            else:assert m['track']=='C' and m['purity']==1
            path=point/'walsh_spectrum.npz'
            if path.exists():
                spectra+=1
                with np.load(path) as a:
                    for label in a.files:
                        if label.startswith('degree_'):continue
                        v=a[label];assert v.ndim==1 and v.min()>=-1e-14 and abs(v.sum()-1)<1e-9
                        if label in results:assert abs(walshness_from_spectrum(v,m['alpha'])-results[label]['W_best'])<1e-8
            else:assert m['track']=='E' and m['n']>6
        return {'thermal_spectra_checked':thermal,'degree_spectra_checked':spectra,'local_candidates':local,
                'phase_circuits':circuits,'digest_policy':'freeze digest fields checked; no new content hashing'}
    check('saved_candidates_spectra_baselines','Budgets, restart arrays, W reconstruction, density spectra and generic/reference baselines are consistent',
          ['<point>/basis_or_circuit.json','<point>/thermal_spectrum.npz','<point>/walsh_spectrum.npz','<point>/representation_baselines.json'],artifacts)
    def precision():
        manifest=read(out/'registration/qutrit_precision_jobs.json');assert len(manifest)==84
        shifts=[]
        for i,job in enumerate(manifest):
            status=read(out/'jobs'/f'qutrit_precision_{i:03d}.json');assert status['status']=='complete';shifts+=status['optimized_W_changes']
            for rel in job['points']:
                assert (out/'precision_archive'/rel/'blind_frozen.json').exists()
                result=read(out/rel/'basis_or_circuit.json')['local_charge_preserving']
                assert result['numerical_convention']=='float64_Z3_kernel_constants'
        return {'repeated_points':len(shifts),'maximum_full_rerun_W_change':max(map(abs,shifts)),
            'note':'optimization trajectories can amplify a small arithmetic change; archived results are superseded'}
    check('qutrit_precision','Every legacy float32-constant optimization is archived and replaced by a full float64-kernel run',
          ['registration/qutrit_precision_correction.json','registration/qutrit_precision_jobs.json','jobs/qutrit_precision_*.json'],precision)
    def phase_records():
        positive=Counter();checked=0
        for p in points:
            m=read(p)
            if m['track']=='E':continue
            d=read(p.parent/'posthoc.json');checked+=1
            for scores in d.get('circuit_scores',{}).values():
                if '1e-08' in scores:assert '1e-12' in scores
                else:assert m['track']=='C' and m['purity']==1
            if d.get('preliminary_non_Marshall_candidates'):positive[m['model']]+=1
            shared=d.get('shared_reference')
            if shared:assert (out/shared/'sign_diagnostics.npz').exists()
            if m['track']=='F' and m['beta']>0:assert (p.parent/'worldline_phase.json').exists()
        return {'point_phase_records':checked,'preliminary_positive_points_by_model':dict(positive),
            'threshold_scope':'two cutoffs; uniform supported pure-loop amplitudes give identical retained sets'}
    check('posthoc_and_QMC','Frozen candidates have phase masks/scores, conservative best-linear comparisons, and F basis-matched worldline diagnostics',
          ['<point>/posthoc.json','<beta>/posthoc_base','<point>/worldline_phase.json','<beta>/physics_F'],phase_records)
    def B():
        root=out/'physical_B';assert len(list((root/'projector_blind').glob('p*/*/alpha*/seed*/transfer.json')))==54
        for p in (root/'projector_blind').glob('p*/*/alpha*/seed*/transfer.json'):
            d=read(p);assert d['target_optimized'] is False and d['target_geometry']['boundary']=='open'
        rows=list((root/'projector_alpha0_recheck').glob('p*/*/alpha*/seed*/metrics.json'));assert len(rows)==54
        for p in rows:
            c=read(p.parent/'basis_or_circuit.json')['local_SU2'];assert np.allclose(c['initial_axes'][1],c['alpha0_axes'])
            assert len(c['W_restarts'])==24 and (p.parent/'representation_baselines.json').exists()
        he=read(root/'projector_HE/summary.json');assert he['source_objects']==6
        control=next(x for x in he['controls'] if not x['real_trial'] and not x['removed_phase'])
        assert control['fits']['linear_plus_HE_three_spin']['test']['circular_loss']<1e-8
        assert len(list((root/'projector_HE').glob('p*/*/degree_moments.json')))==6
        return {'source_projectors':54,'alpha0_full_budget_rechecks':54,'HE_objects':6,'HE_control_phase_loss':control['fits']['linear_plus_HE_three_spin']['test']['circular_loss']}
    check('B_physics','Triangular Gibbs/ground/excited cases, genuine HE controls/fits, textures, K structure factor and frozen open-n12 transfer',
          ['physical_B','triangular/3x3_periodic','operator_walshness/signrule/huse_elser.py'],B)
    def C():
        pair=read(out/'physical_C/thermal_pair/summary.json');assert len(pair['rows'])==6
        assert pair['spectrum_max_difference']<1e-10
        for row in pair['rows']:
            assert row['matched_purity_residual']<1e-10
            for model in row['models'].values():assert len(model['closed_sector_weights'])==4
        assert len(list((out/'negative_controls/loop_phase_scramble').glob('n*/draw*/metrics.json')))==40
        return {'thermal_points':108,'pure_points':36,'phase_scrambles':40,'full_thermal_n':12,'pure_only_n':18,
                'oracle_scope':'closed-loop sectors with independently checked residual; not vertex-defect configurations'}
    check('C_physics','TC/DS matched support/topology, full n12 Gibbs/purity pairing and n18 wavefunctions without dense operators',
          ['physical_C','toric_code','double_semion','negative_controls/loop_phase_scramble'],C)
    def D():
        transfer=read(out/'independent_validation/Levin_Gu_transfer/summary.json')
        assert all(not v['passed_all_target_seeds_temperatures_sizes_thresholds'] for v in transfer['certificates'].values())
        exhaustive=read(out/'independent_validation/Levin_Gu_all_candidates/summary.json')
        assert exhaustive['distinct_preliminary_candidates']==28 and not exhaustive['survivors_needing_size_and_perturbation_tests']
        records=list(out.glob('levin_gu/*/n*/*/*/physics_D/summary.json'));assert len(records)==25
        for p in records:
            d=read(p);assert d['global_Z2_commutator_max']<1e-12 and 'strange_correlator' in d
        return {'LG_thermal_physics_records':len(records),'all_distinct_preliminary_candidates_screened':28,
            'independent_transfer':'failed all-target criterion for all four gate classes','native_n12_perturbed_grid':'conditional gate not met'}
    check('D_physics','Gatewise symmetry plus independent SPT diagnostics and n9-to-n12 frozen validation, without W-defined phase claims',
          ['levin_gu','independent_validation/Levin_Gu_transfer','tests/test_signrule_diagnostics.py'],D)
    def E():
        paths=list((out/'physical_E').glob('**/path_summary.json'))+list((out/'physical_E_large').glob('**/path_summary.json'))
        assert len(paths)==24
        for p in paths:
            d=read(p)
            for row in d['rows']:assert row['phase_string_violations']==row['sigma_nontrivial']==0
        small=list((out/'physical_E').glob('**/beta*/summary.json'));large=list((out/'physical_E_large').glob('**/beta*/summary.json'))
        assert len(small)==84 and len(large)==72
        for p in large:assert read(p)['full_thermal_support_certificate'] is False
        assert len(list((out/'physical_E').glob('**/flux_response.json')))==3
        assert len(list((out/'physical_E_large').glob('**/flux_response.json')))==6
        return {'small_exact_thermal_pairs':84,'large_sampled_kernel_pairs':72,'true_path_rows':96,
            'large_scope':'64 exact propagator columns plus 64 random trace probes, full H graph; no full thermal-support certificate',
            'qudit_metric':'Z3 support degree only, U1-hole plus SU2-spin rotations'}
    check('E_physics','t-J/sigma-t-J all stated chain/ring sizes and ladders: endpoint, true paths, spin-hole and flux diagnostics',
          ['physical_E','physical_E_large','tests/test_signrule_models.py'],E)
    def F():
        a=read(out/'independent_validation/frustration_transfer/summary.json');b=read(out/'independent_validation/frustration_n12_transfer/summary.json')
        assert a['all_temperature_size_seed_threshold_targets_pass'] and b['all_temperature_size_seed_threshold_targets_pass']
        assert a['all_perturbed_targets_pass'] is False
        assert a['passed_mixed_primary_targets']+b['passed_mixed_primary_targets']==48
        double=read(out/'independent_validation/frustration_double_boundary/summary.json');assert double['rows']==40
        for n in [10,12]:assert read(out/f'independent_validation/j1j2_large/n{n}/summary.json')['cases']==4
        return {'primary_mixed_validation_rows':48,'all_target_rows':120,'minimum_primary_phase_gain':min(a['minimum_primary_improvement'],b['minimum_primary_improvement']),
            'interpretation':'bounded lambda0.4 open-boundary CZ correction; fails uniform neighboring-coupling validation',
            'large_J1J2':'8 exact sector-spectrum/H-support checks; native large blind grid remains gated'}
    check('F_physics','Frustration loops and exact worldline gamma; bounded rule validated in independent temperatures/sizes through n12',
          ['controlled_frustration','j1j2','independent_validation/frustration*','independent_validation/j1j2_large'],F)
    def A():
        for n in [10,12]:
            d=read(out/f'independent_validation/large_known_controls/n{n}/summary.json')
            assert d['all_exact_oracle_checks_passed'] and d['all_identity_checks_passed'] and d['beta_points']==36
        return {'native_blind_sizes':[6,8],'full_operator_control_sizes':[10,12],
            'native_large_blind_grid':False,'meaning':'known identity/oracle and frozen-rule tests are distinguished from native blind universal recovery'}
    check('A_controls','Known Marshall/cluster/positive/scramble controls across stated sizes, with recovery and oracle validity distinguished',
          ['independent_validation/large_known_controls','bipartite','cluster','random_positive','tests/test_signrule_references.py'],A)
    def nulls():
        counts={}
        for model in ['controlled_frustration','levin_gu']:
            for family in ['complex_Haar','real_orthogonal_Haar','sector_Haar']:
                rows=list((out/'negative_controls'/model/family).glob('draw*/metrics.json'));assert len(rows)==20
                assert all(read(p)['phase_scores']['1e-08']['versus_test_linear_lower_bound']['ci95'][0]<=0 for p in rows)
                counts[model+'/'+family]=len(rows)
        return counts
    check('same_spectrum_nulls','Selected-point 20-draw Haar, real and symmetry-sector controls at matched blind budgets',
          ['negative_controls','tables/same_spectrum_nulls.csv'],nulls)
    def learning():
        rows=list((out/'learning').glob('n*/beta*/alpha*/w*/metrics.json'));assert len(rows)==36
        for p in rows:
            d=read(p);assert len(d['rows'])==144 and d['training']['seeds']==8
            assert {r['step'] for r in d['rows']}=={2000,10000,40000}
            assert d['training']['hermitian_partner_holdout'] is True
        summary=read(out/'learning/comparison_summary.json');assert summary['paired_comparison_rows']==1944
        assert 'repeated_alpha_caveat' in summary
        for name in ['learning_paired_comparisons.csv','learning_observed_resources.csv','learning_resource_summary.csv']:assert (out/'tables'/name).exists()
        return {'training_instances':1728,'distinct_conditions':768,'matched_architecture_paired_seed_statistics':True,
                'claim':'no universal learned-phase resource advantage; per-entry oracle storage and phase-error floor explicit'}
    check('conditional_learning','Gate-approved six representations, all widths/depths/seeds/checkpoints, phase/Frobenius errors and observed resources',
          ['registration/phase3_gate.json','learning','tables/learning_*.csv'],learning)
    # Link every YAML leaf to its relevant concrete checks; design-only status
    # labels remain specification metadata, rather than execution claims.
    config=yaml.safe_load((ROOT/'operator_walshness_signrule_round3_config.yaml').read_text())
    mapping={'repo':['provenance'],'physics_definition':['numerical_tests','saved_candidates_spectra_baselines'],
        'scan_default':['small_grid_and_adaptive','saved_candidates_spectra_baselines','posthoc_and_QMC'],
        'optimization':['numerical_tests','saved_candidates_spectra_baselines','posthoc_and_QMC','B_physics','D_physics'],
        'measurements':['saved_candidates_spectra_baselines','posthoc_and_QMC','B_physics','C_physics','D_physics','E_physics','F_physics'],
        'tracks.A_known_controls':['A_controls'],'tracks.B_triangular_Huse_Elser':['B_physics'],
        'tracks.C_toric_code_vs_double_semion':['C_physics'],'tracks.D_Levin_Gu_SPT':['D_physics'],
        'tracks.E_tJ_phase_string':['E_physics'],'tracks.F_frustrated_sign_obstruction_and_QMC_control':['F_physics'],
        'secondary_cross_model_controls':['same_spectrum_nulls','saved_candidates_spectra_baselines','C_physics','E_physics'],
        'secondary_application':['conditional_learning'],'execution':['small_grid_and_adaptive','numerical_tests','provenance'],
        'outputs':['delivery','provenance','saved_candidates_spectra_baselines'],'validation':['numerical_tests','saved_candidates_spectra_baselines','posthoc_and_QMC'],
        'scientific_question':['posthoc_and_QMC','F_physics','conditional_learning'],'project':['provenance'],'status':['provenance']}
    def provenance():
        d=read(out/'registration/provenance.json');assert d['git_base'].startswith('a392c34')
        assert all(k in d['libraries'] for k in ['numpy','scipy','torch','pytest'])
        assert (out/'registration/execution_revisions.json').exists()
        return {'initial_base':d['git_base'],'source_digest_policy':d['output_hash_policy'],
            'old_precision_results_archived':True,'previous_rounds':'no round1/2 artifacts modified by this experiment'}
    check('provenance','Source commit, dependency/compiler conventions and implementation corrections remain reconstructible',
          ['registration/provenance.json','registration/execution_revisions.json','registration/source_v1','registration/source_v3_small_offdiagonal'],provenance)
    def delivery():
        for name in ['summary.json','report.html','tables/points.csv','tables/all_metrics.csv','tables/phase_scores.csv','tables/degree_tail_summary.csv',
                     'figures/walsh_gain_vs_purity.png','figures/learning_paired_error_ratios.png']:assert (out/name).exists(),name
        assert read(out/'summary.json')['points_frozen']==3798
        return {'root':'results/signrule_round3','report_status':'scientific finalization and compact publication are checked separately before closing the goal'}
    check('delivery','Human report, JSON, per-track tables, spectra/tails, figures and pytest log exist with current counts',
          ['report.html','summary.json','tables','figures','registration/final_pytest.xml'],delivery)
    leaves=[]
    def walk(value,path=''):
        if isinstance(value,dict):
            for k,v in value.items():walk(v,path+'.'+k if path else k)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,f'{path}[{i}]')
        else:
            prefix=max((k for k in mapping if path==k or path.startswith(k+'.') or path.startswith(k+'[')),key=len)
            leaves.append({'requirement':path,'specified_value':value,'evidence_checks':mapping[prefix]})
    walk(config)
    conditional=[
        {'item':'native n10/n12 control/frustrated and n12 perturbed-LG blind grids','rule':'design section11 and YAML execution phases: enlarge selectively after a non-Marshall signal',
         'decision':'small blind grids plus exact/frozen large checks completed; only controlled-frustration lambda0.4 passed the expansion gate, and it was extended through n12'},
        {'item':'2D 3x3 t-J','rule':'design section7: necessary only if indicated; phase3 conditional on clear robust fingerprint',
         'decision':'not indicated by a blind endpoint fingerprint; all mandatory chains/rings/ladders covered, path identity checked independently'},
        {'item':'large-scale QMC sampling','rule':'F no_new_large_scale_QMC_unless_positive_signal; W is not a universal sign-severity reducer',
         'decision':'exact worldline partition controls suffice; no favorable QMC signal established'},
        {'item':'dense n18 Gibbs, n>6 full qutrit FFT and complete large thermal-support certificate','rule':'explicit n18 prohibition and feasible-spectrum/memory budgets',
         'decision':'n18 wavefunctions only; n8 qutrit exact kernel moments/covariances; n10/12 t-J exact selected columns with sampling scope disclosed'},
        {'item':'optional beam search, chirality, weak symmetry-selection fields','rule':'explicitly optional in design/config',
         'decision':'not invoked; no conclusions rely on these unperformed diagnostics'}]
    passed=all(c['status']=='verified' for c in checks)
    result={'checked_utc':utc(),'status':'numerical_evidence_verified_report_review_pending' if passed else 'incomplete',
        'checks':checks,'yaml_leaf_requirements_mapped':len(leaves),'requirements':leaves,'conditional_scope_decisions':conditional,
        'full_research_goal_complete':False,'completion_note':'numerical audit is necessary, not sufficient: final scientific report and published corrected snapshot must also be reviewed'}
    atomic_json(out/'registration/requirements_audit.json',result)
    print(json.dumps({'status':result['status'],'checks':len(checks),'yaml_leaves':len(leaves),
                      'incomplete':[c for c in checks if c['status']!='verified']},indent=2))
    if not passed:sys.exit(1)


if __name__=='__main__':main()
