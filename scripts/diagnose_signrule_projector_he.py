"""HE template tests on frozen real eigenvectors and explicit variational controls."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from operator_walshness.round2 import atomic_json,schmidt
from operator_walshness.signrule.geometry import triangular
from operator_walshness.signrule.huse_elser import reference_features,trial_wavefunction,fit_circular_phase,axis_texture,structure_factor_K
from operator_walshness.signrule.diagnostics import phase_entries,best_linear_phase,score_phase,paired_improvement_ci
from operator_walshness.signrule.workflow import utc
from operator_walshness.walsh import state_walshness,normalized_expectations
from run_signrule_triangular_projectors import eigenstates


def main():
    out=ROOT/'results/signrule_round3';dest=out/'physical_B/projector_HE';dest.mkdir(parents=True,exist_ok=True)
    g=triangular(3,3);target_g=triangular(3,4,False)
    sz,sub,cubic,reference=reference_features(g)
    target_sz,target_sub,target_cubic,target_ref=reference_features(target_g)
    features={'linear_U1':sz,'three_sublattice_U1':sub,'linear_plus_HE_three_spin':np.column_stack([sz,cubic]),
              'three_sublattice_plus_HE':np.column_stack([sub,cubic])}
    target_features={'linear_U1':target_sz,'three_sublattice_U1':target_sub,
                     'linear_plus_HE_three_spin':np.column_stack([target_sz,target_cubic]),
                     'three_sublattice_plus_HE':np.column_stack([target_sub,target_cubic])}
    rows=[]
    for j2 in [0.,.1,.2]:
        states=eigenstates(g,j2);targets=eigenstates(target_g,j2)
        for label,(psi,state_meta) in states.items():
            directory=dest/f'p{j2:g}'/label
            if (directory/'summary.json').exists():rows.append(json.loads((directory/'summary.json').read_text()));continue
            source=out/'physical_B/projector_blind'/f'p{j2:g}'/label
            frozen=list(source.glob('alpha*/seed*/blind_frozen.json'))
            assert len(frozen)==9 and all(json.loads(p.read_text())['known_label_access'] is False for p in frozen)
            source_metrics=json.loads((source/'alpha1/seed20261027/metrics.json').read_text())
            assert abs(source_metrics['W_fixed']-2*state_walshness(psi,1))<1e-9
            rho=np.outer(psi,psi.conj());target_psi,target_meta=targets[label]
            target_rho=np.outer(target_psi,target_psi.conj());scores={};arrays={}
            for cutoff in [1e-8,1e-12]:
                tag=f'{cutoff:g}';entries=phase_entries(rho,cutoff);target_entries=phase_entries(target_rho,cutoff)
                linear,mask=best_linear_phase(entries,9);lower,_=best_linear_phase(entries,9,'test')
                target_lower,_=best_linear_phase(target_entries,12,'test')
                fits={};phases={};transfer={}
                for name,x in features.items():
                    phase,fit=fit_circular_phase(entries,x,starts=16,maxiter=300)
                    fits[name]=fit;phases[name]=phase;coeff=np.asarray(fit['coefficients'])
                    if name.startswith('linear'):
                        site_coeff=np.array([coeff[(i%3)+3*((i//3)%3)] for i in range(12)])
                        coeff=np.r_[site_coeff,coeff[-1]] if name.endswith('three_spin') else site_coeff
                    target_phase=np.exp(1j*(target_features[name]@coeff))
                    transfer[name]={'target_test':score_phase(target_entries,target_phase),
                        'versus_test_Boolean_linear':paired_improvement_ci(target_entries,target_lower,target_phase),
                        'source_only_coefficients':coeff,'target_fitted':False}
                    arrays[f'{tag}_{name}_phase']=phase
                scores[tag]={'fits':fits,'Boolean_linear_mask':mask,'Boolean_linear_test':score_phase(entries,linear),
                    'HE_versus_continuous_linear':paired_improvement_ci(entries,phases['linear_U1'],phases['linear_plus_HE_three_spin']),
                    'HE_versus_test_Boolean_linear':paired_improvement_ci(entries,lower,phases['linear_plus_HE_three_spin']),
                    'target_frozen_template_transfer':transfer}
            structure,correlations=structure_factor_K(rho,g);arrays['spin_correlations']=correlations
            textures=[]
            for p in source.glob('alpha*/seed*/basis_or_circuit.json'):
                local=json.loads(p.read_text())['local_SU2']
                textures.append({'path':str(p.relative_to(out)),**axis_texture(local['axes_best'],g)})
            _,osee=schmidt(rho)
            row={'created_utc':utc(),'J2':j2,'object':label,'state':state_meta,'target_state':target_meta,
                 'source_n':9,'target_n':12,'purity':1.,'pure_state_carryover':True,'operator_specific':False,
                 'phase_scores':scores,'HE_reference':reference,'target_HE_reference':target_ref,
                 'structure_factor_K':structure,'OSEE_fixed':osee,'local_magnetization':normalized_expectations(rho)[0],
                 'blind_axis_textures':textures,'interpretation':'posthoc HE fits are reference diagnostics, not blind recovery; degenerate eigenvectors are not unique'}
            directory.mkdir(parents=True,exist_ok=True);np.savez_compressed(directory/'phase_fits.npz',**arrays)
            atomic_json(directory/'summary.json',row);rows.append(row);print('PROJECTOR_HE',j2,label,flush=True)
    controls=[]
    for real in [False,True]:
        psi,_=trial_wavefunction(g,real_part=real,n_down=4)
        for positive in [False,True]:
            state=abs(psi) if positive else psi;entries=phase_entries(np.outer(state,state.conj()))
            fits={}
            for name,x in features.items():
                _,fit=fit_circular_phase(entries,x,starts=16,maxiter=300);fits[name]=fit
            controls.append({'real_trial':real,'removed_phase':positive,'fits':fits,
                             'role':'known variational template control; real-part projection need not retain a single U1 HE phase'})
    atomic_json(dest/'summary.json',{'created_utc':utc(),'source_objects':len(rows),'controls':controls,
        'fit_budget':{'starts':16,'max_iterations':300},'source_frozen_before_HE_scoring':True,
        'source_only_transfer':'site coefficients repeat 3x3 source anchors; sublattice/HE scalar coefficients unchanged on open 3x4 geometry',
        'no_new_blind_discovery_claim':True})


if __name__=='__main__':main()
