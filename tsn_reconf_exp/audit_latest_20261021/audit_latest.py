"""Read-only diagnostic analysis of the captured experiment, without retraining or overwriting results."""
from pathlib import Path
import sys, json, dataclasses, hashlib
sys.dont_write_bytecode = True
base = Path(__file__).resolve().parent
snap = base / 'snapshot'
sys.path.insert(0, str(snap))
import numpy as np
import yaml
from scipy.stats import ttest_rel, t
from core_sim import (
    TTFlow, CompiledPortGCL, generate_global_flows, compile_port_gcl,
    local_precheck_port, decode_delta_action, timing_projector_gamma,
    validate_c1_fifo_and_contracts, next_fit_q7_tx_finish,
    _expand_port_instances_np,
)

cfg = yaml.safe_load((snap/'configs/experiment_config.yaml').read_text(encoding='utf-8'))
report = json.loads((snap/'results/full_experimental_results.json').read_text(encoding='utf-8'))
out = {}
methods = report['E2_Core_External_Robustness']['nine_methods']
ours = np.array(methods['Local+Prop+MAPPO']['miss_pct_seeds'])
out['paired_test_recalculation'] = {}
for name in ['StrongStatic','LocalOnly+MAPPO','OwnerProp+MAPPO','Local+Prop+IPPO','Prior+Prop+MAPPO']:
    b = np.array(methods[name]['miss_pct_seeds']); diff = b-ours
    half = t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff))
    out['paired_test_recalculation'][name] = {'delta_percentage_points':float(diff.mean()),
        'relative_reduction_percent':float(100*diff.mean()/b.mean()),
        't95_interval':[float(diff.mean()-half),float(diff.mean()+half)],
        'paired_p':float(ttest_rel(b,ours).pvalue)}

flows = generate_global_flows(cfg)
ports = cfg['topologies']['Topology-v1']['controlled_ports']
pf = {e:[f for f in flows if e in f.path] for e in ports}
x0 = {e:np.array([f.base_offsets_ns[e] for f in pf[e]],float) for e in ports}
g0 = {e:compile_port_gcl(e,pf[e],x0[e],cfg) for e in ports}
H=cfg['network']['hyperperiod_ns']; GB=cfg['network']['guard_band_aligned_ns']

def gate_count(flist,x):
    starts,ends,_=_expand_port_instances_np(flist,x,H)
    pairs=sorted((int(a),int(b)) for a,b in zip(starts,ends) if 0<=a<b<=H)
    regions=[]
    for a,b in pairs:
        if not regions or a-regions[-1][1]>GB: regions.append([a,b])
        else: regions[-1][1]=max(regions[-1][1],b)
    blocked=[(max(0,a-GB),b) for a,b in regions]
    points=sorted({0,H}|{v for ab in pairs+blocked for v in ab})
    states=[]
    for a,b in zip(points,points[1:]):
        m=(a+b)/2
        state=64 if any(s<=m<e for s,e in pairs) else (0 if any(s<=m<e for s,e in blocked) else 145)
        if not states or states[-1]!=state: states.append(state)
    return {'linear_entries':len(states),'cyclic_entries':len(states)-int(len(states)>1 and states[0]==states[-1])}

out['baseline_gate_counts']={e:{'flows':len(pf[e]),'instances':sum(H//f.period_ns for f in pf[e]),
    'compiler_entries':g0[e].num_gcl_entries,**gate_count(pf[e],x0[e])} for e in ports}
f1=TTFlow(9001,'s','d',['probe'],H,H,1000,1020,8160,0,{'probe':20000},{'probe':0},{'probe':H-8160})
f2=dataclasses.replace(f1,flow_id=9002,base_offsets_ns={'probe':32000})
small=compile_port_gcl('probe',[f1,f2],np.array([20000.,32000.]),cfg)
out['small_gap_gcl_probe']={'gap_ns':3840,'compiler_entries':small.num_gcl_entries,**gate_count([f1,f2],np.array([20000.,32000.]))}
all_flex={e:dataclasses.replace(g0[e],num_gcl_entries=1,total_tt_tx_ns=0,total_gb_ns=0,total_flex_ns=H,
    q6_subwindows=[],protected_regions=[],flex_starts_arr=np.array([0.]),flex_ends_arr=np.array([float(H)])) for e in ports}
ok,reason,_=validate_c1_fifo_and_contracts(flows,ports,x0,all_flex,cfg)
out['all_flex_no_tt_validator']={'accepted':bool(ok),'reason':reason}
closed=CompiledPortGCL('closed',[],[],[],1,0,0,0,0,np.array([],float),np.array([],float))
out['closed_gate_nextfit']={'returned_finite_finish_ns':next_fit_q7_tx_finish(100.,0.,100.,0.,closed,H)}

rng=np.random.default_rng(20261021); e='e3u'
E=np.array([f.env_early_ns[e] for f in pf[e]],float); L=np.array([f.env_late_ns[e] for f in pf[e]],float)
_,_,baseline_order=_expand_port_instances_np(pf[e],x0[e],H)
counts={'samples':200,'fallbacks':0,'pass_with_remaining_overlap':0,'pass_with_order_change':0}
for _ in range(200):
    xh=decode_delta_action(rng.uniform(-1,1,len(pf[e])),x0[e],E,L)
    xp,gp,passed,why=local_precheck_port(e,pf[e],xh,x0[e],cfg)
    if not passed: counts['fallbacks']+=1
    else:
        st,en,order=_expand_port_instances_np(pf[e],xp,H)
        counts['pass_with_remaining_overlap']+=int(np.any(st[1:]<en[:-1]))
        counts['pass_with_order_change']+=int(not np.array_equal(order,baseline_order))
out['local_precheck_sample_probe']=counts

import torch
import tsn_env
from predictor_mapper import SetTransformerModernTCN, run_service_mapper_port
from train_predictors import compute_auprc, wasserstein_1d_bins
torch.set_num_threads(2)

secondary_B_mean_lengths=[]
original_build=tsn_env.build_observation
def capture_observation(*args,**kwargs):
    wins=original_build(*args,**kwargs)
    port=args[2] if len(args)>2 else kwargs['egress_port']
    if port=='e4u':
        lens=[ev.length for w in wins for ev in w if ev.flow_id==202]
        secondary_B_mean_lengths.append(float(np.mean(lens)))
    return wins
tsn_env.build_observation=capture_observation
tensors,truth=tsn_env.TSNTrafficAndPredictorDatasetGenerator(cfg,seed=20269999).generate_dataset(400)
tsn_env.build_observation=original_build
out['same_absolute_history_window_consistency']={
    'checked':399,'equal_count':int(sum(np.array_equal(tensors['full'][('A','e3u')][k,1],tensors['full'][('A','e3u')][k+1,0]) for k in range(399)))}
out['secondary_B_flow_202_diagnostic']={
    'feature':'single flow 202 mean frame length over the 32ms history; no fitted model',
    'AP':compute_auprc(np.array([g.occurred['B'] for g in truth]),np.array(secondary_B_mean_lengths)),
    'reported_primary_201_BestSingleFlow_AP':report['E1_and_E15']['e1_six_tasks']['(B, e4u)']['BestSingleFlow']['auprc']}

net=SetTransformerModernTCN().eval()
net.load_state_dict(torch.load(snap/'checkpoints/predictor_SetTrans_ModernTCN.pt',map_location='cpu',weights_only=True))
out['E1_checkpoint_recomputed']={}
with torch.no_grad():
    for g,port,tid in tsn_env.TASK_LIST:
        X=torch.from_numpy(tensors['full'][(g,port)]); ids=torch.full((len(truth),),tid,dtype=torch.long)
        q,pi,_=net(X,ids); q=q.numpy(); pi=pi.numpy()
        y=np.array([x.occurred[g] for x in truth]); bins=np.array([x.true_bin_up[g] if port in ('e3u','e4u') else x.true_bin_down_base[g] for x in truth]); pos=np.where(y>0)[0]
        out['E1_checkpoint_recomputed'][f'({g}, {port})']={
            'auprc':round(compute_auprc(y,q),3),'brier':round(float(np.mean((q-y)**2)),3),
            'nll_bin':round(float(-np.mean(np.log(np.maximum(pi[pos,bins[pos]],1e-6)))),3),
            'matches_saved_AP':round(compute_auprc(y,q),3)==report['E1_and_E15']['e1_six_tasks'][f'({g}, {port})']['SetTrans_ModernTCN']['auprc']}

env=tsn_env.TSNRealClosedLoopEnv(cfg)
counts={'active_packets':0,'downstream_time_before_upstream_arrival':0,'bin_disagrees_with_downstream_time':0}
for gt in truth:
    step=env.simulate_step(gt,env.x0_dict,env.x0_dict,env.gcl0_dict)
    for g in ['A','B','C']:
        if gt.occurred[g]:
            counts['active_packets']+=1
            tq=step['down_arr_time_ns_actual'][g]
            counts['downstream_time_before_upstream_arrival']+=int(tq<gt.src_arrival_ns[g])
            counts['bin_disagrees_with_downstream_time']+=int(int(tq//500000)!=step['down_arr_bins_actual'][g])
out['causal_time_consistency']=counts

pi=np.zeros(16); pi[4]=1
args=dict(cycle_k=0,upstream_port='e3u',local_preds={'A':(.8,pi)},gcl_cur=None,gcl_pre=g0['e3u'],
    residual_block_dist_ns=np.array([0.,3000.]),cfg=cfg,candidate_id='probe',config_hash='audit')
a=run_service_mapper_port(carry_in_backlog_count=0,**args)[('A','e3u','e01')]
b=run_service_mapper_port(carry_in_backlog_count=1,**args)[('A','e3u','e01')]
out['mapper_prefix_probe']={'accepts_none_current_gcl':True,'one_carry_in_changes_probabilities':bool(a.q_prop!=b.q_prop or not np.array_equal(a.pi_prop,b.pi_prop))}
out['message_three_arrays_bytes']=int(a.pi_prop.nbytes+a.bounds_low.nbytes+a.bounds_high.nbytes)
out['source_hash_match']=hashlib.sha256((snap/'configs/global_flows.yaml').read_bytes()).hexdigest()[:16]==report['E0']['config_hash']

(base/'evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=True,indent=2))
