import ast, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch, yaml
from scipy import stats

OUT=Path(__file__).resolve().parent
ROOT=OUT/'snapshot'
sys.path.insert(0,str(ROOT))
import train_rl_and_baselines as rl
from predictor_mapper import SetTransformerModernTCN
from tsn_env import TSNRealClosedLoopEnv
torch.set_num_threads(1)
cfg=yaml.safe_load((ROOT/'configs/experiment_config.yaml').read_text(encoding='utf-8'))
saved=json.loads((ROOT/'results/full_experimental_results.json').read_text(encoding='utf-8'))
env=TSNRealClosedLoopEnv(cfg)
pred=SetTransformerModernTCN()
pred.load_state_dict(torch.load(ROOT/'checkpoints/predictor_SetTrans_ModernTCN.pt',map_location='cpu',weights_only=True))
pred.eval()
pol=rl.TwoLayerDAGPolicy()
pol.load_state_dict(torch.load(ROOT/'checkpoints/rl_Local_Prop_MAPPO.pt',map_location='cpu',weights_only=True)['policy'])
pol.eval()
tree=ast.parse((ROOT/'train_rl_and_baselines.py').read_text(encoding='utf-8'))
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate_method_on_seed')
ns=dict(vars(rl))
ns.update({'cfg':cfg,'sim_env':env,'trained_predictors':{'SetTrans_ModernTCN':pred},'trained_policies':{'Local+Prop+MAPPO':pol},'e1_e15_res':saved['E1_and_E15'],'eval_cycles_per_seed':300,'_seed_dataset_cache':{},'_seed_gen_cache':{},'_seed_pred_cache':{},'_topo_env_cache':{('Topology-v1',22):env},'prior_preds':{},'time':time})
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'train_rl_and_baselines.py'),'exec'),ns)
evaluate=ns['evaluate_method_on_seed']
out={'external_current_checkpoint_environment':{}}
for method in ['Period','E-TSN','ARTSN']:
 vals=[]
 for seed in [3001,3002,3003,3004,3005]:
  v=evaluate(method,seed)
  vals.append(v)
  print(method,seed,v['miss_pct'],flush=True)
 out['external_current_checkpoint_environment'][method]={'seed_results':vals,'mean_miss_pct':float(np.mean([v['miss_pct'] for v in vals])),'mean_ms':float(np.mean([v['mean_e2e_ms'] for v in vals])),'mean_seed_p99_ms':float(np.mean([v['p99_e2e_ms'] for v in vals]))}

orig_step=env.simulate_step
actions={}
for method in ['Local+Prop+Rule','Local+Prop+MAPPO']:
 records={p:[] for p in env.ports}
 def record(gt,*args,**kwargs):
  xraw=args[3]
  for p in env.ports:
   records[p].extend((np.abs(xraw[p]-env.x0_dict[p])/1000.).tolist())
  return orig_step(gt,*args,**kwargs)
 env.simulate_step=record
 evaluate(method,3001)
 env.simulate_step=orig_step
 actions[method]={p:{'mean_abs_baseline_shift_us':float(np.mean(v)),'p95_abs_baseline_shift_us':float(np.percentile(v,95)),'fraction_beyond_13p2us':float(np.mean(np.array(v)>13.2))} for p,v in records.items()}
out['action_magnitude_seed3001']=actions

old=json.loads((OUT/'evidence.json').read_text(encoding='utf-8'))
ours=np.array([s['miss_pct'] for s in old['current_checkpoint_re_evaluation']['Local+Prop+MAPPO']['seeds']])
out['paired_statistics_current']={}
base={'Rule':[s['miss_pct'] for s in old['current_checkpoint_re_evaluation']['Local+Prop+Rule']['seeds']], 'LocalOnly':[s['miss_pct'] for s in old['current_checkpoint_re_evaluation']['LocalOnly+MAPPO']['seeds']]}
for name,res in out['external_current_checkpoint_environment'].items():
 base[name]=[s['miss_pct'] for s in res['seed_results']]
for name,a in base.items():
 diff=np.array(a)-ours
 out['paired_statistics_current'][name]={'baseline_minus_ours_pp':float(diff.mean()),'ci95_halfwidth_pp':float(stats.t.ppf(.975,4)*diff.std(ddof=1)/np.sqrt(5)),'p':float(stats.ttest_rel(a,ours).pvalue)}
(OUT/'followup_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL',json.dumps(out,ensure_ascii=False),flush=True)
