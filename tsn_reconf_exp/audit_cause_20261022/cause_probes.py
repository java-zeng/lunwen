"""Controlled diagnosis; never trains or overwrites original experimental outputs."""
import ast
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent
ROOT = OUT / 'snapshot'
files = list(ORIGINAL.glob('*.py')) + list((ORIGINAL/'configs').glob('*.yaml')) + list((ORIGINAL/'checkpoints').glob('*.pt')) + list((ORIGINAL/'results').glob('*.json'))
manifest = {}
for src in files:
    dst = ROOT/src.relative_to(ORIGINAL)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)
    manifest[str(src.relative_to(ORIGINAL))] = {'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'mtime_ns':src.stat().st_mtime_ns}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
sys.path.insert(0,str(ROOT))
import core_sim as core
import train_rl_and_baselines as rl
from predictor_mapper import SetTransformerModernTCN
from tsn_env import TSNRealClosedLoopEnv, TSNTrafficAndPredictorDatasetGenerator, TASK_LIST

torch.set_num_threads(1)
cfg=yaml.safe_load((ROOT/'configs/experiment_config.yaml').read_text(encoding='utf-8'))
evidence={}
env=TSNRealClosedLoopEnv(cfg)
g_etsn={p:core.compile_etsn_port_gcl(p,env.port_flows[p],env.x0_dict[p],cfg) for p in env.ports}
g_artsn={p:core.compile_artsn_port_gcl(p,env.port_flows[p],env.x0_dict[p],cfg) for p in env.ports}
overlaps={}
for name,gdict in [('E-TSN',g_etsn),('ARTSN',g_artsn)]:
    g=gdict['e3u']
    re=g.protected_regions[4 if name=='ARTSN' else 0][2]
    ready=re-2000.
    finish=core.next_fit_q7_tx_finish(ready,0.,1184.,0.,g,env.H)
    hits=[{'flow_id':fid,'tt_start_ns':s,'tt_finish_ns':e,'overlap_ns':min(e,finish)-max(s,finish-1184.)} for fid,_,s,e in g.q6_subwindows if min(e,finish)>max(s,finish-1184.)]
    overlaps[name]={'C1':core.validate_c1_fifo_and_contracts(env.flows,env.ports,env.x0_dict,gdict,cfg)[:2],'ST_ready_ns':ready,'ST_finish_ns':finish,'TT_overlaps':hits}
evidence['shared_slot_tt_overlap']=overlaps

gen=TSNTrafficAndPredictorDatasetGenerator(cfg,seed=3001)
_,gts=gen.generate_dataset(300)
same_table={}
for name,gs in [('Period_same_x',env.gcl0_dict),('E-TSN_shared',g_etsn),('ARTSN_shared',g_artsn)]:
    alternatives=[(name,gs)]
    if name!='Period_same_x':
        disabled=copy.deepcopy(gs)
        for g in disabled.values():
            g.q7_shared_flex_starts_arr=None
            g.q7_shared_flex_ends_arr=None
        alternatives.append((name+'_sharing_disabled',disabled))
    for label,gdict in alternatives:
        env.reset_episode()
        delays=[]
        misses=[]
        for gt in gts:
            o=env.simulate_step(gt,env.x0_dict,env.x0_dict,gdict)
            delays.extend(o['e2e_delays_ms'])
            misses.extend(o['misses'])
        same_table[label]={'source_frames':len(misses),'miss_pct':100*float(np.mean(misses)),'mean_ms':float(np.mean(delays)),'p99_ms':float(np.percentile(delays,99))}
evidence['same_TT_table_only_permission_changes']=same_table
print('SAME_TABLE',json.dumps(same_table),flush=True)

saved=json.loads((ROOT/'results/full_experimental_results.json').read_text(encoding='utf-8'))
curves=json.loads((ROOT/'results/training_curves_rl.json').read_text(encoding='utf-8'))
evidence['versions']={'saved_full_results':saved['E2_Core_External_Robustness']['rl_training_curves']['_version'],'current_training_curves':curves['_version']}
pred=SetTransformerModernTCN()
pred.load_state_dict(torch.load(ROOT/'checkpoints/predictor_SetTrans_ModernTCN.pt',map_location='cpu',weights_only=True))
pred.eval()
policies={}
for variant in ['Local+Prop+MAPPO','LocalOnly+MAPPO']:
    policy=rl.TwoLayerDAGPolicy()
    ck=torch.load(ROOT/'checkpoints'/('rl_'+variant.replace('+','_')+'.pt'),map_location='cpu',weights_only=True)
    policy.load_state_dict(ck['policy'])
    policy.eval()
    policies[variant]=policy

tree=ast.parse((ROOT/'train_rl_and_baselines.py').read_text(encoding='utf-8'))
fn=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='evaluate_method_on_seed')
ns=dict(vars(rl))
ns.update({'cfg':cfg,'sim_env':env,'trained_predictors':{'SetTrans_ModernTCN':pred},'trained_policies':policies,'e1_e15_res':saved['E1_and_E15'],'eval_cycles_per_seed':300,'_seed_dataset_cache':{},'_seed_gen_cache':{},'_seed_pred_cache':{},'_topo_env_cache':{('Topology-v1',22):env},'prior_preds':{},'time':time})
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'train_rl_and_baselines.py'),'exec'),ns)
evaluate=ns['evaluate_method_on_seed']
current={}
for method in ['Local+Prop+Rule','LocalOnly+MAPPO','Local+Prop+MAPPO']:
    vals=[]
    for seed in [3001,3002,3003,3004,3005]:
        v=evaluate(method,seed)
        vals.append(v)
        print('CURRENT',method,seed,v['miss_pct'],flush=True)
    current[method]={'seeds':vals,'miss_mean_pct':float(np.mean([v['miss_pct'] for v in vals])),'mean_ms':float(np.mean([v['mean_e2e_ms'] for v in vals])),'mean_seed_p99_ms':float(np.mean([v['p99_e2e_ms'] for v in vals]))}
evidence['current_checkpoint_re_evaluation']=current

# Vary only the rule magnitude, retaining the same local predictions and mapper.
class ReplaceMagnitude(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,float) and node.value==.22:
            return ast.copy_location(ast.Name(id='diagnostic_rule_magnitude',ctx=ast.Load()),node)
        return node
rule_fn=ReplaceMagnitude().visit(copy.deepcopy(fn))
rule_fn=ast.fix_missing_locations(rule_fn)
exec(compile(ast.Module(body=[rule_fn],type_ignores=[]),'<rule_magnitude_probe>','exec'),ns)
sweep={}
for mag in [.10,.22,.40,.58,.66,.82]:
    ns['diagnostic_rule_magnitude']=mag
    vals=[ns['evaluate_method_on_seed']('Local+Prop+Rule',s) for s in [3001,3002]]
    sweep[str(mag)]={'miss_pct_seeds':[v['miss_pct'] for v in vals],'miss_mean_pct':float(np.mean([v['miss_pct'] for v in vals])),'p99_ms':float(np.mean([v['p99_e2e_ms'] for v in vals]))}
    print('MAGNITUDE',mag,sweep[str(mag)],flush=True)
evidence['two_seed_rule_magnitude_diagnostic']=sweep

# Compare actual critic predictions with the exact target construction used in code.
critic=rl.CriticMLP(136)
ck=torch.load(ROOT/'checkpoints/rl_Local_Prop_MAPPO.pt',map_location='cpu',weights_only=True)
critic.load_state_dict(ck['critic'])
states=[]
rew=[]
orig_step=env.simulate_step
def capture_step(gt,*args,**kwargs):
    out=orig_step(gt,*args,**kwargs)
    feats=rl.extract_port_pred_features(gt.cycle_k,ns['_seed_pred_cache'][(3001,1.,'SetTrans_ModernTCN','full')])
    states.append(np.concatenate([feats[p] for p in env.ports]))
    rew.append(out['reward'])
    return out
env.simulate_step=capture_step
evaluate('Local+Prop+MAPPO',3001)
env.simulate_step=orig_step
with torch.no_grad():
    v=critic(torch.tensor(np.stack(states),dtype=torch.float32)).numpy()
targets=[]
vused=[]
for start in range(0,288,48):
    vb=v[start:start+48]
    rb=np.array(rew[start:start+48])
    gae=np.zeros(48)
    last=0.
    for t in reversed(range(48)):
        vn=vb[t+1] if t+1<48 else vb[t]
        delta=rb[t]+float(cfg['rl_training']['gamma'])*vn-vb[t]
        last=delta+float(cfg['rl_training']['gamma'])*float(cfg['rl_training']['gae_lambda'])*last
        gae[t]=last
    targets.extend((gae+vb).tolist())
    vused.extend(vb.tolist())
targets=np.array(targets)
vused=np.array(vused)
evidence['critic_scale_diagnostic']={'test_seed':3001,'samples':len(targets),'value_mean':float(vused.mean()),'GAE_target_mean':float(targets.mean()),'target_std':float(targets.std()),'value_std':float(vused.std()),'mse':float(np.mean((targets-vused)**2)),'explained_variance':float(1.-np.var(targets-vused)/np.var(targets))}
evidence['original_sources_unchanged']=all(hashlib.sha256((ORIGINAL/f).read_bytes()).hexdigest()==entry['sha256'] for f,entry in manifest.items())
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL',json.dumps(evidence,ensure_ascii=False),flush=True)
