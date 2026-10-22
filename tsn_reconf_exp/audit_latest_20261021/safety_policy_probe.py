"""Independently apply the existing C1 validator to the main policy's first 64 evaluation decisions."""
from pathlib import Path
import sys,json
from collections import Counter
sys.dont_write_bytecode=True
base=Path(__file__).resolve().parent; snap=base/'snapshot'; sys.path.insert(0,str(snap))
import numpy as np
import yaml,torch
from core_sim import decode_delta_action,local_precheck_port,timing_projector_gamma,compile_port_gcl,validate_c1_fifo_and_contracts
from predictor_mapper import SetTransformerModernTCN
from tsn_env import TASK_LIST,TSNTrafficAndPredictorDatasetGenerator,TSNRealClosedLoopEnv
from train_rl_and_baselines import TwoLayerDAGPolicy,extract_port_pred_features,build_l2_prop_features
cfg=yaml.safe_load((snap/'configs/experiment_config.yaml').read_text(encoding='utf-8'))
torch.set_num_threads(2)
pred=SetTransformerModernTCN().eval(); pred.load_state_dict(torch.load(snap/'checkpoints/predictor_SetTrans_ModernTCN.pt',map_location='cpu',weights_only=True))
policy=TwoLayerDAGPolicy().eval(); policy.load_state_dict(torch.load(snap/'checkpoints/rl_Local_Prop_MAPPO.pt',map_location='cpu',weights_only=True)['policy'])
tensors,truth=TSNTrafficAndPredictorDatasetGenerator(cfg,seed=3001).generate_dataset(64)
env=TSNRealClosedLoopEnv(cfg); preds={}; reasons=Counter(); pre_fails=0
with torch.no_grad():
    for g,e,tid in TASK_LIST:
        q,pi,_=pred(torch.from_numpy(tensors['full'][(g,e)]),torch.full((64,),tid,dtype=torch.long))
        preds[(g,e)]=(q.numpy(),pi.numpy())
    for k in range(64):
        feats=extract_port_pred_features(k,preds,'Local'); shifts={}; xp={}; gp={}
        for e in ['e3u','e4u']:
            action=policy.actors[e](torch.from_numpy(feats[e]).unsqueeze(0))[0].squeeze(0).numpy()
            xh=decode_delta_action(action,env.x0_dict[e],env.E_dict[e],env.L_dict[e])
            xp[e],gp[e],ok,_=local_precheck_port(e,env.port_flows[e],xh,env.x0_dict[e],cfg)
            pre_fails+=int(not ok)
            for slot in [0,1]: shifts[(e,slot)]=env.recover_slot_shift(e,xp[e],slot)
        prop=build_l2_prop_features(feats,shifts,'Local+Prop+MAPPO')
        for e in ['e01','e02']:
            obs=np.concatenate([feats[e],prop[e]])
            action=policy.actors[e](torch.from_numpy(obs).unsqueeze(0))[0].squeeze(0).numpy()
            xh=decode_delta_action(action,env.x0_dict[e],env.E_dict[e],env.L_dict[e])
            xp[e],gp[e],ok,_=local_precheck_port(e,env.port_flows[e],xh,env.x0_dict[e],cfg)
            pre_fails+=int(not ok)
        xt,_=timing_projector_gamma(env.flows,env.ports,xp,env.x0_dict,cfg)
        ge={e:gp[e] if np.allclose(xt[e],xp[e],atol=80.) else compile_port_gcl(e,env.port_flows[e],xt[e],cfg) for e in env.ports}
        ok,reason,_=validate_c1_fifo_and_contracts(env.flows,env.ports,xt,ge,cfg)
        reasons[reason.split(':')[0]]+=1
out={'main_policy': 'Local+Prop+MAPPO','evaluation_seed':3001,'decisions':64,'C1_results':dict(reasons),'local_precheck_failed_port_calls':pre_fails,
     'scope':'First 64 decisions, using captured checkpoint and exact current evaluation proposal logic; no retraining and no full experiment rerun.'}
(base/'safety_policy_evidence.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
