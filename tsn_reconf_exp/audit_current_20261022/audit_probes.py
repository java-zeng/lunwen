"""Read-only probes against the current implementation; writes only audit artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core_sim as core
import predictor_mapper as pm
import tsn_env as envmod
from tsn_env import TSNRealClosedLoopEnv, TSNTrafficAndPredictorDatasetGenerator

torch.set_num_threads(1)
cfg = yaml.safe_load((ROOT / "configs/experiment_config.yaml").read_text(encoding="utf-8"))
OUT = Path(__file__).parent
evidence = {}
source_paths = list(ROOT.glob("*.py")) + list((ROOT / "configs").glob("*.yaml")) + [ROOT / "results/full_experimental_results.json"]
manifest = {str(p.relative_to(ROOT)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "mtime_ns": p.stat().st_mtime_ns} for p in source_paths}

results = json.loads((ROOT / "results/full_experimental_results.json").read_text(encoding="utf-8"))
methods = results["E2_Core_External_Robustness"]["nine_methods"]
ours = np.array(methods["Local+Prop+MAPPO"]["miss_pct_seeds"])
comparisons = {}
for method, res in methods.items():
    if method == "Local+Prop+MAPPO":
        continue
    delta = np.array(res["miss_pct_seeds"]) - ours
    comparisons[method] = {"baseline_minus_ours_pp": float(delta.mean()), "ci95_halfwidth_pp": float(stats.t.ppf(.975, 4) * delta.std(ddof=1) / np.sqrt(5)), "paired_p": float(stats.ttest_rel(res["miss_pct_seeds"], ours).pvalue)}
evidence["recalculated_paired_statistics"] = comparisons

env = TSNRealClosedLoopEnv(cfg)
validate = lambda g: core.validate_c1_fifo_and_contracts(env.flows, env.ports, env.x0_dict, g, cfg)[:2]
checks = {"baseline": validate(env.gcl0_dict)}
g_bad = copy.deepcopy(env.gcl0_dict)
v = list(g_bad["e01"].q6_subwindows[10])
v[2] += 100_000
v[3] += 100_000
g_bad["e01"].q6_subwindows[10] = tuple(v)
checks["interior_q6_window_moved_100us"] = validate(g_bad)
g_bad = copy.deepcopy(env.gcl0_dict)
g_bad["e01"].flex_windows = [(0, env.H)]
g_bad["e01"].flex_starts_arr = np.array([0.0])
g_bad["e01"].flex_ends_arr = np.array([float(env.H)])
checks["all_flex_arrays_with_unchanged_metadata"] = validate(g_bad)
evidence["validator_counterexamples"] = checks

rng = np.random.default_rng(20261022)
passed = rejected = invalid_passes = 0
for _ in range(30):
    for p in env.ports:
        xhat = core.decode_delta_action(rng.uniform(-1, 1, len(env.port_flows[p])), env.x0_dict[p], env.E_dict[p], env.L_dict[p])
        xp, gp, ok, _ = core.local_precheck_port(p, env.port_flows[p], xhat, env.x0_dict[p], cfg)
        passed += int(ok)
        rejected += int(not ok)
        starts, ends, idxs = core._expand_port_instances_np(env.port_flows[p], xp, env.H)
        _, _, base_idxs = core._expand_port_instances_np(env.port_flows[p], env.x0_dict[p], env.H)
        invalid_passes += int(ok and (np.any(starts[1:] < ends[:-1]) or not np.array_equal(idxs, base_idxs)))
evidence["local_precheck_random_stress"] = {"port_proposals": passed + rejected, "pass": passed, "fallback": rejected, "passed_with_overlap_or_changed_order": invalid_passes}

gen = TSNTrafficAndPredictorDatasetGenerator(cfg, seed=3001)
tensors, gts = gen.generate_dataset(60)
overlap_equal = sum(all(gen.last_obs_wins_by_cycle[k][p][1:] == gen.last_obs_wins_by_cycle[k+1][p][:-1] for p in env.ports) for k in range(59))
rates = {}
for p in env.ports:
    wins = [gen.last_obs_wins_by_cycle[k][p][-1] for k in range(60)]
    counts = [len(w) for w in wins]
    bits = sum(sum((ev.length + 20) * 8 for ev in w) for w in wins)
    rates[p] = {"events_per_8ms_mean": float(np.mean(counts)), "all_avb_be_wire_mbps": bits / (60 * .008) / 1e6, "configured_be_alone_mbps": cfg["be_traffic"]["nominal_offered_rate_bps"][p] / 1e6}
evidence["observation_log"] = {"adjacent_shared_histories_equal": overlap_equal, "pairs": 59, "rates": rates}

label_checks = {g: {"active_frames": 0, "bin_disagreement": 0} for g in ("A", "B", "C")}
for gt in gts:
    env.reset_episode()
    out = env.simulate_step(gt, env.x0_dict, env.x0_dict, env.gcl0_dict, update_state=False)
    for g in label_checks:
        if gt.occurred[g]:
            label_checks[g]["active_frames"] += 1
            label_checks[g]["bin_disagreement"] += int(gt.true_bin_down_base[g] != out["down_arr_bins_actual"][g])
evidence["downstream_supervision_vs_baseline_service"] = label_checks

gt0 = replace(gts[10], occurred={"A": 1, "B": 0, "C": 0})
pre_alt = {p: env.x0_dict[p].copy() for p in env.ports}
for p in ("e3u", "e4u"):
    pre_alt[p] = pre_alt[p] + 16_000
env.reset_episode()
out0 = env.simulate_step(gt0, env.x0_dict, env.x0_dict, env.gcl0_dict, update_state=False)
env.reset_episode()
out1 = env.simulate_step(gt0, pre_alt, env.x0_dict, env.gcl0_dict, update_state=False)
evidence["uninstalled_candidate_influences_physics"] = {"same_executed_gcl": True, "only_precheck_candidate_differs": True, "down_A_before_ns": out0["down_arr_time_ns_actual"]["A"], "down_A_after_ns": out1["down_arr_time_ns_actual"]["A"], "reported_e2e_before_ms": out0["e2e_delays_ms"], "reported_e2e_after_ms": out1["e2e_delays_ms"]}

xshift = {p: env.x0_dict[p] + 16_000 for p in env.ports}
gshift = {p: core.compile_port_gcl(p, env.port_flows[p], xshift[p], cfg) for p in env.ports}
violations = []
for gt in gts[:30]:
    gta = replace(gt, occurred={"A": 1, "B": 0, "C": 0})
    env.reset_episode()
    out = env.simulate_step(gta, xshift, xshift, gshift, update_state=False)
    path_lower_ms = (out["down_arr_time_ns_actual"]["A"] - gta.src_arrival_ns["A"]) / 1e6
    if out["e2e_delays_ms"][0] < path_lower_ms:
        violations.append({"cycle": gt.cycle_k, "reported_e2e_ms": out["e2e_delays_ms"][0], "upstream_to_downstream_arrival_alone_ms": path_lower_ms})
evidence["e2e_less_than_its_own_path_time"] = {"samples": 30, "violations": len(violations), "examples": violations[:3], "executed_schedule_C1": core.validate_c1_fifo_and_contracts(env.flows, env.ports, xshift, gshift, cfg)[:2]}

env.reset_episode()
before_credit = env.cbs_credit_bits.copy()
env.simulate_step(gt0, env.x0_dict, env.x0_dict, env.gcl0_dict, update_state=False)
evidence["counterfactual_changes_state"] = {"before_credit": before_credit, "after_credit_despite_update_state_false": env.cbs_credit_bits.copy()}

line = TSNRealClosedLoopEnv(cfg, "Topology-Line-4")
calls = []
original_nextfit = envmod.next_fit_q7_tx_finish
def traced_nextfit(*args, **kwargs):
    gcl = args[4]
    calls.append({"port": gcl.port, "tx_ns": args[2]})
    return original_nextfit(*args, **kwargs)
envmod.next_fit_q7_tx_finish = traced_nextfit
line.simulate_step(gt0, line.x0_dict, line.x0_dict, line.gcl0_dict, update_state=False)
envmod.next_fit_q7_tx_finish = original_nextfit
evidence["line4_q7_service_trace"] = {"expected_A_path": cfg["topologies"]["Topology-Line-4"]["controlled_ports"], "all_nextfit_calls_including_ghost_B_C": calls, "A_frame_and_echo_service_ports": [c["port"] for c in calls if c["tx_ns"] == 1184.0], "missing_ports_in_all_calls": sorted(set(line.ports) - {c["port"] for c in calls})}

pi = np.eye(16)[7]
msgs = pm.run_service_mapper_port(3, "e4u", {"B": (.8, pi), "C": (.6, pi)}, env.gcl0_dict["e4u"], env.gcl0_dict["e4u"], np.array([0., 2800., 6200., 11200.]), 2, cfg, "candidate_A", "config_A", gcl_base=env.gcl0_dict["e4u"])
m = msgs[("B", "e4u", "e01")]
packed = pm.pack_l2_prop_message_bytes(m)
identities = {name: pm.pack_l2_prop_message_bytes(replace(m, **updates)) == packed for name, updates in {"candidate_id": {"candidate_id": "candidate_B"}, "config_hash": {"config_hash": "config_B"}, "mapper_version": {"mapper_version": "another_version"}, "dst_port": {"dst_port": "e02"}}.items()}
evidence["wire_message"] = {"payload_bytes": len(packed), "mode": m.mode, "serialization_unchanged_when_identity_changes": identities, "bounds_are_pmf_plus_minus_point06": bool(np.allclose(m.bounds_low, np.maximum(0., m.pi_prop - .06)) and np.allclose(m.bounds_high, np.minimum(1., m.pi_prop + .06)))}

tail_msg = pm.run_service_mapper_port(3, "e3u", {"A": (1., np.eye(16)[15])}, env.gcl0_dict["e3u"], gshift["e3u"], np.array([0.]), 0, cfg, "c", "h", gcl_base=env.gcl0_dict["e3u"])[("A", "e3u", "e01")]
evidence["mapper_probability_mass"] = {"input_occurrence_probability": 1., "output_q": tail_msg.q_prop, "out_of_window_mass": tail_msg.out_of_window_mass, "q_plus_out": tail_msg.q_prop + tail_msg.out_of_window_mass}

from train_predictors import compute_auprc
y_tie = np.array([1., 1., 0., 0.])
evidence["average_precision_ties"] = {"all_scores": .5, "positive_fraction": .5, "AP_positive_rows_first": compute_auprc(y_tie, np.full(4, .5)), "AP_same_rows_reversed": compute_auprc(y_tie[::-1], np.full(4, .5))}

import run_all_experiments as runner
captured = {}
def profile(frame, event, arg):
    if event == "return" and frame.f_code.co_name == "fast_pred_port_np" and not captured:
        captured["input"] = frame.f_locals["x_ev"].copy()
        captured["tasks"] = frame.f_locals["t_ids"]
        captured["output"] = [(q, pi.copy()) for q, pi in arg]
        sys.setprofile(None)
sys.setprofile(profile)
bench = runner.benchmark_control_loop_timing(cfg, env.flows, env.ports, env.port_flows, env.x0_dict, env.gcl0_dict, "audit", str(ROOT / "results"))
sys.setprofile(None)
predictor = pm.SetTransformerModernTCN()
predictor.load_state_dict(torch.load(ROOT / "checkpoints/predictor_SetTrans_ModernTCN.pt", map_location="cpu", weights_only=True))
predictor.eval()
td = torch.tensor(captured["tasks"], dtype=torch.long)
xx = torch.from_numpy(np.stack([captured["input"] for _ in captured["tasks"]]))
with torch.no_grad():
    qreal, pireal, _ = predictor(xx, td)
qfast = np.array([v[0] for v in captured["output"]])
pifast = np.stack([v[1] for v in captured["output"]])
evidence["timing_predictor_not_equivalent"] = {"tasks": captured["tasks"], "max_q_absolute_diff": float(np.max(np.abs(qfast - qreal.numpy()))), "max_pi_absolute_diff": float(np.max(np.abs(pifast - pireal.numpy()))), "benchmark_total_ms": bench["T_control_total"], "benchmark_line4_ms": bench["T_control_total_Line4"]}

# Public actor action is converted to a baseline-relative absolute target before decoding.
from train_rl_and_baselines import target_norm_to_cur_delta
p = "e3u"
xcur = env.x0_dict[p] + 8_000.
delta = target_norm_to_cur_delta(np.zeros(len(xcur)), xcur, env.x0_dict[p], env.E_dict[p], env.L_dict[p])
xout = core.decode_delta_action(delta, xcur, env.E_dict[p], env.L_dict[p])
evidence["actor_zero_not_keep"] = {"current_minus_baseline_ns": 8000, "output_minus_current_ns_mean": float(np.mean(xout - xcur)), "output_matches_baseline": bool(np.allclose(xout, env.x0_dict[p]))}

evidence["source_unchanged_during_probe"] = all(hashlib.sha256(p.read_bytes()).hexdigest() == manifest[str(p.relative_to(ROOT))]["sha256"] for p in source_paths)
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(evidence, ensure_ascii=False, indent=2))
