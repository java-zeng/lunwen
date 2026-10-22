"""Small diagnostic probes for the captured implementation; does not run or overwrite the experiment suite."""
from pathlib import Path
import sys, json, ast, dataclasses, copy, hashlib
sys.dont_write_bytecode = True
base = Path(__file__).resolve().parent
snap = base / 'snapshot'
sys.path.insert(0, str(snap))
import numpy as np
import yaml
from core_sim import (
    TTFlow, CompiledPortGCL, generate_global_flows, compile_port_gcl,
    local_precheck_port, decode_delta_action, timing_projector_gamma,
    validate_c1_fifo_and_contracts, next_fit_q7_tx_finish,
    _expand_port_instances_np,
)

cfg = yaml.safe_load((snap / 'configs/experiment_config.yaml').read_text(encoding='utf-8'))
results = json.loads((snap / 'results/full_experimental_results.json').read_text(encoding='utf-8'))
runner = (snap / 'run_all_experiments.py').read_text(encoding='utf-8')
tree = ast.parse(runner)
out = {}

assignments = {node.targets[0].id: node for node in ast.walk(tree)
               if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name)}
out['literal_provenance'] = {
    'E0_performance_exactly_matches_literal': ast.literal_eval(assignments['e0_space'].value) == results['E0']['e0_reconfiguration_space'],
    'E1_metrics_exactly_match_literal': ast.literal_eval(assignments['e1_tasks'].value) == results['E1_and_E15']['e1_six_tasks'],
    'E2_audits_exactly_match_literal': ast.literal_eval(assignments['audits'].value) == results['E2_Core_External_Robustness']['audits'],
    'E2_component_ablations_exactly_match_literal': ast.literal_eval(assignments['component_ablations'].value) == results['E2_Core_External_Robustness']['component_ablations'],
}
out['percentile_inconsistencies'] = [
    {'method': name, 'miss_percent': row['miss_mean_pct'], 'p99_ms': row['p99_e2e_ms'], 'deadline_ms': 1.0}
    for name, row in results['E2_Core_External_Robustness']['nine_methods'].items()
    if row['miss_mean_pct'] > 1 and row['p99_e2e_ms'] < 1.0
]

flows = generate_global_flows(cfg)
ports = cfg['topologies']['Topology-v1']['controlled_ports']
pf = {e: [f for f in flows if e in f.path] for e in ports}
x0 = {e: np.array([f.base_offsets_ns[e] for f in pf[e]], dtype=float) for e in ports}
g0 = {e: compile_port_gcl(e, pf[e], x0[e], cfg) for e in ports}
H = cfg['network']['hyperperiod_ns']
GB = cfg['network']['guard_band_aligned_ns']

def canonical_gate_count(flist, x):
    starts, ends, _ = _expand_port_instances_np(flist, x, H)
    pairs = sorted((int(a), int(b)) for a, b in zip(starts, ends) if 0 <= a < b <= H)
    regions = []
    for a, b in pairs:
        if not regions or a - regions[-1][1] > GB:
            regions.append([a, b])
        else:
            regions[-1][1] = max(regions[-1][1], b)
    blocked = [(max(0, a - GB), b) for a, b in regions]
    points = {0, H}
    for a, b in pairs + blocked:
        points.update((a, b))
    points = sorted(points)
    states = []
    for a, b in zip(points, points[1:]):
        m = (a + b) / 2
        state = 64 if any(s <= m < t for s, t in pairs) else (0 if any(s <= m < t for s, t in blocked) else 145)
        if not states or states[-1] != state:
            states.append(state)
    return {'linear_entries': len(states), 'cyclic_merged_entries': len(states) - int(len(states) > 1 and states[0] == states[-1])}

out['workload_and_gate_counts'] = {}
for e in ports:
    out['workload_and_gate_counts'][e] = {
        'flows': len(pf[e]), 'instances_per_H': sum(H // f.period_ns for f in pf[e]),
        'reported_entries': g0[e].num_gcl_entries, **canonical_gate_count(pf[e], x0[e]),
        'empty_q6_subwindow_list': len(g0[e].q6_subwindows) == 0,
        'empty_protected_region_list': len(g0[e].protected_regions) == 0,
    }

f1 = TTFlow(9001, 's', 'd', ['probe'], H, H, 1000, 1020, 8160, 0, {'probe': 20000}, {'probe': 0}, {'probe': H - 8160})
f2 = dataclasses.replace(f1, flow_id=9002, base_offsets_ns={'probe': 32000})
small = compile_port_gcl('probe', [f1, f2], np.array([20000., 32000.]), cfg)
out['small_dark_gap_probe'] = {
    'gap_ns': 32000 - (20000 + 8160),
    'compiler_entries': small.num_gcl_entries,
    **canonical_gate_count([f1, f2], np.array([20000., 32000.])),
}

# An all-Flex GCL with zero TT transmission opportunities must not certify the TT plan.
all_flex = {e: dataclasses.replace(g0[e], q6_subwindows=[], protected_regions=[],
                                 num_gcl_entries=1, total_tt_tx_ns=0, total_gb_ns=0,
                                 total_flex_ns=H, flex_starts_arr=np.array([0.]), flex_ends_arr=np.array([float(H)])) for e in ports}
ok, reason, _ = validate_c1_fifo_and_contracts(flows, ports, x0, all_flex, cfg)
out['all_flex_no_tt_validator_probe'] = {'accepted': bool(ok), 'reason': reason}

rng = np.random.default_rng(20261021)
e = 'e3u'
E = np.array([f.env_early_ns[e] for f in pf[e]], float)
L = np.array([f.env_late_ns[e] for f in pf[e]], float)
_, _, base_order = _expand_port_instances_np(pf[e], x0[e], H)
counts = {'proposals': 200, 'local_precheck_failed': 0, 'passed_with_remaining_overlap': 0, 'passed_with_instance_order_changed': 0, 'raw_preview_entry_overflow': 0}
for _ in range(200):
    delta = rng.uniform(-1, 1, len(pf[e]))
    xhat = decode_delta_action(delta, x0[e], E, L)
    # The ordinary post-fallback statistic in E0 discards overflow candidates.
    xp, gp, passed, why = local_precheck_port(e, pf[e], xhat, x0[e], cfg)
    if not passed:
        counts['local_precheck_failed'] += 1
        if why == 'GCL_ENTRY_LIMIT_EXCEEDED': counts['raw_preview_entry_overflow'] += 1
    if passed:
        st, en, order = _expand_port_instances_np(pf[e], xp, H)
        counts['passed_with_remaining_overlap'] += int(np.any(st[1:] < en[:-1]))
        counts['passed_with_instance_order_changed'] += int(not np.array_equal(order, base_order))
out['local_precheck_probes'] = counts

# A closed gate must not allow a queued frame to transmit.
closed = CompiledPortGCL('closed', [], [], [], 1, 0, 0, 0, 0, np.array([], float), np.array([], float))
out['closed_gate_nextfit_probe'] = {'returned_finish_ns': next_fit_q7_tx_finish(100., 0., 100., 0., closed, H), 'expected': 'no finite service opportunity'}

out['declared_BE_minimum_events_per_8ms'] = {
    e: cfg['be_traffic']['nominal_offered_rate_bps'][e] * (H * 1e-9) / ((1518 + 20) * 8)
    for e in ports
}

import torch
from predictor_mapper import SetTransformerModernTCN, run_service_mapper_port
torch.set_num_threads(1)
torch.manual_seed(17)
net = SetTransformerModernTCN().eval()
original = torch.randn(2, 4, 16, 5)
changed = original.clone()
changed[:, :2] = torch.randn_like(changed[:, :2]) * 20
with torch.no_grad():
    q1, p1, _ = net(original, torch.tensor([0, 1]))
    q2, p2, _ = net(changed, torch.tensor([0, 1]))
out['oldest_16ms_ignored_by_predictor'] = {'q_max_abs_diff': float((q1 - q2).abs().max()), 'pi_max_abs_diff': float((p1 - p2).abs().max())}

pi = np.zeros(16); pi[4] = 1
args = dict(cycle_k=0, upstream_port='e3u', local_preds={'A': (.8, pi)}, gcl_cur=None,
            gcl_pre=g0['e3u'], residual_block_dist_ns=np.array([0., 3000.]), cfg=cfg,
            candidate_id='probe', config_hash='audit')
a = run_service_mapper_port(carry_in_backlog_count=0, **args)[('A', 'e3u', 'e01')]
b = run_service_mapper_port(carry_in_backlog_count=1, **args)[('A', 'e3u', 'e01')]
args['upstream_port'] = 'e12'
line = run_service_mapper_port(carry_in_backlog_count=0, **args)
out['mapper_coverage_probes'] = {'accepts_none_gcl_cur': True, 'one_carry_in_changes_q_or_pi': bool(a.q_prop != b.q_prop or not np.array_equal(a.pi_prop, b.pi_prop)), 'line_port_e12_message_count': len(line)}
out['message_arrays_memory_bytes'] = int(a.pi_prop.nbytes + a.bounds_low.nbytes + a.bounds_high.nbytes)

(base / 'evidence.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(out, ensure_ascii=True, indent=2))
