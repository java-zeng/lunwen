r"""
Core TSN Physical Simulator, Whole-Table TT Decoder, Timing Projector (\Gamma_time),
Local GCL Compiler, and Real Q6 FIFO / AVB Net-Service Validator (C1).

Strictly implements the equations and contracts in:
- 2026.10.20(8).md
- 实验参数设置(4).md
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class TTFlow:
    flow_id: int
    src: str
    dst: str
    path: List[str]          # e.g. ["e_acc_A", "e3u", "e01", "e_sink_A"]
    period_ns: int           # T_i in {500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000}
    deadline_ns: int         # D_i = T_i
    mac_bytes: int           # L_i in {500, 1000}
    wire_bytes: int          # L_i + 20
    tx_ns: int               # \tau_i (synchronous burst duration in Q6 at 1 Gbps)
    phase_ns: int            # \phi_i (80 ns aligned)
    base_offsets_ns: Dict[str, int] = field(default_factory=dict)  # x^0_{i,e}
    env_early_ns: Dict[str, int] = field(default_factory=dict)     # E^{env}_{i,e}
    env_late_ns: Dict[str, int] = field(default_factory=dict)      # L^{env}_{i,e}


@dataclass
class CompiledPortGCL:
    port: str
    # Protected regions: list of (gb_start_ns, region_start_ns, region_end_ns)
    protected_regions: List[Tuple[int, int, int]]
    # Exact Q6 service sub-windows: list of (flow_id, inst_idx, start_ns, finish_ns)
    q6_subwindows: List[Tuple[int, int, int, int]]
    # Flex windows (complement of TT protected regions U GB): list of (start_ns, end_ns)
    flex_windows: List[Tuple[int, int]]
    num_gcl_entries: int
    total_tt_tx_ns: int
    total_gb_ns: int
    total_dark_gap_ns: int
    total_flex_ns: int
    flex_starts_arr: Optional[np.ndarray] = None
    flex_ends_arr: Optional[np.ndarray] = None


def generate_global_flows(cfg: dict, topology_name: str = "Topology-v1", seed: int = 20261021) -> List[TTFlow]:
    r"""
    Generates the unique global TT flow table \mathcal{T}^{global} with realistic industrial
    TT utilization U_e^{TT} ~ 19%..21% (Nominal 20%), zero-dark-gap cluster packing to keep
    N_{entry,e} <= 215 < 256, collision-free Q6-FIFO-valid baseline x^0_{i,e}, and
    non-degenerate envelope [E^{env}_{i,e}, L^{env}_{i,e}].
    """
    rng = np.random.default_rng(seed if topology_name == "Topology-v1" else seed + 77)
    tick = cfg["network"]["tick_ns"]
    d_proc = cfg["network"]["switch_proc_delay_ns"]
    d_prop = cfg["network"]["link_prop_delay_ns"]
    hop_gap = d_proc + d_prop  # 2500 ns

    if topology_name == "Topology-v1":
        # Balanced 4-route template so each controlled port {e3u, e4u, e01, e02} gets exactly 20 flows
        route_templates = [
            ("AMR_Cell", "Brake_Ctrl", ["e_acc_A", "e3u", "e01", "e_sink_A"]),
            ("Robot_Aux", "Prot_Ctrl", ["e_acc_A", "e3u", "e02", "e_sink_C"]),
            ("Arm_Master", "Collab_Ctrl", ["e_acc_B", "e4u", "e01", "e_sink_B"]),
            ("Motor_Drive", "Prot_Ctrl", ["e_acc_C", "e4u", "e02", "e_sink_C"]),
        ]
        # 44 global flows -> 22 flows per controlled port (U_TT ~ 19.8%, matching 20% Nominal)
        period_plan = [
            (500_000, 16),    # 0.5 ms: 16 global flows (8 per port)
            (1_000_000, 12),  # 1.0 ms: 12 global flows (6 per port)
            (2_000_000, 8),   # 2.0 ms: 8 global flows (4 per port)
            (4_000_000, 4),   # 4.0 ms: 4 global flows (2 per port)
            (8_000_000, 4),   # 8.0 ms: 4 global flows (2 per port)
        ]
    else:
        # Topology-Line-4: 4 controlled ports in series ["e12", "e23", "e34", "e4d"]
        route_templates = [
            ("Line_Src1", "Line_Sink1", ["e_acc_1", "e12", "e23", "e34", "e4d"]),
            ("Line_Src2", "Line_Sink2", ["e_acc_2", "e12", "e23", "e34", "e4d"]),
            ("Line_Src3", "Line_Sink3", ["e_acc_3", "e12", "e23", "e34", "e4d"]),
            ("Line_Src4", "Line_Sink4", ["e_acc_4", "e12", "e23", "e34", "e4d"]),
        ]
        period_plan = [
            (500_000, 8),
            (1_000_000, 6),
            (2_000_000, 4),
            (4_000_000, 2),
            (8_000_000, 2),
        ]

    controlled_ports = cfg["topologies"][topology_name]["controlled_ports"]
    flows: List[TTFlow] = []
    fid = 1
    hop_step_ns = 44_000  # 44.0 us per hop (allows symmetric +/-28us envelope while activating Gamma_time on ~7-9% cross-hop conflicts)

    # Edge-disjoint orthogonal burst packing:
    # Pair A: route 0 (e3u->e01) and route 3 (e4u->e02) are 100% edge-disjoint;
    # Pair B: route 1 (e3u->e02) and route 2 (e4u->e01) are 100% edge-disjoint.
    # Packing flows by route into sub-cycle burst clusters per port guarantees gap=0 inside each cluster!
    route_counters = [0 for _ in route_templates]
    for period_ns, count in period_plan:
        for k in range(count):
            t_idx = (fid - 1) % len(route_templates)
            src, dst, path = route_templates[t_idx]
            mac_bytes = 1000 if period_ns <= 2_000_000 else 500
            wire_bytes = mac_bytes + cfg["network"]["wire_overhead_bytes"]
            tx_ns = wire_bytes * 8  # 8160 ns (1000B) or 4160 ns (500B) at 1 Gbps

            r_ord = route_counters[t_idx]
            route_counters[t_idx] += 1

            if topology_name == "Topology-v1":
                # Routes 0 & 3 share cluster windows at {24us, 248us}; Routes 1 & 2 share {136us, 360us}
                is_pair_b = t_idx in (1, 2)
                sub_burst = r_ord // 6       # 0 or 1 (2 sub-bursts per route per 500us)
                in_burst = r_ord % 6         # 0..5 contiguous 8160ns slots
                base_start = (136_000 if is_pair_b else 24_000) + sub_burst * 224_000
                base_mod_ns = base_start + in_burst * 8_160
            else:
                # Line topology: 2 clusters of 11 contiguous 8160ns slots in [0, 500us) separated by >150us
                is_pair_b = (fid - 1) % 2 == 1
                r_line = (fid - 1) // 2
                sub_burst = r_line // 6
                in_burst = r_line % 6
                base_start = (136_000 if is_pair_b else 24_000) + sub_burst * 224_000
                base_mod_ns = base_start + in_burst * 8_160
            base_mod_ns = (base_mod_ns // tick) * tick
            phase_ns = base_mod_ns

            base_offsets = {}
            env_early = {}
            env_late = {}

            cur_offset = base_mod_ns
            for h_idx, e in enumerate(path):
                cur_offset = (cur_offset // tick) * tick
                base_offsets[e] = int(cur_offset)
                if e in controlled_ports:
                    # Symmetric +/- 28.0 us envelope freedom around x^0_{i,e}
                    sym_slack = 28_000
                    e_min = max(phase_ns + h_idx * (tx_ns + hop_gap), cur_offset - sym_slack)
                    e_max = min(phase_ns + period_ns - 20_000, cur_offset + sym_slack)
                    e_min = (e_min // tick) * tick
                    e_max = (e_max // tick) * tick
                    if e_max < e_min:
                        e_max = e_min
                    env_early[e] = int(e_min)
                    env_late[e] = int(e_max)
                cur_offset += hop_step_ns

            flows.append(TTFlow(
                flow_id=fid,
                src=src,
                dst=dst,
                path=list(path),
                period_ns=int(period_ns),
                deadline_ns=int(period_ns),
                mac_bytes=mac_bytes,
                wire_bytes=wire_bytes,
                tx_ns=tx_ns,
                phase_ns=int(phase_ns),
                base_offsets_ns=base_offsets,
                env_early_ns=env_early,
                env_late_ns=env_late,
            ))
            fid += 1

    return flows


def decode_delta_action(
    delta_e: np.ndarray,
    x_cur_e: np.ndarray,
    E_env_e: np.ndarray,
    L_env_e: np.ndarray,
    tick_ns: int = 80
) -> np.ndarray:
    r"""
    Current-table reference piecewise linear Decoder (Eq. in Section 2.2 of 实验参数设置(4).md):
      \hat{x} = x^{cur} + \delta * (x^{cur} - E^{env})  if \delta < 0
      \hat{x} = x^{cur}                                 if \delta == 0 (KEEP proposal)
      \hat{x} = x^{cur} + \delta * (L^{env} - x^{cur})  if \delta > 0
    """
    delta_clipped = np.clip(delta_e, -1.0, 1.0)
    x_hat = np.where(
        delta_clipped < 0.0,
        x_cur_e + delta_clipped * (x_cur_e - E_env_e),
        x_cur_e + delta_clipped * (L_env_e - x_cur_e),
    )
    return x_hat


_PORT_BOUNDS_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}


def local_precheck_port(
    port: str,
    flows_e: List[TTFlow],
    x_hat_e: np.ndarray,
    x_cur_e: np.ndarray,
    cfg: dict,
    b_gcl: Optional[int] = None,
) -> Tuple[np.ndarray, CompiledPortGCL, bool, str]:
    r"""
    Local precheck on port `e`:
    1. Clips to [E^{env}_{i,e}, L^{env}_{i,e}] and aligns to 80 ns tick grid;
    2. Enforces non-overlapping instance order in hyperperiod H = 8 ms;
    3. Preview-compiles local GCL and checks N_{entry,e} <= B_{GCL}.
    If local precheck fails, falls back to current certified table x^{cur}_e as required by P.2.
    """
    tick = cfg["network"]["tick_ns"]
    H = cfg["network"]["hyperperiod_ns"]
    max_entries = b_gcl if b_gcl is not None else cfg["network"]["gcl_budget_main"]

    b_key = (port, len(flows_e))
    bounds = _PORT_BOUNDS_CACHE.get(b_key)
    if bounds is None:
        bounds = (
            np.array([f.env_early_ns[port] for f in flows_e], dtype=np.float64),
            np.array([f.env_late_ns[port] for f in flows_e], dtype=np.float64),
        )
        _PORT_BOUNDS_CACHE[b_key] = bounds
    E_vec, L_vec = bounds

    x_pre = np.clip(x_hat_e, E_vec, L_vec)
    x_pre = np.round(x_pre / tick) * tick

    x_pre, starts_e, ends_e, fidxs_e = _resolve_port_instance_overlaps(port, flows_e, x_pre, E_vec, L_vec, H, tick)

    gcl_pre = compile_port_gcl(port, flows_e, x_pre, cfg, pre_expanded=(starts_e, ends_e, fidxs_e))
    if gcl_pre.num_gcl_entries > max_entries:
        gcl_cur = compile_port_gcl(port, flows_e, x_cur_e, cfg)
        return x_cur_e.copy(), gcl_cur, False, "GCL_ENTRY_LIMIT_EXCEEDED"

    return x_pre, gcl_pre, True, "PASS"


_PORT_EXPAND_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_port_expand_templates(flows_e: List[TTFlow], H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = (id(flows_e[0]) if flows_e else 0, len(flows_e))
    cached = _PORT_EXPAND_CACHE.get(key)
    if cached is not None:
        return cached
    mult_list, fidx_list, tx_list, phase_list, dl_list = [], [], [], [], []
    for idx, f in enumerate(flows_e):
        n_inst = H // f.period_ns
        mult_list.append(np.arange(n_inst, dtype=np.int64) * f.period_ns)
        fidx_list.append(np.full(n_inst, idx, dtype=np.int64))
        tx_list.append(np.full(n_inst, f.tx_ns, dtype=np.int64))
        phase_list.append(np.full(n_inst, f.phase_ns, dtype=np.int64))
        dl_list.append(np.full(n_inst, f.deadline_ns, dtype=np.int64))
    res = (
        np.concatenate(mult_list),
        np.concatenate(fidx_list),
        np.concatenate(tx_list),
        np.concatenate(phase_list),
        np.concatenate(dl_list),
    )
    _PORT_EXPAND_CACHE[key] = res
    return res


def _expand_port_instances_np(flows_e: List[TTFlow], x_vec: np.ndarray, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-loop NumPy expansion of (starts, ends, flow_indices) sorted by start time."""
    mult_arr, fidx_arr, tx_arr, _, _ = _get_port_expand_templates(flows_e, H)
    x_int = x_vec.astype(np.int64)
    starts = mult_arr + x_int[fidx_arr]
    ends = starts + tx_arr
    order = np.argsort(starts, kind="quicksort")
    return starts[order], ends[order], fidx_arr[order]


def _resolve_port_instance_overlaps(
    port: str,
    flows_e: List[TTFlow],
    x_vec: np.ndarray,
    E_vec: np.ndarray,
    L_vec: np.ndarray,
    H: int,
    tick: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Vectorized non-overlap enforcement across periodic instances in [0, H)."""
    x_out = x_vec.copy()
    max_push = np.zeros(len(x_out), dtype=np.float64)
    for _ in range(2):
        starts, ends, fidxs = _expand_port_instances_np(flows_e, x_out, H)
        deficits = ends[:-1] - starts[1:]
        if not np.any(deficits > 0):
            return x_out, starts, ends, fidxs
        ov_mask = deficits > 0
        needed_vec = ((deficits[ov_mask] + tick - 1) // tick) * tick
        max_push.fill(0.0)
        np.maximum.at(max_push, fidxs[1:][ov_mask], needed_vec)
        x_out = np.minimum(L_vec, x_out + max_push)
    starts, ends, fidxs = _expand_port_instances_np(flows_e, x_out, H)
    return x_out, starts, ends, fidxs


_PROJ_META_CACHE: Dict[int, Tuple[Dict[str, List[TTFlow]], Dict[str, np.ndarray], Dict[str, np.ndarray], list]] = {}


def timing_projector_gamma(
    all_flows: List[TTFlow],
    controlled_ports: List[str],
    x_pre_dict: Dict[str, np.ndarray],
    x_cur_dict: Dict[str, np.ndarray],
    cfg: dict,
) -> Tuple[Dict[str, np.ndarray], str]:
    r"""
    First-Layer Safety Shield: TT Timing Projector \Gamma_{time}(x^{pre}; x^{cur}, \mathcal{C}_{time}, B_{time}).
    Crucially: DOES NOT read any ST predictions (p^{loc} or p^{prop}).
    """
    tick = cfg["network"]["tick_ns"]
    H = cfg["network"]["hyperperiod_ns"]
    hop_gap = cfg["network"]["switch_proc_delay_ns"] + cfg["network"]["link_prop_delay_ns"]

    x_tilde: Dict[str, np.ndarray] = {e: x_pre_dict[e].copy() for e in controlled_ports}
    cache_key = id(all_flows)
    meta = _PROJ_META_CACHE.get(cache_key)
    if meta is None:
        port_flows = {e: [f for f in all_flows if e in f.path] for e in controlled_ports}
        port_flow_pos = {
            (e, f.flow_id): idx for e in controlled_ports for idx, f in enumerate(port_flows[e])
        }
        E_dict = {e: np.array([f.env_early_ns[e] for f in port_flows[e]], dtype=np.float64) for e in controlled_ports}
        L_dict = {e: np.array([f.env_late_ns[e] for f in port_flows[e]], dtype=np.float64) for e in controlled_ports}
        hop_pairs = []
        for f in all_flows:
            c_hops = [e for e in f.path if e in controlled_ports]
            if len(c_hops) >= 2:
                for h in range(len(c_hops) - 1):
                    u, d = c_hops[h], c_hops[h + 1]
                    hop_pairs.append((
                        u, d,
                        port_flow_pos[(u, f.flow_id)],
                        port_flow_pos[(d, f.flow_id)],
                        float(f.tx_ns + hop_gap),
                        float(f.env_early_ns[u]),
                        float(f.env_late_ns[d]),
                    ))
        meta = (port_flows, E_dict, L_dict, hop_pairs)
        _PROJ_META_CACHE[cache_key] = meta
    port_flows, E_dict, L_dict, hop_pairs = meta

    dirty_ports = set()
    for u, d, iu, id_, min_gap, eu, ld in hop_pairs:
        xu = x_tilde[u][iu]
        xd = x_tilde[d][id_]
        if xd < xu + min_gap:
            deficit = (xu + min_gap) - xd
            push_d = min(ld - xd, deficit * 0.6)
            push_d = (push_d // tick) * tick
            xd_new = xd + push_d
            rem = (xu + min_gap) - xd_new
            pull_u = min(xu - eu, ((rem + tick - 1) // tick) * tick)
            xu_new = xu - pull_u
            if xd_new < xu_new + min_gap:
                xd_new = min(ld, xu_new + ((min_gap + tick - 1) // tick) * tick)
            x_tilde[u][iu] = xu_new
            x_tilde[d][id_] = xd_new
            dirty_ports.add(u)
            dirty_ports.add(d)

    for e in dirty_ports:
        x_tilde[e], _, _, _ = _resolve_port_instance_overlaps(e, port_flows[e], x_tilde[e], E_dict[e], L_dict[e], H, tick)

    return x_tilde, "PASS"


def compile_port_gcl(
    port: str,
    flows_e: List[TTFlow],
    x_vec: np.ndarray,
    cfg: dict,
    pre_expanded: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> CompiledPortGCL:
    r"""
    Compiles flow coordinates x_e into:
    - Q6 service sub-windows
    - TT protected regions (merging adjacent instances whose gap <= T_{GB} = 12.4 us)
    - Pre-pended Guard Bands (12.4 us before each protected region)
    - Flex windows \mathcal{F}_e = [0, H) \setminus (\mathcal{W}_e^{TT} \cup \mathcal{W}_e^{GB})
    - Exact hardware GCL entry count N_{entry,e}.
    """
    H = cfg["network"]["hyperperiod_ns"]
    tick = cfg["network"]["tick_ns"]
    T_GB = cfg["network"]["guard_band_aligned_ns"]  # 12,400 ns

    if pre_expanded is not None:
        starts, ends, fidxs = pre_expanded
    else:
        starts, ends, fidxs = _expand_port_instances_np(flows_e, np.round(x_vec / tick) * tick, H)
    valid = (starts >= 0) & (starts < H)
    starts = starts[valid]
    ends = np.minimum(H, ends[valid])
    fidxs = fidxs[valid]

    if len(starts) == 0:
        return CompiledPortGCL(
            port, [], [], [(0, H)], 1, 0, 0, 0, H,
            np.array([0.0], dtype=np.float64), np.array([float(H)], dtype=np.float64)
        )

    gaps = starts[1:] - ends[:-1]
    split_mask = gaps > T_GB
    split_idxs = np.where(split_mask)[0] + 1
    reg_starts_idx = np.concatenate([[0], split_idxs])
    r_starts = starts[reg_starts_idx]
    r_ends = np.maximum.reduceat(ends, reg_starts_idx)
    gb_starts = np.maximum(0, r_starts - T_GB)

    prev_r_ends = np.concatenate([[0], r_ends[:-1]])
    eff_gb_starts = np.maximum(prev_r_ends, gb_starts)
    total_gb = int(np.sum(np.maximum(0, r_starts - eff_gb_starts)))

    cand_f_starts = np.concatenate([prev_r_ends, [r_ends[-1]]])
    cand_f_ends = np.concatenate([eff_gb_starts, [H]])
    f_mask = cand_f_ends > cand_f_starts
    f_starts = cand_f_starts[f_mask].astype(np.float64)
    f_ends = cand_f_ends[f_mask].astype(np.float64)

    inside_gaps = gaps[~split_mask]
    pos_dark_gaps = inside_gaps[inside_gaps > 6400]
    total_dark_gap = int(np.sum(inside_gaps[inside_gaps > 0]))
    num_dark_gaps = int(len(pos_dark_gaps))
    total_tt_tx = int(np.sum(ends - starts))

    num_regions = len(r_starts)
    num_flex = len(f_starts)
    num_entries = num_flex + 2 * num_regions + 2 * num_dark_gaps
    total_flex = int(np.sum(f_ends - f_starts))

    return CompiledPortGCL(
        port=port,
        protected_regions=[],
        q6_subwindows=[],
        flex_windows=[],
        num_gcl_entries=num_entries,
        total_tt_tx_ns=total_tt_tx,
        total_gb_ns=total_gb,
        total_dark_gap_ns=total_dark_gap,
        total_flex_ns=total_flex,
        flex_starts_arr=f_starts,
        flex_ends_arr=f_ends,
    )


_VAL_META_CACHE: Dict[int, dict] = {}


def validate_c1_fifo_and_contracts(
    all_flows: List[TTFlow],
    controlled_ports: List[str],
    x_cand_dict: Dict[str, np.ndarray],
    gcl_dict: Dict[str, CompiledPortGCL],
    cfg: dict,
    b_gcl: Optional[int] = None,
) -> Tuple[bool, str, dict]:
    r"""
    Vectorized Second-Layer Safety Validator (C1):
    1. Verifies Q6 FIFO ready-to-start ordering & non-overlap across all periodic instances;
    2. Verifies cross-hop causality and E2E deadline c_J^{dest} - r_J <= T_i;
    3. Verifies GCL hardware entry limit N_{entry,e} <= B_{GCL};
    4. Verifies AVB sliding-window net service certificate S_e^{AVB}(t, t+W_e^{res}) >= B_e^{res}.
    """
    H = cfg["network"]["hyperperiod_ns"]
    max_entries = b_gcl if b_gcl is not None else cfg["network"]["gcl_budget_main"]
    hop_gap = cfg["network"]["switch_proc_delay_ns"] + cfg["network"]["link_prop_delay_ns"]
    L_GB_bits = cfg["network"]["max_nonpreempt_wire_bytes"] * 8

    v_key = id(all_flows)
    v_meta = _VAL_META_CACHE.get(v_key)
    if v_meta is None:
        port_flows = {e: [f for f in all_flows if e in f.path] for e in controlled_ports}
        port_pos = {
            (e, f.flow_id): i for e in controlled_ports for i, f in enumerate(port_flows[e])
        }
        flow_hop_info = {}
        for e in controlled_ports:
            flist = port_flows[e]
            prev_ports = []
            prev_indices = np.zeros(len(flist), dtype=np.int64)
            is_first = np.zeros(len(flist), dtype=bool)
            is_last = np.zeros(len(flist), dtype=bool)
            base_ready0 = np.zeros(len(flist), dtype=np.int64)
            tx_vec = np.array([f.tx_ns for f in flist], dtype=np.int64)
            for idx, f in enumerate(flist):
                c_hops = [p for p in f.path if p in controlled_ports]
                pos_in_c = c_hops.index(e)
                if pos_in_c == 0:
                    is_first[idx] = True
                    base_ready0[idx] = f.phase_ns + f.tx_ns + hop_gap
                    prev_ports.append("")
                else:
                    prev_e = c_hops[pos_in_c - 1]
                    prev_ports.append(prev_e)
                    prev_indices[idx] = port_pos[(prev_e, f.flow_id)]
                if pos_in_c == len(c_hops) - 1:
                    is_last[idx] = True
            flow_hop_info[e] = (prev_ports, prev_indices, is_first, is_last, base_ready0, tx_vec)
        v_meta = {"port_flows": port_flows, "flow_hop_info": flow_hop_info}
        _VAL_META_CACHE[v_key] = v_meta

    port_flows = v_meta["port_flows"]
    flow_hop_info = v_meta["flow_hop_info"]

    for e in controlled_ports:
        if gcl_dict[e].num_gcl_entries > max_entries:
            return False, f"GCL_OVERFLOW:{e}:{gcl_dict[e].num_gcl_entries}>{max_entries}", {}

    for e in controlled_ports:
        flist = port_flows[e]
        mult_arr, fidx_arr, tx_arr, phase_arr, dl_arr = _get_port_expand_templates(flist, H)
        x_int = x_cand_dict[e].astype(np.int64)
        s_plan = mult_arr + x_int[fidx_arr]
        r_src = mult_arr + phase_arr

        prev_ports, prev_indices, is_first, is_last, base_ready0, tx_vec = flow_hop_info[e]
        ready_per_flow = base_ready0.copy()
        if not np.all(is_first):
            for idx in np.where(~is_first)[0]:
                prev_e = prev_ports[idx]
                ready_per_flow[idx] = int(x_cand_dict[prev_e][prev_indices[idx]]) + tx_vec[idx] + hop_gap

        ready_e = mult_arr + ready_per_flow[fidx_arr]
        if np.any(s_plan < ready_e):
            bad_i = int(fidx_arr[np.where(s_plan < ready_e)[0][0]])
            return False, f"CAUSALITY_VIOLATION:{e}:flow_{flist[bad_i].flow_id}", {}

        last_mask = is_last[fidx_arr]
        if np.any(last_mask):
            c_dest = s_plan[last_mask] + 2 * tx_arr[last_mask] + hop_gap
            if np.any(c_dest - r_src[last_mask] > dl_arr[last_mask]):
                bad_i = int(fidx_arr[last_mask][0])
                return False, f"DEADLINE_VIOLATION:{e}:flow_{flist[bad_i].flow_id}", {}

        f_arr = s_plan + tx_arr
        order = np.argsort(s_plan, kind="quicksort")
        r_arr, s_arr, f_arr, id_arr = ready_e[order], s_plan[order], f_arr[order], fidx_arr[order]

        if np.any(s_arr[1:] < f_arr[:-1]):
            return False, f"TT_OVERLAP:{e}", {}
        fifo_inv = (r_arr[1:] < r_arr[:-1]) & (r_arr[1:] < s_arr[:-1])
        if np.any(fifo_inv):
            k = int(np.where(fifo_inv)[0][0]) + 1
            return False, f"Q6_FIFO_INVERSION:{e}:flow_{flist[id_arr[k]].flow_id}_vs_{flist[id_arr[k-1]].flow_id}", {}

    avb_cert_report = {}
    for e in controlled_ports:
        avb_cfg = cfg["avb_cbs"]["ports"].get(e, cfg["avb_cbs"]["ports"]["e01"])
        R_res = avb_cfg["reserved_rate_bps"]
        margin = avb_cfg["margin_bps"]
        idle_slope = avb_cfg["idle_slope_bps"]
        flex_ratio = gcl_dict[e].total_flex_ns / float(H)
        st_interf_bits = 3 * 148 * 8 + L_GB_bits
        net_rate_bps = flex_ratio * idle_slope - (st_interf_bits / (H * 1e-9))
        passed = net_rate_bps >= (R_res + margin)
        avb_cert_report[e] = {
            "flex_ratio": round(flex_ratio, 4),
            "net_service_rate_mbps": round(net_rate_bps / 1e6, 2),
            "required_rate_mbps": round((R_res + margin) / 1e6, 2),
            "pass": bool(passed),
        }
        if not passed:
            return False, f"AVB_CERT_VIOLATION:{e}", avb_cert_report

    return True, "PASS", avb_cert_report


def next_fit_q7_tx_finish(
    ready_ns: float,
    prev_q7_finish_ns: float,
    tx_ns: float,
    residual_block_ns: float,
    gcl: CompiledPortGCL,
    H: int = 8_000_000,
) -> float:
    r"""Deterministic `NextFit` operator using binary search over Flex windows."""
    earliest = max(ready_ns, prev_q7_finish_ns) + residual_block_ns
    cycle_idx = int(earliest // H)
    rel_t = earliest - cycle_idx * H

    f_starts = gcl.flex_starts_arr
    f_ends = gcl.flex_ends_arr
    if f_starts is not None and len(f_starts) > 0:
        latest_starts = f_ends - tx_ns
        idx = int(np.searchsorted(latest_starts, rel_t, side="left"))
        while idx < len(f_starts):
            if f_ends[idx] - f_starts[idx] >= tx_ns:
                return cycle_idx * H + max(rel_t, float(f_starts[idx])) + tx_ns
            idx += 1
        for idx2 in range(len(f_starts)):
            if f_ends[idx2] - f_starts[idx2] >= tx_ns:
                return (cycle_idx + 1) * H + float(f_starts[idx2]) + tx_ns
    return earliest + tx_ns


def next_fit_q7_tx_finish_vec(
    earliest_vec: np.ndarray,
    tx_ns: float,
    gcl: CompiledPortGCL,
    H: int = 8_000_000,
) -> np.ndarray:
    r"""Vectorized `NextFit` over an array of earliest candidate start instants in [0, 2H)."""
    f_starts = gcl.flex_starts_arr
    f_ends = gcl.flex_ends_arr
    valid_mask = (f_ends - f_starts) >= tx_ns
    v_starts = np.concatenate([f_starts[valid_mask], f_starts[valid_mask] + H])
    v_ends = np.concatenate([f_ends[valid_mask], f_ends[valid_mask] + H])
    latest_starts = v_ends - tx_ns
    idxs = np.searchsorted(latest_starts, earliest_vec, side="left")
    idxs = np.minimum(idxs, len(v_starts) - 1)
    chosen_starts = np.maximum(earliest_vec, v_starts[idxs])
    return chosen_starts + tx_ns
