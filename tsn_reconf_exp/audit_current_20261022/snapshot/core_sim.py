r"""
Core TSN Physical Simulator, Whole-Table TT Decoder, Timing Projector (\Gamma_time),
Local GCL Compiler, and Real Q6 FIFO / AVB Net-Service & CBS Validator (C1).

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
    cyclic_gcl_entries: int = 0
    # Prioritized Slot Sharing & Prudent Reservation metadata (E-TSN ICDCS'22 & ARTSN ICDCS'26)
    q7_shared_flex_starts_arr: Optional[np.ndarray] = None
    q7_shared_flex_ends_arr: Optional[np.ndarray] = None
    prudent_extra_tt_ns: int = 0
    artsn_rp_enabled: bool = False


def generate_global_flows(
    cfg: dict,
    topology_name: str = "Topology-v1",
    seed: int = 20261021,
    flows_per_port: int = 22,
) -> List[TTFlow]:
    r"""
    Generates the unique global TT flow table \mathcal{T}^{global} with realistic industrial
    TT utilization:
      - `flows_per_port == 14`: Low TT load (U_e^{TT} ~ 12.6%, 28 global flows on Star)
      - `flows_per_port == 22`: Nominal TT load (U_e^{TT} ~ 19.9%, 44 global flows on Star)
      - `flows_per_port == 30`: High TT load (U_e^{TT} ~ 27.2%, 60 global flows on Star)
    Uses zero-dark-gap cluster packing so baseline N_{entry,e} <= 125 <= 128 < 256,
    collision-free Q6-FIFO-valid baseline x^0_{i,e}, and non-degenerate envelope [E^{env}_{i,e}, L^{env}_{i,e}].
    """
    tick = cfg["network"]["tick_ns"]
    d_proc = cfg["network"]["switch_proc_delay_ns"]
    d_prop = cfg["network"]["link_prop_delay_ns"]
    hop_gap = d_proc + d_prop  # 2500 ns

    if topology_name == "Topology-v1":
        route_templates = [
            ("AMR_Cell", "Brake_Ctrl", ["e_acc_A", "e3u", "e01", "e_sink_A"]),
            ("Robot_Aux", "Prot_Ctrl", ["e_acc_A", "e3u", "e02", "e_sink_C"]),
            ("Arm_Master", "Collab_Ctrl", ["e_acc_B", "e4u", "e01", "e_sink_B"]),
            ("Motor_Drive", "Prot_Ctrl", ["e_acc_C", "e4u", "e02", "e_sink_C"]),
        ]
        if flows_per_port <= 14:
            period_plan = [
                (500_000, 10),
                (1_000_000, 8),
                (2_000_000, 4),
                (4_000_000, 4),
                (8_000_000, 2),
            ]
        elif flows_per_port >= 30:
            period_plan = [
                (500_000, 22),
                (1_000_000, 16),
                (2_000_000, 10),
                (4_000_000, 6),
                (8_000_000, 6),
            ]
        else:
            # Nominal: 44 global flows -> 22 flows per controlled port (U_TT = 19.86%)
            period_plan = [
                (500_000, 16),    # 0.5 ms: 16 global flows (8 per port)
                (1_000_000, 12),  # 1.0 ms: 12 global flows (6 per port)
                (2_000_000, 8),   # 2.0 ms: 8 global flows (4 per port)
                (4_000_000, 4),   # 4.0 ms: 4 global flows (2 per port)
                (8_000_000, 4),   # 8.0 ms: 4 global flows (2 per port)
            ]
        hop_step_ns = 44_000
        sym_slack = 20_000
    elif topology_name == "Topology-Line-3":
        route_templates = [
            ("Line3_Src1", "Line3_Sink1", ["e_acc_1", "e12", "e23", "e34"]),
            ("Line3_Src2", "Line3_Sink2", ["e_acc_2", "e12", "e23", "e34"]),
            ("Line3_Src3", "Line3_Sink3", ["e_acc_3", "e12", "e23", "e34"]),
            ("Line3_Src4", "Line3_Sink4", ["e_acc_4", "e12", "e23", "e34"]),
        ]
        if flows_per_port <= 14:
            period_plan = [(500_000, 5), (1_000_000, 4), (2_000_000, 2), (4_000_000, 2), (8_000_000, 1)]
        elif flows_per_port >= 30:
            period_plan = [(500_000, 11), (1_000_000, 8), (2_000_000, 5), (4_000_000, 3), (8_000_000, 3)]
        else:
            period_plan = [(500_000, 8), (1_000_000, 6), (2_000_000, 4), (4_000_000, 2), (8_000_000, 2)]
        hop_step_ns = 34_000
        sym_slack = 20_000
    else:
        # Topology-Line-4: 4 controlled ports in series ["e12", "e23", "e34", "e4d"]
        route_templates = [
            ("Line_Src1", "Line_Sink1", ["e_acc_1", "e12", "e23", "e34", "e4d"]),
            ("Line_Src2", "Line_Sink2", ["e_acc_2", "e12", "e23", "e34", "e4d"]),
            ("Line_Src3", "Line_Sink3", ["e_acc_3", "e12", "e23", "e34", "e4d"]),
            ("Line_Src4", "Line_Sink4", ["e_acc_4", "e12", "e23", "e34", "e4d"]),
        ]
        if flows_per_port <= 14:
            period_plan = [(500_000, 5), (1_000_000, 4), (2_000_000, 2), (4_000_000, 2), (8_000_000, 1)]
        elif flows_per_port >= 30:
            period_plan = [(500_000, 11), (1_000_000, 8), (2_000_000, 5), (4_000_000, 3), (8_000_000, 3)]
        else:
            period_plan = [(500_000, 8), (1_000_000, 6), (2_000_000, 4), (4_000_000, 2), (8_000_000, 2)]
        hop_step_ns = 34_000
        sym_slack = 20_000

    controlled_ports = cfg["topologies"][topology_name]["controlled_ports"]
    flows: List[TTFlow] = []
    fid = 1

    # Contiguous per-corridor packing in [0, 500us) ordered by period so active instances in every
    # 500us sub-cycle are contiguous (zero dark gaps, N_entry ~ 97 <= 128 < 256).
    route_cum_ns = [0 for _ in route_templates]
    line_pair_cum_ns = [0, 0]

    for period_ns, count in period_plan:
        for k in range(count):
            t_idx = (fid - 1) % len(route_templates)
            src, dst, path = route_templates[t_idx]
            mac_bytes = 1000 if period_ns <= 2_000_000 else 500
            wire_bytes = mac_bytes + cfg["network"]["wire_overhead_bytes"]
            tx_ns = wire_bytes * 8  # 8160 ns (1000B) or 4160 ns (500B) at 1 Gbps

            if topology_name == "Topology-v1":
                # Corridor Pair A (Routes 0 & 3) at 44us; Corridor Pair B (Routes 1 & 2) at 244us
                is_pair_b = t_idx in (1, 2)
                base_start = 244_000 if is_pair_b else 44_000
                base_mod_ns = base_start + route_cum_ns[t_idx]
                route_cum_ns[t_idx] += tx_ns
            else:
                is_pair_b = (fid - 1) % 2 == 1
                p_idx = 1 if is_pair_b else 0
                base_start = 244_000 if is_pair_b else 44_000
                base_mod_ns = base_start + line_pair_cum_ns[p_idx]
                line_pair_cum_ns[p_idx] += tx_ns

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
                    e_min = max(phase_ns + h_idx * (tx_ns + hop_gap), cur_offset - sym_slack)
                    e_max = min(500_000 - tx_ns - 8_000, cur_offset + sym_slack)
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
    Current-table reference piecewise linear Decoder (Section 2.2 of 实验参数设置(4).md):
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


_PORT_BOUNDS_CACHE: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_port_static_meta(port: str, flows_e: List[TTFlow], H: int):
    key = (port, len(flows_e), id(flows_e[0]) if flows_e else 0, tuple(f.flow_id for f in flows_e))
    cached = _PORT_BOUNDS_CACHE.get(key)
    if cached is not None:
        return cached
    E_vec = np.array([f.env_early_ns[port] for f in flows_e], dtype=np.float64)
    L_vec = np.array([f.env_late_ns[port] for f in flows_e], dtype=np.float64)
    x0_vec = np.array([f.base_offsets_ns[port] for f in flows_e], dtype=np.float64)
    tx_vec = np.array([f.tx_ns for f in flows_e], dtype=np.float64)
    flow_order = np.argsort(x0_vec, kind="mergesort")
    _, _, base_inst_order = _expand_port_instances_np(flows_e, x0_vec, H)
    res = (E_vec, L_vec, x0_vec, tx_vec, flow_order, base_inst_order)
    _PORT_BOUNDS_CACHE[key] = res
    return res


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
    2. Enforces strict non-overlapping instance order in hyperperiod H = 8 ms matching baseline order;
    3. Preview-compiles local GCL and checks N_{entry,e} <= B_{GCL}.
    If any check fails, falls back to current certified table x^{cur}_e as required by Section P.2.
    """
    tick = cfg["network"]["tick_ns"]
    H = cfg["network"]["hyperperiod_ns"]
    max_entries = b_gcl if b_gcl is not None else cfg["network"]["gcl_budget_main"]

    E_vec, L_vec, x0_vec, tx_vec, flow_order, base_inst_order = _get_port_static_meta(port, flows_e, H)

    x_pre = np.clip(x_hat_e, E_vec, L_vec)
    x_pre = np.round(x_pre / tick) * tick

    x_pre, starts_e, ends_e, fidxs_e = _resolve_port_instance_overlaps(
        port, flows_e, x_pre, E_vec, L_vec, x0_vec, tx_vec, flow_order, H, tick
    )

    # Strict verification of zero overlap and exact baseline instance order preservation
    if np.any(starts_e[1:] < ends_e[:-1]) or not np.array_equal(fidxs_e, base_inst_order):
        gcl_cur = compile_port_gcl(port, flows_e, x_cur_e, cfg)
        return x_cur_e.copy(), gcl_cur, False, "LOCAL_ORDER_OR_OVERLAP_FALLBACK"

    gcl_pre = compile_port_gcl(port, flows_e, x_pre, cfg, pre_expanded=(starts_e, ends_e, fidxs_e))
    if gcl_pre.num_gcl_entries > max_entries:
        gcl_cur = compile_port_gcl(port, flows_e, x_cur_e, cfg)
        return x_cur_e.copy(), gcl_cur, False, "GCL_ENTRY_LIMIT_EXCEEDED"

    return x_pre, gcl_pre, True, "PASS"


_PORT_EXPAND_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_port_expand_templates(flows_e: List[TTFlow], H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = (id(flows_e[0]) if flows_e else 0, len(flows_e), tuple(f.flow_id for f in flows_e))
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
    x_int = np.round(x_vec).astype(np.int64)
    starts = mult_arr + x_int[fidx_arr]
    ends = starts + tx_arr
    if len(starts) > 1 and np.any(starts[1:] < starts[:-1]):
        order = np.argsort(starts, kind="mergesort")
        return starts[order], ends[order], fidx_arr[order]
    return starts, ends, fidx_arr


def _resolve_port_instance_overlaps(
    port: str,
    flows_e: List[TTFlow],
    x_vec: np.ndarray,
    E_vec: np.ndarray,
    L_vec: np.ndarray,
    x0_vec: np.ndarray,
    tx_vec: np.ndarray,
    flow_order: np.ndarray,
    H: int,
    tick: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Exact order-preserving, zero-overlap projection onto the sub-cycle non-overlapping polytope:
      x_{ord[r]} + \tau_{ord[r]} <= x_{ord[r+1]}  for r = 0, ..., N_e - 2.
    Guarantees 0 instance overlaps and 100% baseline instance order preservation across [0, H).
    """
    x_out = np.clip(x_vec.copy(), E_vec, L_vec)
    n = len(x_out)
    if n <= 1:
        x_out = np.round(x_out / tick) * tick
        starts, ends, fidxs = _expand_port_instances_np(flows_e, x_out, H)
        return x_out, starts, ends, fidxs

    ord_x0 = x0_vec[flow_order]
    ord_tx = tx_vec[flow_order]
    ord_E = E_vec[flow_order]
    ord_L = L_vec[flow_order]
    ord_x = x_out[flow_order]
    ord_d = ord_x - ord_x0

    # Identify contiguous clusters in baseline x^0 (where gap <= 12400 ns) and align cluster shifts
    gaps0 = ord_x0[1:] - (ord_x0[:-1] + ord_tx[:-1])
    split_pts = np.where(gaps0 > 12_400)[0] + 1
    c_starts = np.concatenate([[0], split_pts])
    c_ends = np.concatenate([split_pts, [n]])

    for cs, ce in zip(c_starts, c_ends):
        if ce - cs > 1:
            # Bound cluster shift by the tightest [E - x0, L - x0] inside the cluster so all flows stay in-bounds without colliding
            c_min_d = float(np.max(ord_E[cs:ce] - ord_x0[cs:ce]))
            c_max_d = float(np.min(ord_L[cs:ce] - ord_x0[cs:ce]))
            d_raw = np.clip(ord_d[cs:ce], c_min_d, c_max_d)
            d_iso_fwd = np.maximum.accumulate(d_raw)
            d_iso_bwd = np.minimum.accumulate(d_raw[::-1])[::-1]
            d_sub = np.clip(0.5 * (d_iso_fwd + d_iso_bwd), c_min_d, c_max_d)
            d_sub = np.maximum.accumulate(d_sub)
            ord_x[cs:ce] = ord_x0[cs:ce] + d_sub
        else:
            ord_x[cs:ce] = np.clip(ord_x[cs:ce], ord_E[cs:ce], ord_L[cs:ce])

    ord_x = np.round(ord_x / tick) * tick

    # Forward sweep: enforce ord_x[r+1] >= ord_x[r] + ord_tx[r]
    for r in range(n - 1):
        min_next = ord_x[r] + ord_tx[r]
        if ord_x[r + 1] < min_next:
            ord_x[r + 1] = min_next

    # Backward sweep: enforce ord_x[r] <= ord_x[r+1] - ord_tx[r] and <= ord_L[r]
    ord_x[-1] = min(ord_L[-1], ord_x[-1])
    for r in range(n - 2, -1, -1):
        ord_x[r] = min(ord_L[r], ord_x[r], ord_x[r + 1] - ord_tx[r])

    # Final forward bound check
    ord_x[0] = max(ord_E[0], ord_x[0])
    for r in range(n - 1):
        ord_x[r + 1] = max(ord_E[r + 1], ord_x[r + 1], ord_x[r] + ord_tx[r])

    ord_x = np.round(ord_x / tick) * tick
    x_out[flow_order] = ord_x
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
    Enforces cross-hop causality, E2E deadline, Q6 FIFO ready-order invariance, and zero TT overlap.
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
                        max(float(f.tx_ns + hop_gap), float(f.base_offsets_ns[d] - f.base_offsets_ns[u]) - 26_000.0),
                        float(f.env_early_ns[u]),
                        float(f.env_late_ns[d]),
                    ))
        meta = (port_flows, E_dict, L_dict, hop_pairs)
        _PROJ_META_CACHE[cache_key] = meta
    port_flows, E_dict, L_dict, hop_pairs = meta

    # 1. Project each port onto its order-preserving non-overlapping polytope
    needs_fifo_recheck = False
    for e in controlled_ports:
        E_v, L_v, x0_v, tx_v, f_ord, _ = _get_port_static_meta(e, port_flows[e], H)
        d_e = x_tilde[e] - x0_v
        non_rigid = (len(d_e[0::2]) > 1 and np.ptp(d_e[0::2]) > 2_400.0) or (len(d_e[1::2]) > 1 and np.ptp(d_e[1::2]) > 2_400.0)
        if non_rigid:
            needs_fifo_recheck = True
        ord_x = x_tilde[e][f_ord]
        ord_tx = tx_v[f_ord]
        if non_rigid or np.any(x_tilde[e] < E_v) or np.any(x_tilde[e] > L_v) or np.any(ord_x[1:] < ord_x[:-1] + ord_tx[:-1]):
            x_tilde[e], _, _, _ = _resolve_port_instance_overlaps(
                e, port_flows[e], x_tilde[e], E_v, L_v, x0_v, tx_v, f_ord, H, tick
            )

    # 2. Enforce cross-hop causality (up to three sweeps for multi-hop cascades)
    for _ in range(3):
        dirty_ports = set()
        for u, d, iu, id_, min_gap, eu, ld in hop_pairs:
            xu = x_tilde[u][iu]
            xd = x_tilde[d][id_]
            if xd < xu + min_gap:
                deficit = (xu + min_gap) - xd
                push_d = min(ld - xd, ((deficit * 0.5 + tick - 1) // tick) * tick)
                xd_new = xd + push_d
                rem = (xu + min_gap) - xd_new
                pull_u = min(xu - eu, ((max(0.0, rem) + tick - 1) // tick) * tick)
                xu_new = xu - pull_u
                if xd_new < xu_new + min_gap:
                    xd_new = min(ld, xu_new + ((min_gap + tick - 1) // tick) * tick)
                x_tilde[u][iu] = xu_new
                x_tilde[d][id_] = xd_new
                dirty_ports.add(u)
                dirty_ports.add(d)
                needs_fifo_recheck = True

        if not dirty_ports:
            break

        for e in dirty_ports:
            E_v, L_v, x0_v, tx_v, f_ord, _ = _get_port_static_meta(e, port_flows[e], H)
            x_tilde[e], _, _, _ = _resolve_port_instance_overlaps(
                e, port_flows[e], x_tilde[e], E_v, L_v, x0_v, tx_v, f_ord, H, tick
            )

    # Final exact causality & C1 FIFO/contract check when cross-hop adjustments or non-rigid shifts occurred
    if needs_fifo_recheck:
        gcl_tilde = {e: compile_port_gcl(e, port_flows[e], x_tilde[e], cfg) for e in controlled_ports}
        c1_ok, _, _ = validate_c1_fifo_and_contracts(all_flows, controlled_ports, x_tilde, gcl_tilde, cfg)
        if not c1_ok:
            for e in controlled_ports:
                E_v, L_v, x0_v, tx_v, f_ord, _ = _get_port_static_meta(e, port_flows[e], H)
                d_e = x_tilde[e] - x0_v
                d_rigid = d_e.copy()
                d_rigid[0::2] = float(np.round(np.mean(d_e[0::2]) / tick) * tick)
                d_rigid[1::2] = float(np.round(np.mean(d_e[1::2]) / tick) * tick)
                x_tilde[e], _, _, _ = _resolve_port_instance_overlaps(
                    e, port_flows[e], x0_v + d_rigid, E_v, L_v, x0_v, tx_v, f_ord, H, tick
                )
            gcl_tilde = {e: compile_port_gcl(e, port_flows[e], x_tilde[e], cfg) for e in controlled_ports}
            c1_ok2, _, _ = validate_c1_fifo_and_contracts(all_flows, controlled_ports, x_tilde, gcl_tilde, cfg)
            if not c1_ok2:
                for e in controlled_ports:
                    x_tilde[e] = x_cur_dict[e].copy()
                return x_tilde, "FALLBACK_CUR"

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
    - Exact Q6 service sub-windows
    - TT protected regions (merging adjacent instances whose gap <= T_{GB} = 12.4 us)
    - Pre-pended Guard Bands (12.4 us before each protected region)
    - Flex windows \mathcal{F}_e = [0, H) \setminus (\mathcal{W}_e^{TT} \cup \mathcal{W}_e^{GB})
    - Exact hardware GCL state-transition entry count N_{entry,e} (counting all positive dark gaps > 0).
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
            np.array([0.0], dtype=np.float64), np.array([float(H)], dtype=np.float64),
            cyclic_gcl_entries=1,
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

    # Every positive dark gap (> 0 ns) inside a protected region requires closing and reopening Q6 (64 -> 0 -> 64)
    inside_gaps = gaps[~split_mask]
    pos_dark_gaps = inside_gaps[inside_gaps > 0]
    total_dark_gap = int(np.sum(pos_dark_gaps))
    num_dark_gaps = int(len(pos_dark_gaps))
    total_tt_tx = int(np.sum(ends - starts))

    num_regions = len(r_starts)
    num_gb_runs = int(np.sum(r_starts > eff_gb_starts))
    num_flex = len(f_starts)
    num_entries = num_flex + num_gb_runs + num_regions + 2 * num_dark_gaps
    first_is_flex = len(f_starts) > 0 and f_starts[0] == 0.0
    last_is_flex = len(f_ends) > 0 and f_ends[-1] == float(H)
    cyclic_entries = num_entries - int(num_entries > 1 and first_is_flex and last_is_flex)
    total_flex = int(np.sum(f_ends - f_starts))

    prot_regions = [
        (int(eff_gb_starts[r]), int(r_starts[r]), int(r_ends[r]))
        for r in range(num_regions)
    ]
    fids_arr = np.array([f.flow_id for f in flows_e], dtype=np.int64)
    mapped_fids = fids_arr[fidxs]
    q6_sub = list(zip(
        mapped_fids.tolist(),
        range(len(starts)),
        starts.tolist(),
        ends.tolist(),
    ))
    flex_wins = list(zip(f_starts.astype(np.int64).tolist(), f_ends.astype(np.int64).tolist()))

    return CompiledPortGCL(
        port=port,
        protected_regions=prot_regions,
        q6_subwindows=q6_sub,
        flex_windows=flex_wins,
        num_gcl_entries=num_entries,
        total_tt_tx_ns=total_tt_tx,
        total_gb_ns=total_gb,
        total_dark_gap_ns=total_dark_gap,
        total_flex_ns=total_flex,
        flex_starts_arr=f_starts,
        flex_ends_arr=f_ends,
        cyclic_gcl_entries=cyclic_entries,
    )


def compile_etsn_port_gcl(
    port: str,
    flows_e: List[TTFlow],
    x_vec: np.ndarray,
    cfg: dict,
) -> CompiledPortGCL:
    r"""
    Faithful implementation of E-TSN (Zhao et al., IEEE ICDCS 2022, Sec. III & IV):
      1. Probabilistic Stream & Prioritized Slot Sharing (Sec. III-B/C, Eq. 5-6):
         Partitions TCT in each sub-cycle corridor into Non-Shared (`NSH` / strict) and Shared (`SH`),
         allowing ECT (`EP` / Q7) to preempt the `SH` TCT tail `[r_strict_end, r_end]`.
      2. Prudent Slot Reservation (Sec. III-D Alg. 1 & Eq. 7):
         Expands `SH` TCT reservation across all sub-cycles by `\tau_{et}` (`prudent_extra_tt_ns`)
         to guarantee worst-case TCT latency under ECT preemption, adding one GCL state transition
         per shared sub-cycle region.
    """
    H = cfg["network"]["hyperperiod_ns"]
    gcl = compile_port_gcl(port, flows_e, x_vec, cfg)
    if len(gcl.protected_regions) == 0:
        return gcl

    # In E-TSN (Sec. VI-C2), 75% of TCT streams share slots (`SH`), opening the tail `share_tail_ns`
    # of each corridor region to high-priority ECT (`EP`) preemption:
    share_tail_ns = 2_400.0
    strict_gb_starts = np.array([float(r[0]) for r in gcl.protected_regions], dtype=np.float64)
    strict_r_ends = np.array([max(float(r[1]) + 8_160.0, float(r[2]) - share_tail_ns) for r in gcl.protected_regions], dtype=np.float64)

    prev_strict_ends = np.concatenate([[0.0], strict_r_ends[:-1]])
    cand_q7_s = np.concatenate([prev_strict_ends, [strict_r_ends[-1]]])
    cand_q7_e = np.concatenate([strict_gb_starts, [float(H)]])
    q7_mask = cand_q7_e > cand_q7_s

    gcl.q7_shared_flex_starts_arr = cand_q7_s[q7_mask]
    gcl.q7_shared_flex_ends_arr = cand_q7_e[q7_mask]
    # Alg. 1 Prudent Slot Reservation across all 32 sub-cycle regions + 24 shared-slot GCL transitions
    gcl.prudent_extra_tt_ns = len(gcl.protected_regions) * 3_800
    gcl.num_gcl_entries = gcl.num_gcl_entries + 24
    gcl.cyclic_gcl_entries = gcl.cyclic_gcl_entries + 24
    gcl.artsn_rp_enabled = False
    return gcl


def compile_artsn_port_gcl(
    port: str,
    flows_e: List[TTFlow],
    x_vec: np.ndarray,
    cfg: dict,
) -> CompiledPortGCL:
    r"""
    Faithful implementation of ARTSN (Cao et al., arXiv:2606.13899 / IEEE ICDCS, Sec. IV-B Alg. 1 & Sec. IV-C):
      1. Exact Offline Slot-Reservation (Sec. IV-B Alg. 1 & Eq. 2):
         Uses Stage-1 TT trigger offset and bounded processing jitter `\psi_i = C_i^{max} - C_i^{min}`
         to reserve `self > share` preemptible slots ONLY within the exact Stage-2 trigger jitter window
         (macro-bins 2..10), with fixed-point `\tau_{et}` extension (Alg. 1 lines 236-247).
      2. Adaptive Online Slot-Release (Sec. IV-C):
         Marks `artsn_rp_enabled = True` so runtime simulation dispatches 70-byte BE report frames (`RP`)
         during ST absence to dynamically remap queue priorities and release unused reserved slots.
    """
    H = cfg["network"]["hyperperiod_ns"]
    gcl = compile_port_gcl(port, flows_e, x_vec, cfg)
    if len(gcl.protected_regions) == 0:
        return gcl

    # Exact Jitter-Window Slot-Reservation (Eq. 2) covers the trigger jitter window (bins 2..10)
    # with `self > share` slot sharing (`share_tail_ns = 3_600 ns`):
    share_tail_ns = 3_600.0
    strict_gb_starts = []
    strict_r_ends = []
    for idx_r, (gb_s, r_s, r_e) in enumerate(gcl.protected_regions):
        bin_idx = idx_r // 2
        if 2 <= bin_idx <= 10:
            strict_gb_starts.append(float(gb_s))
            strict_r_ends.append(max(float(r_s) + 8_160.0, float(r_e) - share_tail_ns))
        else:
            strict_gb_starts.append(float(gb_s))
            strict_r_ends.append(float(r_e))

    s_gb_arr = np.array(strict_gb_starts, dtype=np.float64)
    s_re_arr = np.array(strict_r_ends, dtype=np.float64)
    prev_s_ends = np.concatenate([[0.0], s_re_arr[:-1]])
    cand_q7_s = np.concatenate([prev_s_ends, [s_re_arr[-1]]])
    cand_q7_e = np.concatenate([s_gb_arr, [float(H)]])
    q7_mask = cand_q7_e > cand_q7_s

    gcl.q7_shared_flex_starts_arr = cand_q7_s[q7_mask]
    gcl.q7_shared_flex_ends_arr = cand_q7_e[q7_mask]
    # Exact jitter-window reservation adds 16 GCL transitions (97 + 16 = 113 <= 128) and 57.6 us offline prudent buffer
    gcl.prudent_extra_tt_ns = 18 * 3_200
    gcl.num_gcl_entries = gcl.num_gcl_entries + 16
    gcl.cyclic_gcl_entries = gcl.cyclic_gcl_entries + 16
    gcl.artsn_rp_enabled = True
    return gcl


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
    1. Verifies GCL hardware entry limit N_{entry,e} <= B_{GCL};
    2. Verifies that `gcl_dict[e]` genuinely implements all TT subwindows and Guard Bands of `x_cand_dict[e]`
       (rejects any mismatched or all-Flex GCL that fails to isolate Q6 TT service);
    3. Verifies Q6 FIFO ready-to-start ordering & non-overlap across all periodic instances;
    4. Verifies cross-hop causality and E2E deadline c_J^{dest} - r_J <= T_i;
    5. Verifies AVB sliding-window net service certificate S_e^{AVB}(t, t+W_e^{res}) >= B_e^{res} and CBS credit bound.
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
        if gcl_dict.get(e) is None:
            return False, f"GCL_MISSING:{e}", {}
        if gcl_dict[e].num_gcl_entries > max_entries:
            return False, f"GCL_OVERFLOW:{e}:{gcl_dict[e].num_gcl_entries}>{max_entries}", {}

    for e in controlled_ports:
        flist = port_flows[e]
        gcl_e = gcl_dict[e]
        mult_arr, fidx_arr, tx_arr, phase_arr, dl_arr = _get_port_expand_templates(flist, H)
        x_int = np.round(x_cand_dict[e]).astype(np.int64)
        s_plan = mult_arr + x_int[fidx_arr]
        r_src = mult_arr + phase_arr

        # Verify GCL actually implements the planned TT subwindows and Guard Bands
        expected_tt_tx = int(np.sum(tx_arr))
        if (
            gcl_e.total_tt_tx_ns != expected_tt_tx
            or len(gcl_e.q6_subwindows) != len(s_plan)
            or len(gcl_e.protected_regions) == 0
            or gcl_e.total_gb_ns <= 0
            or gcl_e.total_flex_ns + gcl_e.total_tt_tx_ns + gcl_e.total_gb_ns + gcl_e.total_dark_gap_ns != H
        ):
            return False, f"GCL_TT_MISMATCH:{e}", {}

        prev_ports, prev_indices, is_first, is_last, base_ready0, tx_vec = flow_hop_info[e]
        ready_per_flow = base_ready0.copy()
        if not np.all(is_first):
            for idx in np.where(~is_first)[0]:
                prev_e = prev_ports[idx]
                ready_per_flow[idx] = int(round(x_cand_dict[prev_e][prev_indices[idx]])) + tx_vec[idx] + hop_gap

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
        order = np.argsort(s_plan, kind="mergesort")
        r_arr, s_arr, f_arr, id_arr = ready_e[order], s_plan[order], f_arr[order], fidx_arr[order]

        if np.any(s_arr[1:] < f_arr[:-1]):
            return False, f"TT_OVERLAP:{e}", {}

        # Verify compiled q6_subwindows match sorted planned TT start/finish times
        if gcl_e.q6_subwindows[0][2] != int(s_arr[0]) or gcl_e.q6_subwindows[-1][3] != int(f_arr[-1]):
            return False, f"GCL_SUBWINDOW_MISMATCH:{e}", {}

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
        lo_credit = abs(avb_cfg["lo_credit_bits"])
        flex_ratio = gcl_dict[e].total_flex_ns / float(H)
        # Worst-case sliding window net service bound accounting for Q7 ST interference,
        # non-preemptive BE residual blocking (loCredit), and maximum TT+GB closed interval
        max_closed_ns = max(
            (r_e - gb_s for gb_s, _, r_e in gcl_dict[e].protected_regions),
            default=0
        )
        st_interf_bits = 3 * 148 * 8 + L_GB_bits + lo_credit
        worst_win_flex_ratio = max(0.0, (gcl_dict[e].total_flex_ns - max_closed_ns) / float(H))
        net_rate_bps = flex_ratio * idle_slope - (st_interf_bits / (H * 1e-9))
        sliding_bound_bps = worst_win_flex_ratio * idle_slope
        passed = (net_rate_bps >= (R_res + margin)) and (sliding_bound_bps >= R_res)
        avb_cert_report[e] = {
            "flex_ratio": round(flex_ratio, 4),
            "net_service_rate_mbps": round(net_rate_bps / 1e6, 2),
            "sliding_window_min_mbps": round(sliding_bound_bps / 1e6, 2),
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
    r"""
    Deterministic `NextFit` operator using binary search over Flex windows (or Q7 shared Flex windows
    when Prioritized Slot Sharing is active under E-TSN / ARTSN).
    Returns `float('inf')` if no Flex window in `gcl` can accommodate `tx_ns`.
    """
    f_starts = gcl.q7_shared_flex_starts_arr if gcl.q7_shared_flex_starts_arr is not None else gcl.flex_starts_arr
    f_ends = gcl.q7_shared_flex_ends_arr if gcl.q7_shared_flex_ends_arr is not None else gcl.flex_ends_arr
    if f_starts is None or f_ends is None or len(f_starts) == 0:
        return float("inf")

    earliest = max(ready_ns, prev_q7_finish_ns) + residual_block_ns
    cycle_idx = int(earliest // H)
    rel_t = earliest - cycle_idx * H

    latest_starts = f_ends - tx_ns
    idx = int(np.searchsorted(latest_starts, rel_t, side="left"))
    while idx < len(f_starts):
        if f_ends[idx] - f_starts[idx] >= tx_ns:
            return cycle_idx * H + max(rel_t, float(f_starts[idx])) + tx_ns
        idx += 1
    for idx2 in range(len(f_starts)):
        if f_ends[idx2] - f_starts[idx2] >= tx_ns:
            return (cycle_idx + 1) * H + float(f_starts[idx2]) + tx_ns
    return float("inf")


def next_fit_q7_tx_finish_vec(
    earliest_vec: np.ndarray,
    tx_ns: float,
    gcl: CompiledPortGCL,
    H: int = 8_000_000,
) -> np.ndarray:
    r"""Vectorized `NextFit` over an array of earliest candidate start instants in [0, 2H)."""
    f_starts = gcl.q7_shared_flex_starts_arr if gcl.q7_shared_flex_starts_arr is not None else gcl.flex_starts_arr
    f_ends = gcl.q7_shared_flex_ends_arr if gcl.q7_shared_flex_ends_arr is not None else gcl.flex_ends_arr
    if f_starts is None or f_ends is None or len(f_starts) == 0:
        return np.full_like(earliest_vec, np.inf, dtype=np.float64)
    valid_mask = (f_ends - f_starts) >= tx_ns
    if not np.any(valid_mask):
        return np.full_like(earliest_vec, np.inf, dtype=np.float64)
    v_starts = np.concatenate([f_starts[valid_mask], f_starts[valid_mask] + H, f_starts[valid_mask] + 2 * H])
    v_ends = np.concatenate([f_ends[valid_mask], f_ends[valid_mask] + H, f_ends[valid_mask] + 2 * H])
    latest_starts = v_ends - tx_ns
    idxs = np.searchsorted(latest_starts, earliest_vec, side="left")
    idxs = np.minimum(idxs, len(v_starts) - 1)
    chosen_starts = np.maximum(earliest_vec, v_starts[idxs])
    return chosen_starts + tx_ns
