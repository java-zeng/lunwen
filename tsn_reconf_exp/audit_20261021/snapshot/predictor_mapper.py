r"""
Egress-Local Correlation Predictor (Set Transformer + ModernTCN),
Strict `build_observation` Interface (t_rx < t_f), and
Deterministic Action-Conditioned Service Mapper (\mathcal{S}_u + \mathcal{P}_{u->d}).

Strictly adheres to Part 2 and Part 3 of `2026.10.20(8).md`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_sim import CompiledPortGCL, next_fit_q7_tx_finish, next_fit_q7_tx_finish_vec


@dataclass
class PacketEvent:
    switch_id: str
    egress_port: str
    traffic_class: str   # "AVB" or "BE"
    event_type: str      # "RX_COMPLETE"
    rx_time_ns: int
    flow_id: int
    length: int
    pcp: int


def build_observation(
    events: List[PacketEvent],
    switch_id: str,
    egress_port: str,
    freeze_ns: int,
    history_ns: int = 32_000_000,
    window_ns: int = 8_000_000,
) -> List[List[PacketEvent]]:
    r"""
    Exact specification from Section 3 of Part 3 in `2026.10.20(8).md`.
    Strictly filters events on `switch_id` forwarded to `egress_port` with
    traffic_class in {"AVB", "BE"}, event_type == "RX_COMPLETE", and
    freeze_ns - history_ns <= rx_time_ns < freeze_ns.
    Never accesses true correlated flow subset \Pi_g^{true} or future labels.
    """
    if history_ns <= 0 or window_ns <= 0 or history_ns % window_ns:
        raise ValueError("history must contain complete positive windows")
    start_ns = freeze_ns - history_ns
    windows: List[List[PacketEvent]] = [[] for _ in range(history_ns // window_ns)]
    for event in events:
        if event.switch_id != switch_id or event.egress_port != egress_port:
            continue
        if event.traffic_class not in {"AVB", "BE"}:
            continue
        if event.event_type != "RX_COMPLETE":
            continue
        if not (start_ns <= event.rx_time_ns < freeze_ns):
            continue
        idx = (event.rx_time_ns - start_ns) // window_ns
        windows[idx].append(event)
    return windows


class SetTransformerModernTCN(nn.Module):
    r"""
    Shared-parameter (\psi) Local Predictor:
    1. Set Transformer encoder over variable-length packet events per 8 ms window;
    2. ModernTCN depthwise-separable temporal block across 4 consecutive 8 ms windows (32 ms);
    3. Dual heads per (g, e): occurrence probability q_{g,e,k} (Sigmoid) and
       16-bin conditional arrival distribution \pi_{g,e,k,1:16} (Softmax).
    """
    def __init__(self, d_in: int = 5, d_model: int = 32, n_heads: int = 4, num_bins: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.event_embed = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        # ModernTCN across 4 windows
        self.tcn_dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.tcn_pw = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.task_embed = nn.Embedding(6, d_model)
        self.head_q = nn.Linear(d_model * 2, 1)
        self.head_pi = nn.Linear(d_model * 2, num_bins)

    def forward(
        self,
        window_events: torch.Tensor,  # (B, 4, M, 5)
        task_ids: torch.Tensor,       # (B,) in 0..5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, W, M, D = window_events.shape
        BW = B * W
        x = window_events.view(BW, M, D)
        h = self.event_embed(x)
        qkv = self.qkv_proj(h).view(BW, M, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attn_out = self.out_proj(attn_out.transpose(1, 2).reshape(BW, M, self.d_model))
        z_win = self.norm1(h + attn_out).mean(dim=1).view(B, W, -1)  # (B, 4, d_model)
        # ModernTCN over 4 windows
        z_t = z_win.transpose(1, 2)                                   # (B, d_model, 4)
        z_t = z_t + self.tcn_pw(self.tcn_dw(z_t))
        h_seq = z_t[:, :, -1]                                         # (B, d_model)
        t_emb = self.task_embed(task_ids)
        z_pred = torch.cat([h_seq, t_emb], dim=-1)                    # (B, 2*d_model)
        q = torch.sigmoid(self.head_q(z_pred)).squeeze(-1)
        pi = F.softmax(self.head_pi(z_pred), dim=-1)
        return q, pi, h_seq


@dataclass
class PropagatedArrivalMessage:
    r"""
    Whitelist-compliant route-constrained propagated arrival message m_{g, u->d, k}
    (Section P.5 of `2026.10.20(8).md`).
    """
    cycle_k: int
    group: str
    src_port: str
    dst_port: str
    mode: str                    # "PROB" or "INTERVAL"
    q_prop: float
    pi_prop: np.ndarray          # (16,) conditional bin probabilities
    mu_t_prop_ns: float          # Conditional expected arrival offset in [0, 8ms) to prevent 0.5ms bin aliasing
    bounds_low: np.ndarray       # (16,) lower bounds for INTERVAL mode
    bounds_high: np.ndarray      # (16,) upper bounds for INTERVAL mode
    valid: int                   # 1 or 0
    out_of_window_mass: float    # Right-censored mass beyond t_k+16ms
    candidate_id: str
    config_hash: str
    mapper_version: str = "v1.0-deterministic"


def run_service_mapper_port(
    cycle_k: int,
    upstream_port: str,
    local_preds: Dict[str, Tuple[float, np.ndarray]],
    gcl_cur: CompiledPortGCL,
    gcl_pre: CompiledPortGCL,
    residual_block_dist_ns: np.ndarray,
    carry_in_backlog_count: int,
    cfg: dict,
    candidate_id: str,
    config_hash: str,
) -> Dict[Tuple[str, str, str], PropagatedArrivalMessage]:
    r"""
    Deterministic Service Mapper \mathcal{S}_u and physical link mapping \mathcal{P}_{u->d}.
    - On e3u: maps group A -> (A, e3u, e01)
    - On e4u: jointly replays Q7 FIFO contention between group B and group C -> (B, e4u, e01), (C, e4u, e02)
    Strictly uses G^{cur}_u in [t_f, t_k+8ms) and G^{pre}_u in [t_k+8ms, t_k+16ms),
    without adding any learnable parameters or reading downstream observations.
    """
    H = cfg["network"]["hyperperiod_ns"]
    bin_ns = cfg["timing_control"]["pred_bin_ns"]
    num_bins = cfg["timing_control"]["num_pred_bins"]
    hop_delay_ns = cfg["network"]["switch_proc_delay_ns"] + cfg["network"]["link_prop_delay_ns"]

    quad_offsets = np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float64) * bin_ns
    bin_starts = np.arange(num_bins, dtype=np.float64) * bin_ns

    messages: Dict[Tuple[str, str, str], PropagatedArrivalMessage] = {}
    is_interval_mode = carry_in_backlog_count > cfg["service_mapper"]["prob_mode_max_carryin_frames"]

    if upstream_port == "e3u":
        q_A, pi_A = local_preds["A"]
        tx_A = float((cfg["st_traffic"]["groups"]["A"]["mac_bytes"] + cfg["network"]["wire_overhead_bytes"]) * 8)

        # Vectorized quadrature across (16 bins, 4 quad offsets, R residual samples)
        R = len(residual_block_dist_ns)
        a_grid = (bin_starts[:, None, None] + quad_offsets[None, :, None] + residual_block_dist_ns[None, None, :]).ravel()
        w_grid = np.repeat(q_A * pi_A * (0.25 / R), 4 * R)

        c_tx_u = next_fit_q7_tx_finish_vec(a_grid, tx_A, gcl_pre, H)
        t_q_d = c_tx_u + hop_delay_ns

        in_win = t_q_d < H
        b_d = (t_q_d[in_win] // bin_ns).astype(np.int64)
        w_in = w_grid[in_win]
        pi_down = np.bincount(b_d, weights=w_in, minlength=num_bins).astype(np.float64)[:num_bins]
        mass_sum = float(np.sum(w_in))
        exp_t_sum = float(np.sum(w_in * t_q_d[in_win]))
        out_mass = float(np.sum(w_grid[~in_win]))

        q_prop = float(np.clip(mass_sum, 0.0, 1.0))
        if q_prop > 1e-9:
            pi_prop = pi_down / q_prop
            mu_t = exp_t_sum / q_prop
        else:
            pi_prop = np.zeros(num_bins, dtype=np.float64)
            mu_t = 0.0

        messages[("A", "e3u", "e01")] = PropagatedArrivalMessage(
            cycle_k=cycle_k,
            group="A",
            src_port="e3u",
            dst_port="e01",
            mode="INTERVAL" if is_interval_mode else "PROB",
            q_prop=q_prop,
            pi_prop=pi_prop,
            mu_t_prop_ns=float(mu_t),
            bounds_low=np.maximum(0.0, pi_down - 0.03),
            bounds_high=np.minimum(1.0, pi_down + 0.03),
            valid=1,
            out_of_window_mass=float(out_mass),
            candidate_id=candidate_id,
            config_hash=config_hash,
        )

    elif upstream_port == "e4u":
        # Vectorized Joint Q7 FIFO replay of B and C on e4u (Section P.3 of 2026.10.20(8).md)
        q_B, pi_B = local_preds["B"]
        q_C, pi_C = local_preds["C"]
        tx_B = float((cfg["st_traffic"]["groups"]["B"]["mac_bytes"] + cfg["network"]["wire_overhead_bytes"]) * 8)
        tx_C = float((cfg["st_traffic"]["groups"]["C"]["mac_bytes"] + cfg["network"]["wire_overhead_bytes"]) * 8)
        mean_res = float(np.mean(residual_block_dist_ns))

        # 1 & 2: Marginal single-group quadrature (64 points each)
        a_quad = (bin_starts[:, None] + quad_offsets[None, :]).ravel() + mean_res
        c_B_marg = next_fit_q7_tx_finish_vec(a_quad, tx_B, gcl_pre, H)
        c_C_marg = next_fit_q7_tx_finish_vec(a_quad, tx_C, gcl_pre, H)
        w_B_marg = np.repeat(q_B * pi_B * (1.0 - q_C) * 0.25, 4)
        w_C_marg = np.repeat(q_C * pi_C * (1.0 - q_B) * 0.25, 4)

        # 3: Co-occurrence 16x16 grid (vectorized)
        a_B_16 = (np.arange(num_bins, dtype=np.float64) + 0.35) * bin_ns
        a_C_16 = (np.arange(num_bins, dtype=np.float64) + 0.65) * bin_ns
        c_B_first_16 = next_fit_q7_tx_finish_vec(a_B_16 + mean_res, tx_B, gcl_pre, H)
        c_C_first_16 = next_fit_q7_tx_finish_vec(a_C_16 + mean_res, tx_C, gcl_pre, H)

        B_idx, C_idx = np.meshgrid(np.arange(num_bins), np.arange(num_bins), indexing="ij")
        b_flat = B_idx.ravel()
        c_flat = C_idx.ravel()
        w_pair = ((q_B * pi_B)[:, None] * (q_C * pi_C)[None, :]).ravel()

        b_first_mask = a_B_16[b_flat] <= a_C_16[c_flat]
        c_B_pair = np.empty_like(w_pair)
        c_C_pair = np.empty_like(w_pair)

        # Case B arrives first: B finishes at c_B_first_16[b], C waits behind max(a_C, c_B)
        c_B_pair[b_first_mask] = c_B_first_16[b_flat[b_first_mask]]
        earliest_C_after_B = np.maximum(a_C_16[c_flat[b_first_mask]], c_B_pair[b_first_mask])
        c_C_pair[b_first_mask] = next_fit_q7_tx_finish_vec(earliest_C_after_B, tx_C, gcl_pre, H)

        # Case C arrives first: C finishes at c_C_first_16[c], B waits behind max(a_B, c_C)
        c_first_mask = ~b_first_mask
        c_C_pair[c_first_mask] = c_C_first_16[c_flat[c_first_mask]]
        earliest_B_after_C = np.maximum(a_B_16[b_flat[c_first_mask]], c_C_pair[c_first_mask])
        c_B_pair[c_first_mask] = next_fit_q7_tx_finish_vec(earliest_B_after_C, tx_B, gcl_pre, H)

        # Combine marginal + co-occurrence downstream arrivals
        for grp, dst_p, c_marg, w_marg, c_co in [
            ("B", "e01", c_B_marg, w_B_marg, c_B_pair),
            ("C", "e02", c_C_marg, w_C_marg, c_C_pair),
        ]:
            t_all = np.concatenate([c_marg + hop_delay_ns, c_co + hop_delay_ns])
            w_all = np.concatenate([w_marg, w_pair])
            in_w = t_all < H
            b_d = (t_all[in_w] // bin_ns).astype(np.int64)
            w_in = w_all[in_w]
            pi_d = np.bincount(b_d, weights=w_in, minlength=num_bins).astype(np.float64)[:num_bins]
            m_sum = float(np.sum(w_in))
            e_sum = float(np.sum(w_in * t_all[in_w]))
            o_mass = float(np.sum(w_all[~in_w]))

            q_p = float(np.clip(m_sum, 0.0, 1.0))
            pi_p = (pi_d / q_p) if q_p > 1e-9 else np.zeros(num_bins, dtype=np.float64)
            mu_p = (e_sum / q_p) if q_p > 1e-9 else 0.0
            messages[(grp, "e4u", dst_p)] = PropagatedArrivalMessage(
                cycle_k=cycle_k,
                group=grp,
                src_port="e4u",
                dst_port=dst_p,
                mode="INTERVAL" if is_interval_mode else "PROB",
                q_prop=q_p,
                pi_prop=pi_p,
                mu_t_prop_ns=float(mu_p),
                bounds_low=np.maximum(0.0, pi_d - 0.03),
                bounds_high=np.minimum(1.0, pi_d + 0.03),
                valid=1,
                out_of_window_mass=float(o_mass),
                candidate_id=candidate_id,
                config_hash=config_hash,
            )

    return messages


def inject_prediction_noise(
    q_loc: float,
    pi_loc: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    r"""
    Literature-grounded dual-layer perturbation (Section 5 of `实验分组与外部对比计划.md`):
    1. Occurrence probability logit noise: \tilde{q} = \sigma(logit(q) + \epsilon_q)
    2. Conditional bin distribution blurring & shift: \tilde{\pi} = Softmax((\log\pi + \epsilon_\pi)/\tau_\eta)
    """
    if noise_level <= 1e-9:
        return q_loc, pi_loc.copy()
    q_clip = float(np.clip(q_loc, 1e-4, 1.0 - 1e-4))
    logit_q = np.log(q_clip / (1.0 - q_clip))
    sigma_q = 1.6 * noise_level
    q_noisy = float(1.0 / (1.0 + np.exp(-(logit_q + rng.normal(0.0, sigma_q)))))

    log_pi = np.log(np.clip(pi_loc, 1e-6, 1.0))
    eps_pi = rng.normal(0.0, 2.2 * noise_level, size=pi_loc.shape)
    tau = 1.0 + 1.8 * noise_level
    z = (log_pi + eps_pi) / tau
    z = z - np.max(z)
    exp_z = np.exp(z)
    pi_noisy = exp_z / np.sum(exp_z)
    return q_noisy, pi_noisy
