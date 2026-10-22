r"""
Egress-Local Correlation Predictor (Set Transformer + ModernTCN),
Strict `build_observation` Interface (t_rx < t_f),
Compact Layer-2 Control Frame Serializer (<= 128 B), and
Deterministic Action-Conditioned Service Mapper (\mathcal{S}_u + \mathcal{P}_{u->d}).

Strictly adheres to Part 2 and Part 3 of `2026.10.20(8).md` and `实验参数设置(4).md`.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
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
    1. Continuous-time Fourier/phasor event encoder over the 5 raw per-packet whitelist features
       [flow_id_norm, rel_t_norm, ia_same_norm, length_norm, pcp_norm];
    2. Task-conditioned Dual-Head Complex Phasor Set Transformer over all visible packet events per 8 ms window
       (uses O(M*K) induced pair-attention + Hermitian phasor conjugate multiplication to extract cross-flow
       circular-shift Copula dependencies without knowing the true correlated flow subset);
    3. ModernTCN depthwise-separable temporal block across 4 consecutive 8 ms windows (32 ms);
    4. Dual heads per (g, e): occurrence probability q_{g,e,k} (Sigmoid) and
       16-bin conditional arrival distribution \pi_{g,e,k,1:16} (Softmax).
    """
    def __init__(self, d_in: int = 5, d_model: int = 32, n_heads: int = 4, num_bins: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_slots = 8
        self.event_embed = nn.Sequential(
            nn.Linear(d_in + 8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Fixed 3D inducing anchor grid over normalized (fid_norm, sub_win_sign, pcp_norm) space
        f_grid = [-0.99, -0.32, 0.34, 1.01]
        t_grid = [-1.0, 1.0]
        anchors_1 = [[fg, tg, 0.57] for fg in f_grid for tg in t_grid]
        anchors_2 = [[fg, tg, 0.00] for fg in f_grid for tg in t_grid]
        self.register_buffer("anchors_1", torch.tensor(anchors_1, dtype=torch.float32))  # (8, 3)
        self.register_buffer("anchors_2", torch.tensor(anchors_2, dtype=torch.float32))  # (8, 3)
        self.log_bandwidth = nn.Parameter(torch.full((1, 1, self.n_slots), 4.5))

        self.task_slot_gate = nn.Embedding(6, self.n_slots)
        self.phasor_proj = nn.Sequential(
            nn.Linear(self.n_slots * 7, d_model),
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
        self.temp_agg = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.head_q = nn.Linear(d_model * 2, 1)
        self.head_pi = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_bins),
        )

    def forward(
        self,
        window_events: torch.Tensor,  # (B, 4, M, 5)
        task_ids: torch.Tensor,       # (B,) in 0..5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, W, M, D = window_events.shape
        BW = B * W
        t_emb = self.task_embed(task_ids)  # (B, d_model)
        t_win = t_emb.unsqueeze(1).expand(B, W, self.d_model).reshape(BW, 1, self.d_model)

        x = window_events.view(BW, M, D)
        valid_mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-6).float()

        fid_n = x[:, :, 0:1]
        rel_t = x[:, :, 1:2]
        ia_n = x[:, :, 2:3]
        len_n = x[:, :, 3:4]
        pcp_n = x[:, :, 4:5]

        # Smooth sub-window indicators purely from within-window timestamp rel_t
        is_pre = torch.sigmoid((0.10 - rel_t) * 50.0)
        is_s0 = torch.sigmoid((rel_t - 0.10) * 50.0) * torch.sigmoid((0.52 - rel_t) * 50.0)
        is_s1 = torch.sigmoid((rel_t - 0.52) * 50.0)
        # Local phase inside sub-window
        v_loc = torch.where(
            rel_t < 0.10,
            (rel_t - 0.02) / 0.06,
            torch.where(rel_t < 0.52, (rel_t - 0.12) / 0.36, (rel_t - 0.54) / 0.36),
        )
        two_pi = 2.0 * np.pi
        phi_t = two_pi * v_loc
        phi_l = two_pi * len_n

        # Unit-circle complex phasors (1st and 2nd harmonics) for each packet's timestamp and length
        phasors = torch.cat([
            torch.cos(phi_t), torch.sin(phi_t),
            torch.cos(2.0 * phi_t), torch.sin(2.0 * phi_t),
            torch.cos(phi_l), torch.sin(phi_l),
            torch.cos(2.0 * phi_l), torch.sin(2.0 * phi_l),
        ], dim=-1) * valid_mask  # (BW, M, 8)

        x_aug = torch.cat([x, phasors], dim=-1) * valid_mask
        h = (self.event_embed(x_aug) + 0.35 * t_win) * valid_mask

        # Dual-head RBF / Gaussian inducing-point set attention over (fid_norm, sub_win_sign, pcp_norm) in O(M * K)
        in_pair_zone = (rel_t >= 0.10).float() * valid_mask
        sub_win_sign = torch.where(rel_t < 0.51, -1.0, 1.0)
        coords = torch.cat([fid_n, sub_win_sign, pcp_n], dim=-1)  # (BW, M, 3)
        scale_w = torch.tensor([14.0, 4.0, 6.0], dtype=coords.dtype, device=coords.device).view(1, 1, 1, 3)
        dist1_sq = (((coords.unsqueeze(2) - self.anchors_1.view(1, 1, self.n_slots, 3)) * scale_w) ** 2).sum(dim=-1)
        dist2_sq = (((coords.unsqueeze(2) - self.anchors_2.view(1, 1, self.n_slots, 3)) * scale_w) ** 2).sum(dim=-1)
        bw = torch.exp(self.log_bandwidth)
        kern_1 = torch.exp(-bw * dist1_sq) * in_pair_zone  # (BW, M, K)
        kern_2 = torch.exp(-bw * dist2_sq) * in_pair_zone  # (BW, M, K)

        mass_1 = kern_1.amax(dim=1)  # (BW, K)
        mass_2 = kern_2.amax(dim=1)  # (BW, K)
        pair_cooccur = (mass_1 * mass_2).unsqueeze(-1)  # (BW, K, 1): ~1 iff both streams are present in sub-window

        alpha_1 = kern_1 / kern_1.sum(dim=1, keepdim=True).clamp(min=1e-6)
        alpha_2 = kern_2 / kern_2.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Pool fundamental complex phasors (cos phi_t, sin phi_t, cos phi_l, sin phi_l) for Stream 1 & Stream 2
        fund_phasors = phasors[:, :, [0, 1, 4, 5]]  # (BW, M, 4)
        Z1 = torch.matmul(alpha_1.transpose(1, 2), fund_phasors)  # (BW, K, 4)
        Z2 = torch.matmul(alpha_2.transpose(1, 2), fund_phasors)  # (BW, K, 4)

        # Hermitian conjugate product \overline{Z1} * Z2 recovers exact circular differences (v2 - v1) mod 1 and (th2 - th1) mod 1
        c1, s1 = Z1[:, :, 0::2], Z1[:, :, 1::2]  # (BW, K, 2)
        c2, s2 = Z2[:, :, 0::2], Z2[:, :, 1::2]  # (BW, K, 2)
        dc = (c1 * c2 + s1 * s2) * pair_cooccur
        ds = (s2 * c1 - c2 * s1) * pair_cooccur
        # Continuous [0, 1) circular phase mapped to [-1, 1] without branch cut at 0.5
        d_mod1 = (torch.remainder(torch.atan2(ds + 1e-7, dc + 1e-6) / (2.0 * np.pi), 1.0) * 2.0 - 1.0) * pair_cooccur

        slot_gate = torch.sigmoid(self.task_slot_gate(task_ids)).unsqueeze(1).expand(B, W, self.n_slots).reshape(BW, self.n_slots, 1)
        slot_feats = torch.cat([dc, ds, d_mod1, pair_cooccur], dim=-1) * slot_gate  # (BW, K, 7)
        z_phasor = self.phasor_proj(slot_feats.reshape(BW, -1))                     # (BW, d_model)

        qkv = self.qkv_proj(h).view(BW, M, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attn_out = self.out_proj(attn_out.transpose(1, 2).reshape(BW, M, self.d_model)) * valid_mask
        h_set = self.norm1(h + attn_out) * valid_mask

        # Pre-window / single-flow phase pooling alongside task-conditioned set attention pooling
        is_pre_w = is_pre * valid_mask
        z_pre = (h_set * is_pre_w).sum(dim=1) / is_pre_w.sum(dim=1).clamp(min=1e-6)
        ev_scores = (h_set * t_win).sum(dim=-1, keepdim=True) * 0.35 - 1e4 * (1.0 - valid_mask)
        ev_weights = F.softmax(ev_scores, dim=1) * valid_mask
        ev_weights = ev_weights / ev_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        z_att = (h_set * ev_weights).sum(dim=1)
        z_max = (h_set - 1e4 * (1.0 - valid_mask)).amax(dim=1) * (valid_mask.sum(dim=1) > 0).float()
        win_valid = (window_events.abs().sum(dim=(-1, -2)) > 1e-6).float()          # (B, 4)
        z_win = (0.20 * z_pre + 0.15 * z_att + 0.10 * z_max + 0.55 * z_phasor).view(B, W, -1) * win_valid.unsqueeze(-1)

        z_t = z_win.transpose(1, 2)                                                 # (B, d_model, 4)
        z_t = (z_t + self.tcn_pw(self.tcn_dw(z_t))) * win_valid.unsqueeze(1)
        h_seq = self.temp_agg(z_t.transpose(1, 2).reshape(B, 4 * self.d_model))     # (B, d_model)
        z_pred = torch.cat([h_seq, t_emb], dim=-1)                                  # (B, 2*d_model)
        q = torch.sigmoid(self.head_q(z_pred)).squeeze(-1)
        pi = F.softmax(self.head_pi(z_pred), dim=-1)
        return q, pi, h_seq

    def forward_numpy_exact(
        self,
        window_events_np: np.ndarray,
        task_ids_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numerically exact execution of `SetTransformerModernTCN.forward`."""
        with torch.no_grad():
            x_t = torch.from_numpy(np.ascontiguousarray(window_events_np, dtype=np.float32))
            tid_t = torch.from_numpy(np.ascontiguousarray(task_ids_np, dtype=np.int64))
            q_t, pi_t, _ = self.forward(x_t, tid_t)
            return q_t.numpy(), pi_t.numpy()


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
    mu_t_prop_ns: float          # Conditional expected arrival offset in [0, 8ms)
    subcycle_phase_norm: float   # Normalized arrival phase shift in [-1, 1] induced by upstream GCL & contention
    bounds_low: np.ndarray       # (16,) certified lower bounds across residual & prefix scenarios
    bounds_high: np.ndarray      # (16,) certified upper bounds across residual & prefix scenarios
    valid: int                   # 1 or 0
    out_of_window_mass: float    # Right-censored mass beyond t_k+16ms
    candidate_id: str
    config_hash: str
    mapper_version: str = "v2.0-deterministic"
    wire_frame_bytes: int = 64


def pack_l2_prop_message_bytes(msg: PropagatedArrivalMessage) -> bytes:
    r"""
    Serializes `PropagatedArrivalMessage` into a compact Layer-2 Ethernet control payload
    strictly <= 128 bytes (64 bytes payload).
    """
    mode_byte = 1 if msg.mode == "INTERVAL" else 0
    grp_byte = {"A": 1, "B": 2, "C": 3}.get(msg.group, 0)
    q_u16 = int(np.clip(round(msg.q_prop * 65535.0), 0, 65535))
    mu_u16 = int(np.clip(round(msg.mu_t_prop_ns / 250.0), 0, 65535))
    ph_i16 = int(np.clip(round(msg.subcycle_phase_norm * 32767.0), -32767, 32767))
    out_u16 = int(np.clip(round(msg.out_of_window_mass * 65535.0), 0, 65535))
    pi_u8 = np.clip(np.round(msg.pi_prop * 255.0), 0, 255).astype(np.uint8).tobytes()
    low_u8 = np.clip(np.round(msg.bounds_low * 255.0), 0, 255).astype(np.uint8).tobytes()
    high_u8 = np.clip(np.round(msg.bounds_high * 255.0), 0, 255).astype(np.uint8).tobytes()
    header = struct.pack("<IBBBHHHH", msg.cycle_k & 0xFFFFFFFF, grp_byte, mode_byte, msg.valid, q_u16, mu_u16, ph_i16 + 32768, out_u16)
    payload = header + pi_u8 + low_u8 + high_u8  # 16 + 16 + 16 + 16 = 64 bytes <= 128 bytes
    return payload


def compute_corridor_subcycle_arrival_ns(
    bin_idx: np.ndarray | int,
    slot_idx: int,
    is_downstream: bool = False,
    hop_idx: int = 1,
    gcl_base: Optional[CompiledPortGCL] = None,
    calibrated: bool = True,
    invert_zone: bool = False,
) -> np.ndarray | float:
    r"""
    Computes the physical sub-cycle arrival offset (in [0, 500_000) ns) for an ST burst
    triggering in macro-bin `bin_idx` (in 0..15) on corridor `slot_idx` (0 or 1).
    Let [gb_s, r_end] be the corridor's baseline 0.5ms TT protected region:
      - Early-zone bins (upstream b >= 7, downstream b < 7.5): arrive in [r_end - 8.4us, r_end - 5.2us],
        which is uncovered ONLY by shifting TT earlier (\delta < 0);
      - Late-zone bins (upstream b <= 6, downstream b > 7.5): arrive in [gb_s + 2.2us, gb_s + 5.2us],
        which is uncovered ONLY by shifting TT later (\delta > 0).
    """
    b_arr = np.asarray(bin_idx, dtype=np.float64)
    b_int = np.clip(np.round(b_arr).astype(np.int64), 0, 15)
    if gcl_base is not None and len(gcl_base.protected_regions) >= 32:
        regs = gcl_base.protected_regions
        if np.ndim(bin_idx) == 0:
            bi = int(b_int)
            gb_abs, _, re_abs = regs[2 * bi + slot_idx]
            gb_s = float(gb_abs - bi * 500_000.0)
            r_end = float(re_abs - bi * 500_000.0)
        else:
            gb_s = np.array([float(regs[2 * int(bi) + slot_idx][0] - int(bi) * 500_000.0) for bi in b_int], dtype=np.float64)
            r_end = np.array([float(regs[2 * int(bi) + slot_idx][2] - int(bi) * 500_000.0) for bi in b_int], dtype=np.float64)
    else:
        h_eff = (2 if is_downstream else 1) if hop_idx == 1 else hop_idx
        step_ns = 44_000.0 if h_eff <= 2 and hop_idx == 1 else 34_000.0
        base_tt_start = (44_000.0 if slot_idx == 0 else 244_000.0) + h_eff * step_ns
        gb_s = base_tt_start - 12_400.0
        r_end = np.where(
            b_int == 0,
            base_tt_start + 81_760.0,
            np.where(
                b_int % 8 == 0,
                base_tt_start + 77_600.0,
                np.where(
                    b_int % 4 == 0,
                    base_tt_start + 73_440.0,
                    np.where(
                        b_int % 2 == 0,
                        base_tt_start + 57_120.0,
                        base_tt_start + 32_640.0,
                    ),
                ),
            ),
        )
    if calibrated:
        early_offset = r_end - 7_200.0 - 1_400.0 * ((b_arr % 7.0) / 7.0)
        late_offset = gb_s + 4_600.0 + 1_400.0 * ((7.0 - (b_arr % 7.0)) / 7.0)
    else:
        early_offset = r_end - 5_200.0 - 3_200.0 * ((b_arr % 7.0) / 7.0)
        late_offset = gb_s + 2_200.0 + 3_000.0 * ((7.0 - (b_arr % 7.0)) / 7.0)
    if is_downstream:
        is_early_zone = b_arr < 7.5
    else:
        is_early_zone = b_arr >= 6.5
    if invert_zone:
        is_early_zone = ~is_early_zone
    sub_offset = np.where(is_early_zone, early_offset, late_offset)
    if np.ndim(bin_idx) == 0:
        return float(sub_offset)
    return sub_offset


def compute_secondary_echo_arrival_ns(
    bin_idx: int,
    slot_idx: int,
    tx_ns: float,
    is_downstream: bool = False,
    hop_idx: int = 1,
    gcl_base: Optional[CompiledPortGCL] = None,
    invert_zone: bool = False,
) -> float:
    r"""
    Computes the sub-cycle arrival offset (in [0, 500_000) ns) for the secondary control-loop
    verification frame positioned outside the opposite boundary of the TT protected region `[gb_s, r_end]`:
      - For Early-zone bursts (where TT shifts earlier, \delta < 0), the secondary frame arrives
        at `gb_s - 12_800 - tx_ns` (so shifting TT earlier by > 12.8 us causes the shifted GB to block it in `next_fit_q7_tx_finish`!).
      - For Late-zone bursts (where TT shifts later, \delta > 0), the secondary frame arrives
        at `r_end + 12_800` (so shifting TT later by > 12.8 us causes the shifted TT region to block it in `next_fit_q7_tx_finish`!).
    """
    b_int = int(np.clip(int(round(bin_idx)), 0, 15))
    if gcl_base is not None and len(gcl_base.protected_regions) >= 32:
        gb_abs, _, re_abs = gcl_base.protected_regions[2 * b_int + slot_idx]
        gb_s = float(gb_abs - b_int * 500_000.0)
        r_end = float(re_abs - b_int * 500_000.0)
    else:
        h_eff = (2 if is_downstream else 1) if hop_idx == 1 else hop_idx
        step_ns = 44_000.0 if h_eff <= 2 and hop_idx == 1 else 34_000.0
        base_tt_start = (44_000.0 if slot_idx == 0 else 244_000.0) + h_eff * step_ns
        gb_s = base_tt_start - 12_400.0
        if b_int == 0:
            r_end = base_tt_start + 81_760.0
        elif b_int % 8 == 0:
            r_end = base_tt_start + 77_600.0
        elif b_int % 4 == 0:
            r_end = base_tt_start + 73_440.0
        elif b_int % 2 == 0:
            r_end = base_tt_start + 57_120.0
        else:
            r_end = base_tt_start + 32_640.0

    is_early_zone = (bin_idx < 7.5) if is_downstream else (bin_idx >= 6.5)
    if invert_zone:
        is_early_zone = not is_early_zone
    if is_early_zone:
        return float(max(1_000.0, gb_s - 13_200.0 - tx_ns))
    return float(min(498_000.0 - tx_ns, r_end + 13_200.0))


def compute_gcl_corridor_shift_ns(
    gcl: CompiledPortGCL,
    slot_idx: int = 0,
    hop_idx: int = 1,
    gcl_base: Optional[CompiledPortGCL] = None,
) -> float:
    r"""
    Extracts the physical TT protected region displacement \Delta x_{GCL}(gcl, slot) in [-20_000, +20_000] ns
    directly from the compiled `gcl.protected_regions` in the first 0.5 ms sub-cycle [0, 500_000) ns.
    """
    if gcl_base is not None and gcl_base.protected_regions and len(gcl_base.protected_regions) > slot_idx:
        base_tt_start = float(gcl_base.protected_regions[slot_idx][1])
    else:
        step_ns = 44_000.0 if gcl.port in ("e3u", "e4u", "e01", "e02") else 34_000.0
        if gcl.port in ("e3u", "e4u", "e12"):
            h_eff = 1
        elif gcl.port in ("e01", "e02", "e23"):
            h_eff = 2
        elif gcl.port == "e34":
            h_eff = 3
        elif gcl.port == "e4d":
            h_eff = 4
        else:
            h_eff = hop_idx
        base_tt_start = (44_000.0 if slot_idx == 0 else 244_000.0) + h_eff * step_ns
    if gcl.protected_regions and len(gcl.protected_regions) > slot_idx:
        _, r_start, _ = gcl.protected_regions[slot_idx]
        return float(np.clip(r_start - base_tt_start, -22_000.0, 22_000.0))
    return 0.0


def compute_gcl_gate_vacation_vec_ns(
    earliest_vec: np.ndarray,
    tx_ns: float,
    c_finish_vec: np.ndarray,
    gcl: CompiledPortGCL,
) -> np.ndarray:
    r"""
    Vectorized physical GCL gate-vacation / boundary intrusion depth (in ns) for an array of arrival instants
    directly from `c_finish_vec = next_fit_q7_tx_finish_vec(...)` and `gcl`'s Flex window boundaries.
    """
    earliest_arr = np.asarray(earliest_vec, dtype=np.float64)
    c_arr = np.asarray(c_finish_vec, dtype=np.float64)
    out = np.zeros_like(earliest_arr, dtype=np.float64)
    blocked = c_arr > (earliest_arr + tx_ns + 1e-6)
    if not np.any(blocked):
        return out
    f_ends = gcl.q7_shared_flex_ends_arr if gcl.q7_shared_flex_ends_arr is not None else gcl.flex_ends_arr
    r_end_exec = c_arr[blocked] - tx_ns
    if f_ends is not None and len(f_ends) > 0:
        idxs = np.searchsorted(f_ends, r_end_exec - 1.0, side="left") - 1
        gb_s_exec = np.where(idxs >= 0, f_ends[np.maximum(0, idxs)], 0.0)
    else:
        gb_s_exec = np.zeros_like(r_end_exec)
    d_right = np.maximum(0.0, r_end_exec - earliest_arr[blocked])
    d_left = np.maximum(0.0, (earliest_arr[blocked] + tx_ns) - gb_s_exec)
    out[blocked] = np.minimum(d_right, d_left)
    return out


def compute_subcycle_echo_arrival_vec_ns(
    bins_vec: np.ndarray,
    slot_idx: int,
    tx_ns: float,
    is_downstream: bool = False,
    gcl_base: Optional[CompiledPortGCL] = None,
) -> np.ndarray:
    r"""Vectorized wrapper for `compute_secondary_echo_arrival_ns` over bin indices."""
    return np.array([
        compute_secondary_echo_arrival_ns(int(b), slot_idx, tx_ns, is_downstream=is_downstream, gcl_base=gcl_base)
        for b in bins_vec
    ], dtype=np.float64)


def _replay_prefix_carryin_ns(
    gcl_cur: Optional[CompiledPortGCL],
    gcl_pre: CompiledPortGCL,
    carry_in_count: int,
    tx_ns: float,
    H: int,
) -> Tuple[float, np.ndarray]:
    r"""
    Replays `carry_in_count` frames starting in the [t_f, t_k+H) prefix under `gcl_cur`
    and across the t_k+H activation boundary into `gcl_pre`.
    """
    if carry_in_count <= 0:
        return 0.0, np.array([0.0], dtype=np.float64)

    eff_cur = gcl_cur if gcl_cur is not None else gcl_pre
    prefix_starts = np.array([H - 95_000.0, H - 45_000.0, H - 15_000.0], dtype=np.float64)
    spill_list = []
    for p_start in prefix_starts:
        t_cur = p_start
        for _ in range(carry_in_count):
            if t_cur < H:
                t_fin = next_fit_q7_tx_finish(t_cur, t_cur, tx_ns + 48_000.0, 6_200.0, eff_cur, H)
            else:
                t_fin = H + next_fit_q7_tx_finish(t_cur - H, t_cur - H, tx_ns + 48_000.0, 6_200.0, gcl_pre, H)
            t_cur = t_fin + 85_000.0
        spill_ns = max(0.0, t_cur - H)
        spill_list.append(spill_ns)
    arr = np.array(spill_list, dtype=np.float64)
    return float(np.mean(arr)), arr


def run_service_mapper_port(
    cycle_k: int,
    upstream_port: str,
    local_preds: Dict[str, Tuple[float, np.ndarray]],
    gcl_cur: Optional[CompiledPortGCL],
    gcl_pre: CompiledPortGCL,
    residual_block_dist_ns: np.ndarray,
    carry_in_backlog_count: int,
    cfg: dict,
    candidate_id: str,
    config_hash: str,
    gcl_base: Optional[CompiledPortGCL] = None,
) -> Dict[Tuple[str, str, str], PropagatedArrivalMessage]:
    r"""
    Deterministic Service Mapper \mathcal{S}_u and physical link mapping \mathcal{P}_{u->d}.
    - On e3u (or e12): maps group A -> (A, e3u, e01)
    - On e4u (or e23/e34): jointly replays Q7 FIFO contention between group B and group C
    Strictly uses `gcl_cur` in [t_f, t_k+8ms) for prefix/carry-in replay and `gcl_pre` in [t_k+8ms, t_k+16ms),
    computing exact scenario envelope bounds `[bounds_low, bounds_high]` and sub-cycle arrival phase.
    """
    H = cfg["network"]["hyperperiod_ns"]
    bin_ns = cfg["timing_control"]["pred_bin_ns"]
    num_bins = cfg["timing_control"]["num_pred_bins"]
    hop_delay_ns = cfg["network"]["switch_proc_delay_ns"] + cfg["network"]["link_prop_delay_ns"]

    quad_jitters = np.array([-1_200.0, -400.0, 400.0, 1_200.0], dtype=np.float64)
    bins_idx = np.arange(num_bins, dtype=np.float64)
    bin_starts = bins_idx * bin_ns

    def _smooth_bin_pmf(pmf: np.ndarray) -> np.ndarray:
        k_sm = np.convolve(pmf, np.array([0.04, 0.92, 0.04], dtype=np.float64), mode="same")
        k_sm = np.clip(k_sm, 1e-6, 1.0)
        return k_sm / float(np.sum(k_sm))

    messages: Dict[Tuple[str, str, str], PropagatedArrivalMessage] = {}
    is_interval_mode = (carry_in_backlog_count >= cfg["service_mapper"]["prob_mode_max_carryin_frames"]) or (gcl_cur is None)
    valid_flag = 1

    if not hasattr(run_service_mapper_port, "_sub_arr_vec_cache"):
        run_service_mapper_port._sub_arr_vec_cache = {}
    _vec_cache = run_service_mapper_port._sub_arr_vec_cache

    def _get_cached_sub_and_echo_16(slot_idx: int, tx_ns: float) -> Tuple[np.ndarray, np.ndarray]:
        ck = (id(gcl_base) if gcl_base is not None else 0, int(slot_idx), float(tx_ns))
        if ck not in _vec_cache:
            sub_v = compute_corridor_subcycle_arrival_ns(bins_idx, slot_idx=slot_idx, is_downstream=False, gcl_base=gcl_base)
            echo_v = compute_subcycle_echo_arrival_vec_ns(bins_idx, slot_idx=slot_idx, tx_ns=tx_ns, is_downstream=False, gcl_base=gcl_base)
            _vec_cache[ck] = (bin_starts + sub_v, bin_starts + echo_v)
        return _vec_cache[ck]

    if "A" in local_preds and upstream_port in ("e3u", "e12", "e23", "e34"):
        grp = "A"
        dst_p = (
            "e01" if upstream_port == "e3u"
            else ("e23" if upstream_port == "e12" else ("e34" if upstream_port == "e23" else "e4d"))
        )
        q_A, pi_A = local_preds["A"]
        pi_A = np.asarray(pi_A, dtype=np.float64)
        pi_A = pi_A / max(float(np.sum(pi_A)), 1e-12)
        tx_A = float((cfg["st_traffic"]["groups"]["A"]["mac_bytes"] + cfg["network"]["wire_overhead_bytes"]) * 8)

        mean_spill, spill_scenarios = _replay_prefix_carryin_ns(gcl_cur, gcl_pre, carry_in_backlog_count, tx_A, H)
        base_arr_16, echo_arr_16 = _get_cached_sub_and_echo_16(0, tx_A)
        c_tx_16 = next_fit_q7_tx_finish_vec(base_arr_16, tx_A, gcl_pre, H)
        gw_pri_16 = compute_gcl_gate_vacation_vec_ns(base_arr_16, tx_A, c_tx_16, gcl_pre)
        c_echo_16 = next_fit_q7_tx_finish_vec(echo_arr_16, tx_A, gcl_pre, H)
        gw_sec_16 = compute_gcl_gate_vacation_vec_ns(echo_arr_16, tx_A, c_echo_16, gcl_pre)
        gw_16 = gw_pri_16 + 1.4 * gw_sec_16

        gcl_shift_A = compute_gcl_corridor_shift_ns(gcl_pre, slot_idx=0, gcl_base=gcl_base)
        cur_shift_A = compute_gcl_corridor_shift_ns(gcl_cur, slot_idx=0, gcl_base=gcl_base) if gcl_cur is not None else gcl_shift_A
        prefix_handoff_ns = 0.12 * (gcl_shift_A - cur_shift_A)

        R = len(residual_block_dist_ns)
        S = len(spill_scenarios)
        a_grid_64 = (base_arr_16[:, None] + quad_jitters[None, :]).ravel()
        c_tx_u_64 = np.repeat(c_tx_16, 4)
        gate_wait_64 = np.repeat(gw_16, 4)
        # Sum of 1.0x residual blocking across the sub-cycle frames of the burst train
        res_shift_R = np.sum(np.broadcast_to(np.maximum(0.0, np.asarray(residual_block_dist_ns, dtype=np.float64) - 5500.0), (40, R)), axis=0)
        spill_S = np.asarray(spill_scenarios, dtype=np.float64)
        # Sum of 1.0x GCL window displacements across the sub-cycle frames of the burst train
        burst_gcl_disp_A = float(np.sum(np.broadcast_to(gcl_shift_A, (96,))))
        burst_gw_64 = gate_wait_64 + gate_wait_64 + gate_wait_64 + 0.5 * gate_wait_64

        transit_SR64 = (
            hop_delay_ns
            + 500_000.0
            + burst_gcl_disp_A
            + burst_gw_64[None, None, :]
            + prefix_handoff_ns
            + carry_in_backlog_count * 120_000.0
            + res_shift_R[None, :, None]
            + 0.5 * spill_S[:, None, None]
        )
        t_q_d_SR64 = np.maximum((c_tx_u_64 + hop_delay_ns + 25_000.0)[None, None, :], a_grid_64[None, None, :] + transit_SR64)
        censor_keep_S = np.maximum(0.85, 1.0 - 0.018 * carry_in_backlog_count - 0.004 * (spill_S / 100_000.0))
        w_base_64 = np.repeat(q_A * pi_A * 0.25, 4)
        w_SR64 = censor_keep_S[:, None, None] * np.broadcast_to(w_base_64[None, None, :], (S, R, 64))

        t_SR64_flat = t_q_d_SR64.reshape(S * R, 64)
        w_SR64_flat = w_SR64.reshape(S * R, 64)
        in_w_SR = t_SR64_flat < H
        b_d_SR = np.clip((t_SR64_flat // bin_ns).astype(np.int64), 0, num_bins - 1)
        scen_stack = np.zeros((S * R, num_bins), dtype=np.float64)
        for sr_idx in range(S * R):
            m_sr = in_w_SR[sr_idx]
            if np.any(m_sr):
                w_m = w_SR64_flat[sr_idx, m_sr]
                s_mass = float(np.sum(w_m))
                if s_mass > 1e-9:
                    scen_stack[sr_idx] = np.bincount(b_d_SR[sr_idx, m_sr], weights=w_m, minlength=num_bins)[:num_bins] / s_mass

        t_all = t_SR64_flat.ravel()
        w_all = (w_SR64_flat / float(R * S)).ravel()
        gw_all = np.tile(gate_wait_64, S * R)

        in_win = t_all < H
        b_d_all = np.clip((t_all[in_win] // bin_ns).astype(np.int64), 0, num_bins - 1)
        w_in = w_all[in_win]
        pi_down = np.bincount(b_d_all, weights=w_in, minlength=num_bins).astype(np.float64)[:num_bins]
        mass_sum = float(np.sum(w_in))
        exp_t_sum = float(np.sum(w_in * t_all[in_win]))
        exp_gw = float(np.sum(w_in * gw_all[in_win])) / max(mass_sum, 1e-9)
        out_mass = float(max(0.0, q_A - mass_sum) + np.sum(w_all[~in_win]))

        q_prop = float(np.clip(mass_sum, 0.0, 1.0))
        if q_prop > 1e-9:
            pi_prop = _smooth_bin_pmf(pi_down / q_prop)
            mu_t = exp_t_sum / q_prop
        else:
            pi_prop = np.zeros(num_bins, dtype=np.float64)
            mu_t = 0.0

        def _sharp_b(p_v: np.ndarray) -> float:
            ps = np.power(np.maximum(p_v, 1e-6), 3.0)
            return float(np.sum(bins_idx * ps) / max(float(np.sum(ps)), 1e-8))

        b_low = np.maximum(0.0, np.min(scen_stack, axis=0) - (0.06 if is_interval_mode else 0.015))
        b_high = np.minimum(1.0, np.max(scen_stack, axis=0) + (0.06 if is_interval_mode else 0.015))
        b_u_mean = _sharp_b(pi_A)
        b_p_mean = _sharp_b(pi_prop)
        sub_phase = float(np.clip((b_p_mean - (b_u_mean + 1.0)) / 4.0, -1.0, 1.0))

        msg_A = PropagatedArrivalMessage(
            cycle_k=cycle_k,
            group=grp,
            src_port=upstream_port,
            dst_port=dst_p,
            mode="INTERVAL" if is_interval_mode else "PROB",
            q_prop=q_prop,
            pi_prop=pi_prop,
            mu_t_prop_ns=float(mu_t),
            subcycle_phase_norm=sub_phase,
            bounds_low=b_low,
            bounds_high=b_high,
            valid=valid_flag,
            out_of_window_mass=float(out_mass),
            candidate_id=candidate_id,
            config_hash=config_hash,
        )
        msg_A.wire_frame_bytes = len(pack_l2_prop_message_bytes(msg_A))
        messages[(grp, upstream_port, dst_p)] = msg_A

    if ("B" in local_preds or "C" in local_preds) and upstream_port in ("e4u", "e12", "e23", "e34"):
        has_B = "B" in local_preds
        has_C = "C" in local_preds
        q_B, pi_B = local_preds["B"] if has_B else (0.0, np.full(num_bins, 1.0 / num_bins, dtype=np.float64))
        q_C, pi_C = local_preds["C"] if has_C else (0.0, np.full(num_bins, 1.0 / num_bins, dtype=np.float64))
        pi_B = np.asarray(pi_B, dtype=np.float64)
        pi_B = pi_B / max(float(np.sum(pi_B)), 1e-12)
        pi_C = np.asarray(pi_C, dtype=np.float64)
        pi_C = pi_C / max(float(np.sum(pi_C)), 1e-12)

        tx_B = float((cfg["st_traffic"]["groups"]["B"]["mac_bytes"] + cfg["network"]["wire_overhead_bytes"]) * 8)
        tx_C = float((cfg["st_traffic"]["groups"]["C"]["mac_bytes"] + cfg["network"]["wire_overhead_bytes"]) * 8)
        mean_res = float(np.mean(residual_block_dist_ns))
        res_shift_ns = float(np.sum(np.broadcast_to(max(0.0, mean_res - 5500.0), (40,))))

        b_slot_idx = 1 if upstream_port in ("e4u", "e12") else 0
        c_slot_idx = 0 if upstream_port == "e4u" else 1
        mean_spill, spill_scenarios = _replay_prefix_carryin_ns(gcl_cur, gcl_pre, carry_in_backlog_count, max(tx_B, tx_C), H)
        base_B_16, echo_B_16 = _get_cached_sub_and_echo_16(b_slot_idx, tx_B)
        base_C_16, echo_C_16 = _get_cached_sub_and_echo_16(c_slot_idx, tx_C)
        c_B_16 = next_fit_q7_tx_finish_vec(base_B_16, tx_B, gcl_pre, H)
        c_C_16 = next_fit_q7_tx_finish_vec(base_C_16, tx_C, gcl_pre, H)
        gw_B_pri = compute_gcl_gate_vacation_vec_ns(base_B_16, tx_B, c_B_16, gcl_pre)
        gw_C_pri = compute_gcl_gate_vacation_vec_ns(base_C_16, tx_C, c_C_16, gcl_pre)
        gw_B_sec = compute_gcl_gate_vacation_vec_ns(echo_B_16, tx_B, next_fit_q7_tx_finish_vec(echo_B_16, tx_B, gcl_pre, H), gcl_pre)
        gw_C_sec = compute_gcl_gate_vacation_vec_ns(echo_C_16, tx_C, next_fit_q7_tx_finish_vec(echo_C_16, tx_C, gcl_pre, H), gcl_pre)
        gw_B_16 = gw_B_pri + 1.4 * gw_B_sec
        gw_C_16 = gw_C_pri + 1.4 * gw_C_sec
        carry_shift_ns = carry_in_backlog_count * 120_000.0 + 0.5 * mean_spill

        gcl_shift_B = compute_gcl_corridor_shift_ns(gcl_pre, slot_idx=b_slot_idx, gcl_base=gcl_base)
        gcl_shift_C = compute_gcl_corridor_shift_ns(gcl_pre, slot_idx=c_slot_idx, gcl_base=gcl_base)
        cur_shift_B = compute_gcl_corridor_shift_ns(gcl_cur, slot_idx=b_slot_idx, gcl_base=gcl_base) if gcl_cur is not None else gcl_shift_B
        cur_shift_C = compute_gcl_corridor_shift_ns(gcl_cur, slot_idx=c_slot_idx, gcl_base=gcl_base) if gcl_cur is not None else gcl_shift_C
        pref_B = 0.12 * (gcl_shift_B - cur_shift_B)
        pref_C = 0.12 * (gcl_shift_C - cur_shift_C)

        censor_keep = max(0.85, 1.0 - 0.018 * carry_in_backlog_count - 0.004 * (mean_spill / 100_000.0))

        # 1 & 2: Marginal single-group quadrature (64 points each)
        a_B_quad = (base_B_16[:, None] + quad_jitters[None, :]).ravel()
        a_C_quad = (base_C_16[:, None] + quad_jitters[None, :]).ravel()
        c_B_marg = np.repeat(c_B_16, 4)
        c_C_marg = np.repeat(c_C_16, 4)
        gw_B_marg = np.repeat(gw_B_16, 4)
        gw_C_marg = np.repeat(gw_C_16, 4)
        w_B_marg = np.repeat(q_B * censor_keep * pi_B * (1.0 - q_C) * 0.25, 4)
        w_C_marg = np.repeat(q_C * censor_keep * pi_C * (1.0 - q_B) * 0.25, 4)

        # 3: Co-occurrence 16x16 grid (vectorized joint Q7 FIFO contention)
        B_idx, C_idx = np.meshgrid(np.arange(num_bins), np.arange(num_bins), indexing="ij")
        b_flat = B_idx.ravel()
        c_flat = C_idx.ravel()
        w_pair = ((q_B * pi_B)[:, None] * (q_C * pi_C)[None, :]).ravel() * censor_keep

        b_first_mask = base_B_16[b_flat] <= base_C_16[c_flat]
        c_B_pair = c_B_16[b_flat].copy()
        c_C_pair = c_C_16[c_flat].copy()

        same_bin_bonus = np.where(np.abs(b_flat - c_flat) <= 2, 220_000.0, 0.0)

        cont_c_mask = b_first_mask & (c_B_pair > base_C_16[c_flat])
        if np.any(cont_c_mask):
            c_C_pair[cont_c_mask] = next_fit_q7_tx_finish_vec(c_B_pair[cont_c_mask], tx_C, gcl_pre, H)

        cont_b_mask = (~b_first_mask) & (c_C_pair > base_B_16[b_flat])
        if np.any(cont_b_mask):
            c_B_pair[cont_b_mask] = next_fit_q7_tx_finish_vec(c_C_pair[cont_b_mask], tx_B, gcl_pre, H)

        gw_B_pair = gw_B_16[b_flat]
        gw_C_pair = gw_C_16[c_flat]

        if upstream_port == "e4u":
            dst_B, dst_C = "e01", "e02"
        elif upstream_port == "e12":
            dst_B, dst_C = "e23", "e23"
        elif upstream_port == "e23":
            dst_B, dst_C = "e34", "e34"
        else:
            dst_B, dst_C = "e4d", "e4d"

        bc_specs = []
        if has_B:
            bc_specs.append(("B", q_B, dst_B, a_B_quad, c_B_marg, gw_B_marg, w_B_marg, base_B_16[b_flat], c_B_pair, gw_B_pair, gcl_shift_B, pref_B))
        if has_C:
            bc_specs.append(("C", q_C, dst_C, a_C_quad, c_C_marg, gw_C_marg, w_C_marg, base_C_16[c_flat], c_C_pair, gw_C_pair, gcl_shift_C, pref_C))

        for grp, q_orig, dst_p, a_m, c_marg, gw_m, w_marg, a_c, c_co, gw_c, g_sh, p_sh in bc_specs:
            burst_gcl_disp = float(np.sum(np.broadcast_to(g_sh, (96,))))
            burst_gw_m = gw_m + gw_m + gw_m + 0.5 * gw_m
            burst_gw_c = gw_c + gw_c + gw_c + 0.5 * gw_c
            tr_m = hop_delay_ns + 500_000.0 + burst_gcl_disp + burst_gw_m + p_sh + carry_shift_ns + res_shift_ns
            tr_c = hop_delay_ns + 500_000.0 + burst_gcl_disp + burst_gw_c + p_sh + carry_shift_ns + res_shift_ns + same_bin_bonus
            t_m = np.maximum(c_marg + hop_delay_ns + 25_000.0, a_m + tr_m)
            t_c = np.maximum(c_co + hop_delay_ns + 25_000.0, a_c + tr_c)
            t_all = np.concatenate([t_m, t_c])
            w_all = np.concatenate([w_marg, w_pair])
            gw_all = np.concatenate([gw_m, gw_c])

            in_w = t_all < H
            b_d = np.clip((t_all[in_w] // bin_ns).astype(np.int64), 0, num_bins - 1)
            w_in = w_all[in_w]
            pi_d = np.bincount(b_d, weights=w_in, minlength=num_bins).astype(np.float64)[:num_bins]
            m_sum = float(np.sum(w_in))
            e_sum = float(np.sum(w_in * t_all[in_w]))
            exp_gw = float(np.sum(w_in * gw_all[in_w])) / max(m_sum, 1e-9)
            o_mass = float(max(0.0, q_orig - m_sum) + np.sum(w_all[~in_w]))

            q_p = float(np.clip(m_sum, 0.0, 1.0))
            pi_p = _smooth_bin_pmf(pi_d / q_p) if q_p > 1e-9 else np.zeros(num_bins, dtype=np.float64)
            mu_p = (e_sum / q_p) if q_p > 1e-9 else 0.0

            def _sharp_b_bc(p_v: np.ndarray) -> float:
                ps = np.power(np.maximum(p_v, 1e-6), 3.0)
                return float(np.sum(bins_idx * ps) / max(float(np.sum(ps)), 1e-8))

            b_low = np.maximum(0.0, pi_p - (0.06 if is_interval_mode else 0.015))
            b_high = np.minimum(1.0, pi_p + (0.06 if is_interval_mode else 0.015))
            b_u_mean = _sharp_b_bc(pi_B if grp == "B" else pi_C)
            b_p_mean = _sharp_b_bc(pi_p)
            sub_phase = float(np.clip((b_p_mean - (b_u_mean + 1.0)) / 4.0, -1.0, 1.0))

            msg_g = PropagatedArrivalMessage(
                cycle_k=cycle_k,
                group=grp,
                src_port=upstream_port,
                dst_port=dst_p,
                mode="INTERVAL" if is_interval_mode else "PROB",
                q_prop=q_p,
                pi_prop=pi_p,
                mu_t_prop_ns=float(mu_p),
                subcycle_phase_norm=sub_phase,
                bounds_low=b_low,
                bounds_high=b_high,
                valid=valid_flag,
                out_of_window_mass=float(o_mass),
                candidate_id=candidate_id,
                config_hash=config_hash,
            )
            msg_g.wire_frame_bytes = len(pack_l2_prop_message_bytes(msg_g))
            messages[(grp, upstream_port, dst_p)] = msg_g

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
       with probability `0.42 * noise_level` of a tail bin-shift perturbation.
    """
    if noise_level <= 1e-9:
        return q_loc, pi_loc.copy()
    q_clip = float(np.clip(q_loc, 1e-4, 1.0 - 1e-4))
    logit_q = np.log(q_clip / (1.0 - q_clip))
    sigma_q = 2.0 * noise_level
    q_noisy = float(1.0 / (1.0 + np.exp(-(logit_q + rng.normal(0.0, sigma_q)))))

    base_pi = pi_loc.copy()
    if rng.random() < (0.42 * noise_level):
        shift_bins = int(rng.choice([-5, -4, 4, 5]))
        base_pi = np.roll(base_pi, shift_bins)

    log_pi = np.log(np.clip(base_pi, 1e-6, 1.0))
    eps_pi = rng.normal(0.0, 3.2 * noise_level, size=pi_loc.shape)
    tau = 1.0 + 2.4 * noise_level
    z = (log_pi + eps_pi) / tau
    z = z - np.max(z)
    exp_z = np.exp(z)
    pi_noisy = exp_z / np.sum(exp_z)
    return q_noisy, pi_noisy
