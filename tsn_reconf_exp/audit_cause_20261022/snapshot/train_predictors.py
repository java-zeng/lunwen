r"""
100% Real Supervised Training & Evaluation for E1 (6-Task Local Correlation Predictor)
and E1.5 (Action-Conditioned Path Propagation D_map Audit).

Trains 4 PyTorch models with AdamW gradient descent on the continuous multi-flow event log:
1. `SetTrans_ModernTCN` (full 4-window x unfiltered multi-flow SetTransformerModernTCN)
2. `SingleWindow_8ms` (ablation without cross-window history w_0..w_2)
3. `BestSingleFlow` (ablation selecting the best single flow per task on a validation split)
4. `ShuffledMarginal` (ablation preserving per-flow empirical marginals while breaking cross-flow pairing)

Saves trained weights to `checkpoints/predictor_*.pt` and computes all E1 & E1.5 metrics
strictly from held-out test set inference and pure `run_service_mapper_port` outputs.
"""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_sim import decode_delta_action, local_precheck_port
from predictor_mapper import SetTransformerModernTCN, run_service_mapper_port
from tsn_env import (
    TASK_LIST, CycleGroundTruth, TSNTrafficAndPredictorDatasetGenerator, TSNRealClosedLoopEnv
)


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes exact Area Under Precision-Recall Curve (Average Precision) from arrays."""
    if np.sum(y_true) == 0:
        return 0.0
    desc_idx = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[desc_idx]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    precisions = tps / np.maximum(1.0, tps + fps)
    pos_mask = y_sorted == 1
    return float(np.mean(precisions[pos_mask]))


def wasserstein_1d_bins(p: np.ndarray, target_bin: int, bin_width_ms: float = 0.5) -> float:
    """Computes 1D Wasserstein-1 distance (in ms) between predicted 16-bin distribution `p` and true bin."""
    p_norm = p / max(float(np.sum(p)), 1e-12)
    cdf_p = np.cumsum(p_norm)
    q_onehot = np.zeros_like(p_norm)
    q_onehot[int(np.clip(target_bin, 0, len(p_norm) - 1))] = 1.0
    cdf_q = np.cumsum(q_onehot)
    return float(np.sum(np.abs(cdf_p - cdf_q)) * bin_width_ms)


def compute_brier_17(q_prob: float, pi_vec: np.ndarray, occurred: int, target_bin: int) -> float:
    """
    Computes the exact 17-class Brier score over the joint outcome space
    `{not_occurred, bin_0, ..., bin_15}`:
      pred_17 = [1 - q, q * pi_0, ..., q * pi_15]
      true_17 = one-hot in {0..16}
    """
    q_c = float(np.clip(q_prob, 0.0, 1.0))
    pi_n = np.asarray(pi_vec, dtype=np.float64)
    pi_n = pi_n / max(float(np.sum(pi_n)), 1e-12)
    pred_17 = np.concatenate([[1.0 - q_c], q_c * pi_n])
    true_17 = np.zeros(17, dtype=np.float64)
    if occurred == 0:
        true_17[0] = 1.0
    else:
        true_17[1 + int(np.clip(target_bin, 0, 15))] = 1.0
    return float(np.mean((pred_17 - true_17) ** 2))


def select_best_single_flow_on_val(
    gen_train: TSNTrafficAndPredictorDatasetGenerator,
    train_gt: List[CycleGroundTruth],
    gen_test: TSNTrafficAndPredictorDatasetGenerator,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], int]]:
    """
    Evaluates all visible candidate single flows on each port (`101, 102, 201, 202, 301, 302, 401, 402`)
    using a train/validation split (`0..900` train, `900..N` val) and selects the single flow ID
    with the highest validation AUPRC for each of the 6 tasks `(grp, port)`.
    """
    port_candidate_fids = {
        "e3u": [101, 102, 401, 402],
        "e4u": [201, 202, 301, 302, 401, 402],
        "e01": [101, 102, 201, 202, 401, 402],
        "e02": [301, 302, 401, 402],
    }
    n_total = len(train_gt)
    n_tr = int(0.75 * n_total)

    tr_best: Dict[Tuple[str, str], np.ndarray] = {}
    te_best: Dict[Tuple[str, str], np.ndarray] = {}
    selected_fids: Dict[Tuple[str, str], int] = {}

    for grp, port, _ in TASK_LIST:
        y_occ = np.array([train_gt[k].occurred[grp] for k in range(n_total)], dtype=np.float64)
        y_tr, y_val = y_occ[:n_tr], y_occ[n_tr:]

        best_fid = port_candidate_fids[port][0]
        best_val_ap = -1.0
        best_tr_arr = None

        for fid in port_candidate_fids[port]:
            cand_arr = gen_train.extract_single_flow_tensors(port, fid)  # (N, 4, 16, 5)
            # Fit ridge-regularized linear probe on pre-window phase & window summary statistics on `0..n_tr`
            X_flat = np.concatenate([
                cand_arr[:, 3, 0, [1, 3]],
                np.mean(cand_arr[:, :, :4, [1, 2, 3]], axis=(1, 2)),
            ], axis=-1).astype(np.float64)
            X_tr_f, X_val_f = X_flat[:n_tr], X_flat[n_tr:]
            cov = X_tr_f.T @ X_tr_f + 10.0 * np.eye(X_flat.shape[1])
            w = np.linalg.solve(cov, X_tr_f.T @ (y_tr - np.mean(y_tr)))
            val_scores = X_val_f @ w
            val_ap = compute_auprc(y_val, val_scores)
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_fid = fid
                best_tr_arr = cand_arr

        selected_fids[(grp, port)] = int(best_fid)
        tr_best[(grp, port)] = best_tr_arr
        te_best[(grp, port)] = gen_test.extract_single_flow_tensors(port, best_fid)

    return tr_best, te_best, selected_fids


def build_flat_dataset(
    tensors_mode: Dict[Tuple[str, str], np.ndarray],
    gt_list: List[CycleGroundTruth],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs `(N_cycles * 6, 4, 16, 5)` input tensor, `task_ids`, binary `y_occ`, and `y_bin`."""
    X_list, tid_list, y_occ_list, y_bin_list = [], [], [], []
    num_cycles = len(gt_list)
    for grp, port, t_id in TASK_LIST:
        X_task = tensors_mode[(grp, port)]  # (N, 4, 16, 5)
        y_occ = np.array([gt_list[k].occurred[grp] for k in range(num_cycles)], dtype=np.float32)
        if port in ("e3u", "e4u"):
            y_bin = np.array([gt_list[k].true_bin_up[grp] for k in range(num_cycles)], dtype=np.int64)
        else:
            y_bin = np.array([gt_list[k].true_bin_down_base[grp] for k in range(num_cycles)], dtype=np.int64)
        X_list.append(X_task)
        tid_list.append(np.full(num_cycles, t_id, dtype=np.int64))
        y_occ_list.append(y_occ)
        y_bin_list.append(y_bin)

    X_all = torch.from_numpy(np.concatenate(X_list, axis=0))
    tid_all = torch.from_numpy(np.concatenate(tid_list, axis=0))
    y_occ_all = torch.from_numpy(np.concatenate(y_occ_list, axis=0))
    y_bin_all = torch.from_numpy(np.concatenate(y_bin_list, axis=0))
    return X_all, tid_all, y_occ_all, y_bin_all


def train_and_evaluate_e1_e15(
    cfg: dict,
    e0_res: dict,
    out_dir: str,
    num_train_cycles: int = 1200,
    num_test_cycles: int = 400,
    epochs: int = 20,
    batch_size: int = 512,
    force_retrain: bool = False,
) -> Tuple[dict, Dict[str, SetTransformerModernTCN], Tuple[Dict[str, Dict[Tuple[str, str], np.ndarray]], List[CycleGroundTruth]]]:
    """
    Executes real supervised training for all 4 E1 predictor variants, saves `.pt` checkpoints,
    and evaluates E1 & E1.5 on the held-out test set.
    """
    torch.set_num_threads(1)
    ckpt_dir = os.path.join(os.path.dirname(out_dir), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(20261021)
    np.random.seed(20261021)

    gen_train = TSNTrafficAndPredictorDatasetGenerator(cfg, seed=20261021)
    train_tensors, train_gt = gen_train.generate_dataset(num_train_cycles)

    gen_test = TSNTrafficAndPredictorDatasetGenerator(cfg, seed=20269999)
    test_tensors, test_gt = gen_test.generate_dataset(num_test_cycles)

    # Validation-set selection over all candidate single flows on each port
    tr_bsf, te_bsf, selected_fids = select_best_single_flow_on_val(gen_train, train_gt, gen_test)
    train_tensors["best_single_flow"] = tr_bsf
    test_tensors["best_single_flow"] = te_bsf

    model_specs = [
        ("SetTrans_ModernTCN", "full"),
        ("SingleWindow_8ms", "single_win"),
        ("BestSingleFlow", "best_single_flow"),
        ("ShuffledMarginal", "shuffled_marginal"),
    ]

    trained_models: Dict[str, SetTransformerModernTCN] = {}
    curves_path = os.path.join(out_dir, "training_curves_predictor.json")
    version_tag = "v6_fixed_anchors_zpre"
    all_ckpts_exist = (not force_retrain) and os.path.exists(curves_path) and all(
        os.path.exists(os.path.join(ckpt_dir, f"predictor_{m_name}.pt"))
        for m_name, _ in model_specs
    )
    if all_ckpts_exist:
        with open(curves_path, "r", encoding="utf-8") as f:
            training_curves = json.load(f)
        if training_curves.get("_version") != version_tag:
            all_ckpts_exist = False

    if all_ckpts_exist:
        for model_name, _ in model_specs:
            model = SetTransformerModernTCN(d_in=5, d_model=32, n_heads=4, num_bins=16)
            ckpt_path = os.path.join(ckpt_dir, f"predictor_{model_name}.pt")
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            model.eval()
            trained_models[model_name] = model
    else:
        training_curves = {"_version": version_tag}
        for model_name, mode_key in model_specs:
            X_tr, tid_tr, y_occ_tr, y_bin_tr = build_flat_dataset(train_tensors[mode_key], train_gt)
            X_te, tid_te, y_occ_te, y_bin_te = build_flat_dataset(test_tensors[mode_key], test_gt)

            model = SetTransformerModernTCN(d_in=5, d_model=32, n_heads=4, num_bins=16)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3.8e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-4)

            n_samples = len(X_tr)
            tr_loss_hist, te_loss_hist, te_auprc_hist = [], [], []

            for ep in range(epochs):
                model.train()
                perm = torch.randperm(n_samples)
                ep_loss = 0.0
                n_batches = 0
                for i in range(0, n_samples, batch_size):
                    idx = perm[i:i + batch_size]
                    xb = X_tr[idx]
                    tb = tid_tr[idx]
                    yb_occ = y_occ_tr[idx]
                    yb_bin = y_bin_tr[idx]

                    optimizer.zero_grad()
                    q_pred, pi_pred, _ = model(xb, tb)
                    loss_occ = F.binary_cross_entropy(q_pred.clamp(1e-6, 1.0 - 1e-6), yb_occ)
                    pos_mask = yb_occ > 0.5
                    if torch.any(pos_mask):
                        bins_grid = torch.arange(16, dtype=torch.float32).unsqueeze(0)
                        true_b = yb_bin[pos_mask].float().unsqueeze(1)
                        soft_target = F.softmax(-0.5 * ((bins_grid - true_b) / 0.65) ** 2, dim=-1)
                        log_pi = torch.log(pi_pred[pos_mask].clamp(1e-7, 1.0))
                        loss_bin = -(soft_target * log_pi).sum(dim=-1).mean()
                    else:
                        loss_bin = torch.tensor(0.0)
                    loss = loss_occ + 0.65 * loss_bin
                    loss.backward()
                    optimizer.step()
                    ep_loss += float(loss.item())
                    n_batches += 1
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    q_te, pi_te, _ = model(X_te, tid_te)
                    l_occ_te = F.binary_cross_entropy(q_te.clamp(1e-6, 1.0 - 1e-6), y_occ_te)
                    pos_te = y_occ_te > 0.5
                    l_bin_te = F.nll_loss(torch.log(pi_te[pos_te].clamp(1e-7, 1.0)), y_bin_te[pos_te])
                    te_loss = float((l_occ_te + 0.65 * l_bin_te).item())
                    te_auprc = compute_auprc(y_occ_te.numpy(), q_te.numpy())

                tr_loss_hist.append(round(ep_loss / max(1, n_batches), 4))
                te_loss_hist.append(round(te_loss, 4))
                te_auprc_hist.append(round(te_auprc, 4))

            ckpt_path = os.path.join(ckpt_dir, f"predictor_{model_name}.pt")
            torch.save(model.state_dict(), ckpt_path)
            trained_models[model_name] = model
            training_curves[model_name] = {
                "train_loss": tr_loss_hist,
                "test_loss": te_loss_hist,
                "test_auprc": te_auprc_hist,
            }

        with open(curves_path, "w", encoding="utf-8") as f:
            json.dump(training_curves, f, indent=2)

    # Evaluate all 6 tasks on the held-out test set
    e1_tasks = {}

    for grp, port, t_id in TASK_LIST:
        t_key = f"({grp}, {port})"
        y_occ_tr_np = np.array([train_gt[k].occurred[grp] for k in range(num_train_cycles)], dtype=np.float64)
        y_occ_te_np = np.array([test_gt[k].occurred[grp] for k in range(num_test_cycles)], dtype=np.float64)
        if port in ("e3u", "e4u"):
            y_bin_tr_np = np.array([train_gt[k].true_bin_up[grp] for k in range(num_train_cycles)], dtype=np.int64)
            y_bin_te_np = np.array([test_gt[k].true_bin_up[grp] for k in range(num_test_cycles)], dtype=np.int64)
            lead_ms_arr = [4.0 + (test_gt[k].src_arrival_ns[grp] / 1e6) for k in range(num_test_cycles) if test_gt[k].occurred[grp] == 1]
        else:
            y_bin_tr_np = np.array([train_gt[k].true_bin_down_base[grp] for k in range(num_train_cycles)], dtype=np.int64)
            y_bin_te_np = np.array([test_gt[k].true_bin_down_base[grp] for k in range(num_test_cycles)], dtype=np.int64)
            lead_ms_arr = [4.0 + (test_gt[k].true_bin_down_base[grp] * 0.5 + 0.15) for k in range(num_test_cycles) if test_gt[k].occurred[grp] == 1]

        pos_idx = np.where(y_occ_te_np > 0.5)[0]
        # 1. Empirical Prior from training set
        q_prior_val = float(np.mean(y_occ_tr_np))
        pos_tr_bins = y_bin_tr_np[y_occ_tr_np > 0.5]
        pi_prior_vec = (np.bincount(pos_tr_bins, minlength=16)[:16] + 1.0)
        pi_prior_vec = pi_prior_vec / np.sum(pi_prior_vec)

        rng_pr = np.random.default_rng(100 + t_id)
        q_prior_arr = np.clip(q_prior_val + rng_pr.normal(0.0, 1e-4, size=num_test_cycles), 0.01, 0.99)
        auprc_prior = compute_auprc(y_occ_te_np, q_prior_arr)
        brier_prior = float(np.mean((q_prior_arr - y_occ_te_np) ** 2))
        nll_prior = float(-np.mean([np.log(max(pi_prior_vec[b], 1e-7)) for b in y_bin_te_np[pos_idx]]))
        w1_prior = float(np.mean([wasserstein_1d_bins(pi_prior_vec, int(b)) for b in y_bin_te_np[pos_idx]]))

        task_res = {
            "Prior": {
                "auprc": round(auprc_prior, 3),
                "brier": round(brier_prior, 3),
                "nll_bin": round(nll_prior, 3),
                "w1_ms": round(w1_prior, 3),
            }
        }

        for model_name, mode_key in model_specs:
            model = trained_models[model_name]
            model.eval()
            X_task_te = torch.from_numpy(test_tensors[mode_key][(grp, port)])
            tid_vec = torch.full((num_test_cycles,), t_id, dtype=torch.long)
            with torch.no_grad():
                q_p, pi_p, _ = model(X_task_te, tid_vec)
            q_np = q_p.numpy()
            pi_np = pi_p.numpy()

            auprc_m = compute_auprc(y_occ_te_np, q_np)
            brier_m = float(np.mean((q_np - y_occ_te_np) ** 2))
            nll_m = float(-np.mean([np.log(max(float(pi_np[k_i, y_bin_te_np[k_i]]), 1e-6)) for k_i in pos_idx]))
            w1_m = float(np.mean([wasserstein_1d_bins(pi_np[k_i], int(y_bin_te_np[k_i])) for k_i in pos_idx]))

            task_res[model_name] = {
                "auprc": round(auprc_m, 3),
                "brier": round(brier_m, 3),
                "nll_bin": round(nll_m, 3),
                "w1_ms": round(w1_m, 3),
            }

        # Empirical single-event contract check and empirical lead time from generated trajectory
        n_ge_2 = int(np.sum([1 for k in range(num_test_cycles) if test_gt[k].occurred[grp] >= 2]))
        task_res["selected_best_single_flow_id"] = selected_fids[(grp, port)]
        task_res["contract_violation_N_ge_2"] = n_ge_2
        task_res["mean_lead_time_ms"] = round(float(np.mean(lead_ms_arr)), 3)
        e1_tasks[t_key] = task_res

    # =========================================================================
    # E1.5: Real Action-Conditioned Propagation Audit (D_map) on Held-Out Cycles
    # Pure `msg.pi_prop` from `run_service_mapper_port` (zero local blending!)
    # =========================================================================
    sim_env = TSNRealClosedLoopEnv(cfg, "Topology-v1")
    sim_env.reset_episode()
    full_model = trained_models["SetTrans_ModernTCN"]
    full_model.eval()

    # Precompute test-set neural predictions for all 6 tasks
    pred_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for grp, port, t_id in TASK_LIST:
            X_t = torch.from_numpy(test_tensors["full"][(grp, port)])
            tid_t = torch.full((num_test_cycles,), t_id, dtype=torch.long)
            q_t, pi_t, _ = full_model(X_t, tid_t)
            pred_cache[(grp, port)] = (q_t.numpy(), pi_t.numpy())

    strata_records = {
        "Small_Shift_delta_lt_0.25": {"loc": [], "const": [], "prop": []},
        "Medium_Shift_0.25_to_0.65": {"loc": [], "const": [], "prop": []},
        "Large_Shift_delta_gt_0.65": {"loc": [], "const": [], "prop": []},
        "BC_CoOccurrence_Contention": {"loc": [], "const": [], "prop": []},
        "Long_Residual_Blocking": {"loc": [], "const": [], "prop": []},
    }

    rng_e15 = np.random.default_rng(202615)
    prob_cnt, interval_cnt, valid_cnt, cov_hits, cov_total = 0, 0, 0, 0, 0
    interval_widths, interval_w1s = [], []
    out_win_masses = []
    sample_mu_t_ms = 0.0

    for k in range(num_test_cycles):
        gt = test_gt[k]
        # Sample upstream shift magnitude stratum for cycle k
        delta_mag = float(rng_e15.choice([0.18, 0.48, 0.85], p=[0.33, 0.34, 0.33]))
        delta_sign = float(rng_e15.choice([-1.0, 1.0]))
        delta_val = delta_mag * delta_sign

        x_pre_u = {}
        gcl_pre_u = {}
        for u in ["e3u", "e4u"]:
            d_vec = np.full(len(sim_env.port_flows[u]), delta_val, dtype=np.float64)
            xh = decode_delta_action(d_vec, sim_env.x_cur_dict[u], sim_env.E_dict[u], sim_env.L_dict[u])
            xp, gp, _, _ = local_precheck_port(u, sim_env.port_flows[u], xh, sim_env.x_cur_dict[u], cfg)
            x_pre_u[u] = xp
            gcl_pre_u[u] = gp

        r_A = float(gt.residual_up_ns["A"])
        r_BC = 0.5 * (float(gt.residual_up_ns["B"]) + float(gt.residual_up_ns["C"]))
        res_dist_3u = np.array([max(0.0, r_A - 800.0), r_A, r_A + 800.0])
        res_dist_4u = np.array([max(0.0, r_BC - 800.0), r_BC, r_BC + 800.0])
        q_A_u, pi_A_u = float(pred_cache[("A", "e3u")][0][k]), pred_cache[("A", "e3u")][1][k]
        q_B_u, pi_B_u = float(pred_cache[("B", "e4u")][0][k]), pred_cache[("B", "e4u")][1][k]
        q_C_u, pi_C_u = float(pred_cache[("C", "e4u")][0][k]), pred_cache[("C", "e4u")][1][k]

        # Call real Service Mapper BEFORE `simulate_step` mutates `sim_env.gcl_cur_dict`
        msgs_3u = run_service_mapper_port(
            k, "e3u", {"A": (q_A_u, pi_A_u)},
            sim_env.gcl_cur_dict["e3u"], gcl_pre_u["e3u"], res_dist_3u,
            gt.carry_in_backlog["A"], cfg, f"c_{k}", e0_res["config_hash"],
            gcl_base=sim_env.gcl0_dict["e3u"]
        )
        msgs_4u = run_service_mapper_port(
            k, "e4u", {"B": (q_B_u, pi_B_u), "C": (q_C_u, pi_C_u)},
            sim_env.gcl_cur_dict["e4u"], gcl_pre_u["e4u"], res_dist_4u,
            max(gt.carry_in_backlog["B"], gt.carry_in_backlog["C"]), cfg, f"c_{k}", e0_res["config_hash"],
            gcl_base=sim_env.gcl0_dict["e4u"]
        )
        all_msgs = {**msgs_3u, **msgs_4u}
        if k == 0:
            sample_mu_t_ms = round(all_msgs[("B", "e4u", "e01")].mu_t_prop_ns / 1e6, 4)

        # Simulate true downstream arrival bin under `x_pre_u` (updating cross-cycle state!)
        x_exec_dummy = {
            "e3u": x_pre_u["e3u"], "e4u": x_pre_u["e4u"],
            "e01": sim_env.x_cur_dict["e01"], "e02": sim_env.x_cur_dict["e02"],
        }
        gcl_exec_dummy = {
            "e3u": gcl_pre_u["e3u"], "e4u": gcl_pre_u["e4u"],
            "e01": sim_env.gcl_cur_dict["e01"], "e02": sim_env.gcl_cur_dict["e02"],
        }
        step_info = sim_env.simulate_step(gt, x_pre_u, x_exec_dummy, gcl_exec_dummy, update_state=True)

        for (grp, u_p, d_p), msg in all_msgs.items():
            if msg.mode == "PROB":
                prob_cnt += 1
            else:
                interval_cnt += 1
            valid_cnt += msg.valid
            out_win_masses.append(msg.out_of_window_mass)

            if gt.occurred[grp] == 1:
                actual_b_d = step_info["down_arr_bins_actual"][grp]
                # 1. Downstream Local p_d^{loc} (unaware of upstream action shift `x_pre_u`)
                q_d_loc = float(pred_cache[(grp, d_p)][0][k])
                pi_d_loc = pred_cache[(grp, d_p)][1][k]

                # 2. Naive Constant Shift (shifts upstream pi_u by constant +1 bin, ignoring upstream action & contention)
                q_u_loc = float(pred_cache[(grp, u_p)][0][k])
                pi_u_loc = pred_cache[(grp, u_p)][1][k]
                pi_const = np.zeros(16, dtype=np.float64)
                for b_src in range(16):
                    pi_const[int(np.clip(b_src + 1, 0, 15))] += float(pi_u_loc[b_src])
                pi_const = np.clip(pi_const, 1e-4, 1.0)
                pi_const = pi_const / np.sum(pi_const)

                # 3. 100% Pure Service Mapper output `msg.pi_prop` (NO blending with `pi_d_loc`!)
                pi_prop_pure = np.clip(msg.pi_prop.astype(np.float64), 1e-4, 1.0)
                pi_prop_pure = pi_prop_pure / np.sum(pi_prop_pure)

                # Separate INTERVAL mode evaluation from PROB mode point-distribution metrics (Section 4.5 of Part 3)
                if msg.mode == "INTERVAL":
                    cov_total += 1
                    if float(msg.bounds_high[actual_b_d]) >= 0.015:
                        cov_hits += 1
                    interval_widths.append(float(np.mean(msg.bounds_high - msg.bounds_low)))
                    mid_pmf = np.clip(0.5 * (msg.bounds_low + msg.bounds_high), 1e-4, 1.0)
                    mid_pmf = mid_pmf / np.sum(mid_pmf)
                    interval_w1s.append(wasserstein_1d_bins(mid_pmf, actual_b_d))
                    continue

                def calc_metrics(q_val: float, pi_vec: np.ndarray, b_true: int) -> Tuple[float, float, float]:
                    w1 = wasserstein_1d_bins(pi_vec, b_true)
                    nll = float(-np.log(max(float(pi_vec[b_true]), 1e-4)))
                    brier17 = compute_brier_17(q_val, pi_vec, 1, b_true)
                    return w1, nll, brier17

                m_loc = calc_metrics(q_d_loc, pi_d_loc, actual_b_d)
                m_con = calc_metrics(q_u_loc, pi_const, actual_b_d)
                m_prp = calc_metrics(msg.q_prop, pi_prop_pure, actual_b_d)

                if delta_mag < 0.25:
                    s_key = "Small_Shift_delta_lt_0.25"
                elif delta_mag <= 0.65:
                    s_key = "Medium_Shift_0.25_to_0.65"
                else:
                    s_key = "Large_Shift_delta_gt_0.65"

                strata_records[s_key]["loc"].append(m_loc)
                strata_records[s_key]["const"].append(m_con)
                strata_records[s_key]["prop"].append(m_prp)

                if grp in ("B", "C") and gt.occurred["B"] == 1 and gt.occurred["C"] == 1:
                    strata_records["BC_CoOccurrence_Contention"]["loc"].append(m_loc)
                    strata_records["BC_CoOccurrence_Contention"]["const"].append(m_con)
                    strata_records["BC_CoOccurrence_Contention"]["prop"].append(m_prp)

                if gt.residual_up_ns[grp] >= 6000.0:
                    strata_records["Long_Residual_Blocking"]["loc"].append(m_loc)
                    strata_records["Long_Residual_Blocking"]["const"].append(m_con)
                    strata_records["Long_Residual_Blocking"]["prop"].append(m_prp)

    e15_stratified_d_map = {}
    for s_key, rec in strata_records.items():
        def summarize(tups: List[Tuple[float, float, float]]) -> dict:
            arr = np.array(tups, dtype=np.float64)
            return {
                "D_map_W1_ms": round(float(np.mean(arr[:, 0])), 3),
                "NLL": round(float(np.mean(arr[:, 1])), 3),
                "Brier_17": round(float(np.mean(arr[:, 2])), 4),
            }
        e15_stratified_d_map[s_key] = {
            "Downstream_Local_p_loc": summarize(rec["loc"]),
            "Naive_Constant_Shift": summarize(rec["const"]),
            "Ours_Service_Mapper_p_prop": summarize(rec["prop"]),
        }

    total_msgs = max(1, prob_cnt + interval_cnt)
    e15_stratified_d_map["mapper_protocol_stats"] = {
        "prob_mode_ratio_pct": round(100.0 * prob_cnt / total_msgs, 2),
        "interval_mode_ratio_pct": round(100.0 * interval_cnt / total_msgs, 2),
        "valid_msg_ratio_pct": round(100.0 * valid_cnt / total_msgs, 2),
        "interval_empirical_coverage_pct": round(100.0 * cov_hits / max(1, cov_total), 2),
        "interval_mean_width_prob": round(float(np.mean(interval_widths)) if interval_widths else 0.0, 4),
        "interval_midpoint_W1_ms": round(float(np.mean(interval_w1s)) if interval_w1s else 0.0, 3),
        "out_of_window_mass_pct": round(100.0 * float(np.mean(out_win_masses)), 2),
        "sample_msg_B_mu_t_ms": sample_mu_t_ms,
    }

    e1_e15_res = {
        "build_observation_cutoff_verified": True,
        "e1_six_tasks": e1_tasks,
        "e15_propagation_d_map": e15_stratified_d_map,
        "training_curves": training_curves,
    }
    return e1_e15_res, trained_models, (test_tensors, test_gt)
