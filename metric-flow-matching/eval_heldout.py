"""Evaluate a trained MFM model on held-out time points.

Usage (without conditioning):
    conda run -n mfm python eval_heldout.py \
        --ckpt checkpoints/custom_pt/<run_id>/flow_model/<ckpt>.ckpt \
        --data_path mfm/dataloaders/diffusion_2moons_dropout.pt \
        --train_indices 0 5 10 \
        --eval_indices 1 2 3 4 6 7 8 9 \
        --dim 2

Usage (with class conditioning):
    conda run -n mfm python eval_heldout.py \
        --ckpt checkpoints/custom_pt/<run_id>/flow_model/<ckpt>.ckpt \
        --data_path mfm/dataloaders/diffusion_2moons_dropout.pt \
        --train_indices 0 5 10 \
        --eval_indices 1 2 3 4 6 7 8 9 \
        --dim 2 --class_conditioning --categorical --cond_dim 1 --num_categories 2
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchdyn.core import NeuralODE

from mfm.networks.flow_networks.mlp import VelocityNet
from mfm.networks.utils import flow_model_torch_wrapper
import ot


def compute_wasserstein_distance(samples1, samples2):
    """
    Compute Wasserstein distance between two sets of samples using POT library.
    
    Args:
        samples1: JAX array of shape (n_samples1, dim)
        samples2: JAX array of shape (n_samples2, dim)
        
    Returns:
        float: The Wasserstein distance
    """
    # Convert from JAX arrays to numpy for POT library compatibility
    samples1_np = np.array(samples1)
    samples2_np = np.array(samples2)
    

    M = ot.dist(samples1_np, samples2_np)
    a = np.ones(samples1_np.shape[0]) / samples1_np.shape[0]  # uniform weights
    b = np.ones(samples2_np.shape[0]) / samples2_np.shape[0]  # uniform weights
        
    return float(ot.emd2(a, b, M))

# Distinct colors for per-class scatter plots
CLASS_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple",
                "tab:orange", "tab:brown", "tab:pink", "tab:gray"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to flow model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to .pt data file")
    parser.add_argument("--train_indices", nargs="+", type=int, required=True,
                        help="Original time indices used for training (e.g. 0 5 10)")
    parser.add_argument("--eval_indices", nargs="+", type=int, required=True,
                        help="Original time indices to evaluate on (e.g. 1 2 3 4 6 7 8 9)")
    parser.add_argument("--train_flow_times", nargs="+", type=float, default=None,
                        help="Explicit flow times for training indices (must match --train_indices length). "
                             "Defaults to linspace(0,1,len(train_indices)).")
    parser.add_argument("--eval_flow_times", nargs="+", type=float, default=None,
                        help="Explicit flow times for eval indices (must match --eval_indices length). "
                             "Defaults to idx/(T_total-1).")
    parser.add_argument("--dim", type=int, default=2, help="Number of spatial feature dimensions")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 64, 64],
                        help="Hidden dims of the flow network (must match training)")
    parser.add_argument("--activation", type=str, default="selu", help="Activation function")
    parser.add_argument("--num_steps", type=int, default=200, help="ODE integration steps")
    parser.add_argument("--solver", type=str, default="euler", help="ODE solver")
    parser.add_argument("--out_dir", type=str, default="eval_plots", help="Directory to save plots")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to a text file where mean EMD results are appended (TSV format)")

    # Conditioning arguments
    parser.add_argument("--class_conditioning", action=argparse.BooleanOptionalAction,
                        default=False, help="Enable class conditioning")
    parser.add_argument("--categorical", action=argparse.BooleanOptionalAction,
                        default=True, help="Whether condition is categorical (True) or continuous")
    parser.add_argument("--cond_dim", type=int, default=1, help="Dimensionality of condition variable")
    parser.add_argument("--num_categories", type=int, default=None,
                        help="Number of categories (auto-detected from data if not set)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load full dataset
    full_data = torch.load(args.data_path, map_location="cpu")  # [T, N, D_full]
    T_total = full_data.shape[0]

    # Extract spatial and condition columns
    spatial_data = full_data[:, :, :args.dim]  # [T, N, dim]

    if args.class_conditioning:
        cond_data = full_data[:, :, args.dim:args.dim + args.cond_dim]  # [T, N, cond_dim]
        if args.categorical:
            cond_data = cond_data.squeeze(-1)  # [T, N] for categorical
            if args.num_categories is None:
                args.num_categories = int(cond_data.max().item()) + 1
            unique_classes = list(range(args.num_categories))
            print(f"Class conditioning enabled: {args.num_categories} categories detected")
        else:
            unique_classes = None

    # Map original indices to flow time in [0, 1]
    train_indices = sorted(args.train_indices)

    # Build flow-time lookup: idx -> flow_time
    flow_time_map = {}
    if args.train_flow_times is not None:
        assert len(args.train_flow_times) == len(train_indices), \
            "--train_flow_times must match --train_indices length"
        for i, ti in enumerate(train_indices):
            flow_time_map[ti] = args.train_flow_times[i]
    else:
        default_train_times = np.linspace(0, 1, len(train_indices)).tolist()
        for i, ti in enumerate(train_indices):
            flow_time_map[ti] = default_train_times[i]

    if args.eval_flow_times is not None:
        assert len(args.eval_flow_times) == len(args.eval_indices), \
            "--eval_flow_times must match --eval_indices length"
        for i, ei in enumerate(args.eval_indices):
            flow_time_map[ei] = args.eval_flow_times[i]
    else:
        for ei in args.eval_indices:
            flow_time_map[ei] = ei / (T_total - 1)

    def index_to_flow_time(idx):
        return flow_time_map[idx]

    train_flow_times = [index_to_flow_time(i) for i in train_indices]

    # Load trained flow net from checkpoint
    flow_net = VelocityNet(
        dim=args.dim,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        batch_norm=False,
        class_conditioning=args.class_conditioning,
        categorical=args.categorical,
        num_categories=args.num_categories if args.class_conditioning else None,
        cond_dim=args.cond_dim,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = {k.replace("flow_net.", ""): v for k, v in ckpt["state_dict"].items()
                  if k.startswith("flow_net.")}
    flow_net.load_state_dict(state_dict)
    flow_net.eval()

    print(f"\nTraining indices: {train_indices}")
    print(f"Training flow times: {[f'{t:.4f}' for t in train_flow_times]}")
    print(f"Eval indices: {args.eval_indices}")
    print(f"Eval flow times: {[f'{index_to_flow_time(i):.4f}' for i in args.eval_indices]}")
    print(f"Total time points in data: {T_total}")
    print(f"ODE solver: {args.solver}, steps: {args.num_steps}\n")

    all_indices = sorted(set(train_indices + args.eval_indices))

    # ============================================================
    # Conditioning path: run ODE per class, report per-class EMD
    # ============================================================
    if args.class_conditioning and args.categorical:
        all_emds = {}        # idx -> overall EMD
        all_emds_cls = {}    # idx -> {cls: EMD}
        all_preds = {}       # idx -> numpy array [N, dim]
        all_pred_conds = {}  # idx -> numpy array [N]

        header = f"{'':>5} {'Idx':>4} {'Flow t':>7} {'Start':>6} {'t_s':>6}"
        for cls in unique_classes:
            header += f" {'EMD_c' + str(cls):>9}"
        header += f" {'EMD_avg':>9}"
        print(header)
        print("-" * len(header))

        for idx in all_indices:
            t_target = index_to_flow_time(idx)
            start_idx = max([ti for ti in train_indices if ti <= idx])
            t_start = index_to_flow_time(start_idx)

            cls_preds = {}   # cls -> predicted spatial tensor
            cls_emds = {}    # cls -> EMD

            for cls in unique_classes:
                # Get spatial data for this class at the start time
                cls_mask_start = (cond_data[start_idx] == cls)
                x_start_cls = spatial_data[start_idx][cls_mask_start]  # [N_cls, dim]
                cond_vec = torch.full((x_start_cls.shape[0],), cls, dtype=torch.long)

                # Ground truth at target time for this class
                cls_mask_target = (cond_data[idx] == cls)
                x_true_cls = spatial_data[idx][cls_mask_target]

                if idx == start_idx and idx == train_indices[0]:
                    x_pred_cls = x_start_cls.clone()
                elif idx == start_idx:
                    prev_idx = train_indices[train_indices.index(idx) - 1]
                    t_prev = index_to_flow_time(prev_idx)
                    cls_mask_prev = (cond_data[prev_idx] == cls)
                    x_prev_cls = spatial_data[prev_idx][cls_mask_prev]
                    cond_prev = torch.full((x_prev_cls.shape[0],), cls, dtype=torch.long)
                    node_cls = NeuralODE(
                        flow_model_torch_wrapper(flow_net, cond=cond_prev),
                        solver=args.solver, sensitivity="adjoint", atol=1e-5, rtol=1e-5,
                    )
                    t_span = torch.linspace(t_prev, t_target, args.num_steps + 1)
                    with torch.no_grad():
                        traj = node_cls.trajectory(x_prev_cls, t_span=t_span)
                    x_pred_cls = traj[-1]
                else:
                    node_cls = NeuralODE(
                        flow_model_torch_wrapper(flow_net, cond=cond_vec),
                        solver=args.solver, sensitivity="adjoint", atol=1e-5, rtol=1e-5,
                    )
                    t_span = torch.linspace(t_start, t_target, args.num_steps + 1)
                    with torch.no_grad():
                        traj = node_cls.trajectory(x_start_cls, t_span=t_span)
                    x_pred_cls = traj[-1]

                emd_cls = compute_wasserstein_distance(x_pred_cls, x_true_cls)
                cls_preds[cls] = x_pred_cls
                cls_emds[cls] = emd_cls

            # Reassemble full predicted array (all classes concatenated)
            pred_list = [cls_preds[c].numpy() for c in unique_classes]
            cond_list = [np.full(cls_preds[c].shape[0], c) for c in unique_classes]
            all_preds[idx] = np.concatenate(pred_list, axis=0)
            all_pred_conds[idx] = np.concatenate(cond_list, axis=0)
            all_emds_cls[idx] = cls_emds
            all_emds[idx] = float(np.mean([cls_emds[c] for c in unique_classes]))

            tag = "TRAIN" if idx in train_indices else "EVAL "
            row = f"{tag} {idx:>4} {t_target:>7.4f} {start_idx:>6} {t_start:>6.4f}"
            for cls in unique_classes:
                row += f" {cls_emds[cls]:>9.5f}"
            row += f" {all_emds[idx]:>9.5f}"
            print(row)

        eval_emds = {k: v for k, v in all_emds.items() if k in args.eval_indices}
        print("-" * len(header))
        mean_emd = np.mean(list(eval_emds.values()))
        print(f"{'Mean EMD (eval, avg over classes)':>{len(header) - 10}} {mean_emd:>9.5f}")

        # Per-class means for eval indices
        for cls in unique_classes:
            cls_eval_emds = [all_emds_cls[k][cls] for k in args.eval_indices]
            cls_mean = np.mean(cls_eval_emds)
            print(f"{'Mean EMD class ' + str(cls) + ' (eval)':>{len(header) - 10}} {cls_mean:>9.5f}")

        if args.results_file is not None:
            os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
            write_header = not os.path.exists(args.results_file)
            with open(args.results_file, "a") as rf:
                if write_header:
                    rf.write("ckpt\tout_dir\tmean_emd_eval\t" +
                             "\t".join(f"mean_emd_cls{c}" for c in unique_classes) + "\n")
                cls_means_str = "\t".join(
                    f"{np.mean([all_emds_cls[k][c] for k in args.eval_indices]):.6f}"
                    for c in unique_classes
                )
                rf.write(f"{args.ckpt}\t{args.out_dir}\t{mean_emd:.6f}\t{cls_means_str}\n")
            print(f"Results appended to {args.results_file}")

        # ---- Plotting with per-class coloring ----
        all_spatial_np = {idx: spatial_data[idx].numpy() for idx in all_indices}
        all_cond_np = {idx: cond_data[idx].numpy() for idx in all_indices}

        # Shared axis limits
        all_pts = np.concatenate(
            [all_spatial_np[i] for i in all_indices] + [all_preds[i] for i in all_indices], axis=0
        )
        pad = 0.3
        xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

        # Per-timepoint plots
        for idx in all_indices:
            x_true_np = all_spatial_np[idx]
            c_true_np = all_cond_np[idx]
            x_pred_np = all_preds[idx]
            c_pred_np = all_pred_conds[idx]
            is_train = idx in train_indices

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            s = 4; alpha = 0.5

            for cls in unique_classes:
                c_col = CLASS_COLORS[cls % len(CLASS_COLORS)]
                m_t = c_true_np == cls
                axes[0].scatter(x_true_np[m_t, 0], x_true_np[m_t, 1], s=s, alpha=alpha,
                                c=c_col, label=f"class {cls}")
                m_p = c_pred_np == cls
                axes[1].scatter(x_pred_np[m_p, 0], x_pred_np[m_p, 1], s=s, alpha=alpha,
                                c=c_col, label=f"class {cls}")

            axes[0].set_title(f"Ground Truth (t={idx})")
            axes[0].set_xlim(xlim); axes[0].set_ylim(ylim); axes[0].set_aspect("equal")
            axes[0].legend(fontsize=7, markerscale=3)

            emd_strs = ", ".join([f"c{c}={all_emds_cls[idx][c]:.4f}" for c in unique_classes])
            title = f"Predicted (avg EMD={all_emds[idx]:.4f}; {emd_strs})"
            if is_train:
                title = f"Train anchor ({emd_strs})"
            axes[1].set_title(title, fontsize=9)
            axes[1].set_xlim(xlim); axes[1].set_ylim(ylim); axes[1].set_aspect("equal")
            axes[1].legend(fontsize=7, markerscale=3)

            fig.suptitle(f"Time index {idx}  (flow t = {index_to_flow_time(idx):.3f})", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"t{idx:02d}.png"), dpi=150)
            plt.close(fig)

        # Summary grid
        n = len(all_indices)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        if n == 1:
            axes = axes[:, None]
        for j, idx in enumerate(all_indices):
            x_true_np = all_spatial_np[idx]
            c_true_np = all_cond_np[idx]
            x_pred_np = all_preds[idx]
            c_pred_np = all_pred_conds[idx]

            for cls in unique_classes:
                c_col = CLASS_COLORS[cls % len(CLASS_COLORS)]
                m_t = c_true_np == cls
                axes[0, j].scatter(x_true_np[m_t, 0], x_true_np[m_t, 1],
                                   s=1, alpha=0.4, c=c_col)
                m_p = c_pred_np == cls
                axes[1, j].scatter(x_pred_np[m_p, 0], x_pred_np[m_p, 1],
                                   s=1, alpha=0.4, c=c_col)

            axes[0, j].set_xlim(xlim); axes[0, j].set_ylim(ylim)
            axes[0, j].set_aspect("equal"); axes[0, j].set_title(f"t={idx}", fontsize=9)
            if j == 0:
                axes[0, j].set_ylabel("Ground Truth", fontsize=10)

            axes[1, j].set_xlim(xlim); axes[1, j].set_ylim(ylim)
            axes[1, j].set_aspect("equal")
            emd_str = f"{all_emds[idx]:.3f}"
            axes[1, j].set_title(f"EMD={emd_str}", fontsize=8)
            if j == 0:
                axes[1, j].set_ylabel("Predicted", fontsize=10)

        fig.suptitle("Ground Truth vs Predicted (per-class conditioning)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "summary_grid.png"), dpi=150)
        plt.close(fig)

    # ============================================================
    # Original (unconditioned) path
    # ============================================================
    else:
        node = NeuralODE(
            flow_model_torch_wrapper(flow_net),
            solver=args.solver,
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )

        print(f"{'Eval Idx':>8} {'Flow t':>8} {'Start Idx':>10} {'Start t':>8} {'EMD':>10}")
        print("-" * 52)

        all_emds = {}
        all_preds = {}

        for idx in all_indices:
            t_target = index_to_flow_time(idx)
            start_idx = max([ti for ti in train_indices if ti <= idx])
            t_start = index_to_flow_time(start_idx)

            x_start = spatial_data[start_idx]
            x_true = spatial_data[idx]

            if idx == start_idx and idx == train_indices[0]:
                x_pred = x_start.clone()
            elif idx == start_idx:
                prev_idx = train_indices[train_indices.index(idx) - 1]
                t_prev = index_to_flow_time(prev_idx)
                t_span = torch.linspace(t_prev, t_target, args.num_steps + 1)
                with torch.no_grad():
                    traj = node.trajectory(spatial_data[prev_idx], t_span=t_span)
                x_pred = traj[-1]
            else:
                t_span = torch.linspace(t_start, t_target, args.num_steps + 1)
                with torch.no_grad():
                    traj = node.trajectory(x_start, t_span=t_span)
                x_pred = traj[-1]

            emd = compute_wasserstein_distance(x_pred, x_true)
            all_emds[idx] = emd
            all_preds[idx] = x_pred.numpy()

            tag = "TRAIN" if idx in train_indices else "EVAL "
            print(f"{tag} {idx:>5} {t_target:>8.4f} {start_idx:>10} {t_start:>8.4f} {emd:>10.5f}")

        eval_emds = {k: v for k, v in all_emds.items() if k in args.eval_indices}
        print("-" * 58)
        mean_emd = np.mean(list(eval_emds.values()))
        print(f"{'Mean EMD (eval only)':>46} {mean_emd:>10.5f}")

        if args.results_file is not None:
            os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
            write_header = not os.path.exists(args.results_file)
            with open(args.results_file, "a") as rf:
                if write_header:
                    rf.write("ckpt\tout_dir\tmean_emd_eval\n")
                rf.write(f"{args.ckpt}\t{args.out_dir}\t{mean_emd:.6f}\n")
            print(f"Results appended to {args.results_file}")

        # Shared axis limits
        all_points = np.concatenate([spatial_data[i].numpy() for i in all_indices]
                                    + [all_preds[i] for i in all_indices], axis=0)
        pad = 0.3
        xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
        ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

        # Per-timepoint plots
        for idx in all_indices:
            x_true_np = spatial_data[idx].numpy()
            x_pred_np = all_preds[idx]
            is_train = idx in train_indices

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            s = 4; alpha = 0.5

            axes[0].scatter(x_true_np[:, 0], x_true_np[:, 1], s=s, alpha=alpha, c="tab:blue")
            axes[0].set_title(f"Ground Truth (t={idx})")
            axes[0].set_xlim(xlim); axes[0].set_ylim(ylim); axes[0].set_aspect("equal")

            color = "tab:green" if is_train else "tab:orange"
            axes[1].scatter(x_pred_np[:, 0], x_pred_np[:, 1], s=s, alpha=alpha, c=color)
            label = "Train anchor" if is_train else f"Predicted (EMD={all_emds[idx]:.4f})"
            axes[1].set_title(label)
            axes[1].set_xlim(xlim); axes[1].set_ylim(ylim); axes[1].set_aspect("equal")

            fig.suptitle(f"Time index {idx}  (flow t = {index_to_flow_time(idx):.3f})", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"t{idx:02d}.png"), dpi=150)
            plt.close(fig)

        # Summary grid
        n = len(all_indices)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        if n == 1:
            axes = axes[:, None]
        for j, idx in enumerate(all_indices):
            x_true_np = spatial_data[idx].numpy()
            x_pred_np = all_preds[idx]
            is_train = idx in train_indices

            axes[0, j].scatter(x_true_np[:, 0], x_true_np[:, 1], s=1, alpha=0.4, c="tab:blue")
            axes[0, j].set_xlim(xlim); axes[0, j].set_ylim(ylim)
            axes[0, j].set_aspect("equal"); axes[0, j].set_title(f"t={idx}", fontsize=9)
            if j == 0:
                axes[0, j].set_ylabel("Ground Truth", fontsize=10)

            color = "tab:green" if is_train else "tab:orange"
            axes[1, j].scatter(x_pred_np[:, 0], x_pred_np[:, 1], s=1, alpha=0.4, c=color)
            axes[1, j].set_xlim(xlim); axes[1, j].set_ylim(ylim)
            axes[1, j].set_aspect("equal")
            emd_str = f"{all_emds[idx]:.3f}"
            axes[1, j].set_title(f"EMD={emd_str}", fontsize=8)
            if j == 0:
                axes[1, j].set_ylabel("Predicted", fontsize=10)

        fig.suptitle("Ground Truth vs Predicted Pushforward (all time marginals)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "summary_grid.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to {args.out_dir}/")
    print(f"\nAll EMDs: {all_emds}")


if __name__ == "__main__":
    main()
