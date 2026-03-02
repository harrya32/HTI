"""Evaluate a trained MFM model on held-out quantile data instances.

Takes the 180 held-out instances (indices 1200:1380) from timepoint 0,
pushes them forward through the learned flow at specified times, and
computes MSE against ground-truth intermediate timepoints.

Usage:
    python eval_quantile.py \
        --ckpt checkpoints/custom_pt/<run_id>/flow_model/<ckpt>.ckpt \
        --data_path mfm/dataloaders/quantile_data_new.pt
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate flow model on held-out quantile instances")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to flow model checkpoint")
    parser.add_argument("--data_path", type=str, default="mfm/dataloaders/quantile_data_new.pt",
                        help="Path to quantile_data_new.pt")
    parser.add_argument("--dim", type=int, default=3, help="Number of ambient dimensions")
    parser.add_argument("--ambient_start_dim", type=int, default=12,
                        help="Column index where ambient dims begin")
    parser.add_argument("--train_instances", type=int, default=1200,
                        help="Number of training instances (eval starts after this)")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 64, 64],
                        help="Hidden dims of the flow network (must match training)")
    parser.add_argument("--activation", type=str, default="selu", help="Activation function")
    parser.add_argument("--num_steps", type=int, default=200, help="ODE integration steps")
    parser.add_argument("--solver", type=str, default="euler", help="ODE solver")
    parser.add_argument("--out_dir", type=str, default="eval_quantile_plots",
                        help="Directory to save plots")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to append TSV results")
    parser.add_argument("--cond_dim", type=int, default=12,
                        help="Number of condition dimensions (history length)")
    parser.add_argument("--cond_start_dim", type=int, default=0,
                        help="Column index where condition dims begin")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save all model predictions as mfm_preds.pt in out_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Eval flow times, rescaled, and corresponding data indices
    eval_flow_times = [(0.1 - 0.01)/(0.99 - 0.01), (0.25 - 0.01)/(0.99 - 0.01), (0.5 - 0.01)/(0.99 - 0.01), (0.75 - 0.01)/(0.99 - 0.01), (0.9 - 0.01)/(0.99 - 0.01)]
    eval_data_indices = [1, 2, 3, 4, 5]  # indices into the 7-timepoint tensor

    # Load data: [7, 1380, 15]
    full_data = torch.load(args.data_path, map_location="cpu")
    print(f"Loaded data shape: {full_data.shape}")

    sd = args.ambient_start_dim
    ed = sd + args.dim

    # Eval starting points: instances 1200:end from timepoint 0, ambient dims only
    x0_eval = full_data[0, args.train_instances:, sd:ed]  # [180, 3]
    n_eval = x0_eval.shape[0]
    print(f"Eval instances: {n_eval} (indices {args.train_instances}:{full_data.shape[1]})")
    print(f"Ambient dims: columns {sd}:{ed} ({args.dim}D)")

    # Condition: 12-dim history from timepoint 0 (same across all timepoints)
    cd = args.cond_start_dim
    cond_eval = full_data[0, args.train_instances:, cd:cd + args.cond_dim]  # [180, 12]
    print(f"Condition dims: columns {cd}:{cd + args.cond_dim} ({args.cond_dim}D)")

    # Ground truth at intermediate timepoints for the same held-out instances
    gt = {}
    for flow_t, data_idx in zip(eval_flow_times, eval_data_indices):
        gt[flow_t] = full_data[data_idx, args.train_instances:, sd:ed]  # [180, 3]

    # Load trained flow net
    flow_net = VelocityNet(
        dim=args.dim,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        batch_norm=False,
        class_conditioning=True,
        categorical=False,
        num_categories=None,
        cond_dim=args.cond_dim,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # Handle both direct and EMA-wrapped checkpoints
    state_dict = {k.replace("flow_net.", "", 1): v for k, v in ckpt["state_dict"].items()
                  if k.startswith("flow_net.")}
    # If EMA was used, keys will still have "model." prefix — strip it
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()
                      if k.startswith("model.")}
    flow_net.load_state_dict(state_dict)
    flow_net.eval()

    # Build per-instance ODE wrapper with fixed condition
    node = NeuralODE(
        flow_model_torch_wrapper(flow_net, cond=cond_eval),
        solver=args.solver,
        sensitivity="adjoint",
        atol=1e-5,
        rtol=1e-5,
    )

    print(f"\nODE solver: {args.solver}, steps: {args.num_steps}")
    print(f"{'Flow t':>8} {'Data idx':>9} {'MSE':>12} {'RMSE':>12}")
    print("-" * 45)

    all_mses = {}
    all_preds = {}

    for flow_t, data_idx in zip(eval_flow_times, eval_data_indices):
        t_span = torch.linspace(0.0, flow_t, args.num_steps + 1)
        with torch.no_grad():
            traj = node.trajectory(x0_eval, t_span=t_span)
        x_pred = traj[-1]  # [180, 3]

        # MSE per sample, then average
        mse_per_sample = ((x_pred - gt[flow_t]) ** 2).mean(dim=-1)  # [180]
        mean_mse = mse_per_sample.mean().item()
        rmse = mean_mse ** 0.5

        all_mses[flow_t] = mean_mse
        all_preds[flow_t] = x_pred.numpy()

        print(f"{flow_t:>8.2f} {data_idx:>9} {mean_mse:>12.6f} {rmse:>12.6f}")

    print("-" * 45)
    avg_mse = np.mean(list(all_mses.values()))
    avg_rmse = avg_mse ** 0.5
    print(f"{'Average':>8} {'':>9} {avg_mse:>12.6f} {avg_rmse:>12.6f}")

    # Save predictions tensor [n_quantiles, n_samples, 3]
    if args.save_predictions:
        preds_tensor = torch.stack(
            [torch.tensor(all_preds[t], dtype=torch.float32) for t in eval_flow_times]
        )  # [n_quantiles, n_eval, dim]
        save_path = os.path.join(args.out_dir, "mfm_preds.pt")
        torch.save(preds_tensor, save_path)
        print(f"Predictions tensor {list(preds_tensor.shape)} saved to {save_path}")

    # Write results file
    if args.results_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        write_header = not os.path.exists(args.results_file)
        with open(args.results_file, "a") as rf:
            if write_header:
                cols = ["ckpt", "out_dir", "avg_mse"] + [f"mse_t{t}" for t in eval_flow_times]
                rf.write("\t".join(cols) + "\n")
            vals = [args.ckpt, args.out_dir, f"{avg_mse:.6f}"]
            vals += [f"{all_mses[t]:.6f}" for t in eval_flow_times]
            rf.write("\t".join(vals) + "\n")
        print(f"Results appended to {args.results_file}")

    # ---- Plotting ----
    # Time-series forecast plots for specific held-out sample indices
    sample_indices = [56, 88, 72]

    for sample_idx in sample_indices:
        global_idx = args.train_instances + sample_idx

        # History: 12 conditioning dims from timepoint 0 for this instance
        history = full_data[0, global_idx, :sd].numpy()  # [12]
        n_history = len(history)
        history_steps = np.arange(n_history)  # 0..11
        forecast_steps = np.arange(n_history, n_history + args.dim)  # 12, 13, 14

        n_quantiles = len(eval_flow_times)
        fig, axes = plt.subplots(1, n_quantiles, figsize=(4.5 * n_quantiles, 4), sharey=True)
        if n_quantiles == 1:
            axes = [axes]

        for j, (flow_t, data_idx) in enumerate(zip(eval_flow_times, eval_data_indices)):
            ax = axes[j]
            gt_forecast = gt[flow_t][sample_idx].numpy()  # [3]
            pred_forecast = all_preds[flow_t][sample_idx]  # [3]

            # Plot history
            ax.plot(history_steps, history, "o-", color="black", markersize=3,
                    linewidth=1.5, label="History")
            # Connect history to forecasts with a dashed bridge
            ax.plot([history_steps[-1], forecast_steps[0]],
                    [history[-1], gt_forecast[0]], "--", color="tab:blue", alpha=0.4, linewidth=1)
            ax.plot([history_steps[-1], forecast_steps[0]],
                    [history[-1], pred_forecast[0]], "--", color="tab:red", alpha=0.4, linewidth=1)
            # Ground truth forecast
            ax.plot(forecast_steps, gt_forecast, "s-", color="tab:blue", markersize=5,
                    linewidth=1.5, label="GT forecast")
            # Predicted forecast
            ax.plot(forecast_steps, pred_forecast, "^--", color="tab:red", markersize=5,
                    linewidth=1.5, label="Pred forecast")

            ax.set_title(f"Quantile {flow_t}", fontsize=11)
            ax.set_xlabel("Time step")
            if j == 0:
                ax.set_ylabel("Value")
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.axvline(x=n_history - 0.5, color="gray", linestyle=":", alpha=0.5)

        fig.suptitle(f"Forecast vs Ground Truth (instance {global_idx})", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"forecast_instance_{global_idx}.png"), dpi=150)
        plt.close(fig)

    # Summary: MSE vs flow time
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eval_flow_times, [all_mses[t] for t in eval_flow_times], "o-", color="tab:blue")
    ax.set_xlabel("Flow time")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs Flow Time (held-out instances)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "mse_vs_time.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
