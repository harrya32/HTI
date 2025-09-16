import matplotlib.pyplot as plt
import argparse
import jax.numpy as jnp
import csv
import sys
import os
import pickle as pkl
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Evaluate a single ett surrogate.')
parser.add_argument('--lambda_pushforward', type=float, default=0, help='Lambda value for the surrogate function.')
parser.add_argument('--workspace_path', type=str, default="../NLOT/surrogate_models/ett/learned_w_potential/latest_0.pkl", help='Path to the trained OT workspace pickle file.')
parser.add_argument('--all_lambdas', action='store_true', help='Evaluate all lambda values and generate plot')
parser.add_argument('--name', type=str, default="learned_w_potential", help='method to evaluate')
parser.add_argument('--iter', type=int, default=0, help='number to put on results')
sys.path.append('../NLOT')
args = parser.parse_args()
LAMBDA_PUSHFORWARD = args.lambda_pushforward
RUN_NAME = args.name
ITER = args.iter
device = "cuda:0"
PLOT_DIR = f"ett_surrogate_plots/{RUN_NAME}"
os.makedirs(PLOT_DIR, exist_ok=True)
WORKSPACES_DIR = f"../NLOT/surrogate_models/ett/iclr/{RUN_NAME}/"
LAMBDA_VALUES = [1,2,3]

def load_workspace(workspace_path):
    if os.path.exists(workspace_path):
        print(f"Loading OT workspace from {workspace_path}")
        try:
            with open(workspace_path, "rb") as f:
                ws = pkl.load(f)
            print("Workspace loaded successfully")
            return ws
        except Exception as e:
            print(f"Error loading workspace: {e}")
    return None

def pushforward(forecast, input, lambda_val, workspace, final_data):
    """
    Uses trained OT model to push non robust forecast forward to target lambda.
    """
    if workspace is None:
        print("Workspace not available, using original forecast")
        return forecast
        
    #time_points = workspace.time_points
    time_points = [0,4]
    
    if lambda_val in time_points:
        time_idx = list(time_points).index(lambda_val)
        if time_idx == 0:
            return forecast

    current_sample = np.concatenate([forecast.flatten(), input.flatten()])
    
    for k in range(len(time_points) - 1):
        T_k = time_points[k]
        T_k_plus_1 = time_points[k+1]
        
        if T_k <= lambda_val <= T_k_plus_1:
            params_source_map_k = workspace.state_source_maps[k].params
            params_target_potential_k = workspace.state_target_potentials[k].params
            #end_sample = workspace.neural_dual_solver.source_map_apply_jit(
            #    {'params': params_source_map_k},
            #    current_sample
            #)

            #end_sample = workspace.neural_dual_solver.pushforward_jit(
            #    params_source_map_k,
            #    params_target_potential_k,
            #    workspace.params_geometry,
            #    current_sample
            #).solution
            #print("current_sample:", current_sample)
            #print("end_sample:", end_sample)
            end_sample = final_data.numpy()
            if lambda_val < T_k_plus_1:
                s_fraction = (lambda_val - T_k) / (T_k_plus_1 - T_k)
                current_sample = workspace.geometry.apply(
                    {'params': workspace.params_geometry},
                    current_sample,
                    end_sample,
                    s_fraction,
                    method=workspace.geometry.point_on_path
                )
            else:
                current_sample = end_sample
            break
        
        params_source_map_k = workspace.state_source_maps[k].params
        current_sample = workspace.neural_dual_solver.source_map_apply_jit(
            {'params': params_source_map_k},
            current_sample
        )

    forecast_dim = forecast.shape[1] if len(forecast.shape) > 1 else forecast.shape[0]
    pushforward_forecast = current_sample[:forecast_dim].reshape(forecast.shape)

    pushforward_forecast = np.array(pushforward_forecast, dtype=np.float32)
    #print("initial forecast:", forecast)
    #print("pushforward forecast:", pushforward_forecast)
    plt.figure(figsize=(8, 4))
    plt.plot(forecast.flatten(), label="Initial Forecast", marker='o')
    plt.plot(pushforward_forecast.flatten(), label="Pushforward Forecast", marker='x')
    plt.title(f"Forecast Pushforward (lambda={lambda_val})")
    plt.xlabel("Time Step")
    plt.ylabel("Forecast Value")
    plt.legend()
    plot_path = os.path.join(PLOT_DIR, f"pushforward_lambda{lambda_val}.png")
    plt.savefig(plot_path)
    plt.close()

    return pushforward_forecast

def evaluate_lambda(data, lambda_val, workspace):
    """Evaluate the surrogate model by comparing estimates with true forecasts from the relevant lambda"""
    base_data = data[0]
    final_data = data[-1]
    final_forecast_portion = final_data[:,24:]
    final_input_portion = final_data[:,:24]
    final_data_reversed = torch.concat([final_forecast_portion, final_input_portion], dim=-1)
    true_lambda_data = data[lambda_val]

    mses = []
    for i in range(len(base_data)):
        surrogate_forecast = pushforward(base_data[i, 24:], base_data[i, :24], lambda_val, workspace, final_data_reversed[i])
        true_forecast = true_lambda_data[i, 24:]

        mse = np.mean((surrogate_forecast - true_forecast.numpy()) ** 2)
        mses.append(mse)

    return np.mean(mses)

def main():
    print(f"Loading ett data")
    ett_data = torch.load("../robustness/ett_forecasts_iclr.pt")[:,1450:,:]

    print(f"Current working directory: {os.getcwd()}")
    workspace_files = [f for f in os.listdir(WORKSPACES_DIR) if f.endswith('.pkl')]
    print(f"Found {len(workspace_files)} workspace files in {WORKSPACES_DIR}")
    all_workspace_results = {}

    combined_results = {}
    for lambda_val in LAMBDA_VALUES:
        combined_results[lambda_val] = {
            'mse': []
        }

    for workspace_file in workspace_files:
        workspace_path = os.path.join(WORKSPACES_DIR, workspace_file)
        workspace_name = os.path.splitext(os.path.basename(workspace_path))[0]
        print(f"\n===== Processing Workspace: {workspace_name} =====")
        
        workspace = load_workspace(workspace_path)
        if workspace is None:
            print(f"Could not load workspace from {workspace_path}, skipping...")
            continue
            
        lambda_results = []
        for lambda_val in LAMBDA_VALUES:
            lambda_mse = evaluate_lambda(ett_data, lambda_val, workspace)
            lambda_results.append((lambda_val, lambda_mse))
            print(f"Lambda {lambda_val}: MSE = {lambda_mse}")
            combined_results[lambda_val]['mse'].append(lambda_mse)

    return combined_results
if __name__ == "__main__":
    combined_results = main()
    print("Evaluation complete. Results:")
    for lambda_val, results in combined_results.items():
        print(f"Lambda {lambda_val}: MSE = {np.mean(results['mse'])}")