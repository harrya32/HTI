import DTRGym
import gymnasium as gym
import numpy as np
import os
import sys
import pickle as pkl
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import matplotlib.pyplot as plt
import argparse
import jax.numpy as jnp
import csv
from gymnasium.core import RewardWrapper

sys.path.append('/mnt/pdata/hmka3/HTI/NLOT')

AGENT_PATH = "policies/ppo_ghaffari_cancer_model__0.zip" 
AGENT_NAME = "single_agent_eval"
ENV_NAME = 'GhaffariCancerEnv-continuous'
NUM_EVAL_EPISODES = 5
LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PLOT_DIR = "surrogate_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='Evaluate a single agent with a pushforward function.')
parser.add_argument('--lambda_pushforward', type=float, default=0, help='Lambda value for the pushforward function.')
parser.add_argument('--workspace_path', type=str, default="../NLOT/2025.05.14/ours.pkl", help='Path to the trained OT workspace pickle file.')
parser.add_argument('--all_lambdas', action='store_true', help='Evaluate all lambda values and generate plot')
args = parser.parse_args()
LAMBDA_PUSHFORWARD = args.lambda_pushforward

workspace = None

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, lambda_nk: float = 0.5):
        super().__init__(env)
        self.lambda_nk = lambda_nk
        self.init_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.init_obs = np.array(obs, copy=True)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.init_obs is not None:
            N_p  = obs[1]
            N_s  = obs[5]
            N_p0 = self.init_obs[1]
            N_s0 = self.init_obs[5]
            N  = max(np.e,  N_p  + N_s)
            N0 = max(np.e,  N_p0 + N_s0)
            logN, logN0 = np.log(N), np.log(N0)
            nk_penalty = -np.abs(logN / logN0 - 1)
            reward = reward + self.lambda_nk * nk_penalty

        return obs, reward, terminated, truncated, info

#print current working directory
print(f"Current working directory: {os.getcwd()}")
print(args.workspace_path)
if args.workspace_path and os.path.exists(args.workspace_path):
    print(f"Loading OT workspace from {args.workspace_path}")
    try:
        with open(args.workspace_path, "rb") as f:
            workspace = pkl.load(f)
        print("Workspace loaded successfully")
    except Exception as e:
        print(f"Error loading workspace: {e}")
        workspace = None

def pushforward(action, obs, lambda_val):
    """
    Uses trained OT model to push actions forward to target lambda.
    """
    # Check if workspace is None
    if workspace is None:
        print("Workspace not available, using original action")
        return action
        
    time_points = workspace.time_points
    
    if lambda_val in time_points:
        time_idx = list(time_points).index(lambda_val)
        if time_idx == 0:
            return action
    
    current_sample = np.concatenate([action.flatten(), obs.flatten()])
    
    for k in range(len(time_points) - 1):
        T_k = time_points[k]
        T_k_plus_1 = time_points[k+1]
        
        if T_k <= lambda_val <= T_k_plus_1:
            params_source_map_k = workspace.state_source_maps[k].params
            end_sample = workspace.neural_dual_solver.source_map_apply_jit(
                {'params': params_source_map_k},
                current_sample
            )
            
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
    
    action_dim = action.shape[1] if len(action.shape) > 1 else action.shape[0]
    pushforward_action = current_sample[:action_dim].reshape(action.shape)
    
    pushforward_action = np.array(pushforward_action, dtype=np.float32)

    return pushforward_action

def calculate_nk_penalty(current_obs_vec, initial_obs_vec):
    current_obs = current_obs_vec[0]
    initial_obs = initial_obs_vec[0]

    N_p  = current_obs[1]
    N_s  = current_obs[5]
    N_p0 = initial_obs[1]
    N_s0 = initial_obs[5]

    N  = max(np.e,  N_p  + N_s)
    N0 = max(np.e,  N_p0 + N_s0)

    logN = np.log(N)
    logN0 = np.log(N0)

    penalty = -np.abs(logN / logN0 - 1)
    return penalty

def evaluate_lambda(model, lambda_val):
    """Evaluate the agent with a specific lambda pushforward value."""
    def make_env():
        env = gym.make(ENV_NAME)
        env = CustomRewardWrapper(env, lambda_nk=lambda_val * 10)
        return env
    
    env = make_vec_env(make_env, n_envs=1)
    
    overall_average_penalties = []
    episode_rewards = []
    
    print(f'\n--- Evaluating agent {AGENT_NAME} for NK penalty with lambda = {lambda_val} ---')

    for episode_num in range(NUM_EVAL_EPISODES):
        initial_obs_for_episode = env.reset()
        obs = initial_obs_for_episode
        done = False
        penalties_this_episode = []
        episode_reward = 0
        episode_steps = 0
        
        print(f"Starting Episode {episode_num + 1}/{NUM_EVAL_EPISODES}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            modified_action = pushforward(action, obs, lambda_val)
            next_obs, reward, done_array, info = env.step(modified_action)
            penalty = calculate_nk_penalty(next_obs, initial_obs_for_episode)
            penalties_this_episode.append(penalty)
            episode_reward += reward[0] 
            obs = next_obs
            done = done_array[0]
            episode_steps += 1
        
        avg_penalty_for_this_episode = np.mean(penalties_this_episode)
        overall_average_penalties.append(avg_penalty_for_this_episode)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode_num + 1}: Average NK Penalty = {avg_penalty_for_this_episode:.4f}, Steps = {episode_steps}")

    env.close()
    final_avg_nk_penalty = np.mean(overall_average_penalties)
    std_dev_nk_penalty = np.std(overall_average_penalties)
    final_avg_reward = np.mean(episode_rewards) 
    print(f"\n--- Summary for Lambda = {lambda_val} ---")
    print(f"Overall Average NK Penalty across {NUM_EVAL_EPISODES} episodes = {final_avg_nk_penalty:.4f}")
    print(f"Standard Deviation of Per-Episode Average NK Penalties = {std_dev_nk_penalty:.4f}")
    print(f"Overall Average Reward across {NUM_EVAL_EPISODES} episodes = {final_avg_reward:.4f}")

    return final_avg_nk_penalty, std_dev_nk_penalty, final_avg_reward


print(f"--- Loading Agent: {AGENT_NAME} ---")

model = PPO.load(AGENT_PATH)
print(f"Successfully loaded model '{AGENT_NAME}' from '{AGENT_PATH}'")

if args.all_lambdas:
    lambda_results = []
    all_lambda_rewards = [] 
    for lambda_val in LAMBDA_VALUES:
        avg_penalty, std_dev, avg_reward = evaluate_lambda(model, lambda_val)
        lambda_results.append((lambda_val, avg_penalty, std_dev))
        all_lambda_rewards.append(avg_reward)
    
    overall_avg_reward = np.mean(all_lambda_rewards)
    print(f"Average Reward across all lambda values = {overall_avg_reward:.4f}")
    plot_lambdas = [item[0] for item in lambda_results]
    plot_avg_penalties = [item[1] for item in lambda_results]
    plot_std_devs = [item[2] for item in lambda_results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(plot_lambdas, plot_avg_penalties, yerr=plot_std_devs, marker='o', linestyle='-', capsize=5)
    plt.title(f"Overall Average NK Penalty vs. Lambda Value ({NUM_EVAL_EPISODES} Episodes)")
    plt.xlabel("Lambda Value")
    plt.ylabel("Overall Average NK Penalty")
    plt.grid(True)
    plt.xticks(plot_lambdas)
    plt.tight_layout()
    plot_filename = os.path.join(PLOT_DIR, "surrogate_avg_nk_penalty_vs_lambda_ours.png")
    plt.savefig(plot_filename)
    print(f"\nSaved overall average NK penalty plot to {plot_filename}")
    plt.close()
    
    csv_filename = os.path.join(PLOT_DIR, "surrogate_nk_penalty_data_ours.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Lambda', 'Average Penalty', 'Standard Deviation'])
        for row in lambda_results:
            writer.writerow(row)
    print(f"Saved data to {csv_filename}")
    print("\nEvaluation complete.")
else:
    env = make_vec_env(ENV_NAME, n_envs=1)

    print(f'\n--- Evaluating agent {AGENT_NAME} for NK penalty with pushforward function ---')

    overall_average_penalties = [] 
    episode_rewards = []

    for episode_num in range(NUM_EVAL_EPISODES):
        initial_obs_for_episode = env.reset()
        obs = initial_obs_for_episode
        done = False
        penalties_this_episode = []
        episode_reward = 0
        episode_steps = 0
        
        print(f"Starting Episode {episode_num + 1}/{NUM_EVAL_EPISODES}")

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            
            modified_action = pushforward(action, obs, LAMBDA_PUSHFORWARD)
            
            next_obs, reward, done_array, info = env.step(modified_action)
            
            penalty = calculate_nk_penalty(next_obs, initial_obs_for_episode)
            penalties_this_episode.append(penalty)
            episode_reward += reward[0] 
            
            obs = next_obs
            done = done_array[0]
            episode_steps += 1
        
        if penalties_this_episode:
            avg_penalty_for_this_episode = np.mean(penalties_this_episode)
            overall_average_penalties.append(avg_penalty_for_this_episode)
            episode_rewards.append(episode_reward) 
            print(f"Episode {episode_num + 1}: Average NK Penalty = {avg_penalty_for_this_episode:.4f}, Steps = {episode_steps}")
        else:
            print(f"Warning: Agent {AGENT_NAME}, Episode {episode_num + 1} had no penalties recorded.")

    env.close()

    if overall_average_penalties:
        final_avg_nk_penalty = np.mean(overall_average_penalties)
        std_dev_nk_penalty = np.std(overall_average_penalties)
        final_avg_reward = np.mean(episode_rewards) 
        print(f"\n--- Summary for Agent: {AGENT_NAME} ---")
        print(f"Overall Average NK Penalty across {NUM_EVAL_EPISODES} episodes = {final_avg_nk_penalty:.4f}")
        print(f"Standard Deviation of Per-Episode Average NK Penalties = {std_dev_nk_penalty:.4f}")
        print(f"Overall Average Reward across {NUM_EVAL_EPISODES} episodes = {final_avg_reward:.4f}")
    else:
        print(f"\nNo penalties recorded for agent {AGENT_NAME} across {NUM_EVAL_EPISODES} episodes.")

    print("\nEvaluation complete.")