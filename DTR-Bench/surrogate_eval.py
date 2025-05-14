import DTRGym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import matplotlib.pyplot as plt
import argparse

AGENT_PATH = "ppo_ghaffari_cancer_model.zip" 
AGENT_NAME = "single_agent_eval"
ENV_NAME = 'GhaffariCancerEnv-continuous'
NUM_EVAL_EPISODES = 20

parser = argparse.ArgumentParser(description='Evaluate a single agent with a pushforward function.')
parser.add_argument('--lambda_pushforward', type=float, default=1.0, help='Lambda value for the pushforward function.')
args = parser.parse_args()
LAMBDA_PUSHFORWARD = args.lambda_pushforward

def pushforward(action, obs, lambda_val):
    """
    Modifies the agent's action.
    Implement the pushforward logic here.
    For now, it returns the action unmodified.
    """
    modified_action = action * lambda_val
    return modified_action

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

print(f"--- Loading Agent: {AGENT_NAME} ---")
if os.path.exists(AGENT_PATH):
    try:
        model = PPO.load(AGENT_PATH)
        print(f"Successfully loaded model '{AGENT_NAME}' from '{AGENT_PATH}'")
    except Exception as e:
        print(f"Error loading model '{AGENT_NAME}' from '{AGENT_PATH}': {e}")
        model = None
else:
    print(f"Model file not found for '{AGENT_NAME}' at '{AGENT_PATH}'. Exiting.")
    model = None

if model is None:
    exit()

env = make_vec_env(ENV_NAME, n_envs=1)

print(f'\\n--- Evaluating agent {AGENT_NAME} for NK penalty with pushforward function ---')

overall_average_penalties = [] 

for episode_num in range(NUM_EVAL_EPISODES):
    initial_obs_for_episode = env.reset()
    obs = initial_obs_for_episode
    done = False
    penalties_this_episode = []
    episode_steps = 0
    
    print(f"Starting Episode {episode_num + 1}/{NUM_EVAL_EPISODES}")

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        
        modified_action = pushforward(action, obs, LAMBDA_PUSHFORWARD)
        
        next_obs, reward, done_array, info = env.step(modified_action)
        
        penalty = calculate_nk_penalty(next_obs, initial_obs_for_episode)
        penalties_this_episode.append(penalty)
        
        obs = next_obs
        done = done_array[0]
        episode_steps += 1
    
    if penalties_this_episode:
        avg_penalty_for_this_episode = np.mean(penalties_this_episode)
        overall_average_penalties.append(avg_penalty_for_this_episode)
        print(f"Episode {episode_num + 1}: Average NK Penalty = {avg_penalty_for_this_episode:.4f}, Steps = {episode_steps}")
    else:
        print(f"Warning: Agent {AGENT_NAME}, Episode {episode_num + 1} had no penalties recorded.")

env.close()

if overall_average_penalties:
    final_avg_nk_penalty = np.mean(overall_average_penalties)
    std_dev_nk_penalty = np.std(overall_average_penalties)
    print(f"\\n--- Summary for Agent: {AGENT_NAME} ---")
    print(f"Overall Average NK Penalty across {NUM_EVAL_EPISODES} episodes = {final_avg_nk_penalty:.4f}")
    print(f"Standard Deviation of Per-Episode Average NK Penalties = {std_dev_nk_penalty:.4f}")
else:
    print(f"\\nNo penalties recorded for agent {AGENT_NAME} across {NUM_EVAL_EPISODES} episodes.")

print("\\nEvaluation complete.")
