import DTRGym  
import numpy as np
import os
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 
import torch
import matplotlib.pyplot as plt


agent_configs = [
    ("agent_lambda0", "policies/ppo_ghaffari_cancer_model__0.zip"),       
    #("agent_lambda0.1", "ppo_ghaffari_cancer_model.zip"),      
    #("agent_lambda0.25", "ppo_ghaffari_cancer_model_25.zip"),    
    #("agent_lambda0.5", "ppo_ghaffari_cancer_model_05.zip"),     
    #("agent_lambda0.75", "ppo_ghaffari_cancer_model_75.zip"),    
    #("agent_lambda0.9", "ppo_ghaffari_cancer_model_09.zip"),  
    ("agent_lambda1", "policies/ppo_ghaffari_cancer_model__100.zip"),   
    ("agent_lambda2", "policies/ppo_ghaffari_cancer_model__200.zip"),     
    #("agent_lambda2.5", "ppo_ghaffari_cancer_model__250.zip"),
    ("agent_lambda3", "policies/ppo_ghaffari_cancer_model__300.zip"),     
    ("agent_lambda4", "policies/ppo_ghaffari_cancer_model__400.zip"),     
    ("agent_lambda5", "policies/ppo_ghaffari_cancer_model__500.zip"),    
    ("agent_lambda6", "policies/ppo_ghaffari_cancer_model__600.zip"),     
    ("agent_lambda7", "policies/ppo_ghaffari_cancer_model__700.zip"),     
    ("agent_lambda8", "policies/ppo_ghaffari_cancer_model__800.zip"),     
    ("agent_lambda9", "policies/ppo_ghaffari_cancer_model__900.zip"),     
    ("agent_lambda10", "policies/ppo_ghaffari_cancer_model__1000.zip"),   
]


lambda_values = [
    0.0,  
    1.0,
    2.0,  
    3.0,  
    4.0,  
    5.0,  
    6.0,  
    7.0,  
    8.0,  
    9.0,  
    10.0 
]


loaded_models_info = []
print("--- Loading Agents ---")
for i, (name, path) in enumerate(agent_configs):
    if os.path.exists(path):
        try:
            model = PPO.load(path)
            current_lambda = lambda_values[i] if i < len(lambda_values) else float('nan') # Handle potential mismatch
            loaded_models_info.append({"name": name, "model": model, "lambda": current_lambda})
            print(f"Successfully loaded model '{name}' (lambda: {current_lambda}) from '{path}'")
        except Exception as e:
            print(f"Error loading model '{name}' from '{path}': {e}")
    else:
        print(f"Model file not found for '{name}' at '{path}'. Skipping.")

ENV_NAME = 'GhaffariCancerEnv-continuous'
NUM_EVAL_EPISODES = 100         
PLOT_DIR = "nk_penalty_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

env = make_vec_env(ENV_NAME, n_envs=1)

# --- NK Penalty Calculation Function ---
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

print('\n--- Evaluating agents for NK penalty, each using their own actions ---')

agent_overall_average_penalties = {}
penalties_for_plot = []

for agent_data in loaded_models_info:
    agent_name = agent_data["name"]
    model = agent_data["model"]
    agent_lambda = agent_data["lambda"]
    print(f"\n--- Evaluating Agent: {agent_name} (Lambda: {agent_lambda}) ---")
    
    average_penalties_per_episode = []
    
    for episode_num in range(NUM_EVAL_EPISODES):
        initial_obs_for_episode = env.reset()
        obs = initial_obs_for_episode
        done = False
        penalties_this_episode = []
        episode_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False) 
            next_obs, reward, done_array, info = env.step(action)
            
            penalty = calculate_nk_penalty(next_obs, initial_obs_for_episode)
            penalties_this_episode.append(penalty)
            
            obs = next_obs
            done = done_array[0]
            episode_steps += 1
        
        if penalties_this_episode:
            avg_penalty_for_this_episode = np.mean(penalties_this_episode)
            average_penalties_per_episode.append(avg_penalty_for_this_episode)
        else: 
            print(f"Warning: Agent {agent_name}, Episode {episode_num + 1} had no penalties recorded.")

    if average_penalties_per_episode:
        overall_avg_penalty = np.mean(average_penalties_per_episode)
        std_dev_of_episode_averages = np.std(average_penalties_per_episode)
    else:
        overall_avg_penalty = float('nan')
        std_dev_of_episode_averages = float('nan')
        
    agent_overall_average_penalties[agent_name] = overall_avg_penalty
    if not np.isnan(agent_lambda):
        penalties_for_plot.append((agent_lambda, overall_avg_penalty, std_dev_of_episode_averages))
    print(f"Agent {agent_name}: Overall Average NK Penalty across {NUM_EVAL_EPISODES} episodes = {overall_avg_penalty:.4f}")
    print(f"Agent {agent_name}: StdDev of Average Per-Episode NK Penalties = {std_dev_of_episode_averages:.4f}")

env.close()

print("\n--- Summary of Overall Average NK Penalties ---")
for agent_name, overall_avg_penalty in agent_overall_average_penalties.items():
    std_dev_val_for_print = float('nan')
    current_agent_lambda_for_lookup = next((item['lambda'] for item in loaded_models_info if item['name'] == agent_name), None)
    if current_agent_lambda_for_lookup is not None:
        for l_val, avg_val, std_val in penalties_for_plot:
            if l_val == current_agent_lambda_for_lookup:
                std_dev_val_for_print = std_val
                break
    print(f"Agent {agent_name}: {overall_avg_penalty:.4f} (StdDev of Episode Avgs: {std_dev_val_for_print:.4f})")

# --- Plotting Average NK Penalty vs Lambda ---
if penalties_for_plot:
    penalties_for_plot.sort(key=lambda x: x[0])
    
    plot_lambdas = [item[0] for item in penalties_for_plot]
    plot_avg_penalties = [item[1] for item in penalties_for_plot]
    plot_std_devs = [item[2] for item in penalties_for_plot]

    plt.figure(figsize=(12, 7)) 
    
    plt.errorbar(plot_lambdas, plot_avg_penalties, yerr=plot_std_devs, 
                 marker='o', linestyle='-', capsize=5, label='Overall Avg NK Penalty ± StdDev of Episode Avgs')
    
    plt.title(f"Overall Average NK Penalty vs. Lambda Value ({NUM_EVAL_EPISODES} Eval Episodes)")
    plt.xlabel("Lambda_nk Value")
    plt.ylabel("Overall Average NK Penalty")
    plt.grid(True)
    if len(plot_lambdas) > 10:
         plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(plot_lambdas)
    
    plt.legend()
    plt.tight_layout()
    
    plot_filename = os.path.join(PLOT_DIR, "nk_penalty_vs_lambda.png")
    plt.savefig(plot_filename)
    print(f"\nSaved overall average NK penalty plot with episodic std dev error bars to {plot_filename}")
    plt.close()
    import csv
    csv_filename = os.path.join(PLOT_DIR, "nk_penalty_data.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Lambda', 'Average Penalty', 'Standard Deviation'])
        for row in penalties_for_plot:
            writer.writerow(row)
    print(f"Saved NK penalty data to {csv_filename}")
else:
    print("\nNo valid data to plot for NK penalty vs Lambda.")

print("\nEvaluation complete.")