import DTRGym  
import matplotlib.pyplot as plt
import numpy as np
import os
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 
import torch

#load in agent
agent_0 = PPO.load("ppo_ghaffari_cancer_model__0.zip")
#agent_01 = PPO.load("ppo_ghaffari_cancer_model.zip")
#agent_025 = PPO.load("ppo_ghaffari_cancer_model_25.zip")
#agent_05 = PPO.load("ppo_ghaffari_cancer_model_05.zip")
#agent_075 = PPO.load("ppo_ghaffari_cancer_model_75.zip")
#agent_09 = PPO.load("ppo_ghaffari_cancer_model_09.zip")
agent_1 = PPO.load("ppo_ghaffari_cancer_model__100.zip")
agent_2 = PPO.load("ppo_ghaffari_cancer_model__200.zip")
agent_3 = PPO.load("ppo_ghaffari_cancer_model__300.zip")
agent_4 = PPO.load("ppo_ghaffari_cancer_model__400.zip")
agent_5 = PPO.load("ppo_ghaffari_cancer_model__500.zip")
agent_6 = PPO.load("ppo_ghaffari_cancer_model__600.zip")
agent_7 = PPO.load("ppo_ghaffari_cancer_model__700.zip")
agent_8 = PPO.load("ppo_ghaffari_cancer_model__800.zip")
agent_9 = PPO.load("ppo_ghaffari_cancer_model__900.zip")
agent_10 = PPO.load("ppo_ghaffari_cancer_model__1000.zip")
models = [agent_0, agent_1, agent_2, agent_3, agent_4, agent_5, agent_6, agent_7, agent_8, agent_9, agent_10]

# --- Parameters ---
ENV_NAME = 'GhaffariCancerEnv-continuous'
NUM_EVAL_EPISODES = 20         
PLOT_DIR = "reward_weighting_plots"
ACTION_PLOT_DIR = os.path.join(PLOT_DIR, "action_scatter_plots")
REWARD_PLOT_DIR = os.path.join(PLOT_DIR, "reward_plots")
DATASET_DIR = "reward_weighting_data"
os.makedirs(ACTION_PLOT_DIR, exist_ok=True)
os.makedirs(REWARD_PLOT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

env = make_vec_env(ENV_NAME, n_envs=1)

print('Evaluating agents, using states from agent_0')

state_action_data = [[] for _ in range(len(models))] 
total_steps = 0
for episode in range(NUM_EVAL_EPISODES):
    obs = env.reset()
    done = False
    episode_steps = 0
    while not done:
        actions_by_model = []
        for model in models:
            model_actions = []
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=False)
                model_actions.append(action)
            actions_by_model.append(model_actions)

        next_obs, reward, done_array, info = env.step(actions_by_model[0][0])

        state = obs[0]
        for i, model_actions in enumerate(actions_by_model):
            for action in model_actions:
                action_flat = action.flatten() 
                state_action = np.concatenate([action_flat, state])
                state_action_data[i].append(state_action)
        
        obs = next_obs
        done = done_array[0]
        episode_steps += 1
        total_steps += 1
        
        if done:
            print(f"Episode {episode + 1} completed in {episode_steps} steps")

print(f"\nTotal steps across all episodes: {total_steps}")

# Convert data to tensor and reshape
tensor_data = []
for i in range(len(models)):
    agent_data = np.array(state_action_data[i])
    tensor_data.append(torch.tensor(agent_data, dtype=torch.float32))

dataset = torch.stack(tensor_data)
dataset_path = os.path.join(DATASET_DIR, "reward_weighting_data_0_10.pt")
print("Shape of dataset:", dataset.shape)
torch.save(dataset, dataset_path)

# for each model, plot the actions
for i, model in enumerate(models):
    agent_data = state_action_data[i][:1000]
    agent_data = np.array(agent_data)
    actions = agent_data[:, :2]
    #print the average of each action for each model
    print(f"Model {i+1} - Action 1 mean: {np.mean(actions[:, 0])}, Action 2 mean: {np.mean(actions[:, 1])}")
    states = agent_data[:, 2:]
    states = np.concatenate([np.repeat(i, 10) for i in range(10)])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(actions[:, 0], actions[:, 1], alpha=0.3)
    plt.title(f"Actions for Model {i+1}")
    plt.xlabel("Action 1")
    plt.ylabel("Action 2")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.colorbar(label='State')
    plt.grid()
    plt.savefig(os.path.join(ACTION_PLOT_DIR, f"actions_model_{i+1}.png"))
    plt.close()
    
