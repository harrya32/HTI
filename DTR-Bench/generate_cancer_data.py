import gymnasium as gym
from gymnasium.core import RewardWrapper
import DTRGym
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

# --- Parameters ---
ENV_NAME = 'GhaffariCancerEnv-continuous'
TOTAL_TRAINING_TIMESTEPS = 5000
NUM_EVAL_EPISODES = 50  # Run 50 evaluation episodes
REWARD_SHAPING_FACTORS = [0, 0.25, 0.75, 1.0]  # The four different factors
MODEL_SAVE_DIR = "reward_shaping_models"
DATASET_DIR = "reward_shaping_datasets"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Custom Reward Wrapper with different reward shaping factors ---
class CustomRewardWrapper(RewardWrapper):
    """
    Wrapper that modifies the reward according to reward shaping factor
    """
    def __init__(self, env, reward_shaping_factor=0.0):
        super().__init__(env)
        self.reward_shaping_factor = reward_shaping_factor
        
    def reward(self, reward):
        """
        Apply reward shaping based on the factor
        """
        if reward > 50:
                return 1
        elif reward < -50:
            return -1
        else:
            return self.reward_shaping_factor * reward

# --- Train agents with different reward shaping factors ---
def train_agent(env_name, reward_shaping_factor, total_timesteps, save_path):
    """Train an agent with a specific reward shaping factor"""
    print(f"\n--- Training agent with reward_shaping_factor={reward_shaping_factor} ---")
    
    # Create environment with the specific reward shaping factor
    def make_env():
        env = gym.make(env_name)
        env = CustomRewardWrapper(env, reward_shaping_factor=reward_shaping_factor)
        return env
    
    env = make_vec_env(make_env, n_envs=1)
    
    # Create and train the agent
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.99)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    env.close()
    return model

# --- Main script execution ---
def main():
    start_time = time.time()
    
    # Get state and action dimensions
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()
    
    print(f"Environment: {ENV_NAME}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Train all agents
    print("\n=== Training Agents ===")
    models = []
    
    for factor in REWARD_SHAPING_FACTORS:
        model_path = os.path.join(MODEL_SAVE_DIR, f"ppo_agent_rs{factor}")
        model_path = model_path.replace(".", "_")
        model = train_agent(ENV_NAME, factor, TOTAL_TRAINING_TIMESTEPS, model_path)
        models.append(model)
    
    # Create environment for evaluation (using the first reward shaping factor for actual steps)
    def make_eval_env():
        env = gym.make(ENV_NAME)
        env = CustomRewardWrapper(env, reward_shaping_factor=REWARD_SHAPING_FACTORS[0])
        return env
    
    eval_env = make_vec_env(make_eval_env, n_envs=1)
    
    # Collect state-action pairs
    print("\n=== Collecting State-Action Pairs ===")
    state_action_data = [[] for _ in range(len(models))]  # List to store data for each agent
    total_steps = 0
    
    for episode in range(NUM_EVAL_EPISODES):
        obs = eval_env.reset()
        done = False
        episode_steps = 0
        
        print(f"Starting evaluation episode {episode + 1}/{NUM_EVAL_EPISODES}")
        
        while not done:
            # Get actions from all models for the current state
            actions = []
            for model in models:
                action, _ = model.predict(obs, deterministic=False)
                actions.append(action)
            
            # Step the environment using the first model's action
            next_obs, reward, done_array, info = eval_env.step(actions[0])
            
            # Process the state and actions
            state = obs[0]
            for i, action in enumerate(actions):
                # Extract the action from vectorized form for storage
                action_flat = action.flatten()  # Flatten to handle any action shape
                # Combine state and action
                state_action = np.concatenate([state, action_flat])
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
    
    # Stack the data from all agents 
    dataset = torch.stack(tensor_data)
    
    
    # Save dataset
    dataset_path = os.path.join(DATASET_DIR, "agent_state_action_dataset.pt")
    torch.save(dataset, dataset_path)
    
    # Print dataset information
    print(f"\n=== Dataset Information ===")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Expected shape: ({len(models)}, {total_steps}, {state_dim + action_dim})")
    print(f"Dataset saved to: {dataset_path}")

    print(f"Example row from first agent: {dataset[0, 0, :]}")
    print(f"Example row from second agent: {dataset[1, 0, :]}")
    print(f"Example row from third agent: {dataset[2, 0, :]}")
    print(f"Example row from fourth agent: {dataset[3, 0, :]}")
    
    # Print total runtime
    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()