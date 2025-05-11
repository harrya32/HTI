import gymnasium as gym
from gymnasium.core import RewardWrapper
import DTRGym  
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description='Train RL agent with custom reward weighting')
parser.add_argument('--lambda_nk', type=float, default=1.0, help='Lambda value for reward weighting (default: 1.0)')
args = parser.parse_args()

# Use the provided lambda value
lambda_nk = args.lambda_nk

# --- Parameters ---
ENV_NAME = 'GhaffariCancerEnv-continuous'
TOTAL_TRAINING_TIMESTEPS = 500000  
NUM_EVAL_EPISODES = 20         
MODEL_SAVE_PATH = f"ppo_ghaffari_cancer_model__{int(lambda_nk * 100)}"
PLOT_DIR = "rl_agent_plots"
ACTION_PLOT_DIR = os.path.join(PLOT_DIR, "action_scatter_plots")
REWARD_PLOT_DIR = os.path.join(PLOT_DIR, "reward_plots")
os.makedirs(ACTION_PLOT_DIR, exist_ok=True)
os.makedirs(REWARD_PLOT_DIR, exist_ok=True)

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

# --- Create Environment ---
lambda_nk = 0.75
def make_env():
    env = gym.make(ENV_NAME)
    env = CustomRewardWrapper(env, lambda_nk=lambda_nk)
    return env

env = make_vec_env(make_env, n_envs=1)

# --- Define and Train the Agent ---
print(f"--- Training PPO Agent on {ENV_NAME} ---")
gamma = 0.99
model = PPO("MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=os.path.join(PLOT_DIR, "tensorboard_logs"), 
            gamma=gamma
)

# Train the agent
model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS, progress_bar=True)
model.save(MODEL_SAVE_PATH)
print(f"--- Training Complete. Model saved to {MODEL_SAVE_PATH}.zip ---")

# --- Evaluate the Trained Agent ---
print(f"\n--- Evaluating Trained Agent for {NUM_EVAL_EPISODES} episodes ---")

obs = env.reset()
eval_actions = []
eval_episode_rewards = []        
eval_per_timestep_rewards = [] 


for episode in range(NUM_EVAL_EPISODES):
    done = False
    episode_reward = 0
    rewards = []

    if episode > 0:
         obs = env.reset() 

    print(f"Starting Evaluation Episode {episode + 1}")
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        eval_actions.append(action[0])
        obs, reward, done, info = env.step(action)
        scalar_reward = reward[0]
        scalar_done = done[0]
        rewards.append(scalar_reward)
        episode_reward += scalar_reward
        done = scalar_done

        # Optional: Check for truncated if your env uses it (info usually contains it)
        # truncated = info[0].get('TimeLimit.truncated', False) # Example check
        # if truncated: done = True # Treat truncation as done for eval loop

    eval_episode_rewards.append(episode_reward)
    eval_per_timestep_rewards.append(rewards)
    print(f"Finished Evaluation Episode {episode + 1}, Total Reward: {episode_reward}")

env.close()
print("--- Evaluation Complete ---")


# --- Plotting Evaluation Results ---

# --- ACTION PLOT: Evaluation episodes ---
eval_actions = np.array(eval_actions)
orig_env = gym.make(ENV_NAME)
action_dim = orig_env.action_space.shape[0]

if action_dim == 2:
    plt.figure(figsize=(6, 6))
    plt.scatter(eval_actions[:, 0], eval_actions[:, 1], alpha=0.6, s=20)
    plt.xlabel("Action Dimension 1")
    plt.ylabel("Action Dimension 2")
    plt.title(f"Trained Agent Actions across {NUM_EVAL_EPISODES} Eval Episodes")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(ACTION_PLOT_DIR, f"trained_agent_eval_scatter_lambda{lambda_nk}.png"))
    plt.close()
    print(f"Saved evaluation action scatter plot (assuming 2D action space).")
elif action_dim == 1:
     plt.figure(figsize=(8, 4))
     plt.hist(eval_actions[:, 0], bins=50, alpha=0.7)
     plt.xlabel("Action Value")
     plt.ylabel("Frequency")
     plt.title(f"Trained Agent Action Distribution across {NUM_EVAL_EPISODES} Eval Episodes")
     plt.grid(True)
     plt.tight_layout()
     plt.savefig(os.path.join(ACTION_PLOT_DIR, "trained_agent_eval_hist.png"))
     plt.close()
     print(f"Saved evaluation action histogram plot (1D action space).")
else:
    print(f"Skipping action plot: Action dimension is {action_dim}, plotting only implemented for 1D or 2D.")
orig_env.close()

# --- REWARD PLOT 1: Total reward per evaluation episode ---
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EVAL_EPISODES + 1), eval_episode_rewards, marker='o')
plt.xlabel("Evaluation Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode (Trained Agent)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(REWARD_PLOT_DIR, f"trained_agent_total_rewards_lambda{lambda_nk}.png"))
plt.close()

# --- REWARD PLOT 2: Per-timestep reward curves during evaluation ---
plt.figure(figsize=(8, 4))
for i, rewards in enumerate(eval_per_timestep_rewards):
    plt.plot(rewards, label=f"Eval Episode {i+1}")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Reward at Each Timestep per Episode (Trained Agent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(REWARD_PLOT_DIR, f"trained_agent_per_timestep_rewards_lambda{lambda_nk}.png"))
plt.close()

print(f"Saved evaluation reward plots to {REWARD_PLOT_DIR}")

# --- Tensorboard ---
print("\n--- Training Logs ---")
print(f"You can view detailed training logs using tensorboard --logdir {os.path.join(PLOT_DIR, 'tensorboard_logs')}")