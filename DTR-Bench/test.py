import gymnasium as gym
import DTRGym  
import matplotlib.pyplot as plt
import numpy as np
import os
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 

# --- Parameters ---
ENV_NAME = 'GhaffariCancerEnv-continuous'
TOTAL_TRAINING_TIMESTEPS = 100000  
NUM_EVAL_EPISODES = 10         
MODEL_SAVE_PATH = "ppo_ghaffari_cancer_model"
PLOT_DIR = "rl_agent_plots"
ACTION_PLOT_DIR = os.path.join(PLOT_DIR, "action_scatter_plots")
REWARD_PLOT_DIR = os.path.join(PLOT_DIR, "reward_plots")
# --- End Parameters ---


os.makedirs(ACTION_PLOT_DIR, exist_ok=True)
os.makedirs(REWARD_PLOT_DIR, exist_ok=True)

env = make_vec_env(ENV_NAME, n_envs=1)

# --- Define and Train the Agent ---
print(f"--- Training PPO Agent on {ENV_NAME} ---")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(PLOT_DIR, "tensorboard_logs"))

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
    plt.title(f"Trained Agent Actions (Deterministic) across {NUM_EVAL_EPISODES} Eval Episodes")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(ACTION_PLOT_DIR, "trained_agent_eval_scatter.png"))
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
plt.savefig(os.path.join(REWARD_PLOT_DIR, "trained_agent_total_rewards.png"))
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
plt.savefig(os.path.join(REWARD_PLOT_DIR, "trained_agent_per_timestep_rewards.png"))
plt.close()

print(f"Saved evaluation reward plots to {REWARD_PLOT_DIR}")

# --- Tensorboard ---
print("\n--- Training Logs ---")
print(f"You can view detailed training logs using TensorBoard:")
print(f"tensorboard --logdir {os.path.join(PLOT_DIR, 'tensorboard_logs')}")
print("Install TensorBoard if you haven't: pip install tensorboard")