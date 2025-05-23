# Hyperparameter Trajectory Inference (HTI)

This repository contains code to reproduce experimental results for the NeurIPS submission "Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport". 

## 1. Create and activate conda environment

```bash
conda create -n hti_env python=3.10 -y
conda activate hti_env
```

## _Reproducing semicircle results:_

## 2. Install dependencies

```bash
cd NLOT
pip install -r requirements.txt
cd ..
```

### 3. Generate training data (optional)
The synthetic training data is already included in the ```NLOT/data/``` directory. However, if you wish to regenerate it:

```bash
cd NLOT
python generate_synth_data.py
cd ..
```

### 4. Run the CTI methods
Run the following script to train and evalaute each examined CTI method, over five iterations.

```bash
./semicircles.sh
```

The NLL and C.D. results can be seen in the wandb project logs.

## _Reproducing reward weighting results:_

## 5. Install dependencies

```bash
pip install gymnasium==0.28.1
pip install DTRGym==0.1.0
pip install --upgrade typing-extensions
pip install stable_baselines3==2.6.0 --no-deps
```

### 6. Train PPO agents and generate training data
To train PPO agents (for $\lambda_{nk} \in \{0,1,2,3,4,5,6,7,8,9,10\}$):

```bash
cd DTR-bench
python train_ppo_agents.sh
cd ..
```

To then generate the training data by running these PPO agents in the environment, run:

```bash
cd DTR-Bench
python reward_agents_data_gen.py
cd ..
```

This will saved the policy data to ```DTR-Bench/reward_weighting_data/reward_weighting_data_0_10.pt```, which then needs to be moved to ```NLOT/data/reward_weighting_data_0_10.pt``` to run the HTI models on.

```bash
cp DTR-Bench/reward_weighting_data/reward_weighting_data_0_10_testing.pt NLOT/data/reward_weighting_data_0_10_testing.pt
```

### 7. Run the HTI methods
Run the following script to train each examined HTI method, over five iterations.

```bash
./reward_weighting.sh
```

Each model will be saved in their own respective runs, in a folder formatted like ```exp/local/<DATE>/<TIME>.<GEOMETRY>/latest.pkl```. Each ```latest.pkl``` file should be moved to the relevant combined folders ```NLOT/surrogate_models/eucl_no_potential``` ($\mathcal{K}_I$), ```NLOT/surrogate_models/eucl_w_potential``` ($\mathcal{K}_I - \hat{\mathcal{U}}$), ```NLOT/surrogate_models/learned_no_potential``` ($\mathcal{K}_\theta$), ```NLOT/surrogate_models/learned_w_potential``` ($\mathcal{K}_\theta - \hat{\mathcal{U}}$) (make sure to add an identifying number to each ```latest.pkl``` before moving, e.g. ```latest_0.pkl``` so they do not override each other).

### 8. Run evaluation
Run the following script to run each surrogate model in the cancer environment and evaluate the average reward over five iterations.

```bash
cd DTR-Bench
./run_surrogate_eval.sh
cd ..
```

The results will be saved in the relevant ```DTR-Bench/surrogate_plots/<METHOD>``` folder, with each iteration's average reward in the files ```final_avg_reward_<METHOD>_<ITER>.txt```