# Hyperparameter Trajectory Inference (HTI)

This repository contains code to reproduce experimental results in the NeurIPS submission "Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport". 

HTI refers to the problem of learning how the conditional output distribution of a neural network 
changes as you vary a hyperparameter.

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
python NLOT/generate_synth_data.py
```

### 4. Run the CTI methods
Run the following script to train and evalaute each examined CTI method, over five iterations.

```bash
./semicircles.sh
```

The NLL/C.D. results can be seen in the wandb project logs.

## _Reproducing reward weighting results:_

## 5. Install dependencies

```bash
cd DTR-Bench
pip install -r requirements.txt
cd ..
```

### 6. Train PPO agents and generate training data (optional)
The weights for the PPO agents are already included in the ```DTR-Bench/policies/``` directory. However, if you wish to regenerate them (for $\lambda_{nk} \in \{0,1,2,3,4,5,6,7,8,9,10\}$):

```bash
python ./train_ppo_agents.sh
```

To then generate the training data by running these PPO agents in the environment, run:

```bash
python DTR-Bench/reward_agents_data_gen.py
```

This will saved the policy data to ```DTR-Bench/reward_weighting_data/reward_weighting_data_0_10.pt```, which then needs to be moved to ```NLOT/data/reward_weighting_data_0_10.pt``` to run the HTI models on (we have included our data there already).

```bash
cp DTR-Bench/reward_weighting_data/reward_weighting_data_0_10.pt NLOT/data/reward_weighting_data_0_10.pt
```

### 7. Run the HTI methods
Run the following script to train each examined HTI method, over five iterations.

```bash
./reward_weighting.sh
```

This will save the models in ... . Each relevant ```latest.pkl``` file should be moved to the relevant ```NLOT/surrogate_models/eucl_no_potential``` ($\mathcal{K}_I$), ```NLOT/surrogate_models/eucl_w_potential``` ($\mathcal{K}_I - \hat{\mathcal{U}}$), ```NLOT/surrogate_models/learned_no_potential``` ($\mathcal{K}_\theta$), ```NLOT/surrogate_models/learned_w_potential``` ($\mathcal{K}_\theta - \hat{\mathcal{U}}$).

### 8. Run evaluation
Run the following script to evaluate each surrogate model in the cancer environment, and evaluate the average reward over five iterations.

```bash
./run_surrogate_eval.sh
```