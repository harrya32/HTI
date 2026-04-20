# Hyperparameter Trajectory Inference (HTI)

This repository contains code for the ICLR submission "Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport". 

# Training HTI on a new dataset

## 1. Create and activate conda environment

```bash
conda create -n hti_env python=3.10 -y
conda activate hti_env
```

### 2. Install HTI dependencies
```bash
cd NLOT
pip install -r requirements.txt
cd ..
```

### 3. Put your dataset in `NLOT/data/`
The HTI data loader (`NLOT/lagrangian_ot/data.py`) loads a `.pt` file from:

```text
NLOT/data/<your_dataset>.pt
```

Expected shape:

```text
[num_timepoints, num_samples_per_timepoint, D + C]
```

- `D`: ambient/state dimensions used by OT geometry (first `D` columns).
- `C`: conditioning dimensions (last `C` columns).
- If `categorical=True`, the code uses the first conditioning column (`x[:, D]`) as an integer category id in `[0, num_categories-1]` (use `C=1` in this case).
- If `categorical=False`, all `C` conditioning columns are treated as continuous features.

### 5. Register the dataset name
Edit `NLOT/lagrangian_ot/data.py`:

- In `get_samplers(...)`, add your dataset name to `paths`, e.g. `"my_dataset": "my_dataset.pt"`.
- In `get_bounds(...)`, add plotting/interpolation bounds for your dataset.
- Add any required dataset-specific slicing/reformatting in `get_samplers(...)`.

Optional:

- Add custom intermediate-marginal evaluation data in `NLOT/train.py` inside `_get_marginal_eval_data(...)` if you want evaluation beyond training loss.

### 6. Set training settings
Base defaults live in `NLOT/train.yaml`. You can override any field at runtime via CLI.

Common settings to change:

- Dataset name/dimensions: `dataset`, `D`, `C`, `categorical`, `num_categories`
- Training budget: `num_train_iters`, `metric.update_frequency`
- Learned geometry/potential energy: `geometry`, `include_inverse_potential`
- Logging/output: `wandb`, `wandb_project`, `plot_frequency`, `save_frequency`, `collect_save_dir`

### 7. Run training
From repo root:

```bash
python NLOT/train.py
```

To run the full set of HTI variants across seeds, copy/edit one of the scripts in `hti_scripts/` and replace dataset/dimension/hyperparameter values.

### 8. Accessing saved checkpoints
Each run writes to a timestamped Hydra run directory:

```text
exp/local/<YYYY.MM.DD>/<HHMM>.<geometry>/
```

Inside each run directory, the main checkpoint is:

```text
latest.pkl
```

Other run artifacts include `log.csv` and plots. If `collect_save_dir` is set, a copy of the checkpoint is also written there with an auto-generated filename.


# Reproducing experiments from the paper

## _Reproducing semicircle results:_

### 1. Generate training data (optional)
The synthetic training data is already included in the ```NLOT/data/``` directory. However, if you wish to regenerate it:

```bash
cd NLOT
python generate_synth_data.py
cd ..
```

### 2. Run the HTI methods
Run the following script to train and evalaute each examined HTI method, over twenty iterations.

```bash
./hti_scripts/semicircles.sh
```

The NLL and C.D. results can be seen in the wandb project logs.

## _Reproducing cancer reward weighting results:_

## 1. Install dependencies for RL environment

```bash
pip install gymnasium==0.28.1
pip install DTRGym==0.1.0
pip install --upgrade typing-extensions
pip install stable_baselines3==2.6.0 --no-deps
```

### 2. Train PPO agents and generate training data (optional)
Training data is already in the required directory, but this can be re-generated as follows. To train PPO agents (for $\lambda_{nk} \in \{0,1,2,3,4,5,6,7,8,9,10\}$):

```bash
cd DTR-bench
./train_ppo_agents.sh
cd ..
```

or 

```bash
cd DTR-bench
./train_ppo_reward_weighting_hinge.sh
cd ..
```

for the non-linear reward scalarization experiment.


To then generate the training data by running these PPO agents in the environment, run:

```bash
cd DTR-Bench
python reward_agents_data_gen.py
cd ..
```

or 

```bash
cd DTR-Bench
python reward_weighting_hinge_data_gen.py
cd ..
```

This data will be saved to ```DTR-Bench/reward_weighting_data/reward_weighting_data_0_10.pt``` and ```DTR-Bench/reward_weighting_hinge_data/reward_weighting_hinge_data.pt```, which then needs to be moved to ```NLOT/data/reward_weighting_data_0_10.pt``` and ```NLOT/data/reward_weighting_hinge_data.pt``` to run the HTI models on.

### 3. Run the HTI methods
Run the following scripts to train the HTI methods, over twenty iterations.

```bash
./hti_scripts/reward_weighting.sh
```

```bash
./hti_scripts/reward_weighting_hinge.sh
```

Each model will be saved in their own respective runs, in a folder formatted like ```saves/<DATASET>_<GEOMETRY>_<USE_POTENTIAL_ENERGY>_<SEED>_<ALPHA>.pkl```. Each ```.pkl``` file should be moved to a relevant combined folders ```NLOT/surrogate_models/reward_weighting/eucl_no_potential``` ($\mathcal{K}_I$), ```NLOT/surrogate_models/reward_weighting/eucl_w_potential``` ($\mathcal{K}_I - \hat{\mathcal{U}}$), ```NLOT/surrogate_models/reward_weighting/learned_no_potential``` ($\mathcal{K}_\theta$), ```NLOT/surrogate_models/reward_weighting/learned_w_potential``` ($\mathcal{K}_\theta - \hat{\mathcal{U}}$) (or replace ```reward_weighting``` with ```reward_weighting_hinge``` if running non-linear scalarization experiment).

### 4. Run evaluation
Run script to run each surrogate model in the cancer environment and evaluate the average reward.

```bash
cd DTR-Bench
./run_surrogate_eval.sh
cd ..
```
or

```bash
cd DTR-Bench
./run_hinge_surrogate_eval.sh
cd ..
```

The results will be saved in the relevant ```DTR-Bench/surrogate_plots_reward_weighting/<METHOD>``` or ```DTR-Bench/surrogate_plots_hinge/<METHOD>``` folder, with each iteration's average reward in the files ```final_avg_reward_<METHOD>_<ITER>.txt```

## _Reproducing Reacher reward weighting results:_

## 1. Install dependencies for RL environment

```bash
pip install gymnasium[mujoco]
```

### 2. Train PPO agents and generate training data (optional)
To train PPO agents (for $\lambda_{control} \in \{1,2,3,4,5\}$), follow the notebook ```DTR-Bench/reacher.ipynb```

This will save the policy data to ```DTR-Bench/reacher_data_<LAMBDA>.pt```, which then need to be combined and moved to ```NLOT/data/reacher_data.pt``` to run the HTI models on.

### 3. Run the HTI methods
Run the following script to train each examined HTI method, over twenty iterations.

```bash
./hti_scripts/reacher.sh
```

Each model will be saved in ```saves/<DATASET>_<GEOMETRY>_<USE_POTENTIAL_ENERGY>_<SEED>_<ALPHA>.pkl```. Each ```.pkl``` file should be moved to a relevant combined folders ```NLOT/surrogate_models/reacher/eucl_no_potential``` ($\mathcal{K}_I$), ```NLOT/surrogate_models/reacher/eucl_w_potential``` ($\mathcal{K}_I - \hat{\mathcal{U}}$), ```NLOT/surrogate_models/reacher/learned_no_potential``` ($\mathcal{K}_\theta$), ```NLOT/surrogate_models/reacher/learned_w_potential``` ($\mathcal{K}_\theta - \hat{\mathcal{U}}$).

### 4. Run evaluation
Run the following script to run each surrogate model in the Reacher environment and evaluate the average reward.

```bash
cd DTR-Bench
./run_reacher_surrogate_eval.sh
cd ..
```

The results will be saved in the relevant ```DTR-Bench/surrogate_plots_reacher/<METHOD>```, with each iteration's average reward in the files ```final_avg_reward_<METHOD>_<ITER>.txt```

## _Reproducing quantile regression results:_

### 1. Train quantile regression models and generate training data (optional)

Run the following training script to train NN time-series quantile forecasters on ETT data at $\tau \in \{0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99\}$:

```bash
cd quantile_regression
./train_forecasters.sh
cd ..
```

Run the following script to generate forecasts from each model, to act as the HTI training dataset:

```bash
cd quantile_regression
./generate_hti_data.sh
cd ..
```

The data will be saved in ``quantile_regression/hti_data``. Combine the data with the notebook and move the file to ``NLOT/data/quantile_data_new.pt```

### 2. Run the HTI methods
Run the following script to train each examined HTI method, over twenty iterations.

```bash
./hti_scripts/ett_quantiles.sh
```

Each model will be saved in ```saves/<DATASET>_<GEOMETRY>_<USE_POTENTIAL_ENERGY>_<SEED>_<ALPHA>.pkl```. Each ```.pkl``` file should be moved to a relevant combined folders ```NLOT/surrogate_models/ett_quantile/eucl_no_potential``` ($\mathcal{K}_I$), ```NLOT/surrogate_models/ett_quantile/eucl_w_potential``` ($\mathcal{K}_I - \hat{\mathcal{U}}$), ```NLOT/surrogate_models/ett_quantile/learned_no_potential``` ($\mathcal{K}_\theta$), ```NLOT/surrogate_models/ett_quantile/learned_w_potential``` ($\mathcal{K}_\theta - \hat{\mathcal{U}}$).

### 3. Run evaluation
Run the following script to evaluate the difference between each surrogate forecast and the true NN forecasts at unseen quantiles.

```bash
cd DTR-Bench
./run_surrogate_ett_quantile_eval.sh
cd ..
```

The results will be saved in the relevant ```DTR-Bench/surrogate_plots_ett_quantile/<METHOD>```.

## _Reproducing generative dropout results:_

### 1. Train diffusion models and generate training data (optional)

Run the following script to train diffusion models, and generate data from them, on the two moons dataset at $p \in \{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99\}$:

```bash
cd generative_dropout
python generate_2moons_dropout_data.py
cd ..
```

The data will be saved in ``generative_dropout/diffusion_2moons_dropout.pt``. Move the file to ``NLOT/data/diffusion_2moons_dropout.pt```


### 2. Run the HTI methods
Run the following script to train each examined HTI method, over twenty iterations.

```bash
./hti_scripts/generative_dropout.sh
```

The W.D. results can be seen in the wandb project logs.
