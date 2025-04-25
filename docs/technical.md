# Technical Documentation

## Project Overview

### Research Pipeline
1. **Environment Integration**
   - Glucose environment (diabetes management)
   - Gym-Sepsis environment (sepsis treatment)
   - HIV_RL environment (HIV treatment)

2. **RL Policy Training**
   - Neural Network policies
   - Multiple hyperparameter configurations
   - Policy evaluation and behavior collection

3. **Riemann Metric Learning**
   - Neural Networks to estimate the metric of the conditional manifolds of the output distributions from the learned policies
   - To learn geodesics along the manifolds

4. **Conditional Flow Matching Training**
   - Input: Policy outputs and hyperparameter level
   - Output: Velocity field allowing continuous interpolation between hyperparameter settings
   - Goal: Learn the velocity field to match geodesics calculated according to learned metrics from step 3.

## Technology Stack

### Core Technologies
- **Programming Language**: Python 3.8+
- **ML Framework**: PyTorch
- **RL Environments**: Custom environments (glucose, sepsis, HIV)
- **Experiment Tracking**: Weights & Biases
- **Configuration Management**: Hydra
- **Testing Framework**: pytest

### Development Environment
- **Virtual Environment**: conda/venv
- **Code Quality**:
  - Type checking: mypy
  - Linting: flake8
  - Formatting: black
  - Import sorting: isort

## Established Patterns

### 1. Code Organization
```
src/
├── environments/          # Environment wrappers and interfaces
│   ├── glucose/
│   ├── sepsis/
│   └── hiv/
├── policies/             # Original NN Policy implementations
│   ├── networks/
│   ├── training/        # (Training happens during active sampling)
│   ├── evaluation/
│   └── sampling/        # Behavior collection from trained policies
├── geometry/     # Joint metric & geodesic learning
│   ├── architectures/
│   ├── training/
│   └── utils/
├── active_sampling/     # Active sampling logic
│   ├── uncertainty/
│   ├── acquisition/
│   └── sampler/
├── flows/               # Conditional Flow Matcher (Surrogate Model)
│   ├── architectures/
│   ├── training/
│   └── sampling/        # (Inference-time sampling)
├── collected_data/        # Storage for collected (X,Y) pairs from policies
│   ├── glucose/
│   ├── sepsis/
│   └── hiv/
└── utils/               # Shared utilities
```

### 2. Data Management
- Environment data loaded through standard interfaces
- Policy outputs collected and stored efficiently as `(condition, behavior)` or `(x, y)` pairs.
- Collected data organized by environment and hyperparameter (`lambda`).
- Data used for metric/geodesic learning and flow training consists of these `(x, y)` pairs.
- Clear validation splits for flow training.

### 3. Model Development
- Standard policy architectures
- Conditional flow matching to train normalizing flow models
- Checkpoint management with W&B
- Clear version tracking

### 4. Training Pipeline (Overall Workflow)
- **Initialization:**
  1. Define the two initial hyperparameter values (`lambda_0`, `lambda_1`).
  2. Train the original NN policy separately at `lambda_0` and `lambda_1`.
  3. Collect initial behavior samples as `(x, y)` pairs (`(X, Y)_0`, `(X, Y)_1`).
- **Iterative Active Sampling Loop:**
  1. Train/update the joint Metric/Geodesic model using available behavior data (`(x, y)` pairs).
  2. Actively select the next `lambda^*` using uncertainty and acquisition function.
  3. Train the original NN policy at the selected `lambda^*`.
  4. Collect new behaviors as `(x, y)*` pairs.
  5. Repeat steps 1-4 until the active sampling budget is exhausted.
- **Final Flow Training:**
  1. Use the final Metric/Geodesic model and all collected `(x, y)` pairs.
  2. Train the Conditional Flow Matcher (`v_\theta`).
- Hydra used for configuration of all components.
- W&B used for tracking metrics across all stages.

## Implementation Guidelines

### 1. Policy Configuration
```yaml
# Example policy config
policy:
  architecture:
    type: mlp
    hidden_dims: [128, 128]
    activation: relu
  
  training:
    algorithm: ppo
    learning_rate: 3e-4
    n_steps: 2048
    batch_size: 64
    
  hyperparameters:
    # Parameters we want to interpolate between
    discount: 0.99
    entropy_coef: 0.01
```

### 3. Metric & Geodesic Configs
```yaml
# Example Metric/Geodesic Config
metric_geodesic:
  method: nlot_inspired
  metric_architecture: { ... } 
  geodesic_architecture: { ... }
  joint_training:
    batch_size: 64
    learning_rate: 1e-4
    n_epochs_per_update: 10
    loss_weights: {ot_cost: 1.0, density: 0.1, geodesic_align: 0.5}
```

### 5. Active Sampling Configs
```yaml
# Example Active Sampling Config
active_sampling:
  budget: 20
  initial_lambdas: [0.1, 0.9]
  uncertainty_method: ot_distance_heuristic
  acquisition_strategy: max_average_uncertainty
```

### 6. Experiment Logging (W&B)
```python
import wandb

def log_active_sampling_step(iteration, selected_lambda, score):
    wandb.log({"active_sampling/iteration": iteration, "active_sampling/selected_lambda": selected_lambda, "active_sampling/acquisition_score": score})

def log_policy_training(metrics: Dict[str, float], current_lambda: float, step: int):
    wandb.log({f"policy_{current_lambda}/reward": metrics["reward"], "step": step})

def log_metric_geodesic_training(metrics: Dict[str, float], step: int):
    wandb.log({"metric_geodesic/loss": metrics["loss"], "metric_geodesic/ot_cost": metrics["ot_cost"], "step": step})

def log_flow_training(metrics: Dict[str, float], step: int):
    wandb.log({"flow/loss": metrics["loss"], "step": step})
```

## Dependencies

### Core Requirements
```
torch>=1.9.0
wandb>=0.12.0
hydra-core>=1.1.0
pytest>=6.0.0
numpy>=1.21.0
gym>=0.21.0
```

## Common Workflows

### 1. Full HTI Training Run
1. Define initial `lambda_0`, `lambda_1` set and active sampling budget.
2. Configure all components (Policy, Metric/Geodesic, Active Sampler, Flow) via Hydra.
3. Launch the main training script.
   - **Initial Phase:** Train original policy NN at initial `lambda_0` and `lambda_1` & collect data as `(x, y)` pairs.
   - **Active Sampling Loop:** Script executes the iterative loop:
     - Trains/updates metric/geodesic model.
     - Selects `lambda^*`.
     - Trains original policy NN at `lambda^*`.
     - Collects data as `(x, y)*` pairs.
   - **Final Phase:** After loop finishes, trains the final Conditional Flow Matcher.
4. Monitor progress and results on W&B.

### 2. Inference/Interpolation
1. Load trained Conditional Flow Matcher (`v_\theta`).
2. Specify target condition `x` and hyperparameter `\lambda`.
3. Sample behaviors `\hat{y}` using the flow model's sampling method (ODE integration).
4. Analyze or deploy the generated behaviors/policies.
