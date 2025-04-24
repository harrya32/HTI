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
├── policies/             # RL policy implementations
│   ├── networks/
│   ├── training/
│   └── evaluation/
├── flows/               # Normalizing flow models
│   ├── architectures/
│   ├── training/
│   └── interpolation/
└── utils/               # Shared utilities
```

### 2. Data Management
- Environment data loaded through standard interfaces
- Policy outputs collected and stored efficiently
- Metric learning/flow training data organized by environment

### 3. Model Development
- Standard policy architectures
- Conditional flow matching to train normalizing flow models
- Checkpoint management with W&B
- Clear version tracking

### 4. Training Pipeline
- Hydra for experiment configuration
- W&B for experiment tracking
  - Policy training metrics
  - Environment statistics
  - Metric learning progress
  - Flow training progress
  - Visualizations
- Early stopping and model selection
- Resource monitoring and management

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

### 3. Experiment Logging
```python
import wandb

def log_policy_training(metrics: Dict[str, float], step: int):
    """Log policy training metrics."""
    wandb.log({
        "policy/reward": metrics["reward"],
        "policy/value_loss": metrics["value_loss"],
        "policy/entropy": metrics["entropy"],
        "environment/state_stats": metrics["state_stats"],
        "step": step
    })

def log_flow_training(metrics: Dict[str, float], step: int):
    """Log flow training metrics."""
    wandb.log({
        "flow/loss": metrics["loss"],
        "flow/interpolation_quality": metrics["quality_metric"],
        "step": step
    })
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

### 1. Training RL Policies
1. Configure policy and environment
2. Initialize W&B run with hyperparameters
3. Train policy and collect metrics
4. Save policy checkpoints and behavior data

### 2. Riemann Metric Learning 
1. Load policy behavior data
2. Configure metric architecture
3. Train metrics to learn conditional behavior manifolds across input conditions
4. Evaluate metric quality

### 2. Training Flows
1. Load policy behavior data and learned metric
2. Configure flow architecture
3. Train flow to match estimated behaviour manifold
4. Evaluate interpolation quality

### 3. Analyzing Results
1. Generate interpolated policies
2. Evaluate in environments
3. Visualize interpolated policy behaviours