# Development Tasks

## Overview
This document tracks the tasks required to implement and evaluate the Hyperparameter Trajectory Inference (HTI) method.

## Current Tasks (Organized by Priority)

### High Priority (Core Implementation)
- [ ] **Environment Wrappers:** Implement standardized wrappers for Glucose, Sepsis, HIV environments (`src/environments/`).
- [ ] **Original Policy NN:** Implement base policy network architecture (`src/policies/networks/`) and a script to train it for a *single, fixed* `lambda` (`src/policies/training/`).
- [ ] **Behavior Collection:** Implement mechanism to collect and store `(x, y)` pairs from trained policies (`src/policies/sampling/`).
- [ ] **Metric & Geodesic Learner:** Implement joint architecture `MetricGeodesicLearner` (`src/geometry/architectures/`) and its joint training loss (`src/geometry/training/`).
- [ ] **Active Sampler Logic:** Implement the core `ActiveSampler` class, including geodesic prediction calls, uncertainty estimation, and acquisition function (`src/active_sampling/`).
- [ ] **Conditional Flow Matcher:** Implement base flow architecture (`src/flows/architectures/`) and flow matching loss (`src/flows/training/`).
- [ ] **Overall Training Orchestrator:** Script to manage the full HTI loop (initial training, active sampling, final flow training) (`src/main.py`).
- [ ] **Configuration:** Set up Hydra configs for all components and the overall workflow (`configs/`).
- [ ] **W&B Integration:** Integrate W&B logging for all stages (`src/utils/logging.py`).

### Medium Priority (Refinement & Testing)
- [ ] **Unit Tests:** Add unit tests for key components (Metric/Geodesic loss, Flow loss, Sampler logic) (`tests/`).
- [ ] **Integration Tests:** Test the interaction between components (e.g., Active Sampler calling Policy training).
- [ ] **Geodesic Uncertainty:** Refine the heuristic for `estimate_geodesic_uncertainty`.
- [ ] **Acquisition Function:** Implement and test different `acquisition_function` strategies.
- [ ] **Flow Sampling:** Implement the ODE integration for sampling from the trained flow (`src/flows/sampling/`).
- [ ] **Basic Visualization:** Implement simple plots for training metrics and basic interpolation results (`src/visualization/`).
- [ ] **Experiment Setup:** Define specific Hydra config files for the Dynamic Reward and Robustness experiments (`configs/experiment/`).

### Low Priority (Advanced Features & Polish)
- [ ] **Advanced Visualizations:** Behavior space embeddings, interpolation trajectories, performance heatmaps.
- [ ] **Baseline Implementations:** Implement baselines (Direct Training, Linear Interpolation, Uniform Sampling) for comparison.
- [ ] **CI/CD:** Set up continuous integration pipeline.
- [ ] **API Documentation:** Generate API docs using Sphinx or similar.
- [ ] **Code Quality Tools:** Enforce linting, formatting, type checking via pre-commit hooks.

## Completed Tasks
- [x] Initial project structure setup.
- [x] Core documentation established (`architecture.md`, `technical.md`, `status.md`, `experiments/README.md`).
- [x] Environment code moved to `src/environments/`.

## Task Details (Examples)

### Implement Metric & Geodesic Learner
**Status**: Not Started
**Priority**: High
**Dependencies**: Base Policy NN, Behavior Collection
**Requirements**:
- Implement `MetricGeodesicLearner` nn.Module.
- Implement `_build_metric_network` and `_build_geodesic_network`.
- Implement `forward_metric` and `forward_geodesic`.
- Implement `compute_loss` incorporating OT cost, density bias, geodesic alignment.
- Adapt relevant logic from NLOT/MFM codebases.

### Implement Active Sampler Logic
**Status**: Not Started
**Priority**: High
**Dependencies**: Metric & Geodesic Learner
**Requirements**:
- Implement `estimate_geodesic_uncertainty` function.
- Implement `acquisition_function`.
- Implement `ActiveSampler` class with `select_next_lambda` method.
- Handle budget and sampled set management.

### Implement Overall Training Orchestrator
**Status**: Not Started
**Priority**: High
**Dependencies**: All core components (Policy Trainer, Metric/Geodesic Learner, Active Sampler, Flow Trainer)
**Requirements**:
- Script callable via Hydra.
- Manages the initial policy training phase.
- Manages the iterative active sampling loop (calling sampler, policy trainer, metric/geodesic trainer).
- Manages the final flow training phase.
- Handles data flow between components.

## Timeline (Rough Estimate)
- **Sprint 1:** Environment Wrappers, Base Policy NN & Trainer, Behavior Collection.
- **Sprint 2:** Metric & Geodesic Learner (Core), Active Sampler Logic (Core).
- **Sprint 3:** Conditional Flow Matcher (Core), Overall Training Orchestrator, Basic W&B Logging.
- **Sprint 4:** Unit/Integration Testing, Refine Uncertainty/Acquisition, Flow Sampling.
- **Sprint 5+:** Experiment Setup, Visualizations, Baselines, Polish.

## Requirements Tracking

### Core Requirements
1. Reproducible experiments
2. Comprehensive documentation (updated when necessary)
3. Robust model training
4. Clear evaluation metrics

### Documentation Requirements
1. Architecture documentation
2. Experiment documentation
3. Model documentation
4. Task tracking