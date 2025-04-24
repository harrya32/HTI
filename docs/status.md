# Project Status

## Overall Status
**Current Phase**: Initial Setup & Planning
**Last Updated**: [Current Date]

## Component Status

### 1. Infrastructure
| Component | Status | Notes |
|-----------|---------|-------|
| Project Structure | In Progress | Basic directory structure created |
| Documentation | In Progress | Core documentation files initialized |
| Development Environment | Not Started | - |
| Version Control | Complete | Git repository initialized |
| CI/CD | Not Started | - |

### 2. Environment Integration
| Component | Status | Notes |
|-----------|---------|-------|
| Glucose Environment | Medium | Need to make experiment specific alterations |
| Sepsis Environment | Medium | Need to make experiment specific alterations |
| HIV Environment | Medium | Need to make experiment specific alterations |
| Environment Tests | Not Started | - |

### 3. RL Policy Development (Original NN)
| Component | Status | Notes |
|-----------|---------|-------|
| Policy Architecture | Not Started | Need base implementation |
| Base Training Script | Not Started | Script to train policy at a given lambda |
| Evaluation System | Not Started | - |
| Behavior Collection | Not Started | Need format definition |

### 4. Metric & Geodesic Learning
| Component | Status | Notes |
|-----------|---------|-------|
| Joint Architecture | Not Started | Need implementation (NLOT/MFM inspired) |
| Training Pipeline | Not Started | Requires joint loss implementation |
| Geodesic Prediction Module | Not Started | Part of the joint model |

### 5. Active Sampling
| Component | Status | Notes |
|-----------|---------|-------|
| Uncertainty Estimator | Not Started | Heuristic based on geodesics |
| Acquisition Function | Not Started | Strategy to select lambda* |
| Sampler Loop Logic | Not Started | Orchestrates the iterative process |

### 6. Conditional Flow (Surrogate Model)
| Component | Status | Notes |
|-----------|---------|-------|
| Flow Architecture | Not Started | e.g., Real NVP, NSF |
| Training Pipeline | Not Started | Flow matching guided by geodesics |
| Inference/Sampling | Not Started | ODE integration |

### 7. Experiment Tracking (W&B)
| Component | Status | Notes |
|-----------|---------|-------|
| W&B Setup | Not Started | - |
| Active Sampling Logging | Not Started | Track lambda*, scores |
| Policy Training Logging | Not Started | Track metrics per lambda |
| Metric/Geodesic Logging | Not Started | Track joint loss, etc. |
| Flow Logging | Not Started | Track flow matching loss |
| Visualization | Not Started | - |

## Recent Updates

### [Current Date]
- Refined architecture based on detailed method explanation
- Updated documentation to include Active Sampling loop
- Clarified joint Metric/Geodesic learning
- Initialized project structure
- Set up basic version control

## Blockers & Issues
- Need to finalize choice between NLOT/MFM inspired metric learning approaches.

## Next Steps
1. Complete development environment setup.
2. Implement initial Metric & Geodesic Learner architecture.
3. Implement Active Sampling loop logic.
4. Begin Conditional Flow Matcher implementation.
5. Implement base Policy Network and training script (callable by active sampler).

## Notes
- Initial setup phase in progress
- Documentation being actively developed
- Core infrastructure being established 