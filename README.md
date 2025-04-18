# Hyperparameter Trajectory Inference (HTI)

This project implements the code for the NeurIPS submission "Hyperparameter Trajectory Inference" (HTI). HTI is a problem for learning how the conditional output distribution of a neural network changes as you vary a hyperparameter.

## Motivation

Many machine learning models have hyperparameters that significantly affect their behavior (e.g., discount factor in RL). Traditionally, these are tuned once and fixed at training time. HTI aims to understand the *trajectory* of the model's output distribution as a hyperparameter changes. This allows for:

*   Examining model behavior at different hyperparameter settings, without retraining.
*   Dynamically adjusting hyperparameters at inference time based on desired behaviour.

Our method uses conditional flow matching, guided by actively learned Riemannian geometry, to address the HTI problem and model these trajectories efficiently and accurately.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace with your repo URL
    cd hyperparam-trajectory-inference
    ```

2.  **Set up the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate hyperparam-trajectory-inference
    ```
    Alternatively, use the Makefile:
    ```bash
    make conda-setup
    conda activate hyperparam-trajectory-inference
    ```

## Usage

(Add details on how to run experiments, configure settings, etc. later)

**Example "Hello World" Run:**

This command runs the main script with the default configuration, which performs a basic initialization and a dummy forward pass.

```bash
make run
# Or directly:
# python -m hti.main --config-name default
```

Check the output for basic logging and confirmation that the components load correctly.

## Project Structure

```
/hyperparam-trajectory-inference
├── .gitignore
├── environment.yml         # conda environment spec
├── README.md               # project overview, install, usage, structure
├── LICENSE
├── Makefile                # tasks: conda-setup, test, run, clean
├── Dockerfile
├── configs/                # Hydra/YAML configs (default, geometry, flow, active sampling)
├── src/
│   └── hti/                # Main source code package
│       ├── __init__.py
│       ├── data/           # Data loading and processing (dataset.py)
│       ├── geometry/       # Riemannian metric and geodesic learning (metric_model.py, geodesic_model.py)
│       ├── flow/           # Conditional flow matching model (flow_model.py)
│       ├── active/         # Active sampling logic (acquisition.py)
│       ├── experiments/    # Experiment orchestration (run_experiment.py)
│       ├── utils/          # Utility functions (logger.py)
│       └── main.py         # Main entry point using Hydra
└── tests/                  # PyTest tests (test_geometry.py, test_flow.py)
```

*   `configs/`: Contains Hydra configuration files (`.yaml`) defining experiment parameters, model architectures, and other settings.
*   `src/hti/`: The core Python package.
    *   `data/`: Handles dataset loading and preprocessing.
    *   `geometry/`: Implements the conditional Riemannian metric learning (`metric_model.py`) and the geodesic prediction based on the learned metric (`geodesic_model.py`).
    *   `flow/`: Contains the conditional flow matching model (`flow_model.py`) responsible for learning the velocity field.
    *   `active/`: Implements the active learning strategy (`acquisition.py`) for selecting the next hyperparameter to sample.
    *   `experiments/`: Holds the main experiment logic (`run_experiment.py`), orchestrating the training loops, active learning steps, and evaluation.
    *   `utils/`: Common utilities like logging setup.
    *   `main.py`: The main executable script that uses Hydra to parse configurations and launch the experiment runner.
*   `tests/`: Unit and integration tests using PyTest.
*   `environment.yml`: Specifies the Conda environment with all necessary dependencies.
*   `Makefile`: Provides convenient commands (`make conda-setup`, `make test`, `make run`, `make clean`).
*   `Dockerfile`: For containerizing the application.

## Contributing

(Add contribution guidelines if applicable)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

##TODO
- Get Jax working for GPU, so that NLOT is not so slow
- Test combining both metric learning methods (changing the regularisation in NLOT to bias it towards the LAND/RBF MFM metric)
- Add conditioning, and test on toy conditional distributions