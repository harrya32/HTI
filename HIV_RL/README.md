# Structured Treatment Interruption Strategy for HIV Using Reinforcement Learning

This project involves designing a reinforcement learning (RL) agent to optimize a structured treatment interruption strategy for an HIV-infected patient. The simulator class, `HIVPatient`, follows the Gymnasium interface and provides functions to simulate the patient's immune system, allowing the RL agent to make treatment decisions every 5 days over a maximum of 200 time steps. The goal is to maintain the patient's health by balancing drug administration with the natural immune response, while minimizing pharmaceutical side effects and avoiding drug resistance.

## Project Structure

- `HIVPatient.py`: Contains the simulator class for the HIV patient's immune system.
- `train_agent.py`: Script to train the RL agent using the `HIVPatient` simulator.
- `evaluate_agent.py`: Script to evaluate the performance of the trained RL agent.
- `saved_models/`: Directory to store the trained RL agent models.
- `README.md`: Project overview and instructions.

## Getting Started

### Prerequisites

To run this project, the following Python packages are required:

- gymnasium
- numpy
- torch (for deep learning models)
- matplotlib (for plotting results)

You can install these packages using pip:

```bash
pip install gymnasium numpy torch matplotlib
```

### Training the RL Agent

To train the RL agent, run the `train_agent.py` script. This script initializes the `HIVPatient` simulator, sets up the RL agent, and trains it over a series of episodes.

```bash
python train_agent.py
```

The training process will output the progress and save the trained model in the `saved_models/` directory.

### Evaluating the RL Agent

To evaluate the performance of the trained RL agent, run the `evaluate_agent.py` script. This script loads the trained model and tests it on the `HIVPatient` simulator, providing performance metrics and visualizing the results.

```bash
python evaluate_agent.py
```

### Project Files

- **HIVPatient.py**: Implements the `HIVPatient` class, simulating the HIV patient's immune system based on a system of deterministic non-linear equations.
- **train_agent.py**: Contains the training loop for the RL agent. This script initializes the environment and agent, and trains the agent using a selected RL algorithm.
- **evaluate_agent.py**: Evaluates the trained RL agent by simulating its performance on the `HIVPatient` environment. Outputs metrics and visualizations to assess the agent's effectiveness.
- **saved_models/**: Directory where trained models are saved for future use and evaluation.
- **README.md**: Provides an overview of the project, installation instructions, and guidance on training and evaluating the RL agent.

### Notes

- The simulator allows for domain randomization by setting `domain_randomization=True` in the `HIVPatient` constructor. This feature can be used to test the agent on a variety of patient profiles.
- The reward model encourages high values of HIV-specific cytotoxic cells (E) and low values of free virus particles (V), while penalizing drug prescriptions to promote a balanced treatment strategy.

## Conclusion

This project demonstrates the application of reinforcement learning to optimize structured treatment interruption strategies for HIV. By carefully balancing drug administration with the patient's natural immune response, the RL agent aims to maintain the patient's health and minimize side effects.

Feel free to contribute to this project by improving the RL algorithms, experimenting with different reward models, or enhancing the simulation environment.
