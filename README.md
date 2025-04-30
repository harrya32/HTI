# Hyperparameter Trajectory Inference (HTI)

This project implements the code for the NeurIPS submission "Hyperparameter Trajectory Inference" (HTI). HTI is a problem for learning how the conditional output distribution of a neural network changes as you vary a hyperparameter.

## Motivation

Many machine learning models have hyperparameters that significantly affect their behaviour (e.g., reward weighting in RL, perturbation strength in adversarial training). Traditionally, these are tuned once and fixed at training time. HTI aims to understand the *trajectory* of the model's conditional output distribution as a hyperparameter changes. This allows for:

*   Dynamically adjusting hyperparameters at inference time based on desired behaviour.
*   Developing deeper understanding of model behaviour at different hyperparameter settings, without retraining.

Our method uses guided conditional flow matching, matching geodesic velocities estimated via an actively learned conditional Riemannian metric approach, to address the HTI problem and model these trajectories efficiently and accurately.

## TODO
- Diabetes simulator, get RL working, examine/plot inputs/outputs
- Write down/check how our acq. fn. is working/averaging across pairs/conditions
- HIV/sepsis sims