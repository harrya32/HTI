# Hyperparameter Trajectory Inference (HTI)

This project implements the code for the NeurIPS submission "Hyperparameter Trajectory Inference" (HTI). HTI is a problem for learning how the conditional output distribution of a neural network changes as you vary a hyperparameter.

## Motivation

Many machine learning models have hyperparameters that significantly affect their behaviour (e.g., reward weighting in RL, perturbation strength in adversarial training). Traditionally, these are tuned once and fixed at training time. HTI aims to understand the *trajectory* of the model's conditional output distribution as a hyperparameter changes. This allows for:

*   Dynamically adjusting hyperparameters at inference time based on desired behaviour.
*   Developing deeper understanding of model behaviour at different hyperparameter settings, without retraining.

Our method uses guided conditional flow matching, matching geodesic velocities estimated via an actively learned conditional Riemannian metric approach, to address the HTI problem and model these trajectories efficiently and accurately.

## TODO
- Test Gaussian circle data, with inverse potential (it works on uniform data vs. just NLOT metric learning, and purely euclidean cost (need to test other ablation of euclidean, w potential term))
- Define meaningful cancer reward fn., and the hyperparam of interest
- Produce a few cancer datasets, and train NLOT on them to see if it can learn
- Write down/check how our acq. fn. is working/averaging across pairs/conditions
- Adversarial robustness in time series forecasting example, ideally on MIMIC data
- HIV/diabetes/sepsis sims are secondary options which need varying adaptations to get into our setting. Start with HIV, doing similar reward shaping to cancer example.