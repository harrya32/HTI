# Hyperparameter Trajectory Inference (HTI)

This project implements the code for the NeurIPS submission "Hyperparameter Trajectory Inference" (HTI). HTI is a problem for learning how the conditional output distribution of a neural network changes as you vary a hyperparameter.

## Motivation

Many machine learning models have hyperparameters that significantly affect their behaviour (e.g., reward weighting in RL, perturbation strength in adversarial training). Traditionally, these are tuned once and fixed at training time. HTI aims to understand the *trajectory* of the model's conditional output distribution as a hyperparameter changes. This allows for:

*   Dynamically adjusting hyperparameters at inference time based on desired behaviour.
*   Developing deeper understanding of model behaviour at different hyperparameter settings, without retraining.

Our method uses guided conditional flow matching, matching geodesic velocities estimated via an actively learned conditional Riemannian metric approach, to address the HTI problem and model these trajectories efficiently and accurately.

## TODO
- Produce a few cancer datasets for the different n_k weights, and train NLOT on them to see if it can learn. Run on larger lambdas ~ 5.
- Devise a metric for success/behaviour change here (how do we show the surrogate model performs well, outside of wass. distance from the reals) (just check nk avg. nk change per episode, across different surrogate models. Need a way to use surrogate model directly in RL env.)
- Write down/check how our acq. fn. is working/averaging across pairs/conditions
- Plots for synthetic experiment, and write up of it
- Adversarial robustness in time series forecasting example, ideally on MIMIC data
- HIV/diabetes/sepsis sims are secondary options which need varying adaptations to get into our setting. Start with HIV, doing similar reward shaping to cancer example.