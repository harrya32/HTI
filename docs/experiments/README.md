# Experimental Plan

## Goal
To evaluate the effectiveness of the Hyperparameter Trajectory Inference (HTI) method, specifically assessing:
1.  **Interpolation Plausibility:** Does the surrogate model generate realistic and performant behaviors/policies for intermediate hyperparameter values?
2.  **Active Learning Efficiency:** Does the active sampling strategy reduce the number of required original policy trainings compared to uniform sampling?
3.  **Adaptability:** Can the system adapt in real-time to changing hyperparameter preferences at inference time?

## Core Experiments

### 1. RL with Dynamic Reward Trade-offs
*   **Concept:** Learn trajectories for RL policies where the hyperparameter `lambda` controls the trade-off between conflicting reward components (e.g., performance vs. safety, cost vs. efficacy).
*   **Environments:**
    *   `Sepsis`: Balance treatment effectiveness vs. medication side effects/cost.
    *   `HIV`: Balance viral load suppression vs. drug toxicity.
    *   `Glucose`: Balance blood glucose stability vs. insulin dosage amount/frequency.
*   **Hyperparameter (`lambda`):** Weighting factor in a composite reward function `R = lambda * R_1 + (1 - lambda) * R_2`.
*   **Evaluation:**
    *   Sample interpolated policies at various `lambda` values.
    *   Evaluate these policies in the environment and measure performance on *both* reward components (`R_1`, `R_2`).
    *   Visualize the Pareto front of policy performance across `lambda`.
    *   Compare generated policy performance to policies trained directly at intermediate `lambda` values (if feasible).
    *   Assess the smoothness of the transition in policy behavior as `lambda` changes.

### 2. Robustness to Input Noise
*   **Concept:** Learn trajectories for models (potentially supervised or RL) where `lambda` controls a parameter related to robustness, such as the noise level assumed during adversarial training or the strength of a regularization term.
*   **Environments/Tasks:**
    *   Use one of the RL environments (e.g., `Glucose`) and train policies with varying levels of observation noise during training (`lambda` controls noise variance).
    *   Alternatively, a supervised task where `lambda` controls the strength of adversarial noise during training.
*   **Hyperparameter (`lambda`):** Parameter controlling noise level or regularization strength.
*   **Evaluation:**
    *   Sample interpolated models/policies at various `lambda` values.
    *   Evaluate their performance under *different* noise levels at test time.
    *   Measure robustness metrics (e.g., performance drop as test noise increases).
    *   Compare HTI-generated robust models to those trained directly with corresponding `lambda` values.

## Evaluation Metrics

### Interpolation Quality
*   **Plausibility:** Qualitative assessment of generated behaviors/outputs.
*   **Performance:** Task-specific metrics (e.g., reward, accuracy, F1-score) of interpolated models/policies.
*   **Smoothness:** Measure of how rapidly behavior changes w.r.t. `lambda` (e.g., Lipschitz constant estimate along the trajectory).
*   **Geodesic Alignment:** How well the final flow trajectory aligns with the geodesics predicted by the metric/geodesic learner.

### Active Learning Efficiency
*   **Number of Samples:** Total number of original policy trainings (`lambda*` samples) required to reach a target interpolation quality.
*   **Comparison:** Compare performance vs. budget curve against uniform or random sampling of `lambda`.
*   **Acquisition Function Effectiveness:** Correlation between acquisition score and actual improvement in interpolation quality.

### Computational Cost
*   Training time for Metric/Geodesic learner.
*   Training time for Flow Matcher.
*   Total time for the active sampling loop (including original policy trainings).
*   Inference time for sampling from the flow model.

## Baselines
*   **Direct Training:** Train original NN policies/models directly at several intermediate `lambda` values (if computationally feasible).
*   **Linear Interpolation:** Simple linear interpolation in parameter space or behavior space between models trained at endpoint `lambda` values.
*   **Uniform/Random Sampling:** Run the active learning loop but select `lambda*` uniformly or randomly instead of using the acquisition function.

## Implementation Notes
*   Need standardized evaluation protocols for each environment/task.
*   Visualization tools for behavior spaces and interpolation trajectories will be crucial.
*   Careful tracking of computational resources used. 