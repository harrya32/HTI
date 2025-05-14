import os
import pickle
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig 
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from train_ot_scarvelis import Workspace
from lagrangian_ot import data, models, geometries, neuraldual, lagrangian_potentials, metrics
from typing import Optional, List, Tuple
from scipy.stats import vonmises, lognorm

plt.style.use('bmh')

def load_evaluation_data(eval_data_pkl_path: str) -> List[Tuple[float, jnp.ndarray]]:
    """
    Loads evaluation data from a specified .pkl file.
    The .pkl file is expected to contain a list of (time, samples) tuples,
    where samples are JAX/NumPy arrays: [(t0, samples_t0), (t1, samples_t1), ...].
    """
    if not os.path.exists(eval_data_pkl_path):
        raise FileNotFoundError(f"Evaluation data file not found: {eval_data_pkl_path}")

    with open(eval_data_pkl_path, 'rb') as f:
        loaded_object = pickle.load(f)
    
    evaluation_data_tuples = [(float(t), jnp.asarray(s)) for t, s in loaded_object]

    if not evaluation_data_tuples:
        raise ValueError("Loaded data from PKL is empty.")

    print(f"Shape of samples at t0: {evaluation_data_tuples[0][1].shape}")
    if len(evaluation_data_tuples) > 1:
        for i, (t, s) in enumerate(evaluation_data_tuples[1:]):
            print(f"Shape of samples at t_{i+1} (time={t:.4f}): {s.shape}")
        
    return evaluation_data_tuples


def _generate_semicircle_marginal_at_time_jax(
    num_points_per_condition: int,
    time: float,
    radius: float,
    angular_concentration: float,
    radial_std_dev: float
) -> jnp.ndarray:
    """
    Generates semicircle marginal data for all 4 conditions at a single continuous time point.
    Outputs a JAX array.
    """
    data_for_all_conditions = []
    for c in range(4):
        if c == 0:  
            target_angle = time * np.pi
            center_x = -1.0
        elif c == 1:  
            target_angle = -time * np.pi
            center_x = -1.0
        elif c == 2:  
            target_angle = np.pi - time * np.pi
            center_x = 1.0
        else:  
            target_angle = -np.pi + time * np.pi
            center_x = 1.0

        sampled_angles = vonmises.rvs(
            loc=target_angle,
            kappa=angular_concentration,
            size=num_points_per_condition
        )
        
        log_normal_mu = np.log(radius) 
        sampled_radii = lognorm.rvs(
            s=radial_std_dev, 
            loc=0, 
            scale=np.exp(log_normal_mu), 
            size=num_points_per_condition
        )
        
        x_coords = center_x + sampled_radii * np.cos(sampled_angles)
        y_coords = sampled_radii * np.sin(sampled_angles)
        
        points_xy = np.stack((x_coords, y_coords), axis=-1)
        condition_labels = np.full((num_points_per_condition, 1), float(c)) 
        
        condition_data = np.concatenate((points_xy, condition_labels), axis=1)
        data_for_all_conditions.append(condition_data)
        
    concatenated_data = np.concatenate(data_for_all_conditions, axis=0)
    return jnp.array(concatenated_data)


def generate_evaluation_data(
    generation_cfg: DictConfig,
    num_total_time_points_config: int, 
    samples_per_total_tp: int,
    evaluation_time_points_values_config: Optional[List[float] | ListConfig],
    key: jax.random.PRNGKey
) -> List[Tuple[float, jnp.ndarray]]:
    """
    Generates evaluation data. Currently supports "semicircle" type.
    Returns:
        - A list of (time, samples_at_time) tuples.
    """
    print(f"Starting data generation with config: {OmegaConf.to_yaml(generation_cfg)}")

    if generation_cfg.type != "semicircle":
        raise NotImplementedError(
            f"Data generation type '{generation_cfg.type}' is not supported. "
            "Currently, only 'semicircle' is implemented."
        )

    if samples_per_total_tp % 4 != 0:
        raise ValueError(
            "For 'semicircle' generation, 'num_samples_per_time_point' "
            f"({samples_per_total_tp}) must be divisible by 4 (for the 4 conditions)."
        )
    num_points_per_condition = samples_per_total_tp // 4

    times_to_generate = []
    if evaluation_time_points_values_config is not None:
        resolved_times_from_config = [float(t) for t in list(evaluation_time_points_values_config)]
        if len(resolved_times_from_config) == num_total_time_points_config:
            times_to_generate = resolved_times_from_config
            print(f"Using specified evaluation_time_points_values: {times_to_generate}")
        else:
            print(f"Warning: Length of 'evaluation_time_points_values' ({len(resolved_times_from_config)}) "
                  f"does not match 'num_evaluation_time_points' ({num_total_time_points_config}). "
                  "Will fall back to linspace or default for single point.")
    
    if not times_to_generate:
        if num_total_time_points_config == 1:
            times_to_generate = [0.5] # Default for a single point (e.g. mid-time)
            print(f"Generating for a single time point as per num_evaluation_time_points=1, using default t={times_to_generate[0]}")
        elif num_total_time_points_config > 1:
            times_to_generate = np.linspace(0, 1, num_total_time_points_config).tolist()
            print(f"Generating for {num_total_time_points_config} time points using np.linspace(0, 1, ...): {times_to_generate}")
        else: # num_total_time_points_config is 0 or invalid
             raise ValueError(f"Invalid num_evaluation_time_points: {num_total_time_points_config}. Must be >= 1.")


    semicircle_params = generation_cfg.semicircle_params
    print(f"Semicircle generation parameters: Radius={semicircle_params.radius}, "
          f"AngularConcentration={semicircle_params.angular_concentration}, "
          f"RadialStdDev={semicircle_params.radial_std_dev}")
    print(f"Number of points per condition per time point: {num_points_per_condition}")

    full_eval_samples_sequence = []

    evaluation_data_tuples = []

    for t_idx, current_time in enumerate(times_to_generate):
        print(f"Generating data for time point {t_idx + 1}/{len(times_to_generate)}: t = {current_time:.4f}")
        samples_at_t = _generate_semicircle_marginal_at_time_jax(
            num_points_per_condition=num_points_per_condition,
            time=float(current_time),
            radius=semicircle_params.radius,
            angular_concentration=semicircle_params.angular_concentration,
            radial_std_dev=semicircle_params.radial_std_dev
        )
        full_eval_samples_sequence.append(samples_at_t)
        evaluation_data_tuples.append((float(current_time), samples_at_t)) 

    if not evaluation_data_tuples: 
        raise ValueError("Data generation resulted in an empty sequence of (time, samples) tuples.")

    print(f"Shape of generated samples at t0 (time={evaluation_data_tuples[0][0]:.4f}): {evaluation_data_tuples[0][1].shape}")
    if len(evaluation_data_tuples) > 1:
        for i, (t, s) in enumerate(evaluation_data_tuples[1:]):
            print(f"Shape of generated samples at t_{i+1} (time={t:.4f}): {s.shape}")
        
    return evaluation_data_tuples 


@hydra.main(config_path=".", config_name="evaluate_config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    print("Starting evaluation script...")
    print(f"Full evaluation configuration:\n{OmegaConf.to_yaml(cfg)}")

    if Workspace is None:
        print("Workspace class not imported. Cannot proceed. Please check imports and Python path.")
        return

    prng_key = jax.random.PRNGKey(cfg.evaluation_seed) 

    model_path = cfg.model_path
    if not os.path.isabs(model_path) and hydra.utils.get_original_cwd() != os.getcwd():
        model_path = os.path.join(hydra.utils.get_original_cwd(), model_path)
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        with open(model_path, 'rb') as f:
            workspace = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading pickled workspace: {e}")
        return


    if cfg.data_source == "pkl":
        print(f"Preparing evaluation data by loading from PKL file: {cfg.eval_data_pkl_path}")
        try:
            eval_data_path = cfg.eval_data_pkl_path
            if not os.path.isabs(eval_data_path) and hydra.utils.get_original_cwd() != os.getcwd():
                eval_data_path = os.path.join(hydra.utils.get_original_cwd(), eval_data_path)

            evaluation_data_tuples = load_evaluation_data(
                eval_data_pkl_path=eval_data_path
            )
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading evaluation data: {e}")
            return
            
    elif cfg.data_source == "generate":
        evaluation_data_tuples = generate_evaluation_data(
            generation_cfg=cfg.data_generation_config,
            num_total_time_points_config=cfg.num_evaluation_time_points,
            samples_per_total_tp=cfg.num_samples_per_time_point,
            evaluation_time_points_values_config=cfg.get('evaluation_time_points_values', None),
            key=prng_key
        )
    
    num_actual_time_points = len(evaluation_data_tuples) 
    print(f"Successfully obtained data with {num_actual_time_points} time points.")
        
    if cfg.num_evaluation_time_points != num_actual_time_points:
        print(f"Warning: Configured 'num_evaluation_time_points' ({cfg.num_evaluation_time_points}) "
              f"does not match the actual number of time points in the data ({num_actual_time_points}). "
              f"The actual number of time points ({num_actual_time_points}) and derived/configured time values will be used.")


    print("\nStarting evaluation using workspace.evaluate_marginals...")
    try:
        initial_samples_at_t0 = evaluation_data_tuples[0][1] 
        evaluation_points_for_method = evaluation_data_tuples[1:]

        metrics_log = workspace.evaluate_marginals(
            initial_samples_at_t0=initial_samples_at_t0, 
            evaluation_points=evaluation_points_for_method,
            plot_results=cfg.plotting.plot_results, 
            verbose=cfg.verbose_evaluation 
        )
        
        print("\n--- Evaluation Metrics ---")
        for key, value in metrics_log.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nEvaluation script finished.")

if __name__ == "__main__":
    main()
