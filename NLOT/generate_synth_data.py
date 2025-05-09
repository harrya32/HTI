import torch
import os
from torchcfm.optimal_transport import OTPlanSampler
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import vonmises, lognorm
import pickle
import jax.numpy as jnp
import seaborn as sns

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def generate_sphere_data(num_points: int = 5000):
    time_0_samples = np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )
    time_2_samples = 1 - np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )

    x0_ot, x1_ot = OTPlanSampler(method="exact").sample_plan(
        torch.tensor(time_0_samples).unsqueeze(0),
        torch.tensor(time_2_samples).unsqueeze(0),
        replace=False,
    )
    x0_ot, x1_ot = x0_ot.numpy().flatten(), x1_ot.numpy().flatten()
    time_1_samples = (x0_ot + x1_ot) / 2

    phi_0 = np.pi * time_0_samples
    phi_1 = np.pi * time_1_samples
    phi_2 = np.pi * time_2_samples

    theta_0 = 2 * np.pi * np.random.rand(num_points)
    theta_1 = 2 * np.pi * np.random.rand(num_points)
    theta_2 = 2 * np.pi * np.random.rand(num_points)

    x_0 = np.sin(phi_0) * np.cos(theta_0)
    y_0 = np.sin(phi_0) * np.sin(theta_0)
    z_0 = np.cos(phi_0)
    x_1 = np.sin(phi_1) * np.cos(theta_1)
    y_1 = np.sin(phi_1) * np.sin(theta_1)
    z_1 = np.cos(phi_1)
    x_2 = np.sin(phi_2) * np.cos(theta_2)
    y_2 = np.sin(phi_2) * np.sin(theta_2)
    z_2 = np.cos(phi_2)

    # Combining points and creating labels
    points_0 = np.column_stack((x_0, y_0, z_0))
    points_1 = np.column_stack((x_1, y_1, z_1))
    points_2 = np.column_stack((x_2, y_2, z_2))

    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)

    unique_labels = np.unique(labels)
    return points, labels, unique_labels

def generate_arch_data(num_points: int = 5000):
    """Generate synthetic data for the arch dataset."""

    time_0_samples = np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )
    time_2_samples = 1 - np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )

    x0_ot, x1_ot = OTPlanSampler(method="exact").sample_plan(
        torch.tensor(time_0_samples).unsqueeze(0),
        torch.tensor(time_2_samples).unsqueeze(0),
        replace=False,
    )
    x0_ot, x1_ot = x0_ot.numpy().flatten(), x1_ot.numpy().flatten()
    time_1_samples = (x0_ot + x1_ot) / 2

    # Mapping to a semi-circle
    angles_0 = np.pi * (1 - time_0_samples)
    angles_1 = np.pi * (1 - time_1_samples)
    angles_2 = np.pi * (1 - time_2_samples)

    x_0 = np.cos(angles_0)
    y_0 = np.sin(angles_0)
    x_1 = np.cos(angles_1)
    y_1 = np.sin(angles_1)
    x_2 = np.cos(angles_2)
    y_2 = np.sin(angles_2)

    # Adding Gaussian noise
    radius_noise_0 = np.random.normal(0, 0.1, size=num_points)
    radius_noise_1 = np.random.normal(0, 0.1, size=num_points)
    radius_noise_2 = np.random.normal(0, 0.1, size=num_points)

    x_0 = (1 + radius_noise_0) * x_0
    y_0 = (1 + radius_noise_0) * y_0
    x_1 = (1 + radius_noise_1) * x_1
    y_1 = (1 + radius_noise_1) * y_1
    x_2 = (1 + radius_noise_2) * x_2
    y_2 = (1 + radius_noise_2) * y_2

    # Combining points and creating labels
    points_0 = np.column_stack((x_0, y_0))
    points_1 = np.column_stack((x_1, y_1))
    points_2 = np.column_stack((x_2, y_2))

    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)

    # Returning the dataset, labels, and unique labels
    unique_labels = np.unique(labels)
    return points, labels, unique_labels

def generate_conditional_gaussian_data(num_points_per_condition: int, variance: float = 0.1):
    """
    Generates a 2D conditional dataset with 3 time points and 4 conditions.
    t=0: N((0,0), variance*I) for all conditions.
    t=1: N(mean_c_t1, variance*I) where mean_c_t1 is one of {(1,1), (1,-1), (-1,1), (-1,-1)}.
    t=2: N(mean_c_t2, variance*I) where mean_c_t2 is one of {(2,2), (2,-2), (-2,2), (-2,-2)}.

    Returns:
        torch.Tensor: Data tensor of shape (3, num_points_total, 3),
                      where the dimensions are (time, sample, [x, y, condition]).
    """
    num_conditions = 4
    means_t0 = torch.zeros((num_conditions, 2), device=DEVICE)
    means_t1 = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float32, device=DEVICE)
    means_t2 = means_t1 * 2 # Double the means for t=2
    covariance = torch.eye(2, device=DEVICE) * variance
    sqrt_covariance = torch.linalg.cholesky(covariance) # For sampling: mean + sqrt_cov @ randn

    all_data_t0 = []
    all_data_t1 = []
    all_data_t2 = [] # Add list for t=2 data

    for c in range(num_conditions):
        # Sample t=0 data: N(0, variance*I)
        z0 = torch.randn(num_points_per_condition, 2, device=DEVICE)
        samples_t0 = means_t0[c] + z0 @ sqrt_covariance.T

        # Sample t=1 data: N(mean_c_t1, variance*I)
        z1 = torch.randn(num_points_per_condition, 2, device=DEVICE)
        samples_t1 = means_t1[c] + z1 @ sqrt_covariance.T

        # Sample t=2 data: N(mean_c_t2, variance*I)
        z2 = torch.randn(num_points_per_condition, 2, device=DEVICE)
        samples_t2 = means_t2[c] + z2 @ sqrt_covariance.T

        # Create condition vector
        condition_vec = torch.full((num_points_per_condition, 1), float(c), device=DEVICE)

        # Append [x, y, condition]
        all_data_t0.append(torch.cat((samples_t0, condition_vec), dim=1))
        all_data_t1.append(torch.cat((samples_t1, condition_vec), dim=1))
        all_data_t2.append(torch.cat((samples_t2, condition_vec), dim=1)) # Append t=2 data

    # Concatenate across conditions
    final_data_t0 = torch.cat(all_data_t0, dim=0)
    final_data_t1 = torch.cat(all_data_t1, dim=0)
    final_data_t2 = torch.cat(all_data_t2, dim=0) # Concatenate t=2 data

    # Stack time points
    final_data = torch.stack([final_data_t0, final_data_t1, final_data_t2], dim=0) # Stack all three time points

    return final_data

def generate_complex_conditional_gaussian_data(num_points_per_condition: int, variance: float = 0.1):
    """
    Generates a 2D conditional dataset with 5 time points and 4 conditions.
    Each condition follows a specific path defined by Gaussian means at each time point.
    c=0: (0,0), (1,-1), (0,-2), (1,-3), (0,-4)
    c=1: (0,0), (-1,1), (0,-2), (-1,-3), (0,-4)
    c=2: (0,0), (-1,1), (-2,2), (-3,3), (-4,4)
    c=3: (0,0), (1,1), (2,0), (3,1), (4,0)

    Returns:
        torch.Tensor: Data tensor of shape (5, num_points_total, 3),
                      where the dimensions are (time, sample, [x, y, condition]).
    """
    num_conditions = 4
    num_time_points = 5

    # Define means for each condition at each time point
    # Shape: (num_conditions, num_time_points, 2)
    means = torch.zeros((num_conditions, num_time_points, 2), dtype=torch.float32, device=DEVICE)

    # Condition 0: (0,0), (1,-1), (0,-2), (-1,-3), (0,-4)
    means[0, 1] = torch.tensor([1, -1])
    means[0, 2] = torch.tensor([0, -2])
    means[0, 3] = torch.tensor([-1, -3])
    means[0, 4] = torch.tensor([0, -4])

    # Condition 1: (0,0), (-1,-1), (0,-2), (1,-3), (0,-4)
    means[1, 1] = torch.tensor([-1, -1])
    means[1, 2] = torch.tensor([0, -2])
    means[1, 3] = torch.tensor([1, -3])
    means[1, 4] = torch.tensor([0, -4])

    # Condition 2: (0,0), (-1,1), (-2,2), (-3,3), (-4,4)
    means[2, 1] = torch.tensor([-1, 1])
    means[2, 2] = torch.tensor([-2, 2])
    means[2, 3] = torch.tensor([-3, 3])
    means[2, 4] = torch.tensor([-4, 4])

    # Condition 3: (0,0), (1,1), (2,0), (3,1), (4,0)
    means[3, 1] = torch.tensor([1, 1])
    means[3, 2] = torch.tensor([2, 0])
    means[3, 3] = torch.tensor([3, 1])
    means[3, 4] = torch.tensor([4, 0])

    covariance = torch.eye(2, device=DEVICE) * variance
    sqrt_covariance = torch.linalg.cholesky(covariance) # For sampling: mean + sqrt_cov @ randn

    all_data_t = [[] for _ in range(num_time_points)] # List of lists for each time point

    for c in range(num_conditions):
        # Create condition vector once per condition
        condition_vec = torch.full((num_points_per_condition, 1), float(c), device=DEVICE)

        for t in range(num_time_points):
            # Sample data for time t: N(mean_c_t, variance*I)
            z = torch.randn(num_points_per_condition, 2, device=DEVICE)
            samples_t = means[c, t] + z @ sqrt_covariance.T

            # Append [x, y, condition]
            all_data_t[t].append(torch.cat((samples_t, condition_vec), dim=1))

    # Concatenate across conditions for each time point
    final_data_t = [torch.cat(all_data_t[t], dim=0) for t in range(num_time_points)]

    # Stack time points
    final_data = torch.stack(final_data_t, dim=0) # Stack all time points

    return final_data

def generate_velocity_conditioned_data(num_conditions: int, num_points_per_condition: int, timesteps: list[float], variance: float = 0.05, base_speed = 2.0):
    """
    Generates 2D data where trajectories move along the x-axis at different speeds.
    - Samples `num_conditions` distinct speed conditions c ~ Uniform(0, 1).
    - For each condition, generates `num_points_per_condition` points.
    - Mean at time t for condition c: (c * t, 0)
    - Data at time t: N((c*t, 0), variance*I)

    Args:
        num_conditions (int): Number of distinct speed conditions to sample.
        num_points_per_condition (int): Number of points per condition.
        timesteps (list[float]): List of time values to sample data at.
        variance (float): Variance of the Gaussian noise.

    Returns:
        torch.Tensor: Data tensor of shape (len(timesteps), num_conditions * num_points_per_condition, 3),
                      where the dimensions are (time, sample, [x, y, condition_c]).
    """
    total_points = num_conditions * num_points_per_condition
    num_timesteps = len(timesteps)
    timesteps_tensor = torch.tensor(timesteps, dtype=torch.float32, device=DEVICE).view(-1, 1, 1) # Shape (T, 1, 1)

    distinct_conditions_c = torch.rand(1, num_conditions, 1, device=DEVICE)
    conditions_c = distinct_conditions_c.repeat_interleave(num_points_per_condition, dim=1) # Shape (1, N_total, 1)

    # Calculate means for all points at all timesteps
    # Mean_x = c * t
    # Mean_y = 0
    means_x = conditions_c * timesteps_tensor * base_speed
    means_y = torch.zeros_like(means_x)       
    means = torch.cat((means_x, means_y), dim=2)

    covariance = torch.eye(2, device=DEVICE) * variance
    sqrt_covariance = torch.linalg.cholesky(covariance)

    z = torch.randn(num_timesteps, total_points, 2, device=DEVICE)
    samples = means + torch.matmul(z, sqrt_covariance.T)
    conditions_expanded = conditions_c.expand(num_timesteps, -1, -1)
    final_data = torch.cat((samples, conditions_expanded), dim=2)

    return final_data

def generate_rotation_conditioned_data(num_conditions: int, num_points_per_condition: int, timesteps: list[float],
                                       base_radius_speed: float = 1.0,
                                       base_angular_speed: float = math.pi,
                                       condition_range: tuple[float, float] = (-math.pi / 2, math.pi / 2),
                                       variance: float = 0.05):
    """
    Generates 2D data following spiral paths with condition-dependent rotation speed.
    - Samples `num_conditions` distinct angular speed offsets c ~ Uniform(condition_range).
    - For each condition, generates `num_points_per_condition` points.
    - Base path is a spiral: radius = base_radius_speed * t, angle = base_angular_speed * t.
    - Total angular speed = base_angular_speed + c.
    - Mean at time t: (r(t)*cos(theta_total(t)), r(t)*sin(theta_total(t)))
    - Data at time t: N(mean(t), variance*I)

    Args:
        num_conditions (int): Number of distinct angular speed conditions.
        num_points_per_condition (int): Number of points per condition.
        timesteps (list[float]): List of time values to sample data at.
        base_radius_speed (float): Speed at which the spiral radius increases.
        base_angular_speed (float): Base speed of rotation (radians per unit time).
        condition_range (tuple[float, float]): Min and max for the uniform condition sampling.
        variance (float): Variance of the Gaussian noise.

    Returns:
        torch.Tensor: Data tensor of shape (len(timesteps), num_conditions * num_points_per_condition, 3),
                      where the dimensions are (time, sample, [x, y, condition_c]).
    """
    total_points = num_conditions * num_points_per_condition
    num_timesteps = len(timesteps)
    timesteps_tensor = torch.tensor(timesteps, dtype=torch.float32, device=DEVICE).view(-1, 1, 1)
    c_min, c_max = condition_range
    distinct_conditions_c = torch.rand(1, num_conditions, 1, device=DEVICE) * (c_max - c_min) + c_min
    conditions_c = distinct_conditions_c.repeat_interleave(num_points_per_condition, dim=1) 

    # Calculate radius and total angle for all points at all timesteps
    radius = base_radius_speed * timesteps_tensor
    total_angular_speed = base_angular_speed + conditions_c
    total_angle = total_angular_speed * timesteps_tensor

    means_x = radius * torch.cos(total_angle)
    means_y = radius * torch.sin(total_angle)
    means = torch.cat((means_x, means_y), dim=2)

    covariance = torch.eye(2, device=DEVICE) * variance
    sqrt_covariance = torch.linalg.cholesky(covariance) 

    z = torch.randn(num_timesteps, total_points, 2, device=DEVICE) 
    samples = means + torch.matmul(z, sqrt_covariance.T)

    conditions_expanded = conditions_c.expand(num_timesteps, -1, -1)
    final_data = torch.cat((samples, conditions_expanded), dim=2) 
    return final_data

def generate_warped_circle_data(num_points_per_time: int,
                                num_time_points: int = 6,
                                radius: float = 1.0,
                                angular_concentration: float = 15.0,
                                radial_std_dev: float = 0.1,
                                device: torch.device = DEVICE):
    """
    Generates 2D data distributed around a circle at multiple time points,
    with distributions "warped" by the circular geometry using von Mises (angle)
    and Log-Normal (radius) distributions.

    Args:
        num_points_per_time (int): Number of data points for each time step.
        num_time_points (int): Number of discrete time steps.
        radius (float): The target mean radius of the circle.
        angular_concentration (float): Kappa parameter for the von Mises distribution.
                                      Higher values mean tighter angular clustering.
        radial_std_dev (float): Standard deviation of the underlying normal distribution
                                for the Log-Normal radius samples. Controls radial spread.
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (num_time_points, num_points_per_time, 2).
    """
    all_points_t = []
    target_angles = np.linspace(0, 2 * np.pi, num_time_points, endpoint=False)
    log_normal_mu = np.log(radius)
    log_normal_sigma = radial_std_dev

    for t in range(num_time_points):
        target_angle = target_angles[t]
        sampled_angles = vonmises.rvs(loc=target_angle, kappa=angular_concentration, size=num_points_per_time)
        sampled_radii = lognorm.rvs(s=log_normal_sigma, loc=0, scale=np.exp(log_normal_mu), size=num_points_per_time)
        x = sampled_radii * np.cos(sampled_angles)
        y = sampled_radii * np.sin(sampled_angles)
        points_t = np.stack((x, y), axis=-1)
        all_points_t.append(points_t)

    final_data = torch.tensor(np.array(all_points_t), dtype=torch.float32, device=device)

    return final_data

def generate_warped_circle_data_uniform(num_points_per_time: int,
                                        num_time_points: int = 6,
                                        radius: float = 1.0,
                                        angle_range: float = np.pi / 4, 
                                        device: torch.device = DEVICE):
    """
    Generates 2D data distributed uniformly around a circle at multiple time points,
    with the distribution rotated along the circle based on the time point.

    Args:
        num_points_per_time (int): Number of data points for each time step.
        num_time_points (int): Number of discrete time steps.
        radius (float): The target mean radius of the circle.
        angle_range (float): Spread of the distribution on the circumference per time point (in radians).
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (num_time_points, num_points_per_time, 2).
    """
    all_points_t = []
    for t in range(num_time_points):
        center_angle = 2 * np.pi * (t / num_time_points) 

        sampled_angles = np.random.uniform(center_angle - angle_range / 2,
                                            center_angle + angle_range / 2,
                                            size=num_points_per_time)

        sampled_radii = np.random.uniform(radius * 0.9, radius * 1.1, size=num_points_per_time)
        x = sampled_radii * np.cos(sampled_angles)
        y = sampled_radii * np.sin(sampled_angles)
        points_t = np.stack((x, y), axis=-1)
        all_points_t.append(points_t)

    final_data = torch.tensor(np.array(all_points_t), dtype=torch.float32, device=device)

    return final_data

def generate_conditional_circles_data_normal(num_points_per_condition: int,
                                             num_time_points: int = 6,
                                             radius: float = 1.0,
                                             angular_concentration: float = 15.0,
                                             radial_std_dev: float = 0.1,
                                             device: torch.device = DEVICE):
    """
    Generates 4 conditions of warped circle data with independent sampling for each condition.
    Each condition is transformed (time-reversed, x-flipped) to create variations.

    Args:
        num_points_per_condition (int): Number of data points per condition per time step.
        num_time_points (int): Number of discrete time steps.
        radius (float): Mean radius of the circle.
        angular_concentration (float): Concentration parameter for the von Mises distribution.
        radial_std_dev (float): Standard deviation for the radial distribution.
        device (torch.device): Device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (num_time_points + 1, num_points_per_condition * 4, 3).
    """
    all_conditions_data = []
    for _ in range(4):
        condition_data = generate_warped_circle_data(
            num_points_per_time=num_points_per_condition,
            num_time_points=num_time_points,
            radius=radius,
            angular_concentration=angular_concentration,
            radial_std_dev=radial_std_dev,
            device=device
        )
        all_conditions_data.append(torch.cat((condition_data, condition_data[0:1]), dim=0))

    cond0_data = all_conditions_data[0].clone()
    cond0_data[..., 0] -= radius  
    
    cond1_data = all_conditions_data[1].clone()
    cond1_data[..., 0] -= radius  
    cond1_data = torch.flip(cond1_data, dims=[0])  # Time-reverse condition 1
    
    cond2_data = all_conditions_data[2].clone()
    cond2_data[..., 0] -= radius 
    cond2_data[..., 0] *= -1  # Flip x-coordinates for condition 2
    
    cond3_data = all_conditions_data[3].clone()
    cond3_data[..., 0] -= radius  
    cond3_data[..., 0] *= -1  # Flip x-coordinates for condition 3
    cond3_data = torch.flip(cond3_data, dims=[0])  # Time-reverse condition 3

    all_data_t = []
    for t in range(num_time_points + 1):
        data_t_conds = []
        for c, cond_data in enumerate([cond0_data, cond1_data, cond2_data, cond3_data]):
            coords_t = cond_data[t]
            labels_t = torch.full((num_points_per_condition, 1), c, device=device)
            data_t_conds.append(torch.cat((coords_t, labels_t), dim=1))
        all_data_t.append(torch.cat(data_t_conds, dim=0))

    return torch.stack(all_data_t, dim=0)

def generate_conditional_circle_at_time(num_points: int, time: float, radius: float = 1.0, angular_concentration: float = 5.0, radial_std_dev: float = 0.08):
    """
    Generate the conditional circle data for Condition 0 at an arbitrary time point.

    Args:
        num_points (int): Number of points to generate.
        time (float): Time point (normalized between 0 and 1).
        radius (float): Radius of the circle.
        angular_concentration (float): Concentration parameter for the von Mises distribution.
        radial_std_dev (float): Standard deviation for the radial distribution.

    Returns:
        torch.Tensor: Generated data points of shape (num_points, 2).
    """
    # Step 1: Generate base warped circle data at t=0
    base_data = generate_warped_circle_data(
        num_points_per_time=num_points,
        num_time_points=1,  # Only need one time point
        radius=radius,
        angular_concentration=angular_concentration,
        radial_std_dev=radial_std_dev,
        device=DEVICE
    )[0]  # Extract the first (and only) time point

    # Step 2: Convert time to angle
    theta = 2 * math.pi * time  # Map time to angle

    # Step 3: Shift points along the circle
    x, y = base_data[:, 0], base_data[:, 1]
    r = torch.sqrt(x**2 + y**2)  # Compute radius
    phi = torch.atan2(y, x)  # Compute angle
    phi_shifted = phi + theta  # Shift angle

    # Convert back to Cartesian coordinates
    x_shifted = r * torch.cos(phi_shifted)
    y_shifted = r * torch.sin(phi_shifted)

    # Combine shifted points
    shifted_data = torch.stack((x_shifted, y_shifted), dim=1)

    return shifted_data

def generate_conditional_circle_marginal_normal(num_points_per_condition: int,
                                         time: float,
                                         radius: float = 1.0,
                                         angular_concentration: float = 15.0,
                                         radial_std_dev: float = 0.1,
                                         device: torch.device = DEVICE):
    """
    Generate a marginal distribution for all 4 conditions at a single continuous time.
    Each condition uses independently sampled warped circle data.

    Args:
        num_points_per_condition (int): Number of samples per condition.
        time (float): Continuous time input (normalized between 0 and 1).
        radius (float): Radius of the circle.
        angular_concentration (float): Concentration parameter for the von Mises distribution.
        radial_std_dev (float): Standard deviation for the radial distribution.
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (4 * num_points_per_condition, 3),
                      where the last dimension is [x, y, condition].
    """
    def transform_points(points_xy, t_effective, current_radius):
        theta_shift = 2 * math.pi * t_effective  

        x_orig, y_orig = points_xy[:, 0], points_xy[:, 1]
        r_orig = torch.sqrt(x_orig**2 + y_orig**2)  
        phi_orig = torch.atan2(y_orig, x_orig)  

        phi_shifted = phi_orig + theta_shift  

        x_new = r_orig * torch.cos(phi_shifted)
        y_new = r_orig * torch.sin(phi_shifted)

        x_new_shifted = x_new - current_radius 
        return torch.stack((x_new_shifted, y_new), dim=1)

    base_data_0 = generate_warped_circle_data(
        num_points_per_time=num_points_per_condition,
        num_time_points=1,  
        radius=radius, 
        angular_concentration=angular_concentration,
        radial_std_dev=radial_std_dev,
        device=device
    )[0]
    
    base_data_1 = generate_warped_circle_data(
        num_points_per_time=num_points_per_condition,
        num_time_points=1,
        radius=radius,
        angular_concentration=angular_concentration,
        radial_std_dev=radial_std_dev,
        device=device
    )[0]
    
    base_data_2 = generate_warped_circle_data(
        num_points_per_time=num_points_per_condition,
        num_time_points=1,
        radius=radius,
        angular_concentration=angular_concentration,
        radial_std_dev=radial_std_dev,
        device=device
    )[0]
    
    base_data_3 = generate_warped_circle_data(
        num_points_per_time=num_points_per_condition,
        num_time_points=1,
        radius=radius,
        angular_concentration=angular_concentration,
        radial_std_dev=radial_std_dev,
        device=device
    )[0]

    cond0_xy = transform_points(base_data_0, time, radius)
    
    time_rev = 1.0 - time
    cond1_xy = transform_points(base_data_1, time_rev, radius)
   
    cond2_xy = transform_points(base_data_2, time, radius)
    cond2_xy[..., 0] *= -1
    
    cond3_xy = transform_points(base_data_3, time_rev, radius)
    cond3_xy[..., 0] *= -1

    cond0_data = torch.cat((cond0_xy, torch.full((num_points_per_condition, 1), 0, device=device, dtype=cond0_xy.dtype)), dim=1)
    cond1_data = torch.cat((cond1_xy, torch.full((num_points_per_condition, 1), 1, device=device, dtype=cond1_xy.dtype)), dim=1)
    cond2_data = torch.cat((cond2_xy, torch.full((num_points_per_condition, 1), 2, device=device, dtype=cond2_xy.dtype)), dim=1)
    cond3_data = torch.cat((cond3_xy, torch.full((num_points_per_condition, 1), 3, device=device, dtype=cond3_xy.dtype)), dim=1)
    final_data = torch.cat((cond0_data, cond1_data, cond2_data, cond3_data), dim=0)

    return final_data

def generate_conditional_circles_data(num_points_per_condition: int,
                                      num_time_points: int = 6,
                                      radius: float = 1.0,
                                      angle_range: float = np.pi / 4,
                                      device: torch.device = DEVICE):
    """
    Generates 4 conditions based on a warped circle, all starting near (0,0).
    1. Generates a base warped circle trajectory using uniform distribution.
    2. Shifts it so t=0 is centered at (0,0) -> Condition 0.
    3. Time-reverses Condition 0 -> Condition 1.
    4. Negates x-coordinates of Condition 0 -> Condition 2.
    5. Time-reverses Condition 2 -> Condition 3.

    Args:
        num_points_per_condition (int): Number of data points per condition per time step.
        num_time_points (int): Number of discrete time steps for the base trajectory.
        radius (float): The target mean radius of the base circle.
        angle_range (float): Spread of the distribution on the circumference per time point (in radians).
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (num_time_points, num_points_per_condition * 4, 3),
                      where the last dimension is [x, y, condition].
    """
    base_data = generate_warped_circle_data_uniform(
        num_points_per_time=num_points_per_condition,
        num_time_points=num_time_points,
        radius=radius,
        angle_range=angle_range,
        device=device
    )
    base_data = torch.cat((base_data, base_data[0:1]), dim=0)  # Wrap around to first time point

    # 2. Shift data so t=0 is centered at (0,0) -> Condition 0
    cond0_data = base_data.clone()
    cond0_data[..., 0] -= radius

    # 3. Time-reverse Condition 0 -> Condition 1
    cond1_data = torch.flip(cond0_data, dims=[0])

    # 4. Negate x-coordinates of Condition 0 -> Condition 2
    cond2_data = cond0_data.clone()
    cond2_data[..., 0] *= -1

    # 5. Time-reverse Condition 2 -> Condition 3
    cond3_data = torch.flip(cond2_data, dims=[0])

    all_data_t = []
    for t in range(num_time_points + 1):
        data_t_conds = []
        for c, cond_data in enumerate([cond0_data, cond1_data, cond2_data, cond3_data]):
            coords_t = cond_data[t]
            labels_t = torch.full((num_points_per_condition, 1), c, device=device)
            combined_c_t = torch.cat((coords_t, labels_t), dim=1)
            data_t_conds.append(combined_c_t)

        data_t_combined = torch.cat(data_t_conds, dim=0)
        all_data_t.append(data_t_combined)

    final_data = torch.stack(all_data_t, dim=0)

    return final_data

def generate_conditional_circle_marginal(num_points_per_condition: int,
                                         time: float,
                                         radius: float = 1.0,
                                         angle_range: float = np.pi / 4,
                                         device: torch.device = DEVICE):
    """
    Generate a marginal distribution for all 4 conditions at a single continuous time.

    Args:
        num_points_per_condition (int): Number of samples per condition.
        time (float): Continuous time input (normalized between 0 and 1).
        radius (float): Radius of the circle.
        angle_range (float): Spread of the distribution on the circumference per time point (in radians).
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (4 * num_points_per_condition, 3),
                      where the last dimension is [x, y, condition].
    """
    base_data = generate_warped_circle_data_uniform(
        num_points_per_time=num_points_per_condition,
        num_time_points=1,
        radius=radius,
        angle_range=angle_range,
        device=device
    )[0]  # Extract the first (and only) time point

    def transform_points(points_xy, t_effective, current_radius):
        theta_shift = 2 * math.pi * t_effective  # Rotate based on time

        x_orig, y_orig = points_xy[:, 0], points_xy[:, 1]
        r_orig = torch.sqrt(x_orig**2 + y_orig**2)  # Compute radius
        phi_orig = torch.atan2(y_orig, x_orig)  # Compute angle

        phi_shifted = phi_orig + theta_shift  # Shift angle

        x_new = r_orig * torch.cos(phi_shifted)
        y_new = r_orig * torch.sin(phi_shifted)

        x_new_shifted = x_new - current_radius
        return torch.stack((x_new_shifted, y_new), dim=1)

    cond0_xy = transform_points(base_data, time, radius)

    time_rev = 1.0 - time
    cond1_xy = transform_points(base_data, time_rev, radius)

    cond2_xy = cond0_xy.clone()
    cond2_xy[..., 0] *= -1

    cond3_xy = cond1_xy.clone()
    cond3_xy[..., 0] *= -1

    cond0_data = torch.cat((cond0_xy, torch.full((num_points_per_condition, 1), 0, device=device, dtype=cond0_xy.dtype)), dim=1)
    cond1_data = torch.cat((cond1_xy, torch.full((num_points_per_condition, 1), 1, device=device, dtype=cond1_xy.dtype)), dim=1)
    cond2_data = torch.cat((cond2_xy, torch.full((num_points_per_condition, 1), 2, device=device, dtype=cond2_xy.dtype)), dim=1)
    cond3_data = torch.cat((cond3_xy, torch.full((num_points_per_condition, 1), 3, device=device, dtype=cond3_xy.dtype)), dim=1)

    final_data = torch.cat((cond0_data, cond1_data, cond2_data, cond3_data), dim=0)

    return final_data

def generate_conditional_semicircles(num_points_per_condition: int,
                                     num_time_points: int = 6,
                                     radius: float = 1.0,
                                     angular_concentration: float = 15.0,
                                     radial_std_dev: float = 0.1,
                                     device: torch.device = DEVICE):
    """
    Generates 4 conditions of warped semicircle data with independent sampling for each condition.
    Each condition moves along a different semicircle:
    - Condition 0: Top half of circle centered at (-1,0), moving from origin to left
    - Condition 1: Bottom half of circle centered at (-1,0), moving from origin to left
    - Condition 2: Top half of circle centered at (1,0), moving from origin to right
    - Condition 3: Bottom half of circle centered at (1,0), moving from origin to right

    Args:
        num_points_per_condition (int): Number of data points per condition per time step.
        num_time_points (int): Number of discrete time steps.
        radius (float): Mean radius of the semicircles.
        angular_concentration (float): Concentration parameter for the von Mises distribution.
        radial_std_dev (float): Standard deviation for the radial distribution.
        device (torch.device): Device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (num_time_points, num_points_per_condition * 4, 3).
    """
    all_data_t = []
    
    for t in range(num_time_points):
        data_t_conds = []
        
        # Calculate progress from 0 to 1
        progress = t / (num_time_points - 1) if num_time_points > 1 else 0
        
        for c in range(4):
            # Determine the target angle based on condition and progress
            if c == 0:  # Top half of left circle (moving from 0 to π)
                target_angle = progress * np.pi  # 0 to π
                center_x = -1.0
            elif c == 1:  # Bottom half of left circle (moving from 0 to -π)
                target_angle = -progress * np.pi  # 0 to -π
                center_x = -1.0
            elif c == 2:  # Top half of right circle (moving from π to 0)
                target_angle = np.pi - progress * np.pi  # π to 0
                center_x = 1.0
            else:  # c == 3, Bottom half of right circle (moving from -π to 0)
                target_angle = -np.pi + progress * np.pi  # -π to 0
                center_x = 1.0
            
            # Generate warped normal distribution
            sampled_angles = vonmises.rvs(
                loc=target_angle,
                kappa=angular_concentration,
                size=num_points_per_condition
            )
            
            log_normal_mu = np.log(radius)
            log_normal_sigma = radial_std_dev
            sampled_radii = lognorm.rvs(
                s=log_normal_sigma,
                loc=0,
                scale=np.exp(log_normal_mu),
                size=num_points_per_condition
            )
            
            # Convert to Cartesian coordinates
            x = center_x + sampled_radii * np.cos(sampled_angles)
            y = sampled_radii * np.sin(sampled_angles)
            
            # Stack coordinates and add condition labels
            points_t = np.stack((x, y), axis=-1)
            points_tensor = torch.tensor(points_t, dtype=torch.float32, device=device)
            labels_tensor = torch.full((num_points_per_condition, 1), c, device=device)
            
            data_t_conds.append(torch.cat((points_tensor, labels_tensor), dim=1))
        
        all_data_t.append(torch.cat(data_t_conds, dim=0))
    
    return torch.stack(all_data_t, dim=0)

if __name__ == "__main__":

    if True:  # Conditional semicircles
        #----------------------------------------------------#
        #  Conditional Semicircles Data                      #
        #----------------------------------------------------#

        print("\n--- Conditional Semicircles Data ---")
        num_points_per_condition = 100
        num_time_points = 3
        radius = 1.0
        angular_concentration = 15.0  # Higher value -> more concentrated angle
        radial_std_dev = 0.05  # Smaller value -> less spread in radius

        # Generate the conditional semicircles data
        conditional_semicircles_data = generate_conditional_semicircles(
            num_points_per_condition=num_points_per_condition,
            num_time_points=num_time_points,
            radius=radius,
            angular_concentration=angular_concentration,
            radial_std_dev=radial_std_dev,
            device=DEVICE
        )

        print(f"Generated conditional semicircles data shape: {conditional_semicircles_data.shape}")

        # Save the data
        save_path_semicircles = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_semicircles.pt')
        torch.save(conditional_semicircles_data, save_path_semicircles)
        print(f"Conditional semicircles data saved to {save_path_semicircles}")

        # --- Plotting Conditional Semicircles Data ---
        fig_semicircles, axs_semicircles = plt.subplots(1, num_time_points, figsize=(5 * num_time_points, 5.5), 
                                                    sharex=True, sharey=True)
        all_x_semicircles = conditional_semicircles_data[..., 0].cpu().numpy().flatten()
        all_y_semicircles = conditional_semicircles_data[..., 1].cpu().numpy().flatten()
        max_abs_coord_semicircles = max(np.abs(all_x_semicircles).max(), np.abs(all_y_semicircles).max()) * 1.1
        plot_lim_semicircles = (-max_abs_coord_semicircles, max_abs_coord_semicircles)

        colors_semicircles = plt.cm.tab10(np.linspace(0, 1, 4))

        for t in range(num_time_points):
            data_t = conditional_semicircles_data[t].cpu().numpy()
            conditions_t = data_t[:, 2].astype(int)
            axs_semicircles[t].set_title(f"Time t={t}")
            for c in range(4):
                mask = conditions_t == c
                label = f'Cond {c}' if t == 0 else ""
                if c == 0: label += " (Left Top)"
                if c == 1: label += " (Left Bottom)"
                if c == 2: label += " (Right Top)"
                if c == 3: label += " (Right Bottom)"
                axs_semicircles[t].scatter(data_t[mask, 0], data_t[mask, 1], 
                                        color=colors_semicircles[c], label=label, alpha=0.5, s=5)
            axs_semicircles[t].set_xlabel("x")
            if t == 0:
                axs_semicircles[t].set_ylabel("y")
                axs_semicircles[t].legend(markerscale=2, fontsize='small')
            axs_semicircles[t].set_aspect('equal', adjustable='box')
            axs_semicircles[t].grid(True)
            axs_semicircles[t].set_xlim(plot_lim_semicircles)
            axs_semicircles[t].set_ylim(plot_lim_semicircles)
            
            # Draw reference semicircles with condition-specific colors
            from matplotlib.patches import Arc
            
            # Condition 0: Top half of left circle
            top_left_arc = Arc((-1, 0), 2*radius, 2*radius, 
                            theta1=0, theta2=180, 
                            color=colors_semicircles[0], linestyle='-', linewidth=1.5)
            
            # Condition 1: Bottom half of left circle
            bottom_left_arc = Arc((-1, 0), 2*radius, 2*radius, 
                                theta1=180, theta2=360, 
                                color=colors_semicircles[1], linestyle='-', linewidth=1.5)
            
            # Condition 2: Top half of right circle
            top_right_arc = Arc((1, 0), 2*radius, 2*radius, 
                            theta1=0, theta2=180, 
                            color=colors_semicircles[2], linestyle='-', linewidth=1.5)
            
            # Condition 3: Bottom half of right circle
            bottom_right_arc = Arc((1, 0), 2*radius, 2*radius, 
                                theta1=180, theta2=360, 
                                color=colors_semicircles[3], linestyle='-', linewidth=1.5)
            
            axs_semicircles[t].add_patch(top_left_arc)
            axs_semicircles[t].add_patch(bottom_left_arc)
            axs_semicircles[t].add_patch(top_right_arc)
            axs_semicircles[t].add_patch(bottom_right_arc)

        plt.suptitle(f"Conditional Semicircles Data ({num_time_points} Time Points, {num_points_per_condition} Pts/Cond)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path_semicircles = os.path.join(SCRIPT_PATH, 'plots/conditional_semicircles_plot.png')
        plt.savefig(plot_save_path_semicircles)
        print(f"Conditional semicircles plot saved to {plot_save_path_semicircles}")
        plt.close(fig_semicircles)

    if False:  # Eval marginals normal
        num_points_per_condition = 100
        time = 0.25
        radius = 1.0
        angular_concentration = 10.0
        radial_std_dev = 0.08

        marginal_data_normal = generate_conditional_circle_marginal_normal(
            num_points_per_condition=num_points_per_condition,
            time=time,
            radius=radius,
            angular_concentration=angular_concentration,
            radial_std_dev=radial_std_dev,
            device=DEVICE
        )
        print(f"Generated normal marginal data shape: {marginal_data_normal.shape}")

        x = marginal_data_normal[:, 0].cpu().numpy()
        y = marginal_data_normal[:, 1].cpu().numpy()
        conditions = marginal_data_normal[:, 2].cpu().numpy()

        plt.figure(figsize=(8, 8))
        colors = ['blue', 'orange', 'green', 'red']
        labels = ['Condition 0 (Base)', 'Condition 1 (Time-Reversed)', 
                'Condition 2 (X-Flipped)', 'Condition 3 (X-Flipped, Time-Reversed)']

        for c in range(4):
            mask = conditions == c
            plt.scatter(x[mask], y[mask], label=labels[c], color=colors[c], alpha=0.6, s=10)

        circle_ref = plt.Circle((-1, 0), radius, color='black', fill=False, linestyle='--', linewidth=1)
        neg_circle_ref = plt.Circle((1, 0), -radius, color='black', fill=False, linestyle='--', linewidth=1)
        plt.gca().add_patch(circle_ref)
        plt.gca().add_patch(neg_circle_ref)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Normal Marginal Distribution at Time {time}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-1.5, 1.5)

        plt.savefig(os.path.join(SCRIPT_PATH, 'plots', f'conditional_circles_marginal_normal_time_{time}.png'))

        # Define the times for which eval marginals
        times = [0, 0.125, 0.375, 0.625, 0.875]
        time_samples_list_normal = []
        for t in times:
            samples = generate_conditional_circle_marginal_normal(
                num_points_per_condition=num_points_per_condition,
                time=t,
                radius=radius,
                angular_concentration=angular_concentration,
                radial_std_dev=radial_std_dev,
                device=DEVICE
            )
            samples = jnp.array(samples.cpu().numpy())
            time_samples_list_normal.append((t, samples))

        for time, samples in time_samples_list_normal:
            print(f"Time: {time}, Samples Shape: {samples.shape}")

        with open(os.path.join(SCRIPT_PATH, 'eval_marginals_normal.pkl'), 'wb') as f:
            pickle.dump(time_samples_list_normal, f)

    if False:  # Conditional circles normal
        #----------------------------------------------------#
        #  Conditional Circles Data (Normal Distribution)   #
        #----------------------------------------------------#

        print("\n--- Conditional Circles Data (Normal) ---")
        num_points_per_condition = 100
        num_time_points = 4
        radius = 1.0
        angular_concentration = 10.0  # Higher value -> more concentrated angle
        radial_std_dev = 0.08  # Smaller value -> less spread in radius

        # Generate the conditional circles data using the normal distribution
        conditional_circles_data_normal = generate_conditional_circles_data_normal(
            num_points_per_condition=num_points_per_condition,
            num_time_points=num_time_points,
            radius=radius,
            angular_concentration=angular_concentration,
            radial_std_dev=radial_std_dev,
            device=DEVICE
        )

        print(f"Generated conditional circles data (normal) shape: {conditional_circles_data_normal.shape}")

        # Save the data
        save_path_normal = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_circles_normal.pt')
        torch.save(conditional_circles_data_normal, save_path_normal)
        print(f"Conditional circles data (normal) saved to {save_path_normal}")

        # --- Plotting Conditional Circles Data (Normal) ---
        fig_normal, axs_normal = plt.subplots(1, num_time_points, figsize=(5 * num_time_points, 5.5), sharex=True, sharey=True)
        all_x_normal = conditional_circles_data_normal[..., 0].cpu().numpy().flatten()
        all_y_normal = conditional_circles_data_normal[..., 1].cpu().numpy().flatten()
        max_abs_coord_normal = max(np.abs(all_x_normal).max(), np.abs(all_y_normal).max()) * 1.1
        plot_lim_normal = (-max_abs_coord_normal, max_abs_coord_normal)

        colors_normal = plt.cm.tab10(np.linspace(0, 1, 4))

        for t in range(num_time_points):
            data_t = conditional_circles_data_normal[t].cpu().numpy()
            conditions_t = data_t[:, 2].astype(int)
            axs_normal[t].set_title(f"Time t={t}")
            for c in range(4):
                mask = conditions_t == c
                label = f'Cond {c}' if t == 0 else ""
                if c == 0: label += " (Base)"
                if c == 1: label += " (Time Reversed)"
                if c == 2: label += " (X-Flipped)"
                if c == 3: label += " (X-Flipped, Time Reversed)"
                axs_normal[t].scatter(data_t[mask, 0], data_t[mask, 1], color=colors_normal[c], label=label, alpha=0.5, s=5)
            axs_normal[t].set_xlabel("x")
            if t == 0:
                axs_normal[t].set_ylabel("y")
                axs_normal[t].legend(markerscale=2, fontsize='small')
            axs_normal[t].set_aspect('equal', adjustable='box')
            axs_normal[t].grid(True)
            axs_normal[t].set_xlim(plot_lim_normal)
            axs_normal[t].set_ylim(plot_lim_normal)
            circle_ref = plt.Circle((-1, 0), 1.0, color='r', fill=False, linestyle='--', linewidth=1)
            circle_ref_reversed = plt.Circle((1, 0), -1.0, color='b', fill=False, linestyle='--', linewidth=1)
            axs_normal[t].add_patch(circle_ref)
            axs_normal[t].add_patch(circle_ref_reversed)

        plt.suptitle(f"Conditional Circles Data (Normal) ({num_time_points} Time Points, {num_points_per_condition} Pts/Cond)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path_normal = os.path.join(SCRIPT_PATH, 'plots/conditional_circles_normal_plot.png')
        plt.savefig(plot_save_path_normal)
        print(f"Conditional circles plot (normal) saved to {plot_save_path_normal}")
        plt.close(fig_normal)
    
    if False:  #Eval marginals uniform
        # Parameters
        num_points_per_condition = 100
        time = 0.125
        radius = 1.0
        angle_range = np.pi / 2

        # Generate the marginal data for all 4 conditions
        marginal_data = generate_conditional_circle_marginal(
            num_points_per_condition=num_points_per_condition,
            time=time,
            radius=radius,
            angle_range=angle_range,
            device=DEVICE
        )

        print(f"Generated marginal data shape: {marginal_data.shape}")

        # Extract x, y, and condition labels
        x = marginal_data[:, 0].cpu().numpy()
        y = marginal_data[:, 1].cpu().numpy()
        conditions = marginal_data[:, 2].cpu().numpy()

        # Plotting
        plt.figure(figsize=(8, 8))
        colors = ['blue', 'orange', 'green', 'red']
        labels = ['Condition 0 (Base)', 'Condition 1 (Time-Reversed)', 
                'Condition 2 (X-Flipped)', 'Condition 3 (X-Flipped, Time-Reversed)']

        for c in range(4):
            mask = conditions == c
            plt.scatter(x[mask], y[mask], label=labels[c], color=colors[c], alpha=0.6, s=10)

        # Add unit circle for reference
        circle_ref = plt.Circle((-1, 0), radius, color='black', fill=False, linestyle='--', linewidth=1)
        neg_circle_ref = plt.Circle((1, 0), -radius, color='black', fill=False, linestyle='--', linewidth=1)
        plt.gca().add_patch(circle_ref)
        plt.gca().add_patch(neg_circle_ref)

        # Formatting
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Marginal Distribution at Time {time}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-1.5, 1.5)

        # Show the plot
        plt.savefig(os.path.join(SCRIPT_PATH, 'plots', f'conditional_circles_marginal_time_{time}.png'))

        # Define the times for which to generate samples
        times = [0, 0.125, 0.375, 0.625, 0.875]

        # Initialize the list to store (time, samples) tuples
        time_samples_list = []

        # Generate samples for each time
        for t in times:
            samples = generate_conditional_circle_marginal(
                num_points_per_condition=num_points_per_condition,
                time=t,
                radius=radius,
                angle_range=angle_range,
                device=DEVICE
            )
            #convert samples to jnp
            samples = jnp.array(samples.cpu().numpy())

            time_samples_list.append((t, samples))

        # Print the result
        for time, samples in time_samples_list:
            print(f"Time: {time}, Samples Shape: {samples.shape}")

        #save time_samples_list as pkl

        with open(os.path.join(SCRIPT_PATH, 'eval_marginals.pkl'), 'wb') as f:
            pickle.dump(time_samples_list, f)
    
    if False: # KDE
        #----------------------------------------------------#
        #  Conditional Circles KDE #
        #----------------------------------------------------#
        
        num_points_per_condition = 100
        num_time_points = 4
        radius = 1.0
        angle_range = np.pi / 2  # Spread of the distribution on the circumference

        kde_data = generate_conditional_circles_data(
            num_points_per_condition=num_points_per_condition,
            num_time_points=num_time_points,
            radius=radius,
            angle_range=angle_range,
            device=DEVICE
        )[:-1]  # Exclude the last time point for plotting

        #combine all time points
        data = kde_data.view(-1, kde_data.shape[-1]).cpu().numpy()  # Shape: (num_points_per_condition * num_time_points * 4, 3)
        print(data.shape)
        # Separate data by condition
        conditions = data[:, 2].astype(int)
        x = data[:, 0]
        y = data[:, 1]

        # Plot KDE for each condition in separate subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        colors = ['blue', 'orange', 'green', 'red']
        labels = ['Condition 0', 'Condition 1', 'Condition 2', 'Condition 3']

        for c, ax in enumerate(axs.flatten()):
            mask = conditions == c
            sns.kdeplot(x=x[mask], y=y[mask], fill=True, alpha=0.5, label=labels[c], color=colors[c], ax=ax, bw_method=0.25)
            ax.set_title(labels[c])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            ax.legend()

        # Formatting
        plt.suptitle("KDE of Ambient Data for Each Condition")
        plt.tight_layout()
        #plt.savefig(os.path.join(SCRIPT_PATH, 'plots', 'conditional_circles_kde_subplots.png'))

    if False: # Conditional circles uniform
        #----------------------------------------------------#
        #  Conditional Circles Data #
        #----------------------------------------------------#

        print("\n--- Conditional Circles Data ---")
        num_points_per_condition = 100
        num_time_points = 4
        radius = 1.0
        angle_range = np.pi / 2  # Spread of the distribution on the circumference

        conditional_circles_data = generate_conditional_circles_data(
            num_points_per_condition=num_points_per_condition,
            num_time_points=num_time_points,
            radius=radius,
            angle_range=angle_range,
            device=DEVICE
        )

        print(f"Generated conditional circles data shape: {conditional_circles_data.shape}")

        # save
        save_path_smr = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_circles.pt')
        #torch.save(conditional_circles_data, save_path_smr)
        print(f"Conditional circles data saved to {save_path_smr}")

        # --- Plotting SMR Circle Data ---
        fig_smr, axs_smr = plt.subplots(1, num_time_points, figsize=(5 * num_time_points, 5.5), sharex=True, sharey=True)
        all_x_smr = conditional_circles_data[..., 0].cpu().numpy().flatten()
        all_y_smr = conditional_circles_data[..., 1].cpu().numpy().flatten()
        max_abs_coord_smr = max(np.abs(all_x_smr).max(), np.abs(all_y_smr).max()) * 1.1
        plot_lim_smr = (-max_abs_coord_smr, max_abs_coord_smr)

        colors_smr = plt.cm.tab10(np.linspace(0, 1, 4))

        for t in range(num_time_points):
            data_t = conditional_circles_data[t].cpu().numpy()
            conditions_t = data_t[:, 2].astype(int)
            axs_smr[t].set_title(f"Time t={t}")
            for c in range(4):
                mask = conditions_t == c
                label = f'Cond {c}' if t == 0 else ""
                if c == 0: label += " (Base)"
                if c == 1: label += " (Time Reversed)"
                if c == 2: label += " (X-Flipped)"
                if c == 3: label += " (X-Flipped, Time Reversed)"
                axs_smr[t].scatter(data_t[mask, 0], data_t[mask, 1], color=colors_smr[c], label=label, alpha=0.5, s=5)
            axs_smr[t].set_xlabel("x")
            if t == 0:
                axs_smr[t].set_ylabel("y")
                axs_smr[t].legend(markerscale=2, fontsize='small')
            axs_smr[t].set_aspect('equal', adjustable='box')
            axs_smr[t].grid(True)
            axs_smr[t].set_xlim(plot_lim_smr)
            axs_smr[t].set_ylim(plot_lim_smr)
            circle_ref = plt.Circle((-1, 0), 1.0, color='r', fill=False, linestyle='--', linewidth=1)
            circle_ref_reversed = plt.Circle((1, 0), -1.0, color='b', fill=False, linestyle='--', linewidth=1)
            axs_smr[t].add_patch(circle_ref)
            axs_smr[t].add_patch(circle_ref_reversed)

        plt.suptitle(f"Conditional Circles Data ({num_time_points} Time Points, {num_points_per_condition} Pts/Cond)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path_smr = os.path.join(SCRIPT_PATH, 'plots/conditional_circles_plot.png')
        #plt.savefig(plot_save_path_smr)
        print(f"Conditional circles plot saved to {plot_save_path_smr}")
        plt.close(fig_smr)

    if False: # Warped circle
        #------------------------#
        #   WARPED CIRCLE DATA   #
        #------------------------#
        print("\n--- Warped Circle Data ---")
        num_points_circle = 100
        num_times_circle = 4
        angular_conc_circle = 10.0 # Higher value -> more concentrated angle
        radial_std_circle = 0.08  # Smaller value -> less spread in radius

        warped_circle_data = generate_warped_circle_data(
            num_points_per_time=num_points_circle,
            num_time_points=num_times_circle,
            angular_concentration=angular_conc_circle,
            radial_std_dev=radial_std_circle,
            radius=1.0
        )

        #generate uniform circle data
        warped_circle_data = generate_warped_circle_data_uniform(
            num_points_per_time=num_points_circle,
            num_time_points=num_times_circle,
            radius=1.0,
            angle_range=np.pi / 2,
            device=DEVICE
        )

        # save
        #warped_circle_data = torch.cat((warped_circle_data, warped_circle_data[0:1]), dim=0) # Wrap around to first time point
        print(f"Generated warped circle data shape: {warped_circle_data.shape}")

        save_path_circle = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'warped_circle.pt')
        #torch.save(warped_circle_data, save_path_circle)
        #print(f"Warped circle data saved to {save_path_circle}")

        # --- Plotting Warped Circle Data ---
        fig_circle, axs_circle = plt.subplots(1, num_times_circle, figsize=(5 * num_times_circle, 5.5), sharex=True, sharey=True)
        max_abs_coord = warped_circle_data.abs().max().item() * 1.1 # Get max coordinate for consistent plot limits
        plot_lim = (-max_abs_coord, max_abs_coord)

        for t in range(num_times_circle):
            data_t = warped_circle_data[t].cpu().numpy()
            axs_circle[t].scatter(data_t[:, 0], data_t[:, 1], alpha=0.5, s=5)
            axs_circle[t].set_title(f"Time t={t}")
            axs_circle[t].set_xlabel("x")
            if t == 0:
                axs_circle[t].set_ylabel("y")
            axs_circle[t].set_aspect('equal', adjustable='box')
            axs_circle[t].grid(True)
            axs_circle[t].set_xlim(plot_lim)
            axs_circle[t].set_ylim(plot_lim)
            # Draw unit circle for reference
            circle_ref = plt.Circle((0, 0), 1.0, color='r', fill=False, linestyle='--', linewidth=1)
            axs_circle[t].add_patch(circle_ref)


        plt.suptitle(f"Warped Circle Data ({num_times_circle} Time Points, {num_points_circle} Pts/Time)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path_circle = os.path.join(SCRIPT_PATH, 'warped_circle_plot.png')
        plt.savefig(plot_save_path_circle)
        print(f"Warped circle plot saved to {plot_save_path_circle}")
        plt.close(fig_circle)

    if False: # Velocity condition
        #-----------------------------#
        #  VELOCITY CONDITIONED DATA  #
        #-----------------------------#

        print("\n--- Velocity Conditioned Data ---")
        num_cond_vel = 10
        num_points_per_cond_vel = 100
        total_points_vel = num_cond_vel * num_points_per_cond_vel
        timesteps_vel = np.linspace(0, 1, 21)
        variance_vel = 0.01
        velocity_data = generate_velocity_conditioned_data(
            num_conditions=num_cond_vel,
            num_points_per_condition=num_points_per_cond_vel,
            timesteps=timesteps_vel,
            variance=variance_vel
        )
        print(f"Generated velocity conditioned data shape: {velocity_data.shape}")

        #save
        save_path_vel = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_velocity.pt')
        #torch.save(velocity_data, save_path_vel)
        #print(f"Velocity conditioned data saved to {save_path_vel}")

        fig_vel, axs_vel = plt.subplots(1, len(timesteps_vel), figsize=(5 * len(timesteps_vel), 5), sharex=True, sharey=True)
        conditions_vel = velocity_data[0, :, 2].cpu().numpy()
        norm_vel = plt.Normalize(conditions_vel.min(), conditions_vel.max())
        cmap_vel = plt.cm.viridis

        for i, t in enumerate(timesteps_vel):
            data_t = velocity_data[i].cpu().numpy()
            points = axs_vel[i].scatter(data_t[:, 0], data_t[:, 1], c=conditions_vel, cmap=cmap_vel, norm=norm_vel, alpha=0.6, s=10)
            axs_vel[i].set_title(f"Time t={t:.2f}")
            axs_vel[i].set_xlabel("x")
            if i == 0:
                axs_vel[i].set_ylabel("y")
            axs_vel[i].set_aspect('equal', adjustable='box')
            axs_vel[i].grid(True)

        fig_vel.colorbar(points, ax=axs_vel, label='Condition (Speed)')
        plt.suptitle(f"Velocity Conditioned Data ({num_cond_vel} Speeds, {num_points_per_cond_vel} Pts/Speed)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path_vel = os.path.join(SCRIPT_PATH, 'plots/velocity_conditioned_plot.png')
        #plt.savefig(plot_save_path_vel)
        plt.close(fig_vel)

    if False: # Rotation condition
        #-----------------------------#
        #  ROTATION CONDITIONED DATA  #
        #-----------------------------#    
        
        print("\n--- Rotation Conditioned Data ---")
        num_cond_rot = 10 
        num_points_per_cond_rot = 100 
        total_points_rot = num_cond_rot * num_points_per_cond_rot
        timesteps_rot = np.linspace(0, 1, 21)
        variance_rot = 0.03
        rotation_data = generate_rotation_conditioned_data(
            num_conditions=num_cond_rot,
            num_points_per_condition=num_points_per_cond_rot,
            timesteps=timesteps_rot,
            base_radius_speed=0.5,
            base_angular_speed=math.pi,
            condition_range=(0, math.pi),
            variance=variance_rot
        )
        print(f"Generated rotation conditioned data shape: {rotation_data.shape}") 

        #save
        save_path_rot = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_rotation.pt')
        #torch.save(rotation_data, save_path_rot)
        #print(f"Rotation conditioned data saved to {save_path_rot}")

        fig_rot, axs_rot = plt.subplots(1, len(timesteps_rot), figsize=(5 * len(timesteps_rot), 5), sharex=True, sharey=True)
        conditions_rot = rotation_data[0, :, 2].cpu().numpy()
        norm_rot = plt.Normalize(conditions_rot.min(), conditions_rot.max())
        cmap_rot = plt.cm.plasma

        for i, t in enumerate(timesteps_rot):
            data_t = rotation_data[i].cpu().numpy()
            points = axs_rot[i].scatter(data_t[:, 0], data_t[:, 1], c=conditions_rot, cmap=cmap_rot, norm=norm_rot, alpha=0.6, s=10)
            axs_rot[i].set_title(f"Time t={t:.2f}")
            axs_rot[i].set_xlabel("x")
            if i == 0:
                axs_rot[i].set_ylabel("y")
            axs_rot[i].set_aspect('equal', adjustable='box')
            axs_rot[i].grid(True)

        all_x_rot = rotation_data[:, :, 0].cpu().numpy().flatten()
        all_y_rot = rotation_data[:, :, 1].cpu().numpy().flatten()
        x_min_rot, x_max_rot = np.min(all_x_rot), np.max(all_x_rot)
        y_min_rot, y_max_rot = np.min(all_y_rot), np.max(all_y_rot)
        padding_rot = max(abs(x_min_rot), abs(x_max_rot), abs(y_min_rot), abs(y_max_rot)) * 0.1
        axs_rot[0].set_xlim(x_min_rot - padding_rot, x_max_rot + padding_rot)
        axs_rot[0].set_ylim(y_min_rot - padding_rot, y_max_rot + padding_rot)

        fig_rot.colorbar(points, ax=axs_rot, label='Condition (Angular Speed Offset)')
        plt.suptitle(f"Rotation Conditioned Data ({num_cond_rot} Offsets, {num_points_per_cond_rot} Pts/Offset)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path_rot = os.path.join(SCRIPT_PATH, 'plots/rotation_conditioned_plot.png')
        #plt.savefig(plot_save_path_rot)
        plt.close(fig_rot)

    if False: # Gaussian squiggles
        #----------------------------#
        #  COMPLEX CONDITIONAL DATA  #
        #----------------------------#
        num_points_per_cond_5t = 500
        variance_5t = 0.05
        conditional_data_5t = generate_complex_conditional_gaussian_data( 
            num_points_per_condition=num_points_per_cond_5t,
            variance=variance_5t
        )
        print(f"Generated 5T conditional data shape: {conditional_data_5t.shape}")
        print(conditional_data_5t[0])

        #save
        #save_path_5t = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_gaussians_complex.pt')
        #torch.save(conditional_data_5t, save_path_5t)
        #print(f"5T Conditional data saved to {save_path_5t}")

        # --- Plotting 5T Conditional Data ---
        num_time_points_plot = 5
        fig_5t, axs_5t = plt.subplots(1, num_time_points_plot, figsize=(5 * num_time_points_plot, 5), sharex=True, sharey=True)
        colors_5t = plt.cm.viridis(np.linspace(0, 1, 4))

        for t in range(num_time_points_plot):
            data_t = conditional_data_5t[t].cpu().numpy()
            conditions_t = data_t[:, 2].astype(int)
            axs_5t[t].set_title(f"Time t={t}")
            for c in range(4):
                mask = conditions_t == c
                axs_5t[t].scatter(data_t[mask, 0], data_t[mask, 1], color=colors_5t[c], label=f'Cond {c}' if t == 0 else "", alpha=0.6, s=10)
            axs_5t[t].set_xlabel("x")
            if t == 0:
                axs_5t[t].set_ylabel("y")
                axs_5t[t].legend()
            axs_5t[t].set_aspect('equal', adjustable='box')
            axs_5t[t].grid(True)


        all_x = conditional_data_5t[:, :, 0].cpu().numpy().flatten()
        all_y = conditional_data_5t[:, :, 1].cpu().numpy().flatten()
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        padding = 1.0
        axs_5t[0].set_xlim(x_min - padding, x_max + padding)
        axs_5t[0].set_ylim(y_min - padding, y_max + padding)


        plt.suptitle("Generated Complex Conditional Gaussian Data")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path_5t = os.path.join(SCRIPT_PATH, 'plots/conditional_gaussians_complex_plot.png')
        #plt.savefig(plot_save_path_5t)
        #print(f"5T Plot saved to {plot_save_path_5t}")
        plt.close(fig_5t)

    if False: # Gaussian splitting
        #--------------------#
        #  CONDITIONAL DATA  #
        #--------------------#
        num_points_per_cond = 500
        conditional_data = generate_conditional_gaussian_data(num_points_per_condition=num_points_per_cond, variance=0.05)
        total_points = num_points_per_cond * 4
        print(f"Generated conditional data shape: {conditional_data.shape}") 

        # save
        save_path = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_gaussians_3t.pt')
        #torch.save(conditional_data, save_path)
        #print(f"Conditional data saved to {save_path}")

        # --- Plotting Conditional Data ---
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        colors = plt.cm.viridis(np.linspace(0, 1, 4))

        # Plot t=0
        data_t0 = conditional_data[0].cpu().numpy()
        conditions_t0 = data_t0[:, 2].astype(int)
        axs[0].set_title("Time t=0")
        for c in range(4):
            mask = conditions_t0 == c
            axs[0].scatter(data_t0[mask, 0], data_t0[mask, 1], color=colors[c], label=f'Cond {c}', alpha=0.6, s=10)
        axs[0].legend()
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_aspect('equal', adjustable='box')
        axs[0].grid(True)

        # Plot t=1
        data_t1 = conditional_data[1].cpu().numpy()
        conditions_t1 = data_t1[:, 2].astype(int)
        axs[1].set_title("Time t=1")
        for c in range(4):
            mask = conditions_t1 == c
            axs[1].scatter(data_t1[mask, 0], data_t1[mask, 1], color=colors[c], label=f'Cond {c}', alpha=0.6, s=10)
        axs[1].set_xlabel("x")
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].grid(True)

        # Plot t=2
        data_t2 = conditional_data[2].cpu().numpy()
        conditions_t2 = data_t2[:, 2].astype(int)
        axs[2].set_title("Time t=2") 
        for c in range(4):
            mask = conditions_t2 == c
            axs[2].scatter(data_t2[mask, 0], data_t2[mask, 1], color=colors[c], label=f'Cond {c}', alpha=0.6, s=10)
        axs[2].set_xlabel("x")
        axs[2].set_aspect('equal', adjustable='box')
        axs[2].grid(True)

        plt.suptitle("Generated Conditional Gaussian Data (3 Time Steps)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path = os.path.join(SCRIPT_PATH, 'plots/conditional_gaussians_3t_plot.png') 
        #plt.savefig(plot_save_path)
        #print(f"Plot saved to {plot_save_path}")

    if False: # Arch data
        #-------------#
        #  ARCH DATA  #
        #-------------#
        num_points = 5000
        arch_points, labels, unique_labels = generate_arch_data(num_points=num_points)
        print("arch points shape:", arch_points.shape)
        arch_points = np.reshape(arch_points, (len(unique_labels), num_points, 2))

        #save
        arch_points = torch.tensor(arch_points)
        arch_points = arch_points[[0,2]]
        print("arch points shape:", arch_points.shape)
        #torch.save(arch_points, 'scarvelis_data/arch_data.pt')

    if False: # Sphere data
        #---------------#
        #  SPHERE DATA  #
        #---------------#
        sphere_points, labels, unique_labels = generate_sphere_data(num_points=num_points)
        print("sphere points shape:", sphere_points.shape)
        sphere_points = np.reshape(sphere_points, (len(unique_labels), num_points, 3))
        sphere_points = torch.tensor(sphere_points)
        sphere_points = sphere_points[[0,2]]
        print("sphere points shape:", sphere_points.shape)
        #torch.save(sphere_points, 'scarvelis_data/sphere_data.pt')

        #plot 3D sphere points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sphere_points[0, :, 0], sphere_points[0, :, 1], sphere_points[0, :, 2], c='red')
        ax.scatter(sphere_points[1, :, 0], sphere_points[1, :, 1], sphere_points[1, :, 2], c='blue')
        #plt.savefig('plots/sphere_data.png')
