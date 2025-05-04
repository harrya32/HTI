import torch
import os
from torchcfm.optimal_transport import OTPlanSampler
import numpy as np
import matplotlib.pyplot as plt
import math

# Define the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
paths = {
        "scarvelis_circle": "data_gic_24_gaussians_radius_1_std_0p1_100_samples_closed.pt",
        "scarvelis_vee": "data_mass_split_std_1_100_samples_8_intermediate_scale_x10.pt",
        "scarvelis_xpath": "data_xpath_std_0p1_100_samples_8_intermediate.pt",
    }
fname = SCRIPT_PATH + "/scarvelis_data/" + paths['scarvelis_circle']

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


if __name__ == "__main__":

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
    torch.save(velocity_data, save_path_vel)
    print(f"Velocity conditioned data saved to {save_path_vel}")

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
    plot_save_path_vel = os.path.join(SCRIPT_PATH, 'velocity_conditioned_plot.png')
    plt.savefig(plot_save_path_vel)
    plt.close(fig_vel)

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
    torch.save(rotation_data, save_path_rot)
    print(f"Rotation conditioned data saved to {save_path_rot}")

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
    plot_save_path_rot = os.path.join(SCRIPT_PATH, 'rotation_conditioned_plot.png')
    plt.savefig(plot_save_path_rot)
    plt.close(fig_rot)

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
    plot_save_path_5t = os.path.join(SCRIPT_PATH, 'conditional_gaussians_complex_plot.png')
    #plt.savefig(plot_save_path_5t)
    #print(f"5T Plot saved to {plot_save_path_5t}")
    plt.close(fig_5t)


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
    plot_save_path = os.path.join(SCRIPT_PATH, 'conditional_gaussians_3t_plot.png') 
    #plt.savefig(plot_save_path)
    #print(f"Plot saved to {plot_save_path}")

    #-------------#
    #  ARCH DATA  #
    #-------------#
    num_points = 1000
    arch_points, labels, unique_labels = generate_arch_data(num_points=num_points)
    print("arch points shape:", arch_points.shape)
    arch_points = np.reshape(arch_points, (len(unique_labels), num_points, 2))

    #save
    arch_points = torch.tensor(arch_points)
    arch_points = arch_points[[0,2]]
    print("arch points shape:", arch_points.shape)
    #torch.save(arch_points, 'scarvelis_data/arch_data.pt')

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
    #plt.savefig('sphere_data.png')
