import torch
import os
from torchcfm.optimal_transport import OTPlanSampler
import numpy as np
import matplotlib.pyplot as plt

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
    Generates a 2D conditional dataset with 2 time points and 4 conditions.
    t=0: N((0,0), variance*I) for all conditions.
    t=1: N(mean_c, variance*I) where mean_c is one of {(1,1), (1,-1), (-1,1), (-1,-1)}.

    Returns:
        torch.Tensor: Data tensor of shape (2, num_points_total, 3),
                      where the dimensions are (time, sample, [x, y, condition]).
    """
    num_conditions = 4
    means_t0 = torch.zeros((num_conditions, 2), device=DEVICE)
    means_t1 = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float32, device=DEVICE)
    covariance = torch.eye(2, device=DEVICE) * variance
    sqrt_covariance = torch.linalg.cholesky(covariance) # For sampling: mean + sqrt_cov @ randn

    all_data_t0 = []
    all_data_t1 = []

    for c in range(num_conditions):
        # Sample t=0 data: N(0, 0.1*I)
        z0 = torch.randn(num_points_per_condition, 2, device=DEVICE)
        samples_t0 = means_t0[c] + z0 @ sqrt_covariance.T

        # Sample t=1 data: N(mean_c, 0.1*I)
        z1 = torch.randn(num_points_per_condition, 2, device=DEVICE)
        samples_t1 = means_t1[c] + z1 @ sqrt_covariance.T

        # Create condition vector
        condition_vec = torch.full((num_points_per_condition, 1), float(c), device=DEVICE)

        # Append [x, y, condition]
        all_data_t0.append(torch.cat((samples_t0, condition_vec), dim=1))
        all_data_t1.append(torch.cat((samples_t1, condition_vec), dim=1))

    # Concatenate across conditions
    final_data_t0 = torch.cat(all_data_t0, dim=0)
    final_data_t1 = torch.cat(all_data_t1, dim=0)

    # Stack time points
    final_data = torch.stack([final_data_t0, final_data_t1], dim=0)

    return final_data

if __name__ == "__main__":

    #-------------#
    ###CONDITIONAL DATA###
    #-------------#
    num_points_per_cond = 500
    conditional_data = generate_conditional_gaussian_data(num_points_per_condition=num_points_per_cond, variance=0.05) 
    total_points = num_points_per_cond * 4
    print(f"Generated conditional data shape: {conditional_data.shape}") # Should be (2, total_points, 3)

    # Save the conditional data
    save_path = os.path.join(SCRIPT_PATH, 'scarvelis_data', 'conditional_gaussians.pt')
    torch.save(conditional_data, save_path)
    print(f"Conditional data saved to {save_path}")

    # --- Plotting Conditional Data ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
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
    data_t1 = conditional_data[1].cpu().numpy()
    conditions_t1 = data_t1[:, 2].astype(int)
    axs[1].set_title("Time t=1")
    for c in range(4):
        mask = conditions_t1 == c
        axs[1].scatter(data_t1[mask, 0], data_t1[mask, 1], color=colors[c], label=f'Cond {c}', alpha=0.6, s=10)
    axs[1].set_xlabel("x")
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].grid(True)


    plt.suptitle("Generated Conditional Gaussian Data")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plot_save_path = os.path.join(SCRIPT_PATH, 'conditional_gaussians_plot.png')
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")

    #-------------#
    ###ARCH DATA###
    #-------------#
    num_points = 1000
    arch_points, labels, unique_labels = generate_arch_data(num_points=num_points)
    print("arch points shape:", arch_points.shape)

    #group into shape (num_labels, num_points, 2)
    arch_points = np.reshape(arch_points, (len(unique_labels), num_points, 2))

    #save as .pt
    arch_points = torch.tensor(arch_points)
    arch_points = arch_points[[0,2]]
    print("arch points shape:", arch_points.shape)
    #torch.save(arch_points, 'scarvelis_data/arch_data.pt')

    #-------------#
    ###SPHERE DATA###
    #-------------#
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
