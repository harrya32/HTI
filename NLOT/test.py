import torch
import os
from torchcfm.optimal_transport import OTPlanSampler
import numpy as np
import matplotlib.pyplot as plt

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


scarvelis_data = torch.load(fname).detach().cpu().numpy()
print("scarvelis_data shape:", scarvelis_data.shape)


num_points = 100
arch_points, labels, unique_labels = generate_arch_data(num_points=num_points)
print("arch points shape:", arch_points.shape)

#group into shape (num_labels, num_points, 2)
arch_points = np.reshape(arch_points, (len(unique_labels), num_points, 2))

#save as .pt
arch_points = torch.tensor(arch_points)
arch_points = arch_points[[0,2]]
print("arch points shape:", arch_points.shape)
#torch.save(arch_points, 'scarvelis_data/arch_data.pt')

sphere_points, labels, unique_labels = generate_sphere_data(num_points=num_points)
print("sphere points shape:", sphere_points.shape)
sphere_points = np.reshape(sphere_points, (len(unique_labels), num_points, 3))
sphere_points = torch.tensor(sphere_points)
sphere_points = sphere_points[[0,2]]
print("sphere points shape:", sphere_points.shape)
torch.save(sphere_points, 'scarvelis_data/sphere_data.pt')

#plot 3D sphere points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sphere_points[0, :, 0], sphere_points[0, :, 1], sphere_points[0, :, 2], c='red')
ax.scatter(sphere_points[1, :, 0], sphere_points[1, :, 1], sphere_points[1, :, 2], c='blue')
plt.savefig('sphere_data.png')
