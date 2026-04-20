import os
import jax
import jax.numpy as jnp
import numpy as np
import torch as th

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def get_bounds(name):
    if name == "conditional_semicircles":
        xbounds = (-2.5, 2.5)
        ybounds = (-1.5, 1.5)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "reward_weighting_data":
        xbounds = (0, 10)
        ybounds = (0, 8)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "reacher_data":
        bounds = (-1, 1)
        xbounds = ybounds = bounds
    elif name == "quantile_data":
        bounds = (25, 55)
        xbounds = ybounds = bounds
    elif name == "reward_weighting_hinge_data":
        xbounds = (0, 10)
        ybounds = (0, 8)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "2moons_dropout":
        bounds = (-4, 4)
        xbounds = ybounds = bounds
    else:
        raise ValueError(f"Invalid data choice: {name}")

    return bounds, xbounds, ybounds

def get_samplers(geometry_str, num_pairs_requested=None):
    """Load datasets, optionally selecting a subset of pairs.

    Args:
        geometry_str: Name of the dataset (e.g., 'conditional_semicircles').
        num_pairs_requested: The desired number of evenly spaced pairs. If None
                             or >= total available pairs, all pairs are used.

    Returns:
        A list of sampler iterators corresponding to the selected timepoints.
    """
    paths = {
        "conditional_semicircles": "conditional_semicircles.pt",
        "reward_weighting_data": "reward_weighting_data_0_10.pt",
        "reacher_data": "reacher_data.pt",
        "quantile_data" : "quantile_data_new.pt",
        "reward_weighting_hinge_data": "reward_weighting_hinge_data.pt",
        "2moons_dropout": "diffusion_2moons_dropout.pt"
    }
    if geometry_str not in paths:
        raise ValueError(f"Invalid geometry choice: {geometry_str}")

    fname = SCRIPT_PATH + "/../data/" + paths[geometry_str]

    dataset = th.load(fname, map_location="cpu", weights_only=False).detach()

    if geometry_str == "reward_weighting_data":
        dataset = dataset[[0, 5, 10], :1000, :]
        #dataset = dataset[[0,10], :1000, :]
        dataset = jnp.asarray(dataset)
        #add tiny amount of noise for spline stability
        noise = 0.001 * jax.random.normal(jax.random.PRNGKey(0), dataset[:, :, :2].shape)
        dataset = dataset.at[:, :, :2].set(dataset[:, :, :2] + noise)
        dataset = dataset.at[:, :, 0].set(jnp.clip(dataset[:, :, 0], 0, 10))
        dataset = dataset.at[:, :, 1].set(jnp.clip(dataset[:, :, 1], 0, 8))
    elif geometry_str == "reward_weighting_hinge_data":
        dataset = dataset[[0, 1, 2], :1000, :]
        dataset = jnp.asarray(dataset)
        noise = 0.001 * jax.random.normal(jax.random.PRNGKey(0), dataset[:, :, :2].shape)
        dataset = dataset.at[:, :, :2].set(dataset[:, :, :2] + noise)
        dataset = dataset.at[:, :, 0].set(jnp.clip(dataset[:, :, 0], 0, 10))
        dataset = dataset.at[:, :, 1].set(jnp.clip(dataset[:, :, 1], 0, 8))
    elif geometry_str == "quantile_data":
        dataset = jnp.asarray(dataset)
        dataset = dataset[jnp.array([0, -1])]
        dataset_ambient = dataset[:, :1200, 12:]
        dataset_conditioning = dataset[:, :1200, :12]
        dataset = jnp.concatenate((dataset_ambient, dataset_conditioning), axis=2)
    elif geometry_str == "2moons_dropout":
         dataset = dataset[[0,5,-1], :, :]
         dataset = jnp.asarray(dataset)


    print('Dataset shape:', dataset.shape)
    dataset = jnp.asarray(dataset)


    total_timepoints = dataset.shape[0]
    total_pairs = total_timepoints - 1

    if num_pairs_requested is not None and num_pairs_requested > 0 and num_pairs_requested < total_pairs:
        print(f"Selecting {num_pairs_requested} pairs evenly spaced from {total_pairs} total pairs.")
        num_timepoints_to_select = num_pairs_requested + 1
        selected_indices_float = np.linspace(0, total_timepoints - 1, num=num_timepoints_to_select)
        selected_timepoint_indices = np.round(selected_indices_float).astype(int)
        selected_timepoint_indices = np.unique(selected_timepoint_indices)
        
        if len(selected_timepoint_indices) < num_timepoints_to_select:
             selected_timepoint_indices = np.linspace(0, total_timepoints - 1, num=num_timepoints_to_select).astype(int)
             selected_timepoint_indices = np.unique(selected_timepoint_indices)
             print(f"Adjusted selected timepoint indices: {selected_timepoint_indices}")


        print(f"Using timepoint indices: {selected_timepoint_indices}")
        dataset = dataset[selected_timepoint_indices]
    else:
        print(f"Using all {total_pairs} available pairs.")
        selected_timepoint_indices = np.arange(total_timepoints)


    samplers = [
        iter(sampler_from_data(dataset[t])) for t in range(dataset.shape[0])
    ]

    return samplers

def sampler_from_data(data, batch_size=None):
    while True:
        if batch_size is None:
            yield data
        else:
            idx = np.random.choice(data.shape[0], batch_size, replace=False)
            yield data[idx]