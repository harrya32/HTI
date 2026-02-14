import os
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
import requests
import dataclasses
from typing import Iterator
import ott
from abc import ABC, abstractmethod
from lagrangian_ot.geometries import Sphere
import gdown
import scanpy as sc

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def get_bounds(name):
    if name == "scarvelis_circle":
        bounds = (-1.5, 1.5)
        xbounds = ybounds = bounds
    elif name == "scarvelis_vee":
        xbounds = (-2.5, 15)
        ybounds = (-15, 15)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "scarvelis_xpath":
        bounds = (-1.5, 1.5)
        xbounds = ybounds = bounds
    elif name == "gsb_gmm":
        xbounds = ybounds = bounds = (-20, 20)
    elif name == "scarvelis_arch":
        xbounds = (-1.5, 1.5)
        ybounds = (-0.5, 1.5)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "scarvelis_conditional_gaussian":
        bounds = (-4, 4)
        xbounds = ybounds = bounds
    elif name == "scarvelis_conditional_gaussian_complex":
        bounds = (-5,5)
        xbounds = ybounds = bounds
    elif name == "sphere_data":
        bounds = (-2,2)
        xbounds = ybounds = bounds
    elif name == "agent_data":
        bounds = (-1, 9)
        xbounds = ybounds = bounds
    elif name == "conditional_velocity":
        xbounds = (-1, 3)
        ybounds = (-1,1)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "conditional_rotation":
        xbounds = (-1, 1)
        ybounds = (-1, 1)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "warped_circle":
        xbounds = (-1.5, 1.5)
        ybounds = (-1.5, 1.5)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "conditional_circles":
        xbounds = (-2.5, 2.5)
        ybounds = (-1.5, 1.5)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "conditional_circles_normal":
        xbounds = (-2.5, 2.5)
        ybounds = (-1.5, 1.5)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "conditional_semicircles":
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
    elif name == "ett_forecasts":
        bounds = (-5, 55)
        xbounds = ybounds = bounds
    elif name == "reacher_data":
        bounds = (-1, 1)
        xbounds = ybounds = bounds
    elif name == "reacher_all_data":
        bounds = (-1, 1)
        xbounds = ybounds = bounds
    elif name == "quantile_data":
        bounds = (25, 55)
        xbounds = ybounds = bounds
    elif name == "quantile_data_long":
        bounds = (20, 58)
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


def get_samplers(geometry_str, batch_size, key):
    if "lsb" in geometry_str or "babymaze" in geometry_str:
        # return get_lsb_line_finite_samplers(key, batch_size)
        return get_lsb_line_samplers(key, batch_size)
    elif geometry_str in ["gsb_gmm", "gmm"] or geometry_str == "neural_net_metric":
        k1, k2, key = jax.random.split(key, 3)
        scale = 16.0
        variance = 1.0
        source_sampler = iter(
            GaussianMixture(
                name="simple",
                batch_size=batch_size,
                init_rng=k1,
                scale=scale,
                variance=variance,
            )
        )
        target_sampler = iter(
            GaussianMixture(
                name="circle",
                batch_size=batch_size,
                init_rng=k1,
                scale=scale,
                variance=variance,
            )
        )
        return source_sampler, target_sampler
    else:
        if "sq_euclidean" in geometry_str:
            variance = 0.5
            source_mean = jnp.array([-1.0, 0.0])
            target_mean = jnp.array([0.0, 1.0])
        elif geometry_str == "scarvelis_circle":
            variance = 0.3
            source_mean = jnp.array([-1.0, 0.0])
            target_mean = jnp.array([1.0, 0.0])
        elif geometry_str == "gsb_vneck":
            variance = 0.2
            source_mean = jnp.array([-7.0, 0.0])
            target_mean = jnp.array([7.0, 0.0])
        elif geometry_str == "gsb_stunnel":
            variance = 0.5
            source_mean = jnp.array([-11.0, -1.0])
            target_mean = jnp.array([11.0, -1.0])
        else:
            raise ValueError(f"Invalid geometry choice: {geometry_str}")

        k1, k2, key = jax.random.split(key, 3)
        source_sampler = iter(
            Gaussian(
                batch_size=batch_size, init_key=k1, mean=source_mean, variance=variance
            )
        )
        target_sampler = iter(
            Gaussian(
                batch_size=batch_size, init_key=k2, mean=target_mean, variance=variance
            )
        )
        return source_sampler, target_sampler

    raise ValueError(f"Invalid geometry choice: {geometry_str}")


def get_gsb_gmm_sampler(batch_size, key):
    source_sampler = GaussianMixture(
        name="simple", batch_size=batch_size, init_rng=jax.random.PRNGKey(0)
    )
    return train_dataloaders


def get_samplers_scarvelis(geometry_str, num_pairs_requested=None):
    """Load Scarvelis datasets, optionally selecting a subset of pairs.

    Args:
        geometry_str: Name of the Scarvelis dataset (e.g., 'scarvelis_circle').
        num_pairs_requested: The desired number of evenly spaced pairs. If None
                             or >= total available pairs, all pairs are used.

    Returns:
        A list of sampler iterators corresponding to the selected timepoints.
    """
    paths = {
        "scarvelis_circle": "data_gic_24_gaussians_radius_1_std_0p1_100_samples_closed.pt",
        "scarvelis_vee": "data_mass_split_std_1_100_samples_8_intermediate_scale_x10.pt",
        "scarvelis_xpath": "data_xpath_std_0p1_100_samples_8_intermediate.pt",
        "scarvelis_arch": "arch_data.pt",
        "scarvelis_conditional_gaussian": "conditional_gaussians_3t.pt",
        "scarvelis_conditional_gaussian_complex": "conditional_gaussians_complex.pt",
        "sphere_data": "sphere_data.pt",
        "agent_data": "agent_state_action_dataset_no_zeros.pt",
        "conditional_velocity": "conditional_velocity.pt",
        "conditional_rotation": "conditional_rotation.pt",
        "warped_circle": "warped_circle.pt",
        "conditional_circles": "conditional_circles.pt",
        "conditional_circles_normal": "conditional_circles_normal.pt",
        "conditional_semicircles": "conditional_semicircles.pt",
        "reward_weighting_data": "reward_weighting_data_0_10.pt",
        "ett_forecasts": "ett_forecasts_iclr.pt",
        "reacher_data": "reacher_data.pt",
        "reacher_all_data": "reacher_all_data.pt",
        "quantile_data" : "quantile_data_new.pt",
        "quantile_data_long": "quantile_data_long.pt",
        "reward_weighting_hinge_data": "reward_weighting_hinge_data.pt",
        "2moons_dropout": "diffusion_2moons_dropout.pt"
    }
    if geometry_str not in paths:
        raise ValueError(f"Invalid geometry choice: {geometry_str}")

    fname = SCRIPT_PATH + "/../data/" + paths[geometry_str]
    if not os.path.exists(fname):
        os.makedirs(SCRIPT_PATH + "/../data/", exist_ok=True)
        print(f"=== File {fname} does not exist. Trying to download from https://github.com/cscarv/riemannian-metric-learning-ot")
        url = 'https://github.com/cscarv/riemannian-metric-learning-ot/raw/master/data/synthetic/' + paths[geometry_str]
        r = requests.get(url, allow_redirects=True)
        with open(fname, 'wb') as f:
            f.write(r.content)

    dataset = th.load(fname, map_location="cpu", weights_only=False).detach()

    if geometry_str == "reward_weighting_data":
        #dataset = dataset[[0, 5, 10], :1000, :]
        dataset = dataset[[0,10], :1000, :]
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
    elif geometry_str == "ett_forecasts":
        #flip first and last 12 columns, to get the data and conditioning correct
        dataset = jnp.asarray(dataset)
        dataset = dataset[jnp.array([0, 4])]
        dataset_ambient = dataset[:, :, 24:]
        dataset_conditioning = dataset[:, :, :24]
        dataset = jnp.concatenate((dataset_ambient, dataset_conditioning), axis=2)
    elif geometry_str == "quantile_data":
        dataset = jnp.asarray(dataset)
        dataset = dataset[jnp.array([0, -1])]
        dataset_ambient = dataset[:, :1200, 12:]
        dataset_conditioning = dataset[:, :1200, :12]
        dataset = jnp.concatenate((dataset_ambient, dataset_conditioning), axis=2)
    elif geometry_str == "quantile_data_long":
        dataset = jnp.asarray(dataset)
        dataset = dataset[jnp.array([0, -1])]
        dataset_ambient = dataset[:, :1200, 12:]
        dataset_conditioning = dataset[:, :1200, :12]
        dataset = jnp.concatenate((dataset_ambient, dataset_conditioning), axis=2)
    elif geometry_str == "reacher_all_data":
        dataset = dataset[[2,4], :, :]
    elif geometry_str == "2moons_dropout":
         dataset = dataset[[0,5,-1], :, :]
         dataset = jnp.asarray(dataset)


    print('Dataset shape:', dataset.shape)
    dataset = jnp.asarray(dataset)
    if geometry_str == "scarvelis_xpath":
        assert dataset.shape[0] == 2
        dataset = jnp.concatenate((dataset[0], dataset[1]), axis=1)

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


@dataclasses.dataclass
class Gaussian:
    batch_size: int
    init_key: jax.random.PRNGKey
    mean: jnp.ndarray
    variance: float = 0.5

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.
        Returns:
        A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        key = self.init_key
        while True:
            key1, key = jax.random.split(key, 2)
            normal_samples = jax.random.normal(key1, [self.batch_size, 2])
            samples = self.mean + self.variance**2 * normal_samples
            yield samples


@dataclasses.dataclass
class GaussianFiniteSample:
    batch_size: int
    init_key: jax.random.PRNGKey
    mean: jnp.ndarray
    variance: float = 0.5

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.
        Returns:
        A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        key = self.init_key
        normal_samples = jax.random.normal(key, [self.batch_size, 2])
        while True:
            samples = self.mean + self.variance**2 * normal_samples
            yield samples


def get_lsb_line_samplers(key, batch_size):
    def source_generator(key):
        while True:
            k1, k2, key = jax.random.split(key, 3)
            x1 = jax.random.uniform(k1, (batch_size, 1), minval=-1.25, maxval=-1.0)
            x2 = jax.random.uniform(k2, (batch_size, 1), minval=-1.0, maxval=1.0)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    def target_generator(key):
        while True:
            k1, k2, key = jax.random.split(key, 3)
            x1 = jax.random.uniform(k1, (batch_size, 1), minval=1, maxval=1.25)
            x2 = jax.random.uniform(k2, (batch_size, 1), minval=-1.0, maxval=1.0)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    k1, k2, key = jax.random.split(key, 3)
    source_sampler = iter(source_generator(k1))
    target_sampler = iter(target_generator(k2))
    return source_sampler, target_sampler


def get_lsb_line_finite_samplers(key, batch_size):
    def source_generator(key1):
        while True:
            x1 = jax.random.uniform(key1, (batch_size, 1), minval=-1.25, maxval=-1.0)
            x2 = jax.random.uniform(key1, (batch_size, 1), minval=-1.0, maxval=1.0)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    def target_generator(key2):
        while True:
            x1 = jax.random.uniform(key2, (batch_size, 1), minval=1, maxval=1.25)
            x2 = jax.random.uniform(key2, (batch_size, 1), minval=-1.0, maxval=1.0)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    k1, k2, key = jax.random.split(key, 3)
    source_sampler = iter(source_generator(k1))
    target_sampler = iter(target_generator(k2))
    return source_sampler, target_sampler


@dataclasses.dataclass
class GaussianMixture:
    """A mixture of Gaussians.

    Args:
      name: the name specifying the centers of the mixture components:

        - ``simple`` - data clustered in one center,
        - ``circle`` - two-dimensional Gaussians arranged on a circle,
        - ``square_five`` - two-dimensional Gaussians on a square with
          one Gaussian in the center, and
        - ``square_four`` - two-dimensional Gaussians in the corners of a
          rectangle

      batch_size: batch size of the samples
      init_rng: initial PRNG key
      scale: scale of the individual Gaussian samples
      variance: the variance of the individual Gaussian samples
    """

    name: str
    batch_size: int
    init_rng: jax.random.PRNGKey(0)
    scale: float = 5.0
    variance: float = 0.5

    def __post_init__(self):
        gaussian_centers = {
            "simple": np.array([[0, 0]]),
            "circle": np.array(
                [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                    (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                    (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                    (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                ]
            ),
            "square_five": np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]),
            "square_four": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
        }
        if self.name not in gaussian_centers:
            raise ValueError(f"{self.name} is not a valid dataset for GaussianMixture")
        self.centers = gaussian_centers[self.name]

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.

        Returns:
          A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.init_rng
        while True:
            rng1, rng2, rng = jax.random.split(rng, 3)
            means = jax.random.choice(rng1, self.centers, [self.batch_size])
            normal_samples = jax.random.normal(rng2, [self.batch_size, 2])
            samples = self.scale * means + self.variance**2 * normal_samples
            yield samples


def sampler_from_data(data, batch_size=None):
    while True:
        if batch_size is None:
            yield data
        else:
            idx = np.random.choice(data.shape[0], batch_size, replace=False)
            yield data[idx]


@dataclasses.dataclass
class SphereUniform(ABC):
    manifold: Sphere
    batch_size: int
    init_rng: jax.Array

    def __iter__(self) -> Iterator[jnp.array]:
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.init_rng
        while True:
            rng1, rng2, rng = jax.random.split(rng, 3)
            xs = jax.random.normal(rng1, shape=[self.batch_size, self.manifold.D])
            samples = self.manifold.project(xs)
            yield samples


@dataclasses.dataclass
class WrappedNormal(ABC):
    manifold: Sphere
    batch_size: int
    init_rng: jax.Array
    loc: jnp.ndarray
    scale: jnp.ndarray

    def __iter__(self) -> Iterator[jnp.array]:
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.init_rng
        while True:
            rng1, rng2, rng = jax.random.split(rng, 3)
            v = self.scale * jax.random.normal(
                rng1, [self.batch_size, self.manifold.D - 1]
            )
            v = self.manifold.unsqueeze_tangent(v)
            x = self.manifold.zero_like(self.loc)
            u = self.manifold.transp(x, self.loc, v)
            z = self.manifold.exponential_map(self.loc, u)
            yield z

    def __hash__(self):
        return 0  
