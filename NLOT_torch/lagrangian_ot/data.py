# Copyright (c) Meta Platforms, Inc. and affiliates

import os

import torch as th
import numpy as np

import requests

import dataclasses
from typing import Iterator, Optional

import ott

from abc import ABC, abstractmethod

from lagrangian_ot.geometries import Sphere
import gdown

import scanpy as sc

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def get_bounds(name):
    # This could be cleaned up and better-merged with the bounds in geometries.
    if name == "scarvelis_circle":
        bounds = (-1.5, 1.5)
        xbounds = ybounds = bounds
    elif name == "scarvelis_vee":
        xbounds = (-2.5, 15)
        ybounds = (-15, 15)
        bounds = (
            th.tensor((xbounds[0], ybounds[0])),
            th.tensor((xbounds[1], ybounds[1])),
        )
    elif name == "scarvelis_xpath":
        bounds = (-1.5, 1.5)
        xbounds = ybounds = bounds
    elif name == "gsb_gmm":
        xbounds = ybounds = bounds = (-20, 20)
    else:
        raise ValueError(f"Invalid data choice: {name}")

    return bounds, xbounds, ybounds


def get_samplers(geometry_str, batch_size, seed: Optional[int] = None):
    if "lsb" in geometry_str or "babymaze" in geometry_str:
        return get_lsb_line_samplers(batch_size, seed)
    elif geometry_str in ["gsb_gmm", "gmm"] or geometry_str == "neural_net_metric":
        # Create separate seeds for source and target if a seed is provided
        seed1 = seed
        seed2 = seed + 1 if seed is not None else None
        scale = 16.0
        variance = 1.0
        source_sampler = iter(
            GaussianMixture(
                name="simple",
                batch_size=batch_size,
                seed=seed1,
                scale=scale,
                variance=variance,
            )
        )
        target_sampler = iter(
            GaussianMixture(
                name="circle",
                batch_size=batch_size,
                seed=seed2,
                scale=scale,
                variance=variance,
            )
        )
        return source_sampler, target_sampler
    else:
        # Create separate seeds for source and target if a seed is provided
        seed1 = seed
        seed2 = seed + 1 if seed is not None else None
        if "sq_euclidean" in geometry_str:
            variance = 0.5
            source_mean = th.tensor([-1.0, 0.0])
            target_mean = th.tensor([0.0, 1.0])
        elif geometry_str == "scarvelis_circle":
            variance = 0.3
            source_mean = th.tensor([-1.0, 0.0])
            target_mean = th.tensor([1.0, 0.0])
        elif geometry_str == "gsb_vneck":
            variance = 0.2
            source_mean = th.tensor([-7.0, 0.0])
            target_mean = th.tensor([7.0, 0.0])
        elif geometry_str == "gsb_stunnel":
            variance = 0.5
            source_mean = th.tensor([-11.0, -1.0])
            target_mean = th.tensor([11.0, -1.0])
        else:
            raise ValueError(f"Invalid geometry choice: {geometry_str}")

        source_sampler = iter(
            Gaussian(
                batch_size=batch_size, seed=seed1, mean=source_mean, variance=variance
            )
        )
        target_sampler = iter(
            Gaussian(
                batch_size=batch_size, seed=seed2, mean=target_mean, variance=variance
            )
        )
        return source_sampler, target_sampler

    raise ValueError(f"Invalid geometry choice: {geometry_str}")


def get_gsb_gmm_sampler(batch_size, seed: Optional[int] = None):
    source_sampler = GaussianMixture(
        name="simple", batch_size=batch_size, seed=seed
    )
    return source_sampler


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
    }
    if geometry_str not in paths:
        raise ValueError(f"Invalid geometry choice: {geometry_str}")

    fname = SCRIPT_PATH + "/../scarvelis_data/" + paths[geometry_str]
    if not os.path.exists(fname):
        os.makedirs(SCRIPT_PATH + "/../scarvelis_data/", exist_ok=True)
        print(f"=== File {fname} does not exist. Trying to download from https://github.com/cscarv/riemannian-metric-learning-ot")
        url = 'https://github.com/cscarv/riemannian-metric-learning-ot/raw/master/data/synthetic/' + paths[geometry_str]
        r = requests.get(url, allow_redirects=True)
        with open(fname, 'wb') as f:
            f.write(r.content)

    dataset = th.load(fname, map_location="cpu").detach()
    dataset = th.as_tensor(dataset)
    if geometry_str == "scarvelis_xpath":
        assert dataset.shape[0] == 2
        dataset = th.cat((dataset[0], dataset[1]), dim=1)

    # --- Pair selection logic --- START
    total_timepoints = dataset.shape[0]
    total_pairs = total_timepoints - 1

    if num_pairs_requested is not None and num_pairs_requested > 0 and num_pairs_requested < total_pairs:
        print(f"Selecting {num_pairs_requested} pairs evenly spaced from {total_pairs} total pairs.")
        # Calculate indices for k+1 timepoints to get k pairs
        num_timepoints_to_select = num_pairs_requested + 1
        selected_indices_float = np.linspace(0, total_timepoints - 1, num=num_timepoints_to_select)
        selected_timepoint_indices = np.round(selected_indices_float).astype(int)
        selected_timepoint_indices = np.unique(selected_timepoint_indices) # Ensure uniqueness after rounding
        
        # Adjust if rounding resulted in fewer timepoints than needed
        if len(selected_timepoint_indices) < num_timepoints_to_select:
             # This is a fallback, might not be perfectly even spacing but ensures correct number
             selected_timepoint_indices = np.linspace(0, total_timepoints - 1, num=num_timepoints_to_select).astype(int)
             selected_timepoint_indices = np.unique(selected_timepoint_indices)
             # If still not enough due to extreme distribution, we might need a more robust selection
             # but for typical cases, this should suffice.
             print(f"Adjusted selected timepoint indices: {selected_timepoint_indices}")


        print(f"Using timepoint indices: {selected_timepoint_indices}")
        dataset = dataset[selected_timepoint_indices]
    else:
        print(f"Using all {total_pairs} available pairs.")
        selected_timepoint_indices = np.arange(total_timepoints)
    # --- Pair selection logic --- END


    samplers = [
        iter(sampler_from_data(dataset[t])) for t in range(dataset.shape[0])
    ]

    # only keep every 2nd sample
    #samplers = [samplers[i] for i in range(0, len(samplers), 2)]
    return samplers


@dataclasses.dataclass
class Gaussian:
    batch_size: int
    seed: Optional[int] = None # Replaced init_key
    mean: th.Tensor
    variance: float = 0.5

    def __iter__(self) -> Iterator[th.Tensor]:
        """Random sample generator from Gaussian mixture.
        Returns:
        A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[th.Tensor]:
        gen = th.Generator()
        if self.seed is not None:
             gen.manual_seed(self.seed)
        sqrt_variance = th.sqrt(th.tensor(self.variance))
        while True:
             yield self.mean + th.randn(self.batch_size, 2, generator=gen) * sqrt_variance


@dataclasses.dataclass
class GaussianFiniteSample:
    batch_size: int
    seed: Optional[int] = None # Replaced init_key
    mean: th.Tensor
    variance: float = 0.5
    samples: th.Tensor

    def __iter__(self) -> Iterator[th.Tensor]:
        """Random sample generator from Gaussian mixture.
        Returns:
        A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[th.Tensor]:
        gen = th.Generator()
        if self.seed is not None:
             gen.manual_seed(self.seed)
        # Generate samples once
        self.samples = self.mean + th.randn(self.batch_size, 2, generator=gen) * th.sqrt(th.tensor(self.variance))
        while True:
             yield self.samples


def get_lsb_line_samplers(batch_size, seed: Optional[int] = None):
    def source_generator(seed):
        gen = th.Generator()
        if seed is not None: gen.manual_seed(seed)
        while True:
            # Uniform between -0.5, 0.5
            xs = th.rand(batch_size, 1, generator=gen) - 0.5
            ys = th.zeros_like(xs)
            yield th.cat([xs, ys], dim=-1)

    def target_generator(seed):
        gen = th.Generator()
        if seed is not None: gen.manual_seed(seed)
        while True:
            # Uniform between -0.5, 0.5
            ys = th.rand(batch_size, 1, generator=gen) - 0.5
            xs = th.zeros_like(ys)
            yield th.cat([xs, ys], dim=-1)

    # Create separate seeds for source and target if a seed is provided
    seed1 = seed
    seed2 = seed + 1 if seed is not None else None
    return iter(source_generator(seed1)), iter(target_generator(seed2))


def get_lsb_line_finite_samplers(batch_size, seed: Optional[int] = None):
    def source_generator(seed1):
        gen = th.Generator()
        if seed1 is not None: gen.manual_seed(seed1)
        xs = th.rand(batch_size, 1, generator=gen) - 0.5
        ys = th.zeros_like(xs)
        samples = th.cat([xs, ys], dim=-1)
        while True:
            yield samples

    def target_generator(seed2):
        gen = th.Generator()
        if seed2 is not None: gen.manual_seed(seed2)
        ys = th.rand(batch_size, 1, generator=gen) - 0.5
        xs = th.zeros_like(ys)
        samples = th.cat([xs, ys], dim=-1)
        while True:
            yield samples

    # Create separate seeds for source and target if a seed is provided
    seed1 = seed
    seed2 = seed + 1 if seed is not None else None
    return iter(source_generator(seed1)), iter(target_generator(seed2))


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
      seed: initial PRNG key
      scale: scale of the individual Gaussian samples
      variance: the variance of the individual Gaussian samples
    """

    name: str
    batch_size: int
    seed: Optional[int] = None # Replaced init_rng
    scale: float = 5.0
    variance: float = 0.5

    def __post_init__(self):
        # Using numpy arrays initially as torch doesn't directly support float lists like this
        centers_np = {
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
            "square_four": np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]), # Corrected square_four points
        }
        if self.name not in centers_np:
            raise ValueError(f"{self.name} is not a valid dataset for GaussianMixture")
        # Convert numpy array to torch tensor
        self.centers = th.tensor(centers_np[self.name], dtype=th.float32) * self.scale

    def __iter__(self) -> Iterator[th.Tensor]:
        """Random sample generator from Gaussian mixture.

        Returns:
          A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[th.Tensor]:
        gen = th.Generator()
        if self.seed is not None:
             gen.manual_seed(self.seed)
        num_components = self.centers.shape[0]
        sqrt_variance = th.sqrt(th.tensor(self.variance))
        while True:
             noise = th.randn(self.batch_size, 2, generator=gen) * sqrt_variance
             # Use torch.randint for component selection
             center_ix = th.randint(0, num_components, (self.batch_size,), generator=gen)
             yield self.centers[center_ix] + noise


def sampler_from_data(data, batch_size=None):
    while True:
        if batch_size is None:
            yield data
        else:
            idx = np.random.choice(data.shape[0], batch_size, replace=False)
            yield data[idx]
            # assert False # sample from data


@dataclasses.dataclass
class SphereUniform(ABC):
    manifold: Sphere
    batch_size: int
    seed: Optional[int] = None # Replaced init_rng

    def __iter__(self) -> Iterator[th.Tensor]:
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[th.Tensor]:
        gen = th.Generator()
        if self.seed is not None:
             gen.manual_seed(self.seed)
        while True:
             # Standard method for sampling uniformly on a sphere:
             # Sample from a standard Gaussian and normalize.
             # Assumes manifold.dim exists and is the intrinsic dimension (D-1)
             samples = th.randn(self.batch_size, self.manifold.dim + 1, generator=gen)
             samples /= th.linalg.norm(samples, dim=-1, keepdim=True)
             yield samples


@dataclasses.dataclass
class WrappedNormal(ABC):
    manifold: Sphere
    batch_size: int
    seed: Optional[int] = None # Replaced init_rng
    loc: th.Tensor
    scale: th.Tensor # Scale typically represents std dev in tangent space

    def __iter__(self) -> Iterator[th.Tensor]:
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[th.Tensor]:
        gen = th.Generator()
        if self.seed is not None:
             gen.manual_seed(self.seed)
        while True:
            # This requires the manifold.random_riemannian_normal method to be
            # implemented in the PyTorch version of the Sphere class.
            # Placeholder: Return uniform samples on the sphere, as the correct
            # implementation is complex and depends on the specific manifold methods.
            samples = th.randn(self.batch_size, self.manifold.dim + 1, generator=gen)
            samples /= th.linalg.norm(samples, dim=-1, keepdim=True)
            yield samples

    def __hash__(self):
        # Convert tensors to tuples for hashing if they are not hashable directly
        loc_hash = tuple(self.loc.flatten().tolist()) if isinstance(self.loc, th.Tensor) else self.loc
        scale_hash = tuple(self.scale.flatten().tolist()) if isinstance(self.scale, th.Tensor) else self.scale
        return hash((self.manifold, self.batch_size, loc_hash, scale_hash))
