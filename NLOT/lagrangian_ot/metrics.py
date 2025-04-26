# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn

import copy

from typing import Tuple, Optional

from dataclasses import dataclass

from abc import ABC, abstractmethod


plot_cache = {}

@dataclass
class MetricBase(nn.Module):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


@dataclass
class EuclideanMetric(MetricBase):
    def __call__(self, x):
        assert x.ndim == 1
        D = x.shape[0]
        return jnp.eye(D)


@dataclass
class ScarvelisMetric(MetricBase, ABC):
    div_eps: float = 1e-6
    metric_eps: float = 1e-3

    def __call__(self, x):
        assert x.ndim == 1
        D = x.shape[0]
        v_vals = self.v(x)
        return jnp.eye(D) - (1-self.metric_eps)*jnp.outer(v_vals, v_vals)

    @abstractmethod
    def v(self, x):
        raise NotImplementedError


@dataclass
class CircleMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        norm = jnp.clip(jnp.linalg.norm(x), a_min=self.div_eps)
        return jnp.array([-x[1], x[0]]) / norm

@dataclass
class VeeMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        sign_y = jnp.sign(x[1])
        return jnp.array([1./jnp.sqrt(2), sign_y/jnp.sqrt(2)])

@dataclass
class XMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        a = 1.25 * jax.nn.tanh(jax.nn.relu(x[0]*x[1]))
        b = -1.25 * jax.nn.tanh(jax.nn.relu(-x[0]*x[1]))
        v1 = jnp.array([1./jnp.sqrt(2), 1./jnp.sqrt(2)])
        v2 = jnp.array([1./jnp.sqrt(2), -1./jnp.sqrt(2)])
        return (a*v1 + b*v2) / 1.25

@dataclass
class LANDMetric(MetricBase):
    D: int = 2
    gamma: float = 1.0  # Width parameter for the weighting kernel
    rho: float = 1e-3  # Regularization parameter
    alpha: float = 1.0  # Power for the metric
    samples: Optional[jnp.ndarray] = None  # Sample points to compute the metric
    
    def setup(self):
        # Validate that samples are provided
        if self.samples is None:
            raise ValueError("The 'samples' parameter must be provided for LANDMetric")
    
    def _weighting_function(self, x, samples):
        # Compute weights based on distance to samples
        pairwise_sq_diff = (x - samples) ** 2
        pairwise_sq_dist = jnp.sum(pairwise_sq_diff, axis=-1)
        weights = jnp.exp(-pairwise_sq_dist / (2 * self.gamma**2))
        return weights
    
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        
        # Compute weights for each sample based on distance to x
        weights = self._weighting_function(x, self.samples)
        
        # Compute differences between x and samples
        differences = self.samples - x
        squared_differences = differences**2
        
        # Compute the weighted sum of squared differences for each dimension
        M_diag = jnp.zeros(self.D)
        for d in range(self.D):
            M_diag = M_diag.at[d].set(
                jnp.sum(weights * squared_differences[:, d]) + self.rho
            )
        
        # Apply the alpha power (optional parameter for controlling metric strength)
        M_diag = M_diag ** self.alpha
        
        # Convert diagonal to full matrix
        M = jnp.diag(M_diag)
        
        return M

@dataclass
class NeuralNetMetric(MetricBase):
    D = 2

    def setup(self):
        assert self.D == 2
        self.net = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(2)
        ])

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        theta = jnp.arctan2(*self.net(x).squeeze())
        R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
        Q = jnp.array([[1., 0.],
                       [0., 0.1]])

        A = R.T @ Q @ R
        return A


@dataclass
class NeuralNetMetric_direct(MetricBase):
    D = 2

    def setup(self):
        assert self.D == 2
        self.net = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(4)
        ])

    def __call__(self, x, eta = 1e-3):
        assert x.ndim == 1 and x.shape[0] == self.D
        
        nn_out = self.net(x)
        a = nn_out[0]
        b = nn_out[1]
        c = nn_out[2]
        d = nn_out[3]

        Q = jnp.array([[a, b], 
                       [c, d]])

        A = Q.T @ Q + eta * jnp.eye(2)

        return A

@dataclass
class NeuralNetMetricEig(MetricBase):
    D: int = 2
    min_eigenvalue: float = 0.1
    max_eigenvalue: float = 10.0
    
    def setup(self):
        # Network outputs eigenvalues and parameters for rotation
        # For D dimensional space: D eigenvalues + D(D-1)/2 rotation parameters
        # For D=2, we output eigenvalues and a 2D vector for arctan2
        if self.D == 2:
            output_size = self.D + 2  # D eigenvalues + 2 for rotation via arctan2
        else:
            output_size = self.D + (self.D * (self.D - 1)) // 2
        
        self.net = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(output_size)
        ])
    
    def __call__(self, x, eta=1e-3):
        assert x.ndim == 1 and x.shape[0] == self.D
        
        nn_out = self.net(x)
        
        # Extract eigenvalues (first D outputs)
        # Apply sigmoid + scaling to keep eigenvalues in reasonable range
        raw_eigenvalues = nn_out[:self.D]
        eigenvalues = self.min_eigenvalue + jax.nn.sigmoid(raw_eigenvalues) * (self.max_eigenvalue - self.min_eigenvalue)
        
        # Create rotation matrix from remaining parameters
        rotation = self._create_rotation_matrix(nn_out[self.D:])
        
        # Create diagonal matrix from eigenvalues
        diagonal = jnp.diag(eigenvalues)
        
        # Compute metric tensor: R^T D R
        A = rotation.T @ diagonal @ rotation
        
        return A
    
    def _create_rotation_matrix(self, params):
        """
        Create a rotation matrix from parameters.
        For D=2: uses arctan2 from 2D vector like original implementation
        For D=3: 3 angle parameters (Euler angles)
        For D>3: D(D-1)/2 parameters for generalized rotation
        """
        if self.D == 2:
            # Use arctan2 like the original implementation
            theta = jnp.arctan2(params[1], params[0])
            rotation = jnp.array([
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)]
            ])
            return rotation
        else:
            # For higher dimensions, use a series of Givens rotations
            # This is one approach to parameterize SO(n) with D(D-1)/2 parameters
            rotation = jnp.eye(self.D)
            param_idx = 0
            
            for i in range(self.D):
                for j in range(i+1, self.D):
                    # Create a Givens rotation in the (i,j) plane
                    angle = params[param_idx]
                    param_idx += 1
                    
                    # Create rotation matrix for this plane
                    givens = jnp.eye(self.D)
                    givens = givens.at[i, i].set(jnp.cos(angle))
                    givens = givens.at[j, j].set(jnp.cos(angle))
                    givens = givens.at[i, j].set(-jnp.sin(angle))
                    givens = givens.at[j, i].set(jnp.sin(angle))
                    
                    # Apply this rotation
                    rotation = rotation @ givens
                    
            return rotation
        

