from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import copy
import jax
import jax.numpy as jnp
from jax import nn
import numpy as np
import flax

@dataclass
class LagrangianPotentialBase(flax.linen.Module):
    D: int = 2
    C: int = 0

    M_bounds = (0., 0.01)
    temp_bounds = (1e-1, 1e-2)

    def setup(self):
        self.M = self.param('M', lambda key: jnp.full((1,), self.M_bounds[1]))
        self.temp = self.param('temp', lambda key: jnp.full((1,), self.temp_bounds[1]))

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @classmethod
    def get_annealed_params(cls, t):
        assert 0 <= t and t <= 1
        if 1.-t < 1e-3:
            t = 1.
        elif t < 1e-3:
            t = 0.
        else:
            t = nn.sigmoid(10.*(t-0.5))

        M_start, M_end = cls.M_bounds
        temp_start, temp_end = cls.temp_bounds
        new_M = M_start + (M_end - M_start) * t
        new_temp = temp_start + (temp_end - temp_start) * t
        new_params = {
            'M': jnp.array([new_M]),
            'temp': jnp.array([new_temp]),
        }
        return new_params

# https://github.com/take-koshizuka/NLSB/blob/main/models/potential_2d.py
class BoxPotential(LagrangianPotentialBase):
    xmin: float = -0.5
    xmax: float = 0.5
    ymin: float = -0.5
    ymax: float = 0.5

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux = (nn.sigmoid((x[0] - self.xmin) / self.temp) - \
              nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin) / self.temp) - \
              nn.sigmoid((x[1] - self.ymax) / self.temp))
        U = -Ux * Uy
        return self.M*U


class SlitPotential(LagrangianPotentialBase):
    xmin: float = -0.1
    xmax: float = 0.1
    ymin: float = -0.25
    ymax: float = 0.25
    M_bounds = (0., 1.)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux = (nn.sigmoid((x[0] - self.xmin) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax) / self.temp)) - 1.
        U = Ux * Uy
        return U

class BabyMazePotential(LagrangianPotentialBase):
    xmin1: float = -0.5
    xmax1: float = -0.3
    ymin1: float = -1.99
    ymax1: float = -0.15
    xmin2: float = 0.3
    xmax2: float = 0.5
    ymin2: float = 0.15
    ymax2: float = 1.99
    M_bounds = (0., 10.)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux1 = (nn.sigmoid((x[0] - self.xmin1) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax1) / self.temp))
        Ux2 = (nn.sigmoid((x[0] - self.xmin2) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax2) / self.temp))

        Uy1 = (nn.sigmoid((x[1] - self.ymin1) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax1) / self.temp)) - 1.

        Uy2 = (nn.sigmoid((x[1] - self.ymin2) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax2) / self.temp)) - 1.
        U = Ux1 * Uy1 + Ux2 * Uy2
        return self.M*U

@dataclass
class WellPotential(LagrangianPotentialBase):
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        U = -jnp.sum(x**2)
        return self.M*U

@dataclass
class HillPotential(LagrangianPotentialBase):
    M_bounds = (0., 0.05)
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        U = -jnp.exp(-jnp.sum(x**2))
        return self.M*U


@dataclass
class GSB_GMM_Potential(LagrangianPotentialBase):
    centers = jnp.array([[6,6], [6,-6], [-6,-6]])
    radius = 1.5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        V = 0.
        for i in range(self.centers.shape[0]):
            dist = jnp.linalg.norm(x - self.centers[i])
            V -= self.M * nn.sigmoid((self.radius - dist) / self.temp)
        return V


@dataclass
class GSB_VNeck_Potential(LagrangianPotentialBase):
    c_sq = 0.36
    coef = 5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        xs_sq = x*x
        d = self.coef * xs_sq[0] - xs_sq[1]

        return - self.M * nn.sigmoid((-self.c_sq - d) / self.temp)


@dataclass
class GSB_STunnel_Potential(LagrangianPotentialBase):
    a, b, c = 20, 1, 90
    centers = [[5,6], [-5,-6]]
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        V = 0.0
        d = self.a*(x[0]-self.centers[0][0])**2 + \
            self.b*(x[1]-self.centers[0][1])**2
        V -= self.M * nn.sigmoid((self.c - d) / self.temp)

        d = self.a*(x[0]-self.centers[1][0])**2 + \
            self.b*(x[1]-self.centers[1][1])**2
        V -= self.M * nn.sigmoid((self.c - d) / self.temp)

        return V
    
    
@dataclass
class DoubleCirclePotential(LagrangianPotentialBase):
    center_left: jnp.ndarray = jnp.array([-1.0, 0.0])
    center_right: jnp.ndarray = jnp.array([1.0, 0.0])
    radius: float = 1.0
    offset: float = 0.1
    M_bounds = (0.0, 1.0)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D + self.C, "Input x must have shape (D + C,)"

        x_ambient = x[:self.D]
        condition = x[self.D].astype(int)

        center = jnp.where(
            (condition == 0) | (condition == 1),
            self.center_left,
            self.center_right
        )

        dist = jnp.linalg.norm(x_ambient - center)

        U = nn.sigmoid((dist - (self.radius - self.offset)) / 0.1) - \
            nn.sigmoid((dist - (self.radius + self.offset)) / 0.1)

        return self.M * U
    

@dataclass
class DensityPotentialCircles(LagrangianPotentialBase):
    """
    U(x) = -lambda_repel / log(p̂(x) + epsilon),     p̂(x) = 1/N ∑_i ϕ((x - samples[i])/bandwidth)
    """
    samples: jnp.ndarray = None
    bandwidth: float = 1.0
    lambda_repel: float = 0.01
    epsilon: float = 1e-8

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D + self.C, "Input x must have shape (D + C,)"
        assert self.samples is not None, "Samples must be provided before calling the potential."
        assert self.samples.ndim == 2 and self.samples.shape[1] == self.D + self.C, "Samples must have shape (N, D + C)."

        x_ambient = x[:self.D]
        x_cond = x[self.D:]
        all_samples_ambient = self.samples[:, :self.D]

        if self.C > 0:
            mask = jnp.all(self.samples[:, self.D:] == x_cond, axis=1)
            num_cond_samples = jnp.sum(mask, dtype=jnp.float32)
        else:
            mask = jnp.ones(self.samples.shape[0], dtype=bool)
            num_cond_samples = jnp.array(self.samples.shape[0], dtype=jnp.float32)

        safe_num_cond_samples_denom = jnp.maximum(num_cond_samples, 1.0)
        diffs = (all_samples_ambient - x_ambient[None, :]) / self.bandwidth
        sq_norms = jnp.sum(diffs**2, axis=1)
        kernel_norm_factor = (2 * jnp.pi * self.bandwidth**2)**(-self.D / 2.0)
        kernel_vals = kernel_norm_factor * jnp.exp(-0.5 * sq_norms)
        masked_kernel_vals = jnp.where(mask, kernel_vals, 0.0)
        density_p_hat = jnp.sum(masked_kernel_vals) / safe_num_cond_samples_denom
        potential = self.lambda_repel * jnp.log(density_p_hat + self.epsilon)
        return potential

@dataclass
class DensityPotential(LagrangianPotentialBase):
    """
    U(y | x) = alpha * log p̂(y | x),     p̂(y | x) = 1/N ∑_i ϕ((y - samples[i])/bandwidth)
    If C > 0 (conditional dimensions exist):
      p̂_eff(x) is an estimate of the conditional density p(x_ambient | x_cond),
      calculated using a Nadaraya-Watson kernel regression (a form of conditional KDE).
    If C = 0 (no conditional dimensions):
      p̂_eff(x) is an estimate of the marginal density p(x_ambient), 
      calculated using standard Kernel Density Estimation.
    """
    samples: jnp.ndarray = None
    bandwidth: float = 1.0 
    conditional_bandwidth: float = 1.0 
    lambda_repel: float = 0.01
    epsilon: float = 1e-8 
    denominator_epsilon: float = 1e-8

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D + self.C, f"Input x must have shape (D+C,), got {x.shape}"
        assert self.samples is not None, "Samples must be provided before calling the potential."
        assert self.samples.ndim == 2 and self.samples.shape[1] == self.D + self.C, f"Samples must have shape (N, D+C,), got {self.samples.shape}"

        x_ambient = x[:self.D]
        all_samples_ambient = self.samples[:, :self.D]

        # Gaussian kernel values for ambient dimensions
        # K_h_a(x_a - s_a_i) = (2*pi*h_a^2)^(-D/2) * exp(-0.5 * ||(x_a-s_a_i)/h_a||^2 )
        diffs_ambient = (all_samples_ambient - x_ambient[None, :]) / self.bandwidth
        sq_norms_ambient = jnp.sum(diffs_ambient**2, axis=1)
        ambient_kernel_norm_factor = (2 * jnp.pi * self.bandwidth**2)**(-self.D / 2.0)
        ambient_kernel_vals = ambient_kernel_norm_factor * jnp.exp(-0.5 * sq_norms_ambient)

        if self.C > 0:
            x_cond = x[self.D:]
            all_samples_cond = self.samples[:, self.D:]

            # Gaussian kernel values for conditional dimensions
            # K_h_c(x_c - s_c_i) = (2*pi*h_c^2)^(-C/2) * exp(-0.5 * ||(x_c-s_c_i)/h_c||^2 )
            diffs_cond = (all_samples_cond - x_cond[None, :]) / self.conditional_bandwidth
            sq_norms_cond = jnp.sum(diffs_cond**2, axis=1)
            cond_kernel_norm_factor = (2 * jnp.pi * self.conditional_bandwidth**2)**(-self.C / 2.0)
            cond_kernel_weights = cond_kernel_norm_factor * jnp.exp(-0.5 * sq_norms_cond)
            
            # Estimate p̂(x_ambient | x_cond) using Nadaraya-Watson estimator:
            # sum_i K_h_a(x_a - s_a_i) * K_h_c(x_c - s_c_i) / sum_j K_h_c(x_c - s_c_j)
            numerator = jnp.sum(ambient_kernel_vals * cond_kernel_weights)
            denominator = jnp.sum(cond_kernel_weights)
            
            density_p_hat = numerator / jnp.maximum(denominator, self.denominator_epsilon)

        else: # C == 0, standard KDE for p̂(x_ambient)
            # p̂(x_ambient) = (1/N) * sum_i K_h_a(x_a - s_a_i)
            density_p_hat = jnp.mean(ambient_kernel_vals)

        potential = self.lambda_repel * jnp.log(density_p_hat + self.epsilon)
        return potential