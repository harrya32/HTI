# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import torch as th
import torch.nn as nn
import functools

from dataclasses import dataclass

from abc import ABC, abstractmethod

import copy
from typing import Callable, Optional, Tuple, Dict


from enum import Enum

from lagrangian_ot import (
    metrics,
    geodesics,
    lagrangian_potentials,
    splines,
    spline_amortizer,
)


class DistanceModes(Enum):
    GEODESIC = "geodesic"
    SQUARED_GEODESIC = "sq_geodesic"
    LAGRANGIAN = "lagrangian"


def get(name, geometry_kwargs):
    if name == "sq_euclidean":
        return SqEuclidean()
    elif name == "gmm":
        return SqEuclidean(bounds=(-20, 20))
    elif name == "sq_euclidean_manifold":
        return MetricManifold(
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.EuclideanMetric,
            **geometry_kwargs,
        )
    elif name == "scarvelis_circle":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.CircleMetric,
            **geometry_kwargs,
        )
    elif name == "scarvelis_vee":
        xbounds = (-2.5, 15)
        ybounds = (-15, 15)
        bounds = (
            th.tensor((xbounds[0], ybounds[0]), dtype=th.float32),
            th.tensor((xbounds[1], ybounds[1]), dtype=th.float32),
        )
        return MetricManifold(
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.VeeMetric,
            xbounds=xbounds,
            ybounds=ybounds,
            bounds=bounds,
            **geometry_kwargs,
        )
    elif name == "scarvelis_xpath":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.XMetric,
            **geometry_kwargs,
        )
    elif name == "lsb_box":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.BoxPotential,
            **geometry_kwargs,
        )
    elif name == "lsb_slit":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.SlitPotential,
            **geometry_kwargs,
        )
    elif name == "lsb_hill":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.HillPotential,
            **geometry_kwargs,
        )
    elif name == "lsb_well":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.WellPotential,
            **geometry_kwargs,
        )
    elif name == "gsb_gmm":
        return MetricManifold(
            bounds=(-20, 20),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.GSB_GMM_Potential,
            **geometry_kwargs,
        )
    elif name == "gsb_vneck":
        xbounds = (-10, 10)
        ybounds = (-8, 8)
        bounds = (
            th.tensor((xbounds[0], ybounds[0]), dtype=th.float32),
            th.tensor((xbounds[1], ybounds[1]), dtype=th.float32),
        )
        return MetricManifold(
            xbounds=xbounds,
            ybounds=ybounds,
            bounds=bounds,
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.GSB_VNeck_Potential,
            **geometry_kwargs,
        )
    elif name == "gsb_stunnel":
        xbounds = (-15, 15)
        ybounds = (-7.5, 7.5)
        bounds = (
            th.tensor((xbounds[0], ybounds[0]), dtype=th.float32),
            th.tensor((xbounds[1], ybounds[1]), dtype=th.float32),
        )
        return MetricManifold(
            xbounds=xbounds,
            ybounds=ybounds,
            bounds=bounds,
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.GSB_STunnel_Potential,
            **geometry_kwargs,
        )
    elif name == "babymaze":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.BabyMazePotential,
            **geometry_kwargs,
        )
    elif name == "neural_net_metric":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.NeuralNetMetric,
            **geometry_kwargs,
        )
    else:
        raise ValueError(f"Unknown geometry: {name}")


@dataclass
class GeometryBase(ABC):
    D: int = 2  # dimension of the ambient space
    bounds: Tuple = (-2, 2)  # bounds of the measures

    # for 2d geometries
    xbounds: Optional[Tuple] = None
    ybounds: Optional[Tuple] = None

    def __post_init__(self):
        # Set default bounds if not provided
        if self.xbounds is None:
            self.xbounds = self.bounds
        if self.ybounds is None:
            self.ybounds = self.bounds

    @abstractmethod
    def cost(self, x, y):
        pass

    @abstractmethod
    def path(self, x, y, num_points=20):
        pass

    @abstractmethod
    def project(self, x):
        pass

    def add_plot_background(self, ax, xlims, ylims=None, **kwargs):
        pass


eps = 1e-5
divsin = lambda x: x / th.sin(x)
sindiv = lambda x: th.sin(x) / (x + eps)
divsinh = lambda x: x / th.sinh(x)
sinhdiv = lambda x: th.sinh(x) / (x + eps)


class Sphere(GeometryBase):
    jitter: float = 1e-2

    def exponential_map(self, x, v):
        v_norm = th.linalg.norm(v, dim=-1, keepdim=True)
        return x * th.cos(v_norm) + v * sindiv(v_norm)

    def log(self, x, y):
        xy = (x * y).sum(dim=-1, keepdim=True)
        xy = th.clamp(xy, a_min=-1 + 1e-6, a_max=1 - 1e-6)
        val = th.acos(xy)
        return divsin(val) * (y - xy * x)

    def tangent_projection(self, x, u):
        x_dot_u = th.sum(x * u, dim=-1, keepdim=True)
        proj_u = u - x * x_dot_u
        return proj_u

    def tangent_orthonormal_basis(self, x, dF):
        assert x.ndim == 2

        if x.shape[1] == 2:
            E = x[:, [1, 0]] * th.tensor([-1.0, 1.0], device=x.device, dtype=x.dtype)
            E = E.unsqueeze(-1)
        elif x.shape[1] == 3:
            norm_v = dF / th.linalg.norm(dF, dim=-1, keepdim=True)
            cross_prod = th.cross(x, norm_v, dim=1)
            E = th.stack([norm_v, cross_prod], dim=2)
        else:
            raise NotImplementedError()

        return E

    def dist(self, x, y):
        if x.ndim == 2 and y.ndim == 2:
            inner = th.matmul(x, y.transpose(-1, -2))
        elif x.ndim == 2 and y.ndim == 1:
            inner = th.matmul(x, y)
        elif x.ndim == 1 and y.ndim == 1:
            inner = th.dot(x, y)
        else:
            inner = th.sum(x * y, dim=-1)

        inner = inner / (1 + self.jitter)
        inner = th.clamp(inner, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return th.acos(inner)

    def cost(self, x, y):
        d = self.dist(x, y)
        return d ** 2 / 2.0

    def project(self, x):
        norm = th.linalg.norm(x, dim=-1, keepdim=True)
        return x / (norm + eps)

    def transp(self, x, y, u):
        yu = th.sum(y * u, dim=-1, keepdim=True)
        xy = th.sum(x * y, dim=-1, keepdim=True)
        xy_clipped = th.clamp(xy, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return u - yu / (1 + xy_clipped) * (x + y)

    def logdetexp(self, x, u):
        norm_u = th.linalg.norm(u, dim=-1)
        log_sindiv = th.log(th.abs(sindiv(norm_u)))
        return (self.D - 1) * log_sindiv

    def zero(self):
        return th.zeros(self.D)

    def zero_like(self, x):
        return th.zeros_like(x)

    def squeeze_tangent(self, x):
        return x

    def unsqueeze_tangent(self, x):
        return x

    def path(self, x, y, num_points=20):
        angle = self.dist(x, y)
        ts = th.linspace(0, 1, num_points, device=x.device, dtype=x.dtype).unsqueeze(-1)
        if angle.ndim < ts.ndim:
            angle = angle.unsqueeze(0)
        sin_angle = th.sin(angle)
        path = (th.sin((1 - ts) * angle) * x.unsqueeze(0) + th.sin(ts * angle) * y.unsqueeze(0)) / (sin_angle.unsqueeze(0) + eps)
        return path


class SqEuclidean(GeometryBase):
    def cost(self, x, y):
        return 0.5 * th.sum((x - y) ** 2, dim=-1)

    def path(self, x, y, num_points=20):
        ts = th.linspace(0, 1, num_points, device=x.device, dtype=x.dtype).unsqueeze(-1)
        path = (1 - ts) * x.unsqueeze(0) + ts * y.unsqueeze(0)
        return path

    def project(self, x):
        if isinstance(self.bounds, tuple) and len(self.bounds) == 2 and isinstance(self.bounds[0], (int, float)):
            return th.clamp(x, min=self.bounds[0], max=self.bounds[1])
        elif isinstance(self.bounds, tuple) and len(self.bounds) == 2 and isinstance(self.bounds[0], th.Tensor):
            return th.max(th.min(x, self.bounds[1]), self.bounds[0])
        else:
            return x


@dataclass
class MetricManifold(GeometryBase, nn.Module):
    distance_mode: DistanceModes = DistanceModes.SQUARED_GEODESIC
    metric_initializer_fn: Callable = metrics.EuclideanMetric
    spline_model_initializer_fn: Callable = spline_amortizer.SplineMLP
    lagrangian_potential_initializer_fn: Optional[Callable] = None
    spline_solver_kwargs: Optional[Dict] = None

    def __init__(self, D=2, bounds=(-2, 2), xbounds=None, ybounds=None,
                 distance_mode=DistanceModes.SQUARED_GEODESIC,
                 metric_initializer_fn=metrics.EuclideanMetric,
                 spline_model_initializer_fn=spline_amortizer.SplineMLP,
                 lagrangian_potential_initializer_fn=None,
                 spline_solver_kwargs=None, **kwargs):

        self.D = D
        self.bounds = bounds
        self.xbounds = xbounds if xbounds is not None else bounds
        self.ybounds = ybounds if ybounds is not None else bounds
        if self.xbounds is None: self.xbounds = self.bounds
        if self.ybounds is None: self.ybounds = self.bounds

        super().__init__()

        self.distance_mode = distance_mode
        self.metric_initializer_fn = metric_initializer_fn
        self.spline_model_initializer_fn = spline_model_initializer_fn
        self.lagrangian_potential_initializer_fn = lagrangian_potential_initializer_fn
        self.spline_solver_kwargs = spline_solver_kwargs if spline_solver_kwargs is not None else {}

        self.spline_geodesic_solver = geodesics.SplineSolver(
            D=self.D, **self.spline_solver_kwargs
        )
        self.spline_amortizer = spline_amortizer.SplineAmortizer(
            self, self.spline_geodesic_solver
        )

        self.metric_module = self.metric_initializer_fn()
        self.lagrangian_potential_module = None
        if self.lagrangian_potential_initializer_fn is not None:
            self.lagrangian_potential_module = self.lagrangian_potential_initializer_fn()

        num_params = self.spline_geodesic_solver.num_spline_params
        self.spline_model = self.spline_model_initializer_fn(num_params)

    def predict_spline_params(self, x, y):
        return self.spline_model(x, y)

    def metric(self, x):
        return self.metric_module(x)

    def lagrangian_potential(self, x):
        if self.lagrangian_potential_module is not None:
            return self.lagrangian_potential_module(x)
        else:
            return th.zeros((x.shape[0],) if x.ndim > 1 else (), device=x.device, dtype=x.dtype)

    def path(self, x, y, num_points=20):
        assert x.ndim == 1 and y.ndim == 1

        with th.no_grad():
            init_spline_params = self.predict_spline_params(x, y)

        out = self.spline_geodesic_solver.solve(
            self.curve_energy,
            x,
            y,
            init_params=init_spline_params.detach(),
            num_final_points=num_points,
        )
        return out.mu

    def energy_at_point(self, x, v):
        M = self.metric(x)

        if x.ndim == 1:
            M = M.squeeze(0) if M.ndim == 3 else M
            v = v.unsqueeze(0)
            kinetic_term = v @ M @ v.T
        else:
            kinetic_term = v.unsqueeze(1) @ M @ v.unsqueeze(2)
            kinetic_term = kinetic_term.squeeze(-1).squeeze(-1)

        if (
            self.distance_mode == DistanceModes.GEODESIC
            or self.distance_mode == DistanceModes.SQUARED_GEODESIC
        ):
            kinetic = th.sqrt(th.clamp(kinetic_term, min=eps))
        else:
            kinetic = 0.5 * kinetic_term

        potential = self.lagrangian_potential(x)

        return kinetic - potential

    def curve_energy(self, xs):
        assert xs.ndim == 2
        vs = xs[1:] - xs[:-1]
        Es = self.energy_at_point(xs[:-1], vs)
        return Es.sum()

    def cost(self, x, y):
        assert x.ndim == 1 and y.ndim == 1
        gamma = self.path(x, y)
        E = self.curve_energy(gamma)
        if self.distance_mode == DistanceModes.SQUARED_GEODESIC:
            E = 0.5 * E**2
        return E

    def project(self, x):
        if isinstance(self.bounds, tuple) and len(self.bounds) == 2 and isinstance(self.bounds[0], (int, float)):
            return th.clamp(x, min=self.bounds[0], max=self.bounds[1])
        elif isinstance(self.bounds, tuple) and len(self.bounds) == 2 and isinstance(self.bounds[0], th.Tensor):
            return th.max(th.min(x, self.bounds[1]), self.bounds[0])
        else:
            return x

    def add_plot_background(self, ax, xlims, ylims=None, alpha=1.0):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = th.device("cpu")

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy()

        if issubclass(
            self.metric_initializer_fn, metrics.ScarvelisMetric
        ) or issubclass(self.metric_initializer_fn, metrics.NeuralNetMetric):
            grid_size = 21
            assert len(xlims) == 2
            if ylims is None:
                ylims = xlims

            xflat_np, x1_np, x2_np = _get_grid(xlims, ylims, grid_size)
            xflat = th.tensor(xflat_np, dtype=th.float32, device=device)

            with th.no_grad():
                A_batch = self.metric(xflat)
                A_batch_cpu = A_batch.cpu()
                vals, vecs = th.linalg.eigh(A_batch_cpu)

            u = to_numpy(vecs[:, 0, 0]).reshape(x1_np.shape)
            v = to_numpy(vecs[:, 1, 0]).reshape(x1_np.shape)

            ax.quiver(x1_np, x2_np, u, v, alpha=alpha)
            ax.quiver(x1_np, x2_np, -u, -v, alpha=alpha)

            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
        elif self.lagrangian_potential_module is not None:
            grid_size = 201
            assert len(xlims) == 2
            if ylims is None:
                ylims = xlims

            xflat_np, x1_np, x2_np = _get_grid(xlims, ylims, grid_size)
            xflat = th.tensor(xflat_np, dtype=th.float32, device=device)

            with th.no_grad():
                vals_pot = self.lagrangian_potential(xflat)
                vals = -to_numpy(vals_pot).reshape(x1_np.shape)

            CS = ax.contourf(x1_np, x2_np, vals, cmap="Blues")

            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)


def _get_grid(xlims: Tuple[float, float], ylims: Tuple[float, float], grid_size=21):
    x1 = np.linspace(xlims[0], xlims[1], grid_size)
    x2 = np.linspace(ylims[0], ylims[1], grid_size)
    x1, x2 = np.meshgrid(x1, x2)
    xflat = np.stack([x1.ravel(), x2.ravel()], axis=-1)
    return xflat, x1, x2
