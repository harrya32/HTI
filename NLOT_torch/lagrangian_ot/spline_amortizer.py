# Copyright (c) Meta Platforms, Inc. and affiliates

import torch as th
import torch.nn as nn
import torch.optim as optim

from typing import Any, TYPE_CHECKING, Optional

from dataclasses import dataclass

from . import splines, meters, geodesics

# Type hint for GeometryBase to avoid circular import
if TYPE_CHECKING:
    from .geometries import MetricManifold

# Convert SplineMLP to PyTorch nn.Module
class SplineMLP(nn.Module):
    def __init__(self, out_dims: int, input_dims: int, num_hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            # Input dimension is doubled (x and y concatenated)
            nn.Linear(input_dims * 2, num_hidden),
            nn.ReLU(), # Use ReLU as in Flax example
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, out_dims)
        )

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        assert x.ndim == 2
        z = th.cat([x, y], dim=1) # Concatenate along feature dimension
        z = self.net(z)
        if squeeze:
            z = z.squeeze(0)
        return z


# Define as a regular class without dataclass or complex type hints in body
class SplineAmortizer:
    # Attributes will be defined in __init__

    def __init__(self, geometry, spline_geodesic_solver):
        # Type check geometry (optional but recommended)
        # from .geometries import MetricManifold # Import locally if needed for check
        # assert isinstance(geometry, MetricManifold)
        # assert isinstance(spline_geodesic_solver, geodesics.SplineSolver)

        self.geometry = geometry
        self.spline_geodesic_solver = spline_geodesic_solver
        self.D = self.spline_geodesic_solver.D
        self.basis = self.spline_geodesic_solver.spline_basis
        # Ensure basis is a tensor
        if not isinstance(self.basis, th.Tensor):
             self.basis = th.tensor(self.basis) # Convert if necessary
        self.num_params_spline = self.basis.shape[-1] * self.D

        # Initialize optimizer related attributes
        self.grad_clip = 1.0
        self.optimizer = None # Will be initialized in train

    def loss_fn(self, xs: th.Tensor, ys: th.Tensor) -> th.Tensor:
        # Ensure basis is on the correct device
        device = xs.device
        # Make sure basis is moved to device inside the loss function
        basis = self.basis.to(device)

        # Predict spline parameters using the geometry's spline_model
        params_spline = self.geometry.predict_spline_params(xs, ys)

        # Calculate spline energy for each pair (x, y) in the batch
        total_energy = 0.0
        num_samples = xs.shape[0]

        ts = th.linspace(0., 1., num=self.spline_geodesic_solver.num_spline_points_eval, device=device)
        for i in range(num_samples):
            x_i = xs[i]
            y_i = ys[i]
            params_spline_i = params_spline[i]

            # Ensure compute_spline takes tensors
            path = splines.compute_spline(
                x=x_i, y=y_i, basis=basis, params=params_spline_i, ts=ts)
            # Ensure curve_energy takes tensor and returns scalar tensor
            E = self.geometry.curve_energy(path)
            total_energy += E

        # Return mean energy as a scalar tensor
        return total_energy / num_samples

    def train(self, source_sampler, target_sampler, max_iter,
              lr=1e-4, grad_norm_threshold=None, callback=None):
        print('Fitting spline amortizer')
        if not isinstance(self.geometry, nn.Module) or not hasattr(self.geometry, 'spline_model'):
             raise ValueError("Geometry must be an nn.Module with a 'spline_model' attribute")

        # Check if spline_model has parameters before initializing optimizer
        try:
            model_params = list(self.geometry.spline_model.parameters())
            if not model_params:
                print("Warning: spline_model has no parameters. Skipping training.")
                return # Exit if no parameters to optimize
            device = model_params[0].device
        except StopIteration:
            print("Warning: spline_model has no parameters. Skipping training.")
            return # Exit if no parameters to optimize
        except AttributeError:
             raise ValueError("Geometry's spline_model does not seem to be a valid nn.Module")

        # Initialize optimizer only if it hasn't been initialized
        if self.optimizer is None:
            self.optimizer = optim.Adam(model_params, lr=lr)

        loss_meter = meters.EMAMeter(0.9)
        loss_grad_meter = meters.EMAMeter(0.9)

        self.geometry.spline_model.train()

        for i in range(max_iter):
            try:
                xs = next(source_sampler).to(device)
                ys = next(target_sampler).to(device)
            except StopIteration:
                print("Warning: Sampler exhausted before reaching max_iter.")
                break

            self.optimizer.zero_grad()
            loss = self.loss_fn(xs, ys)
            loss.backward()

            total_norm = nn.utils.clip_grad_norm_(model_params, self.grad_clip)

            self.optimizer.step()

            loss_meter.update(loss.item())
            loss_grad_meter.update(total_norm.item())

            if i % 1000 == 0:
                print(f'[{i}] loss: {loss_meter.value:.2e} grad_norm: {loss_grad_meter.value:.2e}')
            if grad_norm_threshold is not None and loss_grad_meter.value < grad_norm_threshold:
                print(f'Stopping early at iter {i} due to grad norm threshold')
                break

            if callback is not None:
                callback(i)

        self.geometry.spline_model.eval()
        print('Finished fitting spline amortizer')
