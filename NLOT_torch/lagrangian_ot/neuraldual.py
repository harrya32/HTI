# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Sequence,
    TYPE_CHECKING
)

import functools
import torch as th
import torch.nn as nn
import torch.optim as optim
# Removed JAX/Flax/Optax imports

import collections

# Removed Flax imports

from copy import copy

# Removed OTT import (unused)

import matplotlib as mpl
import matplotlib.pyplot as plt

# Assuming these modules are converted to PyTorch
from lagrangian_ot import ctransform_solvers, models, geometries, meters # Added meters

if TYPE_CHECKING:
    # Use standard PyTorch types
    from torch.optim.optimizer import Optimizer
    # Import converted geometry/model types
    from .geometries import GeometryBase
    from .models import ModelBase
    from .ctransform_solvers import CTransformSolver

# Use Dicts and Lists for logs, standard types for others
Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable
# CTransformSolver type hint used directly
# Potential_t not used directly in this file after conversion

# Replace jnp.ndarray with th.Tensor in NamedTuple
Info = collections.namedtuple("Info", "dual_loss amor_loss num_ctransform_iter target_hat")
# UpdateOut no longer needed as updates happen in-place
# UpdateOut = collections.namedtuple("UpdateOut", "states info")


class ManifoldW2NeuralDual:
    def __init__(
            self,
            geometry: "GeometryBase",
            target_potential: "ModelBase",
            source_map: "ModelBase",
            ctransform_solver: Optional["CTransformSolver"] = None, # Use Optional
            amortization_loss: Literal["objective", "regression"] = "regression",
            device: Optional[Union[str, th.device]] = None # Add device
    ):
        self.geometry = geometry
        self.amortization_loss = amortization_loss

        # Use default if None
        self.ctransform_solver = ctransform_solver if ctransform_solver is not None else ctransform_solvers.DEFAULT_CTRANSFORM_SOLVER
        # Geometry passed to solver during solve call in PyTorch version
        # self.ctransform_solver.geometry = geometry

        # Ensure models are nn.Module
        assert isinstance(target_potential, nn.Module)
        assert isinstance(source_map, nn.Module)
        self.target_potential = target_potential
        self.source_map = source_map

        # Determine device
        if device is None:
            # Try to infer from model parameters, default to CPU
            try:
                device = next(self.target_potential.parameters()).device
            except StopIteration:
                try:
                     device = next(self.source_map.parameters()).device
                except StopIteration:
                     device = th.device("cpu")
                     print("Warning: No parameters found in models, defaulting to CPU.")
        self.device = th.device(device)

        # Move models to device
        self.geometry.to(self.device) # If geometry has buffers/params
        self.target_potential.to(self.device)
        self.source_map.to(self.device)

        # JIT compilation removed, PyTorch uses tracing or scripting if needed
        # Pre-compiled/jitted functions are removed

    # Removed initialize_states, state handled directly with models/optimizers

    # Removed state_from_dicts/state_to_dicts (use PyTorch state_dict)

    def loss_fn(self, batch: Dict[str, th.Tensor]) -> Tuple[th.Tensor, Info]:
        """Loss function for both potentials using PyTorch."""
        source, target = batch["source"].to(self.device), batch["target"].to(self.device)

        # Forward pass through source map (potential network)
        # Ensure models are in correct mode (train/eval)
        init_target_hat = self.source_map(source)

        # Partial function for target potential evaluation
        # Requires target_potential to be callable
        target_potential_eval = lambda t: self.target_potential(t)

        # C-transform calculation
        num_ctransform_iter_val = 0 # Default value
        if self.ctransform_solver is not None:
            # Solve C-transform for each source point
            # Note: The original implementation used vmap. PyTorch requires explicit batch handling or loop.
            # Assuming ctransform_solver.solve can handle batches or we loop here.
            # For simplicity, let's assume solve handles batches or use a placeholder loop.

            # Placeholder: Using init_target_hat directly if solver batching is complex
            # This bypasses the finetuning step, adjust if solver works with batches
            # --- Start Potential Loop/Batch --- #
            target_hats = []
            iters = []
            # Loop approach (less efficient than potential batching in solver)
            for i in range(source.shape[0]):
                out = self.ctransform_solver.solve(
                    self.geometry,
                    target_potential_eval,
                    source[i], # Single source point
                    target_init=init_target_hat[i] # Single init point
                )
                target_hats.append(out.solution)
                iters.append(out.num_iter)

            if target_hats:
                target_hat_tensor = th.stack(target_hats)
                num_ctransform_iter = th.tensor(iters, dtype=th.float32).mean()
                num_ctransform_iter_val = num_ctransform_iter.item()
                # Detach target_hat from computation graph for dual loss calculation
                target_hat_detach = target_hat_tensor.detach()
                # Projection might be needed depending on geometry
                # target_hat_detach = self.geometry.project(target_hat_detach)
            else: # Handle case where loop didn't run (e.g., batch size 0)
                 target_hat_detach = init_target_hat.detach() # Use initial guess
                 num_ctransform_iter = th.tensor(0.0, device=self.device)
            # --- End Potential Loop/Batch --- #

        else:
            target_hat_detach = init_target_hat.detach()
            num_ctransform_iter = th.tensor(0.0, device=self.device)

        # Calculate dual loss components
        target_potential_vals = target_potential_eval(target)
        # Cost function requires batch handling or loop
        # Assuming geometry.cost handles batches [N, D], [N, D] -> [N]
        cost_vals = self.geometry.cost(source, target_hat_detach)
        target_potential_hat_vals = target_potential_eval(target_hat_detach)

        source_potential = cost_vals - target_potential_hat_vals
        dual_source = source_potential.mean()
        dual_target = target_potential_vals.mean()
        dual_loss = -dual_source - dual_target

        # Calculate amortization loss
        if self.amortization_loss == "regression":
            # Use init_target_hat (output of source_map) and detached target_hat
            amor_loss = th.mean((init_target_hat - target_hat_detach) ** 2)
        elif self.amortization_loss == "objective":
            # This part needs careful re-implementation based on the exact objective
            # Original code used ipdb.set_trace(), suggesting it might be incomplete/complex
            warnings.warn("Amortization loss 'objective' not fully implemented for PyTorch conversion.")
            # Placeholder: Use regression loss for now
            amor_loss = th.mean((init_target_hat - target_hat_detach) ** 2)
        else:
            raise ValueError("Amortization loss has been misspecified.")

        loss = dual_loss + amor_loss
        info = Info(
            dual_loss=dual_loss.item(),
            amor_loss=amor_loss.item(),
            num_ctransform_iter=num_ctransform_iter_val,
            target_hat=target_hat_detach # Return the detached tensor
        )
        return loss, info

    # update_fn replaced by standard PyTorch training step
    def training_step(self, batch, optimizer_target, optimizer_source):
        self.target_potential.train()
        self.source_map.train()

        optimizer_target.zero_grad()
        optimizer_source.zero_grad()

        loss, info = self.loss_fn(batch)

        loss.backward()

        optimizer_target.step()
        optimizer_source.step()

        return info # Return collected info

    # Simplify plotting functions to use PyTorch tensors directly
    @th.no_grad() # Disable gradients for plotting
    def plot_forward_map(
            self,
            source: th.Tensor,
            target: th.Tensor,
            ax: Optional["plt.Axes"] = None,
            legend: bool = True,
            scatter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["plt.Figure", "plt.Axes"]:
        if mpl is None:
            raise RuntimeError("Please install `matplotlib` first.")

        if scatter_kwargs is None:
            scatter_kwargs = {"alpha": 0.5}

        if ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="white")
        else:
            fig = ax.get_figure()

        # Move data to device and detach for numpy conversion
        source_np = source.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        source_dev = source.to(self.device)

        # plot the source and target samples
        label_transport = r"transported"
        source_color, target_color = "#1A254B", "#A7BED3"

        ax.scatter(
                source_np[:, 0],
                source_np[:, 1],
                color=source_color,
                label="source",
                **scatter_kwargs,
        )
        ax.scatter(
                target_np[:, 0],
                target_np[:, 1],
                color=target_color,
                label="target",
                **scatter_kwargs,
        )

        # Ensure models are in eval mode for prediction
        self.target_potential.eval()
        self.source_map.eval()

        # Calculate transported samples
        # Placeholder: Looping for pushforward (similar to loss_fn)
        transported_samples_list = []
        for i in range(source_dev.shape[0]):
             # The original pushforward used solve directly
             out = self.ctransform_solver.solve(
                 self.geometry,
                 lambda t: self.target_potential(t), # Pass callable
                 source_dev[i],
                 target_init=self.source_map(source_dev[i]) # Use model directly
             )
             transported_samples_list.append(out.solution)

        if transported_samples_list:
            transported_samples = th.stack(transported_samples_list).detach().cpu().numpy()
            ax.scatter(
                    transported_samples[:, 0],
                    transported_samples[:, 1],
                    color="#F2545B",
                    label=label_transport,
                    **scatter_kwargs,
            )

        if legend:
            ax.legend()
        ax.set_title(r'$\mathcal{W}_2$ Neural Dual')
        return fig, ax

    @th.no_grad() # Disable gradients
    def plot_target_potential(
            self,
            source: th.Tensor,
            target: th.Tensor,
            quantile: float = 0.05,
            ax: Optional["mpl.axes.Axes"] = None,
            x_bounds: Tuple[float, float] = (-6, 6),
            y_bounds: Tuple[float, float] = (-6, 6),
            num_grid: int = 50,
            contourf_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["mpl.figure.Figure", "mpl.axes.Axes"]:
        if mpl is None:
            raise RuntimeError("Please install `matplotlib` first.")

        if contourf_kwargs is None:
            contourf_kwargs = {}

        if ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="white")
        else:
            fig = ax.get_figure()

        # Create grid
        x_coords = th.linspace(x_bounds[0], x_bounds[1], num_grid, device=self.device)
        y_coords = th.linspace(y_bounds[0], y_bounds[1], num_grid, device=self.device)
        grid_x, grid_y = th.meshgrid(x_coords, y_coords, indexing='xy')
        grid = th.stack([grid_x.ravel(), grid_y.ravel()], dim=1)

        # Evaluate potential on grid
        self.target_potential.eval() # Set model to eval mode
        potential_values = self.target_potential(grid).detach().cpu().numpy()
        potential_values = potential_values.reshape(grid_x.shape)

        # Determine contour levels
        vmin = np.quantile(potential_values, quantile)
        vmax = np.quantile(potential_values, 1.0 - quantile)
        levels = np.linspace(vmin, vmax, 10)

        contourf_kwargs.setdefault("levels", levels)
        contourf_kwargs.setdefault("cmap", "viridis")
        contourf_kwargs.setdefault("alpha", 0.5)

        # Plot potential contours
        CS = ax.contourf(
            grid_x.cpu().numpy(), grid_y.cpu().numpy(), potential_values, **contourf_kwargs
        )
        fig.colorbar(CS, ax=ax)

        # Plot source and target samples (optional)
        source_np = source.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        ax.scatter(source_np[:, 0], source_np[:, 1], alpha=0.3, label="source")
        ax.scatter(target_np[:, 0], target_np[:, 1], alpha=0.3, label="target")

        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_title("Target Potential")
        ax.legend()
        return fig, ax
