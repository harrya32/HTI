# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
from typing import Callable, Literal, NamedTuple, Optional, TYPE_CHECKING
import dataclasses
import collections

import torch as th # Use th alias
import torch.optim as optim

# Removed jax, optax, flax, jaxopt imports

# Removed ott.utils import (unused)
# Assuming geometries module is converted
from lagrangian_ot import geometries

# Remove TYPE_CHECKING block if string hints cause issues
# if TYPE_CHECKING:
#     from .geometries import GeometryBase # For type hint


# Keep NamedTuple, replace jnp.ndarray with th.Tensor
class CTransformResults(NamedTuple):
    val: float
    solution: th.Tensor
    num_iter: int


# Base class remains similar
class CTransformSolver(abc.ABC):
    # Removed geometry attribute hint from base class def

    # Pass geometry as an object, no type hint here
    @abc.abstractmethod
    def solve(
        self,
        geometry, # Removed type hint
        f: Callable[[th.Tensor], th.Tensor],
        source: th.Tensor,
        target_init: Optional[th.Tensor] = None
    ) -> CTransformResults:
        pass

@dataclasses.dataclass
class CTransformLBFGS(CTransformSolver):
    gtol: float = 1e-3
    max_iter: int = 10
    max_eval: Optional[int] = None # LBFGS param
    # Linesearch params may differ in torch.optim.LBFGS, adjust if needed
    # Using PyTorch LBFGS defaults where possible
    lr: float = 1.0 # Default learning rate for LBFGS
    history_size: int = 100 # Default LBFGS history size

    def solve(
        self,
        geometry, # Removed type hint
        f: Callable[[th.Tensor], th.Tensor],
        source: th.Tensor,
        target_init: Optional[th.Tensor] = None
    ) -> CTransformResults:
        assert source.ndim == 1
        device = source.device

        # Initialize target variable to optimize
        if target_init is None:
            target_init = source.clone()
        # Ensure target requires gradients
        target = target_init.detach().clone().requires_grad_(True).to(device)

        # Cost function using the geometry object directly
        cost_fn = lambda trgt: geometry.cost(source, trgt)

        # Define the objective function for PyTorch LBFGS
        # LBFGS optimizer needs a closure that re-evaluates the model and returns the loss
        def closure():
            optimizer.zero_grad()
            # Project target within the optimization step if needed by geometry
            target_projected = geometry.project(target)
            loss = cost_fn(target_projected) - f(target_projected)
            loss.backward() # Compute gradients
            return loss

        # Initialize PyTorch LBFGS optimizer
        optimizer = optim.LBFGS(
            [target],
            lr=self.lr,
            max_iter=self.max_iter,
            max_eval=self.max_eval,
            tolerance_grad=self.gtol, # Use gtol for gradient tolerance
            tolerance_change=1e-9, # Default PyTorch value
            history_size=self.history_size,
            line_search_fn="strong_wolfe" # Or None for default backtracking
        )

        # Run the optimization
        # LBFGS requires multiple steps within the closure
        optimizer.step(closure)

        # Get final results
        final_target = geometry.project(target.detach())
        # Re-evaluate final objective value
        with th.no_grad():
             final_val = (cost_fn(final_target) - f(final_target)).item()

        # PyTorch LBFGS doesn't easily expose iteration count within closure steps
        # We can estimate or just return max_iter if converged or not tracked precisely.
        # For simplicity, returning max_iter or a placeholder. Actual iter count is complex.
        num_iter = self.max_iter # Placeholder

        return CTransformResults(
            val=final_val, solution=final_target, num_iter=num_iter
        )

@dataclasses.dataclass
class CTransformAdam(CTransformSolver):
    gtol: float = 1e-3
    max_iter: int = 10

    adam_kwargs: Optional[dict] = None
    lr_schedule_kwargs: Optional[dict] = None # May need adapting for PyTorch schedulers
    init_lr: float = 0.1 # Add base LR for scheduler

    def __post_init__(self):
        if self.adam_kwargs is None:
            self.adam_kwargs = {'betas': (0.9, 0.999)} # PyTorch uses 'betas' tuple
        if self.lr_schedule_kwargs is None:
            # Parameters for CosineAnnealingLR
            self.lr_schedule_kwargs = {
                'T_max': self.max_iter,
                'eta_min': self.init_lr * 1e-4 # Equivalent to alpha * init_value
            }

    def solve(
        self,
        geometry, # Removed type hint
        f: Callable[[th.Tensor], th.Tensor],
        source: th.Tensor,
        target_init: Optional[th.Tensor] = None
    ) -> CTransformResults:
        assert source.ndim == 1
        device = source.device

        # Initialize target variable to optimize
        if target_init is None:
            target_init = source.clone()
        target = target_init.detach().clone().requires_grad_(True).to(device)

        # Cost function
        cost_fn = lambda trgt: geometry.cost(source, trgt)

        # Objective function
        def objective(trgt):
             trgt_proj = geometry.project(trgt) # Project within objective
             return cost_fn(trgt_proj) - f(trgt_proj)

        # Set up optimizer and scheduler
        optimizer = optim.Adam([target], lr=self.init_lr, **self.adam_kwargs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **self.lr_schedule_kwargs)

        i = 0
        grad_norm = th.inf

        # PyTorch optimization loop
        while i < self.max_iter and grad_norm > self.gtol:
            optimizer.zero_grad()
            current_obj = objective(target)
            current_obj.backward()

            if target.grad is not None:
                # Use L-infinity norm for gradient check as in original cond_fun
                grad_norm = th.linalg.norm(target.grad, ord=float('inf'))
            else:
                grad_norm = th.tensor(0.0, device=device)
                break # Stop if no gradient

            optimizer.step()
            scheduler.step()

            i += 1
            # Optional: print progress
            # print(f"Iter: {i}, Obj: {current_obj.item():.4f}, GradNorm: {grad_norm.item():.4g}")

        # Final results
        final_target = geometry.project(target.detach())
        final_obj = objective(final_target).item()

        return CTransformResults(
            val=final_obj, solution=final_target, num_iter=i
        )


# Update default solver
DEFAULT_CTRANSFORM_SOLVER = CTransformAdam(
    gtol=1e-5,
    max_iter=20,
    init_lr=0.1 # Ensure init_lr is provided if using default Adam kwargs
)
