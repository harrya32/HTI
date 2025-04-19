# Copyright (c) Meta Platforms, Inc. and affiliates

import torch as th # Use th alias
import torch.optim as optim # For optimizers

from collections import namedtuple

from dataclasses import dataclass
from abc import ABC, abstractmethod

from lagrangian_ot import splines

GPState = namedtuple("GPState", "i ddc mu dmu cost")
SplineState = namedtuple("SplineState", "i params opt_state E grad_norm")
SolverOut = namedtuple("SolverOut", "mu dmu num_iter cost")

@dataclass
class SolverBase(ABC):
    D: int

    @abstractmethod
    def solve(self, geo_eq, x0, x1):
        pass


@dataclass
class SplineSolver(SolverBase):
    num_spline_nodes: int = 20
    grad_tol: float = 1e-5
    init_lr: float = 1e-2
    num_spline_points_eval: int = 21
    max_iter: int = 20

    def __post_init__(self):
        self.spline_basis = splines.get_basis(
            self.D, num_nodes=self.num_spline_nodes)
        self.num_spline_params = self.spline_basis.shape[-1] * self.D

    def solve(self, energy_fn, x0: th.Tensor, x1: th.Tensor, init_params: th.Tensor, num_final_points=None):
        device = x0.device
        x0, x1, init_params = x0.to(device), x1.to(device), init_params.to(device)
        spline_basis = self.spline_basis.to(device)

        ts_eval = th.linspace(0., 1., num=self.num_spline_points_eval, device=device)

        params = init_params.detach().clone().requires_grad_(True)

        if self.max_iter == 0:
            final_ts = ts_eval
            if num_final_points is not None:
                final_ts = th.linspace(0., 1., num=num_final_points, device=device)
            with th.no_grad():
                xs = splines.compute_spline(
                    x=x0, y=x1, basis=spline_basis, params=params, ts=final_ts)
                E = energy_fn(xs)
            return SolverOut(mu=xs.detach(), dmu=None, num_iter=0, cost=E.detach())

        def F(p):
            xs = splines.compute_spline(
                x=x0, y=x1, basis=spline_basis, params=p, ts=ts_eval)
            E = energy_fn(xs)
            return E

        optimizer = optim.Adam([params], lr=self.init_lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=self.init_lr*1e-1)

        grad_norm = th.inf
        current_E = th.inf
        i = 0

        while i < self.max_iter and grad_norm > self.grad_tol:
            optimizer.zero_grad()
            current_E = F(params)
            current_E.backward()

            if params.grad is not None:
                grad_norm = th.linalg.norm(params.grad)
            else:
                grad_norm = th.tensor(0.0, device=device)
                break

            optimizer.step()
            scheduler.step()

            i += 1

        final_params = params.detach()
        num_iter = i
        final_E = current_E.detach()

        final_ts = ts_eval
        if num_final_points is not None:
            final_ts = th.linspace(0., 1., num=num_final_points, device=device)

        with th.no_grad():
            final_xs = splines.compute_spline(
                x=x0, y=x1, basis=spline_basis, params=final_params, ts=final_ts)

        return SolverOut(mu=final_xs, dmu=None, num_iter=num_iter, cost=final_E)
