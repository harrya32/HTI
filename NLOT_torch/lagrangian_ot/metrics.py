# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import torch as th
import torch.nn as nn

import copy

from typing import Tuple

from dataclasses import dataclass

from abc import ABC, abstractmethod


plot_cache = {}

@dataclass
class MetricBase(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class EuclideanMetric(MetricBase):
    def forward(self, x):
        D = x.shape[-1]
        if x.ndim == 1:
            return th.eye(D, device=x.device, dtype=x.dtype)
        elif x.ndim == 2:
            batch_size = x.shape[0]
            return th.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            raise ValueError("Input tensor must have 1 or 2 dimensions")


@dataclass
class ScarvelisMetric(MetricBase, ABC):
    div_eps: float = 1e-6
    metric_eps: float = 1e-3

    def __post_init__(self):
        super().__init__()

    def forward(self, x):
        assert x.ndim <= 2
        D = x.shape[-1]

        v_vals = self.v(x)

        if x.ndim == 1:
            identity = th.eye(D, device=x.device, dtype=x.dtype)
            outer_prod = th.outer(v_vals, v_vals)
        elif x.ndim == 2:
            batch_size = x.shape[0]
            identity = th.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            outer_prod = th.bmm(v_vals.unsqueeze(2), v_vals.unsqueeze(1))
        else:
            raise ValueError("Input tensor must have 1 or 2 dimensions")

        return identity - (1 - self.metric_eps) * outer_prod

    @abstractmethod
    def v(self, x):
        raise NotImplementedError


@dataclass
class CircleMetric(ScarvelisMetric):
    def __post_init__(self):
        super().__post_init__()

    def v(self, x):
        assert x.shape[-1] == 2
        norm = th.linalg.norm(x, dim=-1, keepdim=True)
        norm = th.clamp(norm, min=self.div_eps)
        if x.ndim == 1:
            v = x[[1, 0]] * th.tensor([-1., 1.], device=x.device, dtype=x.dtype)
        else:
            v = x[:, [1, 0]] * th.tensor([-1., 1.], device=x.device, dtype=x.dtype)
        return v / norm

@dataclass
class VeeMetric(ScarvelisMetric):
    def __post_init__(self):
        super().__post_init__()

    def v(self, x):
        assert x.shape[-1] == 2
        sign_y = th.sign(x[..., 1]).unsqueeze(-1)
        val = 1. / th.sqrt(th.tensor(2.0, device=x.device, dtype=x.dtype))
        v = th.cat([th.full_like(sign_y, val), sign_y * val], dim=-1)
        return v

@dataclass
class XMetric(ScarvelisMetric):
    def __post_init__(self):
        super().__post_init__()

    def v(self, x):
        assert x.shape[-1] == 2
        x0, x1 = x[..., 0], x[..., 1]
        a = 1.25 * th.tanh(th.relu(x0 * x1))
        b = -1.25 * th.tanh(th.relu(-x0 * x1))

        val = 1. / th.sqrt(th.tensor(2.0, device=x.device, dtype=x.dtype))
        v1 = th.tensor([val, val], device=x.device, dtype=x.dtype)
        v2 = th.tensor([val, -val], device=x.device, dtype=x.dtype)

        if x.ndim == 2:
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)

        return (a * v1 + b * v2) / 1.25

class NeuralNetMetric(MetricBase):
    def __init__(self, D=2, hidden_dim=128):
        super().__init__()
        self.D = D
        assert self.D == 2, "NeuralNetMetric currently only supports D=2"
        self.net = nn.Sequential(
            nn.Linear(self.D, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.D)
        )

    def forward(self, x):
        assert x.shape[-1] == self.D
        is_batched = x.ndim == 2
        if not is_batched:
            x = x.unsqueeze(0)

        net_out = self.net(x)
        theta = th.atan2(net_out[:, 1], net_out[:, 0])

        cos_theta = th.cos(theta)
        sin_theta = th.sin(theta)

        R = th.stack([
            th.stack([cos_theta, -sin_theta], dim=1),
            th.stack([sin_theta, cos_theta], dim=1)
        ], dim=1)

        Q = th.tensor([[1., 0.], [0., 0.1]], device=x.device, dtype=x.dtype)
        Q_batch = Q.unsqueeze(0).repeat(x.shape[0], 1, 1)

        A = th.bmm(th.bmm(R.transpose(1, 2), Q_batch), R)

        if not is_batched:
            A = A.squeeze(0)

        return A
