# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import copy

import torch as th
import torch.nn as nn
import numpy as np


@dataclass
class LagrangianPotentialBase(nn.Module):
    D: int = 2

    M_bounds = (0., 0.01)
    temp_bounds = (1e-1, 1e-2)

    def __init__(self, D: int = 2):
        super().__init__()
        self.D = D
        self.M = nn.Parameter(th.full((1,), self.M_bounds[1]))
        self.temp = nn.Parameter(th.full((1,), self.temp_bounds[1]))

    @abstractmethod
    def forward(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    @classmethod
    def get_annealed_params(cls, t):
        assert 0 <= t and t <= 1
        if 1.-t < 1e-3:
            t = 1.
        elif t < 1e-3:
            t = 0.
        else:
            t = th.sigmoid(10.*(th.tensor(t, dtype=th.float32)-0.5))

        M_start, M_end = cls.M_bounds
        temp_start, temp_end = cls.temp_bounds
        new_M = M_start + (M_end - M_start) * t
        new_temp = temp_start + (temp_end - temp_start) * t
        new_params = {
            'M': th.tensor([new_M]),
            'temp': th.tensor([new_temp]),
        }
        return new_params

    def set_annealed_params(self, t):
        with th.no_grad():
            new_params_dict = self.get_annealed_params(t)
            self.M.copy_(new_params_dict['M'])
            self.temp.copy_(new_params_dict['temp'])


# https://github.com/take-koshizuka/NLSB/blob/main/models/potential_2d.py
class BoxPotential(LagrangianPotentialBase):
    xmin: float = -0.5
    xmax: float = 0.5
    ymin: float = -0.5
    ymax: float = 0.5

    def __init__(self, D: int = 2):
        super().__init__(D=D)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        x0, x1 = x[..., 0], x[..., 1]
        Ux = (th.sigmoid((x0 - self.xmin) / self.temp) -
              th.sigmoid((x0 - self.xmax) / self.temp))
        Uy = (th.sigmoid((x1 - self.ymin) / self.temp) -
              th.sigmoid((x1 - self.ymax) / self.temp))
        U = -Ux * Uy
        return self.M * U


class SlitPotential(LagrangianPotentialBase):
    xmin: float = -0.1
    xmax: float = 0.1
    ymin: float = -0.25
    ymax: float = 0.25
    M_bounds = (0., 1.)

    def __init__(self, D: int = 2):
        super().__init__(D=D)
        self.M = nn.Parameter(th.full((1,), self.M_bounds[1]))

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        x0, x1 = x[..., 0], x[..., 1]
        Ux = (th.sigmoid((x0 - self.xmin) / self.temp) -
                th.sigmoid((x0 - self.xmax) / self.temp))
        Uy = (th.sigmoid((x1 - self.ymin) / self.temp) -
                th.sigmoid((x1 - self.ymax) / self.temp)) - 1.
        U = Ux * Uy
        return self.M * U

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

    def __init__(self, D: int = 2):
        super().__init__(D=D)
        self.M = nn.Parameter(th.full((1,), self.M_bounds[1]))

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        x0, x1 = x[..., 0], x[..., 1]
        Ux1 = (th.sigmoid((x0 - self.xmin1) / self.temp) -
                th.sigmoid((x0 - self.xmax1) / self.temp))
        Ux2 = (th.sigmoid((x0 - self.xmin2) / self.temp) -
                th.sigmoid((x0 - self.xmax2) / self.temp))

        Uy1 = (th.sigmoid((x1 - self.ymin1) / self.temp) -
                th.sigmoid((x1 - self.ymax1) / self.temp)) - 1.

        Uy2 = (th.sigmoid((x1 - self.ymin2) / self.temp) -
                th.sigmoid((x1 - self.ymax2) / self.temp)) - 1.
        U = Ux1 * Uy1 + Ux2 * Uy2
        return self.M * U

class WellPotential(LagrangianPotentialBase):
    def __init__(self, D: int = 2):
        super().__init__(D=D)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        U = -th.sum(x**2, dim=-1)
        return self.M * U

class HillPotential(LagrangianPotentialBase):
    M_bounds = (0., 0.05)
    def __init__(self, D: int = 2):
        super().__init__(D=D)
        self.M = nn.Parameter(th.full((1,), self.M_bounds[1]))

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        U = -th.exp(-th.sum(x**2, dim=-1))
        return self.M * U


class GSB_GMM_Potential(LagrangianPotentialBase):
    centers_np = np.array([[6,6], [6,-6], [-6,-6]])
    radius = 1.5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __init__(self, D: int = 2):
        super().__init__(D=D)
        self.M = nn.Parameter(th.full((1,), self.M_bounds[1]))
        self.temp = nn.Parameter(th.full((1,), self.temp_bounds[1]))
        self.register_buffer('centers', th.tensor(self.centers_np, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist = th.linalg.norm(diff, dim=-1)

        V = -self.M * th.sigmoid((self.radius - dist) / self.temp)
        V_sum = th.sum(V, dim=-1)
        return V_sum


class GSB_VNeck_Potential(LagrangianPotentialBase):
    c_sq = 0.36
    coef = 5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __init__(self, D: int = 2):
        super().__init__(D=D)
        self.M = nn.Parameter(th.full((1,), self.M_bounds[1]))
        self.temp = nn.Parameter(th.full((1,), self.temp_bounds[1]))

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        x0, x1 = x[..., 0], x[..., 1]
        xs_sq = x * x
        d = self.coef * xs_sq[..., 0] - xs_sq[..., 1]

        return - self.M * th.sigmoid((-self.c_sq - d) / self.temp)


class GSB_STunnel_Potential(LagrangianPotentialBase):
    a, b, c = 20, 1, 90
    centers_np = np.array([[5,6], [-5,-6]])
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __init__(self, D: int = 2):
        super().__init__(D=D)
        self.M = nn.Parameter(th.full((1,), self.M_bounds[1]))
        self.temp = nn.Parameter(th.full((1,), self.temp_bounds[1]))
        self.register_buffer('centers', th.tensor(self.centers_np, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.D
        x0, x1 = x[..., 0], x[..., 1]
        center0_x, center0_y = self.centers[0, 0], self.centers[0, 1]
        center1_x, center1_y = self.centers[1, 0], self.centers[1, 1]

        V = 0.0
        d0 = self.a * (x0 - center0_x)**2 + self.b * (x1 - center0_y)**2
        V -= self.M * th.sigmoid((self.c - d0) / self.temp)

        d1 = self.a * (x0 - center1_x)**2 + self.b * (x1 - center1_y)**2
        V -= self.M * th.sigmoid((self.c - d1) / self.temp)

        return V
