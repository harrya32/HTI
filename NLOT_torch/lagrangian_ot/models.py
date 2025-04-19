# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn
import torch.optim as optim

PotentialValueFn_t = Callable[[th.Tensor], th.Tensor]
PotentialGradientFn_t = Callable[[th.Tensor], th.Tensor]

class ModelBase(abc.ABC, nn.Module):
    """Base class for the neural solver models."""

    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def is_potential(self) -> bool:
        """Indicates if the module defines the potential's value or the gradient.

        Returns:
            ``True`` if the module defines the potential's value, ``False``
            if it defines the gradient.
        """
        pass

class MLP(ModelBase):
    """A non-convex MLP.

    Args:
        dim_hidden: sequence specifying size of hidden dimensions. The output
            dimension of the last layer is automatically set to 1 if
            :attr:`is_potential` is ``True``, or the dimension of the input otherwise
        is_potential: Model the potential if ``True``, otherwise
            model the gradient of the potential
        dim_input: The input dimension (required for PyTorch layer definition)
    """

    def __init__(self, dim_input: int, dim_hidden: Sequence[int], is_potential: bool = True):
        super().__init__()
        self._is_potential = is_potential

        layers = []
        last_dim = dim_input
        for n_hidden in dim_hidden:
            layers.append(nn.Linear(last_dim, n_hidden))
            layers.append(nn.LeakyReLU())
            last_dim = n_hidden

        if self.is_potential:
            layers.append(nn.Linear(last_dim, 1))
            self.net = nn.Sequential(*layers)
        else:
            self.final_layer = nn.Linear(last_dim, dim_input)
            self.feature_net = nn.Sequential(*layers)

    @property
    def is_potential(self) -> bool:
        return self._is_potential

    def forward(self, x: th.Tensor) -> th.Tensor:
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        assert x.ndim == 2, x.ndim

        if self.is_potential:
            z = self.net(x)
            z = z.squeeze(-1)
        else:
            features = self.feature_net(x)
            residual = self.final_layer(features)
            z = x + residual

        return z.squeeze(0) if squeeze else z
