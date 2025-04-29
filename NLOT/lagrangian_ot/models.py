# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state
from jax.nn import initializers

from ott.math import matrix_square_root

PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]

class ModelBase(abc.ABC, nn.Module):
    """Base class for the neural solver models."""

    @property
    @abc.abstractmethod
    def is_potential(self) -> bool:
        """Indicates if the module defines the potential's value or the gradient.

        Returns:
            ``True`` if the module defines the potential's value, ``False``
            if it defines the gradient.
        """

class MLP(ModelBase):
    """A non-convex MLP.

    Args:
        dim_hidden: sequence specifying size of hidden dimensions. The output
            dimension of the last layer is automatically set to 1 if
            :attr:`is_potential` is ``True``, or the dimension of the input otherwise
        is_potential: Model the potential if ``True``, otherwise
            model the gradient of the potential
    """

    dim_hidden: Sequence[int]
    is_potential: bool = True
    D: int = 2
    C: int = 0
    categorical: Optional[bool] = False 
    num_categories: Optional[int] = 4 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:    # noqa: D102
        squeeze = x.ndim == 1
        if squeeze:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2, x.ndim
        assert x.shape[1] == self.D + self.C

        x_ambient = x[:, :self.D]

        if self.categorical:
            category_index = x[:, self.D].astype(jnp.int32)
            category_one_hot = jax.nn.one_hot(category_index, num_classes=self.num_categories)
            z = jnp.concatenate([x_ambient, category_one_hot], axis=1)
        else:
            c = x[:, self.D:]
            z = jnp.concatenate([x_ambient, c], axis=1)
        
        for n_hidden in self.dim_hidden:
            Wx = nn.Dense(n_hidden, use_bias=True)
            z = nn.leaky_relu(Wx(z))

        if self.is_potential:
            Wx = nn.Dense(1, use_bias=True)
            z = Wx(z).squeeze(-1)
        else:
            Wx = nn.Dense(self.D, use_bias=True)
            z = x[:, :self.D] + Wx(z) 
            
            # Add condition to the end
            z = jnp.concatenate([z, x[:, self.D:]], axis=-1)

        return z.squeeze(0) if squeeze else z
