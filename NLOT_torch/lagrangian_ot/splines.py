# Copyright (c) Meta Platforms, Inc. and affiliates

import torch as th

from dataclasses import dataclass

import stochman

def get_basis(D, num_nodes):
    stochman_spline = stochman.curves.CubicSpline(
        begin=th.zeros(D), end=th.ones(D), num_nodes=num_nodes)
    basis = stochman_spline.basis.detach()
    return basis


def compute_spline(x: th.Tensor, y: th.Tensor, basis: th.Tensor, params: th.Tensor, ts: th.Tensor) -> th.Tensor:
    assert x.ndim == 1 and y.ndim == 1
    assert isinstance(x, th.Tensor) and isinstance(y, th.Tensor)
    assert isinstance(basis, th.Tensor) and isinstance(params, th.Tensor)
    assert isinstance(ts, th.Tensor)

    degree = 4
    D = x.shape[0]
    num_edges = basis.shape[0] // degree
    params = params.reshape(num_edges + 1, D)

    coeffs = (basis @ params).reshape(num_edges, degree, D)
    idx = th.floor(ts * num_edges).clamp(0, num_edges - 1).long()
    power = th.arange(0, degree, device=ts.device, dtype=ts.dtype).reshape(1, -1)
    tpow = ts.reshape(-1, 1) ** power
    coeffs_idx = coeffs[idx]
    retval = tpow.unsqueeze(-1) * coeffs_idx
    retval = th.sum(retval, dim=-2)
    ts_expanded = ts.unsqueeze(-1)
    retval += x * (1 - ts_expanded) + y * ts_expanded
    return retval
