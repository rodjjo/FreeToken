from __future__ import annotations

from typing import Tuple

import torch
from freetoken.layers import BaseOP


class GroupedRMSNorm(BaseOP):
    """Grouped RMSNorm matching K2HorizonRMSNorm semantics.

    Hidden states of shape (..., hidden_size) are partitioned into `n_groups`
    along the last dimension. Variance is computed for each group independently,
    normalized, and then scaled by `weight`.
    """

    def __init__(self, size: int, eps: float, n_groups: int = 2) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        self.n_groups = n_groups
        assert size % n_groups == 0
        self.group_size = size // n_groups
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_fp32 = x.float().view(*orig_shape[:-1], self.n_groups, self.group_size)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = (x_fp32 * torch.rsqrt(variance + self.eps)).view(orig_shape)
        return (self.weight.float() * normed).to(x.dtype)


class GroupedRMSNormFused(BaseOP):
    """Grouped RMSNorm with residual accumulation matching RMSNormFused interface."""

    def __init__(self, size: int, eps: float, n_groups: int = 2) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        self.n_groups = n_groups
        assert size % n_groups == 0
        self.group_size = size // n_groups
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
            residual = x
        else:
            residual = x

        orig_shape = x.shape
        x_fp32 = x.float().view(*orig_shape[:-1], self.n_groups, self.group_size)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = (x_fp32 * torch.rsqrt(variance + self.eps)).view(orig_shape)
        return (self.weight.float() * normed).to(x.dtype), residual


__all__ = ["GroupedRMSNorm", "GroupedRMSNormFused"]
