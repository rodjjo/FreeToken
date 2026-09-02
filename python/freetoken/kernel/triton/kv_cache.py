"""Dynamically-scaled E4M3 paged-KV cache writes.

The cache keeps one FP32 scale per physical token row and KV head, independently
for K and V. Attention dequantizes in registers; model activations remain BF16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from freetoken.kernel.triton.e4m3_compat import round_e4m3


FP8_MAX = tl.constexpr(448.0)


@triton.jit
def _store_fp8_cache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    k_ptr,
    v_ptr,
    stride_kcs,
    stride_kch,
    stride_vcs,
    stride_vch,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(indices_ptr + token)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D

    k = tl.load(
        k_ptr + token * stride_kt + head * stride_kh + offs_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + token * stride_vt + head * stride_vh + offs_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    k_absmax = tl.max(tl.abs(k), axis=0)
    v_absmax = tl.max(tl.abs(v), axis=0)
    k_s = tl.maximum(k_absmax / FP8_MAX, 1.0e-12)
    v_s = tl.maximum(v_absmax / FP8_MAX, 1.0e-12)
    k_q = round_e4m3(tl.maximum(tl.minimum(k / k_s, FP8_MAX), -FP8_MAX))
    v_q = round_e4m3(tl.maximum(tl.minimum(v / v_s, FP8_MAX), -FP8_MAX))

    tl.store(
        k_cache_ptr + slot * stride_kcs + head * stride_kch + offs_d,
        k_q,
        mask=mask,
    )
    tl.store(
        v_cache_ptr + slot * stride_vcs + head * stride_vch + offs_d,
        v_q,
        mask=mask,
    )
    tl.store(k_scale_ptr + slot * stride_kss + head * stride_ksh, k_s)
    tl.store(v_scale_ptr + slot * stride_vss + head * stride_vsh, v_s)


def store_fp8_cache(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Quantize BF16 K/V rows into E4M3 cache storage and save dynamic scales."""
    fp8 = torch.float8_e4m3fn
    assert k_cache.is_cuda and v_cache.is_cuda
    assert k_scale.is_cuda and v_scale.is_cuda and indices.is_cuda
    assert k_cache.dtype == fp8 and v_cache.dtype == fp8
    assert k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32
    assert k_cache.dim() == 3 and v_cache.shape == k_cache.shape
    assert k_scale.dim() == 2 and v_scale.shape == k_scale.shape
    assert k_scale.shape == k_cache.shape[:2]
    assert k.dim() == 3 and v.shape == k.shape
    assert k.shape[1:] == k_cache.shape[1:]
    assert k.shape[0] == indices.numel()

    num_tokens, num_heads, head_dim = k.shape
    block_d = triton.next_power_of_2(head_dim)
    _store_fp8_cache_kernel[(num_tokens, num_heads)](
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        indices,
        k,
        v,
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        D=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


__all__ = ["store_fp8_cache"]
