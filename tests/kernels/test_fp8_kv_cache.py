from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _quantized_cache(k: torch.Tensor, v: torch.Tensor):
    from freetoken.kernel.triton.kv_cache import store_fp8_cache

    slots, heads, dim = k.shape
    k_cache = torch.empty((slots, heads, dim), dtype=torch.float8_e4m3fn, device=k.device)
    v_cache = torch.empty_like(k_cache)
    k_scale = torch.empty((slots, heads), dtype=torch.float32, device=k.device)
    v_scale = torch.empty_like(k_scale)
    indices = torch.arange(slots, dtype=torch.int32, device=k.device)
    store_fp8_cache(
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        indices=indices,
        k=k,
        v=v,
    )
    return k_cache, v_cache, k_scale, v_scale


def test_fp8_kv_store_matches_dynamic_per_head_reference():
    from freetoken.kernel.triton.kv_cache import store_fp8_cache

    torch.manual_seed(21)
    device = torch.device("cuda")
    tokens, slots, heads, dim = 5, 11, 2, 256
    k = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device=device) * 3
    v = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device=device) * 2
    k[0].zero_()
    v[1].zero_()
    indices = torch.tensor([7, 1, 9, 3, 5], dtype=torch.int32, device=device)
    k_cache = torch.empty(slots, heads, dim, dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.empty_like(k_cache)
    k_scale = torch.empty(slots, heads, dtype=torch.float32, device=device)
    v_scale = torch.empty_like(k_scale)

    store_fp8_cache(
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        indices=indices,
        k=k,
        v=v,
    )

    for source, cache, scales in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        ref_scale = (source.float().abs().amax(dim=-1) / 448.0).clamp(min=1e-12)
        ref = (source.float() / ref_scale[..., None]).clamp(-448, 448)
        ref = ref.to(torch.float8_e4m3fn)
        assert torch.equal(cache[indices.long()].view(torch.uint8), ref.view(torch.uint8))
        torch.testing.assert_close(scales[indices.long()], ref_scale)


def test_scaled_fp8_cache_matches_explicitly_dequantized_attention_paths():
    from freetoken.kernel.triton.attention import (
        decode_paged_attention,
        extend_paged_attention,
        paged_attention,
    )

    torch.manual_seed(22)
    device = torch.device("cuda")
    head_dim, q_heads, kv_heads, num_kv = 256, 16, 2, 64
    k = torch.randn(num_kv, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    k_cache, v_cache, k_scale, v_scale = _quantized_cache(k, v)
    k_dequant = (k_cache.float() * k_scale[..., None]).to(torch.bfloat16)
    v_dequant = (v_cache.float() * v_scale[..., None]).to(torch.bfloat16)

    q = torch.randn(1, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    indptr = torch.tensor([0, num_kv], dtype=torch.int32, device=device)
    indices = torch.arange(num_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor([num_kv - 1], dtype=torch.int64, device=device)
    q_to_req = torch.zeros(1, dtype=torch.int32, device=device)
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q, k_cache, v_cache, indptr, indices, q_to_req, q_positions, sm_scale,
        k_scale=k_scale, v_scale=v_scale,
    )
    expected = paged_attention(
        q, k_dequant, v_dequant, indptr, indices, q_to_req, q_positions, sm_scale,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    max_splits = 8
    scratch = torch.empty(1, q_heads, max_splits, head_dim, device=device)
    lse = torch.empty(1, q_heads, max_splits, device=device)
    splits = torch.full((1,), max_splits, dtype=torch.int32, device=device)
    actual = decode_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_positions,
        scratch, lse, splits, max_splits, sm_scale,
        k_scale=k_scale, v_scale=v_scale,
    )
    expected = decode_paged_attention(
        q, k_dequant, v_dequant, indptr, indices, q_positions,
        torch.empty_like(scratch), torch.empty_like(lse), splits, max_splits, sm_scale,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    extend_len = 4
    prefix_len = num_kv - extend_len
    q_extend = torch.randn(
        extend_len, q_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([prefix_len], dtype=torch.int32, device=device)
    actual = extend_paged_attention(
        q_extend, k_cache, v_cache, qo_indptr, indptr, indices, prefix_lens,
        extend_len, sm_scale, k_extend=k[prefix_len:], v_extend=v[prefix_len:],
        k_scale=k_scale, v_scale=v_scale,
    )
    expected = extend_paged_attention(
        q_extend, k_dequant, v_dequant, qo_indptr, indptr, indices, prefix_lens,
        extend_len, sm_scale, k_extend=k[prefix_len:], v_extend=v[prefix_len:],
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
