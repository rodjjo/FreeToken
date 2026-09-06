"""KV-cache quantization and store routines: FP8 (e4m3) and sub-byte (Q4_0, Q6_0, Q8_0).

FP8 (e4m3) KV-cache storage: per-token, per-head symmetric quantization.
One KV row is one (token, kv_head) slice of head_dim elements. It is stored
as head_dim e4m3 bytes plus ONE fp32 scale shared by the whole row:
    scale = max(amax(row) / 448, eps)          # 448 == e4m3 finite max
    code  = round_e4m3(clamp(row / scale))     # RNE onto the e4m3 grid
    read  = code.to(f32) * scale               # in the attention kernels

Sub-byte and 8-bit block quantization (q8, q4, q6):
A program handles one (token, kv_head) pair: it loads that head's head_dim
values as a [head_dim // BLOCK, BLOCK] tile, reduces max-abs along the block, and
writes the quantized values plus one scale per block. K and V are done in the same
program.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_f32_to_u8,
    round_e4m3,
)

FP8 = torch.float8_e4m3fn
KV_SCALE_DTYPE = torch.float32
KV_QUANT_FP8 = "fp8"



def kv_codes_dtype() -> torch.dtype:
    """Storage dtype of one quantized KV element: e4m3 bytes in a uint8 buffer.

    A constant, deliberately. Every attempt to answer this per target -- triton's
    compile-time probe, then testing the pointer's element type at the load -- ended up
    picking an arm that did not match the buffer the host had just allocated (see the
    module header). torch still reads these bytes as fp8 whenever real numbers are
    wanted: :func:`codes_to_f32`.
    """
    return torch.uint8


def alloc_codes(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """A zero-filled code buffer of :func:`kv_codes_dtype` -- bytes, on every arch.

    Zero-filling matters because the pools read slots that were never written: a stale
    byte decodes to a real number (0x7F/0xFF even to NaN once reinterpreted as fp8),
    while 0x00 is exactly 0.0 (same reasoning as kvcache/bsa_pool.py).
    """
    return torch.zeros(shape, dtype=kv_codes_dtype(), device=device)


def codes_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """Decode a code buffer to fp32 ON THE HOST (torch's own e4m3 cast).

    Works on either storage dtype -- uint8 bytes are reinterpreted as fp8 first -- so
    a test or a debugging tool reads the same numbers on sm_86 as on sm_90, and does
    it through torch rather than through the software decoder it is checking.
    """
    if codes.dtype is not FP8:
        codes = codes.view(FP8)
    return codes.to(torch.float32)


@triton.jit
def _kv_quant_scatter_kernel(
    k_src,
    v_src,
    k_dst,
    v_dst,
    k_scale,
    v_scale,
    idx_ptr,
    stride_xs,  # K source row pitch, in elements (the qkv slice is wider than one row)
    stride_vx,  # V source row pitch. K and V need not share one: a .clamp() on one side
                # leaves it densely packed, so reusing K's pitch reads V off its rows.
    stride_kd,  # K cache row pitch, in elements (== HEADS * D)
    stride_vd,
    stride_ks,  # scale row pitch, in elements (== HEADS)
    stride_vs,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (token, kv_head): quantize the row and write it to slot
    ``idx_ptr[token]`` (the same ``out_loc`` the bf16 store scatters through)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(idx_ptr + t).to(tl.int64)  # int64: slots * row can pass 2**31
    d = tl.arange(0, BLOCK_D)
    mask = d < D

    off = h * D + d
    xk = tl.load(k_src + t * stride_xs + off, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(v_src + t * stride_vx + off, mask=mask, other=0.0).to(tl.float32)

    # 448 == e4m3 finite max; 1e-10 is the amax floor of the activation quant in
    # kernel/triton/fp8_block_linear.py (literals keep the kernel self-contained).
    sk = tl.maximum(tl.max(tl.abs(xk), axis=0), 1e-10) / 448.0
    sv = tl.maximum(tl.max(tl.abs(xv), axis=0), 1e-10) / 448.0
    qk = tl.clamp(xk / sk, -448.0, 448.0)
    qv = tl.clamp(xv / sv, -448.0, 448.0)

    # Straight-line, like the reader: round onto the e4m3 grid in ONE step (RNE) and
    # pack the bits into the byte buffer. No fp8 type on either side -- the reason this
    # feature stopped compiling twice is explained in kv_codes_dtype's docstring.
    out_k = e4m3_f32_to_u8(round_e4m3(qk))
    out_v = e4m3_f32_to_u8(round_e4m3(qv))

    tl.store(k_dst + pos * stride_kd + h * D + d, out_k, mask=mask)
    tl.store(v_dst + pos * stride_vd + h * D + d, out_v, mask=mask)
    tl.store(k_scale + pos * stride_ks + h, sk)
    tl.store(v_scale + pos * stride_vs + h, sv)


def quantize_kv_to_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize fresh K/V rows into an fp8 KV pool.

    ``k``/``v``  : ``[T, num_kv_heads * head_dim]`` compute-dtype rows -- exactly what
        the attention backends hand to ``store_kv`` (a slice of the qkv projection,
        so the row pitch may be wider than the row itself).
    ``out_loc``  : ``[T]`` device slot index per row (int32 or int64).
    ``*_cache``  : ``[num_slots, num_kv_heads, head_dim]`` of :func:`kv_codes_dtype`.
    ``*_scale``  : ``[num_slots, num_kv_heads]`` fp32, indexed by the SAME slot.
    """
    tokens = k.shape[0]
    if tokens == 0:
        return
    assert k.dim() == 2 and v.shape == k.shape, (k.shape, v.shape)
    assert k.stride(1) == 1 and v.stride(1) == 1, "K/V rows must be contiguous"
    heads, dim = k_cache.shape[1], k_cache.shape[2]
    assert k.shape[1] == heads * dim, (tuple(k.shape), tuple(k_cache.shape))
    assert k_cache.shape == v_cache.shape, (tuple(k_cache.shape), tuple(v_cache.shape))
    assert k_scale.shape == v_scale.shape == (k_cache.shape[0], heads), (
        tuple(k_scale.shape),
        tuple(k_cache.shape),
    )
    assert k_cache.dtype == v_cache.dtype == kv_codes_dtype(), (
        k_cache.dtype,
        kv_codes_dtype(),
    )
    assert k_scale.dtype == KV_SCALE_DTYPE, k_scale.dtype
    _kv_quant_scatter_kernel[(tokens, heads)](
        k,
        v,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        out_loc,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        k_scale.stride(0),
        v_scale.stride(0),
        D=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=1,
    )


from freetoken.kvcache.quant import BLOCK
from freetoken.kvcache.quant import LAYOUT_Q4 as _LAYOUT_Q4
from freetoken.kvcache.quant import LAYOUT_Q6 as _LAYOUT_Q6
from freetoken.kvcache.quant import LAYOUT_Q8 as _LAYOUT_Q8

# Bind to tl.constexpr so the @triton.jit kernel can use them in `if LAYOUT == ...`
# (Triton 3.7.1 forbids reading plain Python globals from inside a jit function).
LAYOUT_Q8 = tl.constexpr(_LAYOUT_Q8)
LAYOUT_Q4 = tl.constexpr(_LAYOUT_Q4)
LAYOUT_Q6 = tl.constexpr(_LAYOUT_Q6)


@triton.jit
def _store_kv_quant_kernel(
    k_ptr,  # [tokens, heads, D] source, compute dtype (bf16)
    v_ptr,
    kc_ptr,  # [slots, heads, D_PHYSICAL] destination, storage dtype
    vc_ptr,
    ks_ptr,  # [slots, heads, D // BLOCK] scales, fp16
    vs_ptr,
    indices_ptr,  # [tokens] destination slot per token
    stride_kt,
    stride_kh,
    stride_ct,
    stride_ch,
    stride_st,
    stride_sh,
    D: tl.constexpr,  # logical head_dim
    D_PHYSICAL: tl.constexpr,  # packed byte count (== D for q8)
    MAX_MAG: tl.constexpr,
    IS_INT: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,  # D // BLOCK
    LAYOUT: tl.constexpr,  # "q8" | "q4" | "q6"
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(indices_ptr + tok).to(tl.int64)

    # [NBLOCK, BLOCK] tile over head_dim: rows are quant blocks, columns the elements
    # sharing one scale. (memory: "BLK_STRIDE = payload_bytes_per_block" for the
    # WRITER; the read side just uses D.)
    offs = tl.arange(0, NBLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    scale_offs = tl.arange(0, NBLOCK)

    for is_v in tl.static_range(2):
        src_ptr = v_ptr if is_v else k_ptr
        dst_ptr = vc_ptr if is_v else kc_ptr
        sc_ptr = vs_ptr if is_v else ks_ptr

        x = tl.load(src_ptr + tok * stride_kt + head * stride_kh + offs).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        # An all-zero block quantizes to zeros under any positive scale; 1.0 keeps the
        # division finite.
        scale = tl.where(amax > 0, amax / MAX_MAG, 1.0)
        # Round to the stored precision before dividing, so the value written here and
        # the value the attention kernels read back are scaled by the identical number.
        scale = scale.to(sc_ptr.dtype.element_ty).to(tl.float32)
        # div_rn, not `/`: the plain operator is free to lower to a reciprocal multiply,
        # which disagrees with the torch reference on values sitting exactly between two
        # quantization steps. IEEE round-to-nearest divide makes the two bit-identical.
        q = tl.math.div_rn(x, scale[:, None])
        if IS_INT:
            # Round half away from zero (what GGUF's Q8_0 / Q4_0 / Q6_0 do), then clamp.
            # The 4-bit signed range is [-8, 7] (16 levels, stored as unsigned
            # 0..15 -- 8 maps to -8 in the XOR-sub sign extension) and 6-bit
            # signed is [-32, 31] (64 levels). For Q4, MAX_MAG=8 and the
            # writer must clamp at MAX_MAG-1=7 so the dequant sees a real
            # signed value; for Q6, MAX_MAG=31 already aligns the storage
            # with the 6-bit signed range so we clamp at MAX_MAG.
            q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
            if LAYOUT == LAYOUT_Q4:
                q = tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG - 1.0)
            else:
                q = tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG)
        else:
            # The native fp32 -> float8e4nv downcast does not round to nearest on
            # every arch (it lowers as a truncating fp32 -> fp16 -> e4m3 double-round
            # on sm_89), so values just above a grid midpoint collapse downward and
            # disagree with the RNE torch reference. Round explicitly first.
            from freetoken.kernel.triton.e4m3_compat import round_e4m3

            q = round_e4m3(tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG))

        # ---- pack into the storage dtype ----
        if LAYOUT == LAYOUT_Q8:
            # Direct: one value per byte.
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + offs,
                q.to(dst_ptr.dtype.element_ty),
            )
        elif LAYOUT == LAYOUT_Q4:
            # Pack two 4-bit values per byte: low nibble = even index, high = odd.
            qi = q.to(tl.int8)  # [NBLOCK, 32] in [-8, 7]
            low4 = qi & 0xF  # [NBLOCK, 32] each in [0, 15]
            # Reshape to [NBLOCK, 16, 2] (last dim = 2 to satisfy tl.split), then
            # split the last dim into the even/odd halves, each [NBLOCK, 16].
            pairs = tl.reshape(low4, (NBLOCK, 16, 2))
            lo, hi = tl.split(pairs)  # each [NBLOCK, 16]
            packed = (lo | (hi << 4)).to(tl.uint8)
            byte_offs = tl.arange(0, NBLOCK)[:, None] * 16 + tl.arange(0, 16)[None, :]
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + byte_offs,
                packed,
            )
        elif LAYOUT == LAYOUT_Q6:
            # 32 values -> 16-byte low plane (nibbles) + 8-byte high plane (2-bit tops
            # at bit positions 0, 2, 4, 6). Layout per block: lo first, then hi.
            qi = q.to(tl.int8)  # [NBLOCK, 32] in [-32, 31]
            lo4 = qi & 0xF  # low 4 bits
            hi2 = (qi >> 4) & 0x3  # top 2 bits
            lo_pairs = tl.reshape(lo4, (NBLOCK, 16, 2))
            lo, hi_lo = tl.split(lo_pairs)  # each [NBLOCK, 16]
            packed_lo = (lo | (hi_lo << 4)).to(tl.uint8)
            hi_groups = tl.reshape(hi2, (NBLOCK, 8, 4))
            # masks for bit positions 0, 2, 4, 6 = 1, 4, 16, 64; broadcasts to
            # [NBLOCK, 8, 4] for elementwise multiply along the value axis.
            idx = tl.arange(0, 4)
            masks = tl.where(
                idx == 0, 1,
                tl.where(idx == 1, 4,
                         tl.where(idx == 2, 16, 64)),
            )
            packed_hi = tl.sum(hi_groups * masks[None, None, :], axis=2).to(tl.uint8)
            # [NBLOCK, 8]
            base = tl.arange(0, NBLOCK)[:, None] * 24
            lo_offs = base + tl.arange(0, 16)[None, :]
            hi_offs = base + 16 + tl.arange(0, 8)[None, :]
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + lo_offs,
                packed_lo,
            )
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + hi_offs,
                packed_hi,
            )
        else:
            tl.static_assert(False, f"unknown LAYOUT {LAYOUT!r}")

        # Scales: one per block (logical, not packed). Same address math as Q8.
        tl.store(
            sc_ptr + slot * stride_st + head * stride_sh + scale_offs,
            scale.to(sc_ptr.dtype.element_ty),
        )


def store_kv_quant(
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec,
) -> None:
    """Quantize ``k``/``v`` ``[tokens, heads, D]`` into the pool slots ``indices``.

    ``k_cache``/``v_cache`` are ``[slots, heads, D_PHYSICAL]`` in the spec's storage
    dtype, where ``D_PHYSICAL == D`` for 8-bit and ``D * bits // 8`` for sub-byte.
    ``k_scale``/``v_scale`` are ``[slots, heads, D // BLOCK]`` in fp16.
    """
    from freetoken.kvcache.quant import BLOCK

    num_tokens, num_heads, head_dim = k.shape
    if num_tokens == 0:
        return
    assert head_dim % BLOCK == 0, f"head_dim {head_dim} not a multiple of {BLOCK}"
    # The cache's last axis is the PACKED byte count, which differs from the source's
    # head_dim for sub-byte schemes. We pass both as constexprs to the kernel.
    d_physical = k_cache.shape[-1]
    expected_physical = spec.physical_head_dim(head_dim) if spec.enabled else head_dim
    assert d_physical == expected_physical, (
        f"cache physical dim {d_physical} != spec {spec.name} expected {expected_physical}"
    )
    _store_kv_quant_kernel[(num_tokens, num_heads)](
        k,
        v,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        indices,
        k.stride(0),
        k.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        D=head_dim,
        D_PHYSICAL=d_physical,
        MAX_MAG=spec.max_magnitude,
        IS_INT=spec.is_integer,
        BLOCK=BLOCK,
        NBLOCK=head_dim // BLOCK,
        LAYOUT=spec.layout,
        num_warps=4,
    )


__all__ = [
    "FP8",
    "KV_QUANT_FP8",
    "KV_SCALE_DTYPE",
    "alloc_codes",
    "codes_to_f32",
    "kv_codes_dtype",
    "quantize_kv_to_cache",
    "store_kv_quant",
]
