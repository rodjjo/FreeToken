"""Qwen3.5-family vision tower (Qwen3-VL encoder + patch merger).

Shared by every checkpoint the ``qwen3_5_moe`` module serves -- the MoE variants
(Ornith-1.5-35B-A3B, Qwen3.6-35B-A3B) and the dense ones (Qwen3.8-27B) carry a
byte-identical tower: 27 blocks, hidden 1152, patch 16, spatial merge 2. Only
``out_hidden_size`` differs, matching each text stack's hidden width.

Two facts about this tower drive the implementation:

* **It is never quantized.** All 333 visual tensors in
  ``ornith-ai/Ornith-1.5-35B-A3B-NVFP4`` are bf16 with no scales anywhere -- 0.832 GiB
  resident, unaffected by the NVFP4 export. So this file has no quant paths: plain
  ``nn.Linear``-shaped buffers throughout, and the tower is opt-in (``--vision``)
  precisely because those bytes are pure overhead for text-only serving.

* **The learned position table is a 48x48 grid, not a hard limit.**
  ``pos_embed.weight`` is ``[2304, 1152]`` and ``2304 = 48**2``; each image's
  ``(h, w)`` patch grid resamples it bilinearly. A 2880x1800 screenshot needs 20,160
  pre-merge patches -- 8.8x the table -- and reads correctly, which is only possible
  because of that resampling. Getting the interpolation wrong therefore does not fail
  loudly; it quietly degrades every image above 768x768.

Numerics follow ``transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`` exactly:
pre-norm residual blocks with LayerNorm (eps 1e-6, with bias -- not RMSNorm), a
GELU-tanh MLP with bias on both projections, non-causal attention with 2D RoPE applied
in fp32, and a merger that normalizes at 1152 *before* concatenating each 2x2 block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, OPList


@dataclass
class Qwen3_5VisionConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    in_channels: int
    out_hidden_size: int
    num_position_embeddings: int
    layer_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def grid_per_side(self) -> int:
        """Side of the square learned position grid (48 for num_position_embeddings=2304)."""
        return int(round(self.num_position_embeddings**0.5))

    @property
    def merged_size(self) -> int:
        return self.hidden_size * self.spatial_merge_size**2


def parse_vision_config(vc: Any, out_hidden_size: int) -> Qwen3_5VisionConfig | None:
    """``vision_config`` -> :class:`Qwen3_5VisionConfig`, or None when there is none."""
    if vc is None:
        return None
    return Qwen3_5VisionConfig(
        hidden_size=int(vc.hidden_size),
        num_layers=int(getattr(vc, "depth", getattr(vc, "num_hidden_layers", 0))),
        num_heads=int(vc.num_heads),
        intermediate_size=int(vc.intermediate_size),
        patch_size=int(vc.patch_size),
        temporal_patch_size=int(getattr(vc, "temporal_patch_size", 2)),
        spatial_merge_size=int(getattr(vc, "spatial_merge_size", 2)),
        in_channels=int(getattr(vc, "in_channels", 3)),
        out_hidden_size=int(getattr(vc, "out_hidden_size", out_hidden_size)),
        num_position_embeddings=int(getattr(vc, "num_position_embeddings", 2304)),
    )


# ======================================================================================
# Position handling: the bilinear resample of the learned grid, and the 2D RoPE.
# ======================================================================================
def interpolation_taps(
    grid_thw: torch.Tensor, grid_per_side: int, merge: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indices, weights)`` of shape ``[total_patches, 4]``: the bilinear taps that resample
    the ``grid_per_side x grid_per_side`` learned table onto each image's ``(h, w)`` grid.

    Equivalent to ``F.interpolate(mode="bilinear", align_corners=True)`` with border padding,
    expressed as a gather so the table stays an embedding lookup. Patches are emitted in
    spatial-merge-block order (the order the merger consumes), not raster order.
    """
    idx_out: list[torch.Tensor] = []
    w_out: list[torch.Tensor] = []
    side = grid_per_side
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        r_tap, r_wt = _axis_taps(torch.arange(h, device=device), h, side)
        c_tap, c_wt = _axis_taps(torch.arange(w, device=device), w, side)
        # 2D separable: the outer product of the per-axis taps, in (r, c) tap order.
        idx = (r_tap[:, None, :, None] * side + c_tap[None, :, None, :]).reshape(h, w, 4)
        wts = (r_wt[:, None, :, None] * c_wt[None, :, None, :]).reshape(h, w, 4)
        idx, wts = _to_merge_order(idx, h, w, merge), _to_merge_order(wts, h, w, merge)
        idx_out.append(idx.repeat(t, 1))
        w_out.append(wts.repeat(t, 1))
    return torch.cat(idx_out), torch.cat(w_out)


def _axis_taps(index: torch.Tensor, size: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One axis of the bilinear resample: ``(taps, weights)`` of shape ``[len(index), 2]``.

    ``src`` is deliberately NOT clamped before ``floor``: only the resulting taps are. Clamping
    the coordinate instead moves the sample point, which is invisible while upsampling (h < side,
    where src stays in range) and wrong while downsampling -- a 1080p grid puts col 0 at
    src = -0.1, whose correct weights are the border-clamped pair, not (1, 0)."""
    # align_corners=True, which is what Qwen3_5MoeVisionModel sets
    # (``interpolation_align_corners = True``): endpoints map to 0 and side-1, the closed form
    # of ``torch.linspace(0, side-1, size)[index]``. The align_corners=False half-pixel form
    # gives a DIFFERENT resample -- it agrees only when upsampling small grids, so a test that
    # picks the wrong flag on both sides passes while the model quietly degrades.
    src = index.to(torch.float32) * (side - 1) / max(size - 1, 1)
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)  # padding="border"
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    return taps, (1 - distance).clamp(min=0)


def _to_merge_order(x: torch.Tensor, h: int, w: int, merge: int) -> torch.Tensor:
    """``[h, w, taps]`` -> ``[h*w, taps]`` reordered so each 2x2 merge block is contiguous."""
    taps = x.shape[-1]
    x = x.view(h // merge, merge, w // merge, merge, taps)
    return x.permute(0, 2, 1, 3, 4).reshape(h * w, taps)


def rope_position_ids(
    grid_thw: torch.Tensor, merge: int, device: torch.device
) -> torch.Tensor:
    """``[total_patches, 2]`` (row, col) ids in merge-block order, for the 2D RoPE."""
    out: list[torch.Tensor] = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        rows = torch.arange(h, device=device).view(h, 1).expand(h, w)
        cols = torch.arange(w, device=device).view(1, w).expand(h, w)
        ids = torch.stack([rows, cols], dim=-1)
        out.append(_to_merge_order(ids, h, w, merge).repeat(t, 1))
    return torch.cat(out)


def rope_cos_sin(
    position_ids: torch.Tensor, head_dim: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE tables for ``[N, 2]`` (row, col) ids, shaped ``[N, head_dim]``.

    The frequency ladder is built over ``head_dim // 2`` -- HALF the head width -- and the two
    position axes each take half of it, so ``(row, col)`` freqs flatten to exactly
    ``head_dim // 2`` columns. Duplicating that once gives the full width, which is what
    ``rotate_half`` expects. Building the ladder over the full ``head_dim`` instead (or
    duplicating twice) yields the right SHAPE with the wrong frequencies, so it fails only as a
    quiet accuracy loss."""
    half = head_dim // 2                                     # 36 for head_dim 72
    inv = 1.0 / (theta ** (torch.arange(0, half, 2, device=position_ids.device,
                                        dtype=torch.float32) / half))   # 18 freqs
    freqs = position_ids.float()[..., None] * inv            # [N, 2, 18]
    emb = freqs.flatten(1)                                   # [N, 36]
    emb = torch.cat((emb, emb), dim=-1)                      # [N, 72] = head_dim
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


# ======================================================================================
# Modules
# ======================================================================================
class Qwen3_5VisionMLP(BaseOP):
    """GELU-tanh MLP, bias on both projections (not a SwiGLU)."""

    def __init__(self, vc: Qwen3_5VisionConfig):
        self.linear_fc1 = _Linear(vc.hidden_size, vc.intermediate_size)
        self.linear_fc2 = _Linear(vc.intermediate_size, vc.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x), approximate="tanh"))


class Qwen3_5VisionAttention(BaseOP):
    """Non-causal full attention over one image's patches, with 2D RoPE in fp32."""

    def __init__(self, vc: Qwen3_5VisionConfig):
        self.num_heads = vc.num_heads
        self.head_dim = vc.head_dim
        self.qkv = _Linear(vc.hidden_size, vc.hidden_size * 3)
        self.proj = _Linear(vc.hidden_size, vc.hidden_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cu_seqlens: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        qkv = self.qkv.forward(x).view(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # fp32 rotation, as the reference does, then back to the activation dtype
        c, s = cos[:, None, :].float(), sin[:, None, :].float()
        qf, kf = q.float(), k.float()
        q = ((qf * c) + (_rotate_half(qf) * s)).to(x.dtype)
        k = ((kf * c) + (_rotate_half(kf) * s)).to(x.dtype)
        # Each image is its own attention window: no patch may attend across cu_seqlens.
        out = torch.empty_like(q)
        bounds = cu_seqlens.tolist()
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi <= lo:
                continue
            qs = q[lo:hi].transpose(0, 1)[None]     # [1, heads, len, dim]
            ks = k[lo:hi].transpose(0, 1)[None]
            vs = v[lo:hi].transpose(0, 1)[None]
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
            out[lo:hi] = o[0].transpose(0, 1)
        return self.proj.forward(out.reshape(n, -1))


class Qwen3_5VisionBlock(BaseOP):
    """Pre-norm residual block: x + attn(norm1(x)), then x + mlp(norm2(x))."""

    def __init__(self, vc: Qwen3_5VisionConfig):
        self.norm1 = _LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.norm2 = _LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.attn = Qwen3_5VisionAttention(vc)
        self.mlp = Qwen3_5VisionMLP(vc)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cu_seqlens: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin, cu_seqlens)
        return x + self.mlp.forward(self.norm2.forward(x))


class Qwen3_5VisionPatchEmbed(BaseOP):
    """Conv3d over ``[temporal_patch, patch, patch]`` with kernel == stride."""

    def __init__(self, vc: Qwen3_5VisionConfig):
        self.vc = vc
        self.proj = _Conv3d(vc.in_channels, vc.hidden_size,
                            (vc.temporal_patch_size, vc.patch_size, vc.patch_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vc = self.vc
        x = pixel_values.view(-1, vc.in_channels, vc.temporal_patch_size,
                              vc.patch_size, vc.patch_size)
        return self.proj.forward(x.to(self.proj.weight.dtype)).view(-1, vc.hidden_size)


class Qwen3_5VisionPatchMerger(BaseOP):
    """Normalize at hidden_size, then fold each 2x2 block into out_hidden_size.

    The norm runs BEFORE the concatenation (``use_postshuffle_norm=False`` upstream), which is
    why ``merger.norm`` is 1152-wide while ``linear_fc1`` is 4608 -- reading it the other way
    round produces plausible-looking output that is subtly wrong.
    """

    def __init__(self, vc: Qwen3_5VisionConfig):
        self.merged = vc.merged_size
        self.norm = _LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.linear_fc1 = _Linear(vc.merged_size, vc.merged_size)
        self.linear_fc2 = _Linear(vc.merged_size, vc.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x).view(-1, self.merged)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


class Qwen3_5VisionModel(BaseOP):
    """patch_embed -> resampled pos_embed -> 2D RoPE -> 27 blocks -> merger."""

    def __init__(self, vc: Qwen3_5VisionConfig):
        self.vc = vc
        self.patch_embed = Qwen3_5VisionPatchEmbed(vc)
        self.pos_embed = _Embedding(vc.num_position_embeddings, vc.hidden_size)
        # OPList, not a plain list: BaseOP.load_state_dict walks __dict__ and only
        # recurses into BaseOP values, so a bare list leaves every block unloaded.
        self.blocks = OPList([Qwen3_5VisionBlock(vc) for _ in range(vc.num_layers)])
        self.merger = Qwen3_5VisionPatchMerger(vc)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        vc = self.vc
        dev = pixel_values.device
        x = self.patch_embed.forward(pixel_values)

        idx, wts = interpolation_taps(grid_thw, vc.grid_per_side, vc.spatial_merge_size, dev)
        pos = (self.pos_embed.forward(idx) * wts[:, :, None]).sum(1)
        x = x + pos.to(x.dtype)

        pids = rope_position_ids(grid_thw, vc.spatial_merge_size, dev)
        cos, sin = rope_cos_sin(pids, vc.head_dim)

        cu = _cu_seqlens(grid_thw, dev)
        for blk in self.blocks.op_list:
            x = blk.forward(x, cos, sin, cu)
        return self.merger.forward(x)


def _cu_seqlens(grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Attention window boundaries: one window per frame, so patches never attend across
    images (or across a video's frames)."""
    lens: list[int] = []
    for t, h, w in grid_thw.tolist():
        lens.extend([int(h) * int(w)] * int(t))
    return F.pad(torch.tensor(lens, device=device, dtype=torch.int32).cumsum(0), (1, 0))


# ======================================================================================
# Buffer holders. Plain bf16 parameters -- this tower carries no quantization.
# ======================================================================================
class _Linear(BaseOP):
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.empty(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class _LayerNorm(BaseOP):
    def __init__(self, size: int, eps: float):
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class _Embedding(BaseOP):
    def __init__(self, num: int, dim: int):
        self.weight = torch.empty(num, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)


class _Conv3d(BaseOP):
    def __init__(self, in_ch: int, out_ch: int, kernel: tuple[int, int, int]):
        self.kernel = kernel
        self.weight = torch.empty(out_ch, in_ch, *kernel)
        self.bias = torch.empty(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.weight, self.bias, stride=self.kernel)


__all__ = [
    "Qwen3_5VisionConfig",
    "Qwen3_5VisionModel",
    "parse_vision_config",
]
