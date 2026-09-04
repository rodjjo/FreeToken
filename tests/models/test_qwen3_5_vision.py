"""Qwen3.5-family vision tower against the transformers reference.

Every one of these pins a mistake that was actually made while writing the tower, and each
one is invisible from the output: a wrong resample degrades large images only, a wrong
`align_corners` degrades everything above 768x768, a bare list of blocks loads no weights at
all yet raises nothing until a shape check much later.

The interpolation tests compare against `transformers.vision_utils` directly rather than
re-deriving the formula, because re-deriving it is exactly how the `align_corners` bug
survived: the first version of this file passed `align_corners=False` to BOTH sides, so two
wrong things agreed.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from freetoken.models.qwen3_5_moe.vision import (
    Qwen3_5VisionConfig,
    interpolation_taps,
    rope_cos_sin,
    rope_position_ids,
)

DEV = torch.device("cuda")
SIDE = 48          # sqrt(2304), the learned position grid's side
MERGE = 2


def _cfg(**over) -> Qwen3_5VisionConfig:
    base = dict(hidden_size=1152, num_layers=27, num_heads=16, intermediate_size=4304,
                patch_size=16, temporal_patch_size=2, spatial_merge_size=MERGE, in_channels=3,
                out_hidden_size=2048, num_position_embeddings=SIDE * SIDE)
    base.update(over)
    return Qwen3_5VisionConfig(**base)


# --------------------------------------------------------------------------- config
def test_grid_per_side_is_the_square_root_of_the_table():
    """2304 entries are a 48x48 grid, not a cap on patches: an image with more patches than
    that resamples the grid. Reading it as a limit is what makes large images silently wrong."""
    assert _cfg().grid_per_side == 48
    assert _cfg().merged_size == 1152 * 4        # 2x2 block concatenated -> 4608
    assert _cfg().head_dim == 72                 # 1152 / 16


# ------------------------------------------------------------------- the resample
@pytest.mark.parametrize("grid", [
    [[1, 4, 6]],                  # upsampling: fewer patches than the grid's 48 per side
    [[1, 34, 60]],                # downsampling: ~1080p, where the two align_corners differ
    [[1, 90, 56]],                # a Retina screenshot, 5040 tokens
    [[1, 4, 6], [1, 8, 8]],       # two images packed in one batch
])
def test_interpolation_matches_the_reference(grid):
    """align_corners=True, which is what Qwen3_5MoeVisionModel sets. The False (half-pixel)
    form agrees while upsampling and diverges while downsampling, so the 4x6 case alone would
    not catch it -- 34x60 is here for exactly that reason."""
    g = torch.tensor(grid, device=DEV)
    ref_i, ref_w = get_vision_interpolation_indices_and_weights(
        g, num_grid_per_side=SIDE, mode="bilinear", align_corners=True,
        spatial_merge_size=MERGE, padding="border",
    )
    got_i, got_w = interpolation_taps(g, SIDE, MERGE, DEV)
    assert tuple(got_i.shape) == tuple(ref_i.shape)
    assert torch.equal(got_i.to(ref_i.dtype), ref_i)
    assert (got_w.float() - ref_w.float()).abs().max().item() < 1e-4


def test_the_false_align_corners_form_would_disagree_on_a_downsampled_grid():
    """A guard on the guard: if this ever stops differing, the test above has stopped being
    able to tell the two resamples apart and no longer pins anything."""
    g = torch.tensor([[1, 34, 60]], device=DEV)
    a, _ = get_vision_interpolation_indices_and_weights(
        g, num_grid_per_side=SIDE, mode="bilinear", align_corners=True,
        spatial_merge_size=MERGE, padding="border")
    b, _ = get_vision_interpolation_indices_and_weights(
        g, num_grid_per_side=SIDE, mode="bilinear", align_corners=False,
        spatial_merge_size=MERGE, padding="border")
    assert not torch.equal(a, b)


def test_patches_come_out_in_merge_block_order():
    """The merger folds each contiguous run of 4 rows into one token, so the encoder has to
    emit 2x2 blocks together. Raster order gives the right shape and scrambles the image."""
    g = torch.tensor([[1, 4, 6]], device=DEV)
    ids = rope_position_ids(g, MERGE, DEV)
    # first block covers rows 0-1, cols 0-1
    assert ids[:4].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    # second block steps across, not down
    assert ids[4:8].tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]


def test_position_ids_match_the_reference():
    for grid in ([[1, 4, 6]], [[1, 34, 60]], [[1, 4, 6], [1, 8, 8]]):
        g = torch.tensor(grid, device=DEV)
        ref = get_vision_position_ids(g, MERGE).reshape(-1, 2)
        got = rope_position_ids(g, MERGE, DEV)
        assert torch.equal(got, ref.to(got.dtype))


# ------------------------------------------------------------------------- 2D RoPE
def test_rope_ladder_is_built_over_half_the_head_width():
    """The frequency ladder spans head_dim // 2 and the two axes split it, so (row, col)
    flatten to head_dim // 2 columns and duplicate to head_dim. Building it over the full
    head_dim yields the right SHAPE with the wrong frequencies -- an accuracy loss only."""
    ids = torch.tensor([[0, 0], [1, 2], [7, 3]], device=DEV)
    cos, sin = rope_cos_sin(ids, head_dim=72)
    assert cos.shape == (3, 72) and sin.shape == (3, 72)
    # duplicated halves
    assert torch.equal(cos[:, :36], cos[:, 36:])
    # position (0, 0) is the identity rotation
    assert torch.allclose(cos[0], torch.ones_like(cos[0]))
    assert torch.allclose(sin[0], torch.zeros_like(sin[0]))


def test_rope_matches_the_reference_construction():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionRotaryEmbedding,
    )

    head_dim = 72
    ids = get_vision_position_ids(torch.tensor([[1, 4, 6]], device=DEV), MERGE).reshape(-1, 2)
    ref_rot = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2).to(DEV)
    half = ref_rot(ids)                                  # [N, head_dim // 2]
    emb = torch.cat((half, half), dim=-1)
    got_cos, got_sin = rope_cos_sin(ids, head_dim)
    assert torch.allclose(emb.cos(), got_cos, atol=1e-5)
    assert torch.allclose(emb.sin(), got_sin, atol=1e-5)


# ------------------------------------------------------------------------ structure
def test_blocks_are_an_oplist_so_their_weights_load():
    """BaseOP.load_state_dict walks __dict__ and recurses only into BaseOP values. A plain
    list of blocks leaves all 324 block tensors unconsumed -- which surfaces as an
    'Unexpected keys' error far from the cause, or silently as random weights if the caller
    passes strict=False."""
    from freetoken.layers import OPList
    from freetoken.models.qwen3_5_moe.vision import Qwen3_5VisionModel
    from freetoken.utils import torch_dtype

    with torch_dtype(torch.bfloat16):
        model = Qwen3_5VisionModel(_cfg(num_layers=2))
    assert isinstance(model.blocks, OPList)
    keys = set(model.state_dict())
    for i in (0, 1):
        assert f"blocks.{i}.attn.qkv.weight" in keys
        assert f"blocks.{i}.norm1.bias" in keys
    assert "merger.linear_fc2.weight" in keys and "pos_embed.weight" in keys
