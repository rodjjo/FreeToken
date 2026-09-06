from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx

from .vision import Qwen3_5VisionModel
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # None for a text-only build, so _merge_multimodal returns immediately.
        self._image_token_id = config.image_token_id if config.is_multimodal else None

    def _merge_multimodal(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Overwrite the embeddings at image-token positions with the vision tower's output.

        Scatters PER REQUEST, over that request's own slice of the batch. A batch-wide mask
        would count every image token in the forward against one concatenated tensor -- and
        the image token has a printable spelling (``<|image_pad|>`` tokenizes straight to
        248056 on Qwen3-VL), so any other request in the batch could add to that count just by
        containing the string. That made one request's text able to break another's prefill,
        and the mismatch surfaces inside the forward, where there is no per-request recovery.

        ``req.mm_embeds`` is set by the scheduler (which owns the model, so it is the only
        place that can run the tower) and covers the whole prompt; this forward takes the
        window of it belonging to the placeholders inside its own chunk. Text-only batches --
        every decode step, and every prefill without an image -- take the early return.
        """
        if self._image_token_id is None:
            return x
        batch = get_global_ctx().batch
        if not getattr(batch, "has_images", False):
            return x  # nothing in this batch scatters; skip the per-request walk
        offset = 0
        for req in batch.reqs:
            n = req.extend_len
            if req.mm_embeds is not None:
                span = slice(offset, offset + n)
                mask = input_ids[span] == self._image_token_id
                n_slots = int(mask.sum())
                if n_slots:
                    # ``mm_embeds`` holds one row per image token in the WHOLE prompt, in prompt
                    # order; this forward covers ``input_ids[cached_len:]``. Rows for placeholders
                    # an earlier chunk (or a prefix-cache hit) already consumed sit in front, so
                    # the window starts past them. That is what lets a prompt be chunked BETWEEN
                    # its images instead of having to hold every image in one chunk -- the whole
                    # span, first image to last, no longer has to fit ``--max-prefill-length``.
                    # getattr + the zero test: nothing is cached on a first chunk, which is
                    # every unchunked prompt, so the scan is skipped there -- and the scheduler
                    # tests drive this with request stubs that carry neither field.
                    cached = getattr(req, "cached_len", 0)
                    ids = getattr(req, "input_ids", None)
                    before = (
                        int((ids[:cached] == self._image_token_id).sum())
                        if cached and ids is not None
                        else 0
                    )
                    embeds = req.mm_embeds[before : before + n_slots]
                    # A raise, not an assert: `python -O` strips asserts, and what follows a
                    # stripped one here is masked_scatter silently taking the wrong number of
                    # rows -- one request's image landing in another's tokens.
                    if n_slots != embeds.shape[0]:
                        raise RuntimeError(
                            f"request {req.uid}: image-token slots ({n_slots}) != vision features "
                            f"({embeds.shape[0]}) for this chunk (rows {before}..{before + n_slots} "
                            f"of {req.mm_embeds.shape[0]}); the processor's token expansion and "
                            "the tower's output disagree"
                        )
                    x[span] = x[span].masked_scatter(mask.unsqueeze(-1), embeds.to(x.dtype))
            offset += n
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = self._merge_multimodal(input_ids, x)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        # Built only when parse_config kept a vision_config, i.e. only under --vision: the
        # tower is ~0.8 GiB of bf16 that a text-only deployment never reads.
        self.visual = Qwen3_5VisionModel(config.vision_config) if config.is_multimodal else None
        super().__init__()

    def encode_images(self, mm: dict[str, torch.Tensor]) -> torch.Tensor:
        """Takes the processor bundle and reads the keys this family needs -- the caller
        does not know which those are, which is the point."""
        pixel_values = mm["pixel_values"]
        image_grid_thw = mm.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError(
                "qwen3_5_moe.encode_images needs 'image_grid_thw' in mm_inputs; the "
                f"processor produced {sorted(mm)}"
            )
        """Run the vision tower. Returns ``[num_soft_tokens, text_hidden]``.

        ``pixel_values``: ``[total_patches, in_ch * temporal * patch**2]`` -- every image's
        patches packed into one tensor, which is why the split lives in ``image_grid_thw``
        ``[num_images, 3]`` (t, h, w in PRE-merge patches). The merger folds each 2x2 block,
        so the result has ``sum(t*h*w) / 4`` rows.
        """
        if self.visual is None:
            raise RuntimeError(
                "this model was built without a vision tower -- pass --vision (or set "
                "FREETOKEN_LOAD_VISION=1) so parse_config keeps the checkpoint's vision_config"
            )
        return self.visual.forward(pixel_values, image_grid_thw)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5MoEForCausalLM"]
