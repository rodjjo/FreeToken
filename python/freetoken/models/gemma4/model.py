from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.utils import nvtx_annotate

from freetoken.models.blocks import BaseLLMModel

from .attention import Gemma4Attention
from .moe import Gemma4DenseMLP, Gemma4MLP
from .vision import Gemma4MultimodalEmbedder, Gemma4VisionModel

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Gemma4DecoderLayer(BaseOP):
    """Gemma 4 decoder block: attention sandwich + feed-forward sandwich, scaled by a
    per-layer ``layer_scalar``. The feed-forward is the dual (shared MLP || routed MoE)
    branch for MoE checkpoints, or a single dense MLP branch for dense checkpoints."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self.self_attn = Gemma4Attention(config, layer_id)
        self.feed_forward = (
            Gemma4MLP(config, layer_id) if config.is_moe else Gemma4DenseMLP(config)
        )

        eps = config.rms_norm_eps
        H = config.hidden_size
        self.input_layernorm = GemmaRMSNorm(H, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(H, eps=eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- attention sandwich ---
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h)
        h = self.post_attention_layernorm.forward(h)
        pre_ff, x = self.pre_feedforward_layernorm.forward_add_residual(h, residual)
        return self.feed_forward.forward(pre_ff, x)


class Gemma4Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            embed_scale=config.embedding_scale,
        )
        self.layers = OPList(
            [Gemma4DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._image_token_id = config.image_token_id

    def _merge_multimodal(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scatter precomputed image soft-token embeddings at image-token positions.

        Per REQUEST, over that request's own slice of the batch. A batch-wide mask counts
        every image token in the forward against one request's features -- and the image
        placeholder has a printable spelling, so any other request in the batch can add to
        that count just by containing the string. The mismatch surfaces inside the forward,
        which has no per-request recovery, so one request's text would take the worker down.

        ``req.mm_embeds`` is set by the scheduler from that request's vision features and
        covers the whole prompt; this forward takes the window of it belonging to the
        placeholders inside its own chunk. Decode batches never carry images and take the
        early return.
        """
        batch = get_global_ctx().batch
        if not getattr(batch, "has_images", False) or self._image_token_id is None:
            return x
        offset = 0
        for req in batch.reqs:
            n = req.extend_len
            if req.mm_embeds is not None:
                span = slice(offset, offset + n)
                mask = input_ids[span] == self._image_token_id
                n_slots = int(mask.sum())
                if n_slots:
                    # Rows for placeholders an earlier chunk (or a prefix-cache hit) already
                    # consumed sit in front of this window; see qwen3_5_moe._merge_multimodal.
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
                    # stripped one is masked_scatter quietly taking the wrong number of rows.
                    if n_slots != embeds.shape[0]:
                        raise RuntimeError(
                            f"request {req.uid}: image-token slots ({n_slots}) != vision features "
                            f"({embeds.shape[0]}) for this chunk (rows {before}..{before + n_slots} "
                            f"of {req.mm_embeds.shape[0]})"
                        )
                    x[span] = x[span].masked_scatter(mask.unsqueeze(-1), embeds.to(x.dtype))
            offset += n
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = self._merge_multimodal(input_ids, x)
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class Gemma4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Gemma4Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._final_logit_softcapping = config.final_logit_softcapping
        if config.is_multimodal:
            self.vision_tower = Gemma4VisionModel(config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config)
        super().__init__()

        # GGUF checkpoints carry native block-quantized weights: swap the dense
        # projections + embedding for GGUF-quant ops (experts stay on the offload cache).
        from .gguf import convert_gemma4_to_gguf, is_gguf_model

        if is_gguf_model(config):
            convert_gemma4_to_gguf(self, config)

    @torch.inference_mode()
    def encode_images(self, mm: dict[str, torch.Tensor]) -> torch.Tensor:
        """Takes the processor bundle and reads the keys this family needs -- the caller
        does not know which those are, which is the point."""
        pixel_values = mm["pixel_values"]
        image_position_ids = mm.get("image_position_ids")
        if image_position_ids is None:
            raise RuntimeError(
                "gemma4.encode_images needs 'image_position_ids' in mm_inputs; the "
                f"processor produced {sorted(mm)}"
            )
        """Run the vision tower + projector. Returns ``[num_valid_soft_tokens, text_hidden]``.

        ``pixel_values``: ``[num_images, num_patches, 3*patch**2]``;
        ``image_position_ids``: ``[num_images, num_patches, 2]`` with ``(-1, -1)`` padding.
        """
        features = self.vision_tower.forward(pixel_values, image_position_ids)
        return self.embed_vision.forward(features)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        if self._final_logit_softcapping is not None:
            cap = self._final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits


__all__ = ["Gemma4ForCausalLM"]
