from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import K2HorizonAttention
from .moe import K2HorizonDenseMLP, K2HorizonSparseBlock
from .norm import GroupedRMSNormFused

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class K2HorizonDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self._layer_id = layer_id
        is_sparse = layer_id >= config.first_k_dense_replace
        self.self_attn = K2HorizonAttention(config, layer_id, is_sparse_layer=is_sparse)
        if is_sparse:
            self.mlp = K2HorizonSparseBlock(config, layer_id)
        else:
            self.mlp = K2HorizonDenseMLP(config)

        self.input_layernorm = GroupedRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
            n_groups=config.layernorm_num_groups,
        )
        self.post_attention_layernorm = GroupedRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
            n_groups=config.layernorm_num_groups,
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class K2HorizonModel(BaseOP):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [K2HorizonDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GroupedRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
            n_groups=config.layernorm_num_groups,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class K2HorizonForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = K2HorizonModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["K2HorizonForCausalLM", "K2HorizonModel", "K2HorizonDecoderLayer"]
