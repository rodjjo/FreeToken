from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.models.blocks import GatedMLP as K2HorizonDenseMLP
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class K2HorizonSharedExpert(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(
            intermediate_size,
            hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("SharedExpert")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        return self.down_proj.forward(silu_and_mul(gate_up))


class K2HorizonSparseBlock(BaseOP):
    """K2-Horizon sparse MoE block: 100 routed experts (top-8 selected) + 1 shared expert.

    Router logic:
    1. router_logits = x @ gate.weight.T (bias is not applied to logits)
    2. scores = sigmoid(router_logits)
    3. selection_scores = scores + gate.bias
    4. top-k on selection_scores
    5. gather scores at selected indices, renormalize, and scale by routed_scaling_factor (2.5)
    6. routed forward through experts
    7. add shared expert output
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=True)

        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
        )

        shared_intermediate = config.moe_intermediate_size * max(1, config.n_shared_experts)
        self.shared_experts = K2HorizonSharedExpert(config.hidden_size, shared_intermediate)

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = torch.sigmoid(logits)
        scores_for_choice = scores + self.gate.bias.float()

        _, topk_ids = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

    @nvtx_annotate("K2HorizonMoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(flat_hidden)
        out = self.experts.routed_forward(flat_hidden, topk_weights, topk_ids)
        out = out + self.shared_experts.forward(flat_hidden)
        return out.view(num_tokens, hidden_dim)


__all__ = ["K2HorizonDenseMLP", "K2HorizonSparseBlock", "K2HorizonSharedExpert"]
