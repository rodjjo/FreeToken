from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearOProj, LinearReplicated
from freetoken.layers.linear import _LinearTPImpl
from freetoken.layers.rotary import get_rope
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LinearColParallel(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        tp_info = get_tp_info()
        local_output_size = div_even(output_size, tp_info.size)
        super().__init__(input_size, output_size, input_size, local_output_size, has_bias)


class K2HorizonAttention(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        is_sparse_layer: bool,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.is_sparse_layer = is_sparse_layer
        head_dim = config.head_dim
        self.head_dim = head_dim
        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        self.num_qo_heads = div_even(config.num_qo_heads, self.tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, self.tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim

        self.q_proj = LinearColParallel(
            config.hidden_size, config.num_qo_heads * head_dim, has_bias=False
        )
        self.k_proj = LinearColParallel(
            config.hidden_size, config.num_kv_heads * head_dim, has_bias=False
        )
        self.o_proj = LinearOProj(
            config.num_qo_heads * head_dim, config.hidden_size, has_bias=False
        )
        self.gate_proj = LinearColParallel(
            config.hidden_size, config.num_qo_heads * head_dim, has_bias=False
        )

        if not is_sparse_layer:
            # Dense layers (0, 1, 2) have standard v_proj
            self.v_proj = LinearColParallel(
                config.hidden_size, config.num_kv_heads * head_dim, has_bias=False
            )
            self.v_router = None
            self.v_experts = None
        else:
            # Sparse MoVA layers (3..47)
            self.v_proj = None
            self.mova_num_experts = config.mova_num_experts
            self.mova_top_k = config.mova_num_experts_per_tok
            self.router_scaling_factor = config.routed_scaling_factor
            self.v_router = LinearReplicated(
                config.hidden_size, self.mova_num_experts, has_bias=True
            )
            # Stacked expert weights: [mova_num_experts, kv_attn_dim, hidden_size]
            from freetoken.moe import is_offload_moe_backend
            v_device = "cpu" if is_offload_moe_backend(config.moe_backend) else None
            self.v_experts = torch.empty(
                self.mova_num_experts,
                self.kv_attn_dim,
                config.hidden_size,
                device=v_device,
            )

        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )

    def _compute_mova_v(self, x: torch.Tensor) -> torch.Tensor:
        # router logits: x @ v_router.weight.T
        router_logits = F.linear(x.float(), self.v_router.weight.float())
        scores = torch.sigmoid(router_logits)
        scores_for_choice = scores + self.v_router.bias.float()

        topk_weights, topk_ids = torch.topk(scores_for_choice, self.mova_top_k, dim=-1)
        topk_weights = torch.gather(scores, dim=-1, index=topk_ids)
        topk_weights = (
            topk_weights
            / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
            * self.router_scaling_factor
        )

        num_tokens = x.shape[0]
        v_experts = self.v_experts
        device = x.device

        if num_tokens == 1:
            ids = topk_ids[0]
            if v_experts.device != device:
                w = v_experts[ids.cpu()].to(device, non_blocking=True)
            else:
                w = v_experts[ids]
            out = F.silu(F.linear(x, w.view(-1, x.shape[-1]))).view(1, self.mova_top_k, self.kv_attn_dim)
            v = (out * topk_weights.unsqueeze(-1).to(out.dtype)).sum(dim=1)
            return v

        # Prefill / multi-token path
        final_v = torch.zeros((num_tokens, self.kv_attn_dim), dtype=x.dtype, device=device)
        expert_mask = F.one_hot(topk_ids, num_classes=self.mova_num_experts).permute(2, 1, 0)
        expert_hits = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()

        for exp_idx in expert_hits:
            idx_int = int(exp_idx)
            topk_pos, tok_pos = torch.where(expert_mask[idx_int])
            exp_w = v_experts[idx_int]
            if exp_w.device != device:
                exp_w = exp_w.to(device, non_blocking=True)
            exp_states = F.silu(F.linear(x[tok_pos], exp_w))
            exp_states = exp_states * topk_weights[tok_pos, topk_pos, None].to(exp_states.dtype)
            final_v.index_add_(0, tok_pos, exp_states)

        return final_v

    @nvtx_annotate("K2HorizonAttention")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        if self.v_proj is not None:
            v = self.v_proj.forward(x)
        else:
            v = self._compute_mova_v(x)

        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)

        gate = self.gate_proj.forward(x)
        gate = F.softplus(gate, beta=math.log(2))
        o = o.view(-1, self.qo_attn_dim) * gate
        return self.o_proj.forward(o)


__all__ = ["K2HorizonAttention", "LinearColParallel"]
