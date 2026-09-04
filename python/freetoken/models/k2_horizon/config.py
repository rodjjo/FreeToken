from __future__ import annotations

from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig


def parse_config(hf_config: Any) -> ModelConfig:
    num_kv_heads = getattr(
        hf_config,
        "num_key_value_heads",
        hf_config.num_attention_heads,
    )
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )

    rope_theta = 10000000.0
    rope_params = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        rope_theta = float(rope_params["rope_theta"])
    elif hasattr(hf_config, "rope_theta") and hf_config.rope_theta is not None:
        rope_theta = float(hf_config.rope_theta)

    mlp_only_layers = getattr(hf_config, "mlp_only_layers", [0, 1, 2])
    first_k_dense = len(mlp_only_layers) if mlp_only_layers else 0

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=getattr(hf_config, "rope_head_dim", head_dim) or head_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=getattr(hf_config, "rope_scaling", None),
        ),
        num_experts=getattr(hf_config, "num_experts", 100),
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 8),
        moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", 768),
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "k2_horizon"),
        architectures=getattr(hf_config, "architectures", ["K2HorizonForCausalLM"]),
        moe_enabled=True,
        first_k_dense_replace=first_k_dense,
        n_shared_experts=getattr(hf_config, "num_shared_experts", 1),
        routed_scaling_factor=float(getattr(hf_config, "router_scaling_factor", 2.5)),
        has_router_bias=bool(getattr(hf_config, "moe_gate_bias", True)),
        layernorm_num_groups=int(getattr(hf_config, "layernorm_num_groups", 2)),
        mova_num_experts=int(getattr(hf_config, "mova_num_experts", 64)),
        mova_num_experts_per_tok=int(getattr(hf_config, "mova_num_experts_per_tok", 4)),
        attention_gate_func=getattr(hf_config, "attention_gate_func", "softplus"),
    )


__all__ = ["parse_config"]
