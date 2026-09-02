from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.models.config import KVCacheGroupSpec


def _qwen_config(**overrides):
    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=tuple(range(10)),
        num_kv_heads=2,
        head_dim=256,
        attn_type=AttnType.FULL,
        sliding_window=None,
    )
    model = SimpleNamespace(
        model_type="qwen3_5_moe",
        single_stream_only=False,
        is_moe=False,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=True,
        num_layers=40,
        rotary_config=SimpleNamespace(max_position=262144),
    )
    model.kv_cache_group_specs = lambda: (spec,)
    kwargs = dict(
        model_path="/tmp/qwen",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        attention_backend="auto",
    )
    kwargs.update(overrides)
    config = EngineConfig(**kwargs)
    object.__setattr__(config, "model_config", model)
    return config


def test_fp8_kv_auto_selects_triton(monkeypatch):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_arch_supported", lambda major, minor=0: (major, minor) <= (8, 9))
    engine._adjust_config(config := _qwen_config())
    assert config.attention_backend == "triton"


def test_fp8_kv_rejects_non_triton_backend(monkeypatch):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_arch_supported", lambda *_: True)
    with pytest.raises(ValueError, match="attention-backend triton"):
        engine._adjust_config(_qwen_config(attention_backend="fi"))


def test_fp8_kv_rejects_pre_sm89(monkeypatch):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_arch_supported", lambda *_: False)
    with pytest.raises(ValueError, match="compute capability 8.9"):
        engine._adjust_config(_qwen_config(attention_backend="triton"))


def test_qwen_fp8_kv_bytes_include_dynamic_scales():
    from freetoken.kvcache.base import spec_kv_bytes_per_token

    config = _qwen_config(attention_backend="triton")
    (spec,) = config.model_config.kv_cache_group_specs()
    # FP8 data: K/V * 10 layers * 2 heads * 256. Scales: K/V * layers * heads * FP32.
    assert spec_kv_bytes_per_token(spec, config) == 2 * 10 * 2 * 256 + 2 * 10 * 2 * 4


def test_server_parser_maps_fp8_kv_dtype():
    from freetoken.server.args import parse_args

    hf = SimpleNamespace(
        to_dict=lambda: {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "dtype": "bfloat16",
        }
    )
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args, _ = parse_args(
            ["--model", "/models/qwen", "--kv-cache-dtype", "fp8_e4m3"]
        )
    assert args.kv_cache_dtype == torch.float8_e4m3fn
