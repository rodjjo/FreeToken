from __future__ import annotations

import re
from typing import Iterable, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    iter_stacked_experts,
    iter_weight_files,
    shard_tensor,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_MLP_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.mlp\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_MOVA_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.self_attn\.v_experts)\.(?P<idx>\d+)\.weight$")

def iter_k2_merged_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    merge_buf: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors:
        # Only merge gate_proj and up_proj inside mlp (dense, shared, and routed experts)
        if ".mlp." in name and (
            name.endswith(".gate_proj.weight") or name.endswith(".up_proj.weight")
        ):
            is_gate = name.endswith(".gate_proj.weight")
            merged_key = name.replace(
                ".gate_proj.weight", ".gate_up_proj.weight"
            ).replace(".up_proj.weight", ".gate_up_proj.weight")
            slot = "gate" if is_gate else "up"
            slots = merge_buf.setdefault(merged_key, {})
            slots[slot] = tensor
            if "gate" in slots and "up" in slots:
                parts = [slots["gate"], slots["up"]]
                del merge_buf[merged_key]
                yield merged_key, torch.cat(parts, dim=0)
        else:
            yield name, tensor

    assert not merge_buf, f"k2_horizon: Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"


def _is_mlp_expert(name: str) -> bool:
    return _MLP_EXPERT_PATTERN.match(name) is not None


def _is_mova_expert(name: str) -> bool:
    return _MOVA_EXPERT_PATTERN.match(name) is not None


def iter_stacked_mova_experts(
    tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    num_experts: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    expert_buf: dict[str, dict[int, torch.Tensor]] = {}
    for name, tensor in tensors:
        match = _MOVA_EXPERT_PATTERN.match(name)
        if match is None:
            yield name, tensor
            continue
        prefix = match.group("prefix")
        idx = int(match.group("idx"))
        slots = expert_buf.setdefault(prefix, {})
        slots[idx] = tensor
        if len(slots) == num_experts:
            experts = [slots[i] for i in range(num_experts)]
            del expert_buf[prefix]
            yield prefix, torch.stack(experts, dim=0)

    assert not expert_buf, f"Incomplete MoVA expert tensors: {list(expert_buf.keys())}"


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = raw_name.removeprefix("language_model.")
                    is_expert = _is_mlp_expert(name)
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue

                    raw = f.get_tensor(raw_name)

                    if _is_mova_expert(name) and tp_info.size > 1:
                        # Shard MoVA value expert along output dimension (dim 0)
                        tensor = raw.chunk(tp_info.size, dim=0)[tp_info.rank].clone()
                    elif "self_attn.v_router" in name or "mlp.gate" in name:
                        # Router weights and biases are replicated
                        tensor = raw.clone()
                    else:
                        tensor = shard_tensor(
                            name,
                            raw,
                            rank=tp_info.rank,
                            world_size=tp_info.size,
                            num_kv_heads=config.num_kv_heads,
                        )
                    del raw
                    yield name, tensor

    # Merge gate and up projections for dense MLPs, shared experts, and routed experts
    merged = iter_k2_merged_tensors(sharded_tensors())

    # Stack MoVA V-experts (if non-moe included)
    if include_non_moe:
        stacked_mova = iter_stacked_mova_experts(
            merged,
            num_experts=config.mova_num_experts,
        )
    else:
        stacked_mova = merged

    # Stack MLP experts (if moe experts included)
    if include_moe_experts:
        yield from iter_stacked_experts(
            stacked_mova,
            num_experts=config.num_experts,
            model_name="k2_horizon",
            expert_pattern=_MLP_EXPERT_PATTERN,
        )
    else:
        yield from stacked_mova


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only iter_weights: raw experts read via the common chunked multi-threaded
    O_DIRECT reader, then same merge+stack pipeline."""
    assert include_moe_experts and not include_non_moe, (
        "k2_horizon parallel reader is experts-only (used by load_moe_expert_sources)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def raw_experts() -> Iterator[tuple[str, torch.Tensor]]:
        for raw_name, raw in iter_expert_tensors_parallel(
            model_path, _is_mlp_expert, workers=workers, chunk=chunk
        ):
            name = raw_name.removeprefix("language_model.")
            tensor = shard_tensor(
                name,
                raw,
                rank=tp_info.rank,
                world_size=tp_info.size,
                num_kv_heads=config.num_kv_heads,
            )
            yield name, tensor

    merged = iter_k2_merged_tensors(raw_experts())
    yield from iter_stacked_experts(
        merged,
        num_experts=config.num_experts,
        model_name="k2_horizon",
        expert_pattern=_MLP_EXPERT_PATTERN,
    )


__all__ = ["iter_weights", "iter_weights_parallel"]
