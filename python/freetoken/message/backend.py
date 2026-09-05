from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings (GPU tensor). Only used by
    # the in-process offline path, which owns the model and can run the vision tower
    # itself; the serialized online path sends the processor tensors below instead and
    # lets the scheduler encode them.
    mm_embeds: torch.Tensor | None = None
    # Online image path: whatever this checkpoint's HF processor produced for the request's
    # images, as CPU tensors, kept as one opaque bundle. The transport does not open it --
    # only the model that will consume it knows which keys it needs (Qwen3-VL wants
    # ``image_grid_thw``, gemma4 wants ``image_position_ids``, both want ``pixel_values``),
    # so a new modality is a key here rather than a field on every layer in between.
    mm_inputs: dict[str, torch.Tensor] | None = None


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
