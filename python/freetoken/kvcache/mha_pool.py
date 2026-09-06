from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool
from .quant import NONE, KVQuantSpec
from .quant_storage import QuantizedKVStorageMixin


def _kv_store_dtype(dtype: torch.dtype, kv_quant: str) -> torch.dtype:
    """Storage dtype of the KV buffer for a quantization mode."""
    if kv_quant in ("none", "auto", "bf16"):
        return dtype
    if kv_quant == "fp8":
        from freetoken.kernel.triton.kv_quant import kv_codes_dtype

        return kv_codes_dtype()
    raise ValueError(f"unknown kv_quant {kv_quant!r}")


class MHAKVCache(QuantizedKVStorageMixin, BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.

    ``kv_quant="fp8"`` halves the cache: rows become e4m3 codes and every
    ``(token, slab, layer, kv head)`` row carries one fp32 scale (see
    :mod:`freetoken.kernel.triton.kv_quant`). The codes buffer keeps the exact same
    shape as the 16-bit one, so ``k_cache``/``v_cache`` and every index into them are
    unchanged -- only the element type, and ``store_kv``'s write path, differ.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_quant: str = "none",
        quant: KVQuantSpec = NONE,
    ) -> None:
        if isinstance(kv_quant, KVQuantSpec):
            quant = kv_quant
            kv_quant = quant.name
        elif quant is not None and isinstance(quant, KVQuantSpec) and quant.enabled:
            if kv_quant == "none":
                kv_quant = quant.name

        self._quant = quant
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self.kv_quant = kv_quant
        self.is_fp8 = (kv_quant == "fp8")
        self._compute_dtype = dtype
        self._head_dim = head_dim
        self._device = device
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._alloc(num_pages, page_size, num_storage_layers, local_kv_heads, head_dim)

    def _alloc(
        self,
        num_pages: int,
        page_size: int,
        num_storage_layers: int,
        local_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Allocate the code buffer (and, when quantized, the scale buffer)."""
        device = self._device
        if self.is_fp8:
            from freetoken.kernel.triton.kv_quant import alloc_codes

            shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
            self._kv_buffer = alloc_codes(shape, device)
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._scale_buffer = torch.zeros(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads),
                device=device,
                dtype=torch.float32,
            )
            self._k_scale = self._scale_buffer[0]
            self._v_scale = self._scale_buffer[1]
            self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        elif self._quant.enabled:
            last_dim = self._quant.physical_head_dim(head_dim)
            kv_shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, last_dim)
            self._kv_buffer = torch.empty(
                kv_shape, device=device, dtype=self._buffer_dtype(self._compute_dtype)
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._scale_buffer = self._alloc_scales(kv_shape, device)
            self._k_scale = self._scale_buffer[0] if self._scale_buffer is not None else None
            self._v_scale = self._scale_buffer[1] if self._scale_buffer is not None else None
            self._storage_shape = (num_pages * page_size, local_kv_heads, last_dim)
        else:
            shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
            self._kv_buffer = torch.empty(
                shape, device=device, dtype=self._compute_dtype
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._scale_buffer = None
            self._k_scale = None
            self._v_scale = None
            self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE."""
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, _ = self._kv_buffer.shape
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        self._k_scale = None
        self._v_scale = None
        self._scale_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._alloc(num_pages, page_size, num_storage_layers, local_kv_heads, self._head_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        total = buf.numel() * buf.element_size()
        if self._scale_buffer is not None:
            total += self._scale_buffer.numel() * self._scale_buffer.element_size()
        return int(total) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def k_scale(self, index: int) -> torch.Tensor | None:
        if self.is_fp8:
            return self._scale_buffer[0][self._dense(index)] if self._scale_buffer is not None else None
        return None if self._k_scale is None else self._k_scale[self._dense(index)]

    def v_scale(self, index: int) -> torch.Tensor | None:
        if self.is_fp8:
            return self._scale_buffer[1][self._dense(index)] if self._scale_buffer is not None else None
        return None if self._v_scale is None else self._v_scale[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        if self.is_fp8:
            from freetoken.kernel.triton.kv_quant import quantize_kv_to_cache

            quantize_kv_to_cache(
                k=k,
                v=v,
                out_loc=out_loc,
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                k_scale=self._scale_buffer[0][dense],
                v_scale=self._scale_buffer[1][dense],
            )
            return
        if self._quant.enabled:
            scale_shape = (self._storage_shape[0], self._storage_shape[1], -1)
            self._store_kv_into(
                self._k_buffer[dense].view(self._storage_shape),
                self._v_buffer[dense].view(self._storage_shape),
                None if self._k_scale is None else self._k_scale[dense].view(scale_shape),
                None if self._v_scale is None else self._v_scale[dense].view(scale_shape),
                out_loc,
                k,
                v,
            )
            return
        from freetoken.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The COMPUTE dtype (kvcache/base.py): what ``store_kv`` receives and what
        backends size their scratch with -- still 16-bit on an fp8 pool."""
        return self._compute_dtype

    @property
    def store_dtype(self) -> torch.dtype:
        """Element type of ``_kv_buffer``: fp8/uint8 codes when quantized, else the
        compute dtype. Reading the buffer itself needs THIS, and a backend that cannot
        apply the row scales never gets a quantized pool (supports_fp8_kv gate)."""
        return self._kv_buffer.dtype

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._compute_dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
