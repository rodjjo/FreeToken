from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    #: Token id the soft embeddings are scattered at, when this request carries images. The
    #: adder needs it to find where in input_ids they sit; None for a text-only request.
    image_token_id: int | None = None

    @cached_property
    def mm_span(self) -> tuple[int, int] | None:
        """``[first, last+1)`` over input_ids covering EVERY image token, or None.

        The model's scatter matches image tokens inside one chunk against the whole of
        ``mm_embeds``, so what a chunk boundary must not do is cut this span -- it does not
        have to keep the entire prompt together. In an agent turn that distinction is the
        difference between a 200-token sprite forcing a 166k-token prompt into one chunk and
        it riding along in whichever chunk happens to contain it.

        Cached: the adder asks once per chunk per scheduling pass, and the scan is over the
        WHOLE prompt -- which for the case this exists to serve is 166k tokens.
        """
        if self.mm_embeds is None or self.image_token_id is None:
            return None
        hits = (self.input_ids == self.image_token_id).nonzero()
        if hits.numel() == 0:
            return None
        return int(hits[0]), int(hits[-1]) + 1

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
