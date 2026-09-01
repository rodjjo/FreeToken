"""Image input on the online (server) path.

The pieces that used to silently swallow pictures are the ones pinned here: the API
layer's content flattening, the tokenizer worker's split, and the message serializer --
each of which turned an image into nothing without ever raising.
"""

from __future__ import annotations

import base64

import pytest
import torch

from freetoken.message.utils import deserialize_type, serialize_type
from freetoken.server.generation import _render_content_parts
from freetoken.tokenizer.tokenize import split_image_parts

PNG = base64.b64encode(b"not-a-real-png").decode()
DATA_URL = f"data:image/png;base64,{PNG}"


def _image_part(url: str = DATA_URL) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


# --------------------------------------------------------------------------- API layer
def test_text_only_content_still_collapses_to_a_string():
    """The long-standing behaviour every chat template depends on."""
    out = _render_content_parts([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert out == "ab"


def test_content_with_an_image_survives_as_a_list():
    parts = [_image_part(), {"type": "text", "text": "what is this?"}]
    assert _render_content_parts(parts) == parts


def test_unknown_content_part_is_still_rejected():
    with pytest.raises(ValueError, match="Unsupported content part type"):
        _render_content_parts([_image_part(), {"type": "audio_url"}])


# ---------------------------------------------------------------- tokenizer-side split
def test_split_replaces_images_with_template_placeholders():
    msgs = [{"role": "user", "content": [_image_part(), {"type": "text", "text": "q"}]}]
    rendered, images = split_image_parts(msgs)
    # The template sees a bare {"type": "image"} -- that is what expands into image tokens.
    assert rendered == [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}
    ]
    assert images == [b"not-a-real-png"]


def test_split_preserves_image_order_across_messages():
    a = base64.b64encode(b"first").decode()
    b = base64.b64encode(b"second").decode()
    msgs = [
        {"role": "user", "content": [_image_part(f"data:image/png;base64,{a}")]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [_image_part(f"data:image/png;base64,{b}")]},
    ]
    _, images = split_image_parts(msgs)
    assert images == [b"first", b"second"]  # processor consumes them in template order


def test_split_leaves_text_only_messages_untouched():
    msgs = [{"role": "user", "content": "plain"},
            {"role": "user", "content": [{"type": "text", "text": "t"}]}]
    rendered, images = split_image_parts(msgs)
    assert rendered == msgs and images == []


@pytest.mark.parametrize("url", ["https://example.com/x.png", "data:image/png,notbase64"])
def test_non_inline_images_are_rejected(url):
    """A remote URL would make the tokenizer worker fetch a caller-supplied address."""
    with pytest.raises(ValueError):
        split_image_parts([{"role": "user", "content": [_image_part(url)]}])


# ------------------------------------------------------------------------ serialization
def test_multidimensional_tensors_round_trip():
    """Processor output is 3-D; the queue used to assert 1-D and kill the worker."""
    for t in (torch.randn(1, 2520, 768), torch.zeros(2, 4, 2, dtype=torch.int64)):
        got = deserialize_type({}, serialize_type(t))
        assert got.shape == t.shape and got.dtype == t.dtype and torch.equal(got, t)


def test_one_d_tensors_round_trip_unchanged():
    t = torch.arange(5, dtype=torch.int32)
    got = deserialize_type({}, serialize_type(t))
    assert torch.equal(got, t) and got.dtype == t.dtype


def test_non_contiguous_tensor_round_trips_by_value():
    t = torch.randn(3, 4).t()  # a view: strides do not survive, values must
    got = deserialize_type({}, serialize_type(t))
    assert got.shape == t.shape and torch.equal(got, t.contiguous())


def test_user_msg_carries_image_tensors():
    from freetoken.core import SamplingParams
    from freetoken.message import UserMsg

    msg = UserMsg(
        uid=1,
        input_ids=torch.arange(3, dtype=torch.int32),
        sampling_params=SamplingParams(),
        pixel_values=torch.randn(1, 8, 12),
        image_position_ids=torch.zeros(1, 8, 2, dtype=torch.int64),
    )
    got = deserialize_type({"UserMsg": UserMsg, "SamplingParams": SamplingParams},
                           serialize_type(msg))
    assert got.pixel_values.shape == (1, 8, 12)
    assert got.image_position_ids.shape == (1, 8, 2)
    assert torch.equal(got.input_ids, msg.input_ids)


# --------------------------------------------------- chunking around the image tokens
# The rule is NOT "a multimodal prompt must fit one chunk" -- that made a 200-token sprite
# force its 166k-token agent turn into a single prefill and get a 400 back. What one forward
# must see whole is the IMAGE-TOKEN SPAN, because the model scatters the whole of mm_embeds
# into the image tokens that chunk contains, and asserts the counts match.


class _NoSwa:  # non-sliding-window cache: chunking is capped by token_budget alone
    def __init__(self):
        self.swa_paged = False
        self.prefill_chunk_align = 1  # no snapshot-boundary alignment to respect


class _Tables:  # the token_pool row _add_one_req stages the chunk's ids into
    def __init__(self, length: int):
        import torch

        self.token_pool = torch.zeros(1, length, dtype=torch.int32)


def _pending(prompt_len: int, *, img_at: slice | None, n_img: int = 0, tok_id: int | None = None):
    """A PendingReq whose prompt has `n_img` image tokens laid at `img_at`."""
    import torch

    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    ids = torch.zeros(prompt_len, dtype=torch.int32)
    if img_at is not None:
        ids[img_at] = tok_id
    return PendingReq(
        uid=9,
        input_ids=ids,
        sampling_params=SamplingParams(),
        mm_embeds=torch.randn(n_img, 16) if n_img else None,
        image_token_id=tok_id,
    )


def _adder(budget: int, prompt_len: int):
    from freetoken.scheduler.prefill import PrefillAdder

    return PrefillAdder(
        token_budget=budget, reserved_size=0, cache_manager=_NoSwa(),
        table_manager=_Tables(prompt_len),
    )


def test_a_long_prompt_with_a_small_image_chunks_instead_of_being_rejected():
    """The regression this whole section exists for. 9383 tokens, 8192 of budget, and a
    196-token image sitting near the front: the span fits chunk one, so the prompt chunks
    like any other and the first chunk is the one that scatters."""
    pending = _pending(9383, img_at=slice(100, 296), n_img=196, tok_id=151655)
    got = _adder(8192, 9383)._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is not None, "must be admitted, not rejected"
    assert got.extend_len == 8192, "chunked at the token budget like a text prompt"
    assert got.mm_scatter is True
    assert got.mm_embeds is not None


def test_only_the_chunk_holding_the_image_tokens_scatters():
    """Every chunk keeps mm_embeds -- the cache manager reads it as 'keep this out of the
    shared prefix cache' -- but a chunk with no image tokens must not scatter, or the
    model's slot-count assert trips on 0 slots vs 196 features."""
    pending = _pending(9383, img_at=slice(8500, 8696), n_img=196, tok_id=151655)
    first = _adder(8192, 9383)._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert first is not None and first.mm_scatter is False
    assert first.mm_embeds is not None, "the multimodal marker survives on every chunk"

    second = _adder(8192, 9383)._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=8192
    )
    assert second is not None and second.mm_scatter is True


def test_the_final_chunk_does_not_scatter_an_earlier_chunks_image():
    """The last chunk of a chunked prompt has chunk_size == remain_len, so a guard keyed on
    "is this chunk chunked?" skips it and the default scatter fires into 0 image-token slots.
    That is the assert that took the scheduler worker down on a live 6k-token prompt."""
    pending = _pending(9383, img_at=slice(100, 296), n_img=196, tok_id=151655)
    first = _adder(8192, 9383)._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert first is not None and first.mm_scatter is True
    last = _adder(8192, 9383)._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=8192
    )
    assert last is not None
    assert last.extend_len == 1191 == 9383 - 8192, "this chunk runs to the end of the prompt"
    assert last.mm_scatter is False, "the image was scattered by chunk one"


def test_a_boundary_inside_the_span_pulls_back_instead_of_rejecting():
    """The 8192 boundary lands inside [8100, 8296), so the chunk ends at 8100 and the span
    rides chunk two whole. Rejecting here would 400 a perfectly servable request ~0.4% of
    the time (span width / chunk width) purely on where the image happened to land."""
    pending = _pending(9383, img_at=slice(8100, 8296), n_img=196, tok_id=151655)
    adder = _adder(8192, 9383)
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is not None, adder.rejected
    assert got.extend_len == 8100, "pulled back to just before the first image token"
    assert got.mm_scatter is False
    assert adder.rejected == []

    second = _adder(8192, 9383)._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=8100
    )
    assert second is not None and second.mm_scatter is True


def test_a_span_wider_than_one_chunk_is_rejected_not_fatal():
    """Nothing to pull back to: the image tokens alone outrun the chunk. Terminal for the
    REQUEST -- return None so the caller's release path runs, because raising from the
    chunker took the whole scheduler worker down."""
    pending = _pending(20000, img_at=slice(100, 10100), n_img=10000, tok_id=151655)
    adder = _adder(8192, 20000)
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is None
    assert [uid for uid, _ in adder.rejected] == [9]
    reason = adder.rejected[0][1]
    assert "[100, 10100)" in reason and "cannot be split" in reason and "10000" in reason


def test_the_pull_back_respects_the_chunk_alignment():
    """A model with snapshot boundaries (prefill_chunk_align) needs the pulled-back end
    aligned too, or the continuation resumes its carry mid-unit."""
    pending = _pending(9383, img_at=slice(8100, 8296), n_img=196, tok_id=151655)
    adder = _adder(8192, 9383)
    adder.cache_manager.prefill_chunk_align = 256
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is not None and got.extend_len == 7936 == 31 * 256, got.extend_len
    assert got.mm_scatter is False


def test_an_unlocatable_image_span_keeps_the_all_or_nothing_rule():
    """No image_token_id (text-only build) or a prompt whose placeholder expansion never
    happened: refuse to guess, and require the whole prompt in one chunk as before.

    The bound is not just --max-prefill-length: on a sliding-window model the swa pool caps
    the chunk too, which is how a 33k-token agent prompt still chunked at 40960.
    """
    from freetoken.scheduler.prefill import PrefillManager

    pending = _pending(9383, img_at=None, n_img=266, tok_id=None)
    assert pending.mm_span is None
    adder = _adder(8192, 9383)
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is None, "must return None so the caller releases the pass's resources"
    assert [uid for uid, _ in adder.rejected] == [9]
    reason = adder.rejected[0][1]
    assert "9383" in reason and "8192" in reason and "one prefill chunk" in reason

    mgr = PrefillManager(cache_manager=None, table_manager=None, decode_manager=None)
    mgr.rejections.extend(adder.rejected)
    assert mgr.drain_rejections() == [(9, reason)]
    assert mgr.drain_rejections() == [], "draining is one-shot"


def test_an_unchunked_multimodal_prompt_is_untouched():
    """The common case -- prompt fits one chunk -- must not go near the span logic."""
    pending = _pending(500, img_at=slice(10, 206), n_img=196, tok_id=151655)
    got = _adder(8192, 500)._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is not None and got.mm_scatter is True and got.extend_len == 500


# ------------------------------------------------------------------------ the span itself
def test_span_covers_from_the_first_image_token_to_the_last():
    """Two images with text between them: the span is one interval spanning BOTH, because
    mm_embeds is a single concatenated tensor scattered in one forward."""
    p = _pending(400, img_at=None, n_img=8, tok_id=151655)
    p.input_ids[50:54] = 151655
    p.input_ids[300:304] = 151655
    assert p.mm_span == (50, 304)
    assert int((p.input_ids == 151655).sum()) == 8  # the text between is not an image token


def test_span_is_none_when_the_request_carries_no_images():
    assert _pending(400, img_at=None, n_img=0, tok_id=151655).mm_span is None


def test_span_is_none_when_the_prompt_holds_no_placeholder():
    """Embeddings but no tokens to scatter into: caller decides (today: all-or-nothing)."""
    assert _pending(400, img_at=None, n_img=4, tok_id=151655).mm_span is None


# ------------------------------------------------------------------ vision-capable gate
def test_text_only_family_never_gets_an_image_processor(monkeypatch):
    """A multimodal CHECKPOINT served by a text-only family (Ornith / qwen3_5_moe, whose
    parse_config pins vision_config=None) ships processor files but has no encode_images.
    Accepting the image at the tokenizer would only fail later, further from the cause."""
    from types import SimpleNamespace

    import freetoken.models.weight as weight_mod
    from freetoken.tokenizer.server import _image_processor_path

    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    monkeypatch.setattr(
        weight_mod, "_spec_for_model_path",
        lambda path: (SimpleNamespace(is_multimodal=False), None),
    )
    assert _image_processor_path("some/text-only-model") is None


def test_probe_failure_disables_images_rather_than_blocking_startup(monkeypatch):
    import freetoken.models.weight as weight_mod
    from freetoken.tokenizer.server import _image_processor_path

    def _boom(path):
        raise RuntimeError("unresolvable path")

    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    monkeypatch.setattr(weight_mod, "_spec_for_model_path", _boom)
    assert _image_processor_path("nonexistent/model") is None
