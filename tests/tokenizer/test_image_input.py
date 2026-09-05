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
        mm_inputs={
            "pixel_values": torch.randn(1, 8, 12),
            "image_position_ids": torch.zeros(1, 8, 2, dtype=torch.int64),
        },
    )
    got = deserialize_type({"UserMsg": UserMsg, "SamplingParams": SamplingParams},
                           serialize_type(msg))
    assert got.mm_inputs["pixel_values"].shape == (1, 8, 12)
    assert got.mm_inputs["image_position_ids"].shape == (1, 8, 2)
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
    adder = _adder(8192, 20000)  # span 10000 > 8192, the largest chunk this server schedules
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is None
    assert [uid for uid, _, _ in adder.rejected] == [9]
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
    assert [uid for uid, _, _ in adder.rejected] == [9]
    reason = adder.rejected[0][1]
    assert "9383" in reason and "8192" in reason and "one prefill chunk" in reason

    mgr = PrefillManager(cache_manager=None, table_manager=None, decode_manager=None)
    mgr.rejections.extend(adder.rejected)
    assert mgr.drain_rejections() == [(9, reason, None)]
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


# ------------------------------------------------- budget: transient vs terminal
def test_losing_this_passs_budget_race_declines_instead_of_rejecting():
    """token_budget is the pass's REMAINDER -- earlier admissions in the same pass spend it.
    A prompt whose span would fit a full chunk must not be killed for being second in line:
    decline (None, stays queued) and let it retry at the front of the queue."""
    pending = _pending(9383, img_at=slice(100, 5000), n_img=4900, tok_id=151655)
    adder = _adder(8192, 9383)
    adder.token_budget = 3000  # what an earlier admission this pass left behind
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is None
    assert adder.rejected == [], "the span fits 8192; nothing about it is terminal"


def test_a_span_over_the_full_budget_is_still_terminal():
    """The other half: no pass can ever hold this one, so waiting is pointless."""
    pending = _pending(20000, img_at=slice(100, 10100), n_img=10000, tok_id=151655)
    adder = _adder(8192, 20000)
    assert adder.max_chunk_budget == 8192
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=0
    )
    assert got is None
    assert [uid for uid, _, _ in adder.rejected] == [9]
    assert "largest chunk this server will schedule is 8192" in adder.rejected[0][1]


def test_rejecting_a_continuation_hands_back_the_prior_chunk():
    """Earlier chunks already forwarded: dropping the request from the queue leaves nobody to
    return their KV pages, table row and GDN slots, so the rejection must carry the Req that
    owns them. A first-chunk rejection has nothing to hand back (the adder released it)."""
    pending = _pending(20000, img_at=slice(9000, 19000), n_img=10000, tok_id=151655)
    adder = _adder(8192, 20000)
    prior = object()
    got = adder._add_one_req(
        pending_req=pending, cache_handle=None, table_idx=0, cached_len=8192,
        chunked_req=prior,
    )
    assert got is None
    uid, _, handed_back = adder.rejected[0]
    assert uid == 9 and handed_back is prior


# --------------------------------------------- the batch-wide image-token count
def test_a_stray_placeholder_in_another_request_cannot_break_this_one(monkeypatch):
    """`<|image_pad|>` has a printable spelling and tokenizes straight to the image id, so a
    plain text request can carry it. Under a batch-wide mask its tokens counted against the
    image request's embeddings -- a mismatch inside the forward, which has no per-request
    recovery, so one request's text took the whole worker down."""
    import torch
    from types import SimpleNamespace

    import freetoken.models.qwen3_5_moe.model as mod

    IMG = 151655
    img_req = SimpleNamespace(
        uid=1, extend_len=6, mm_embeds=torch.full((2, 4), 9.0), mm_scatter=True)
    text_req = SimpleNamespace(uid=2, extend_len=3, mm_embeds=None, mm_scatter=True)
    # request 1: [t, IMG, IMG, t, t, t]   request 2 (attacker): [t, IMG, t]
    input_ids = torch.tensor([5, IMG, IMG, 5, 5, 5, 7, IMG, 7])
    x = torch.zeros(9, 4)
    batch = SimpleNamespace(reqs=[img_req, text_req], has_images=True)
    monkeypatch.setattr(mod, "get_global_ctx", lambda: SimpleNamespace(batch=batch))

    out = mod.Qwen3_5Model._merge_multimodal(SimpleNamespace(_image_token_id=IMG), input_ids, x)
    assert out[1].tolist() == [9.0] * 4 and out[2].tolist() == [9.0] * 4
    assert out[7].tolist() == [0.0] * 4, "the other request's placeholder is left alone"


def test_a_count_mismatch_raises_rather_than_asserts(monkeypatch):
    """`python -O` strips asserts, and what follows a stripped one is masked_scatter quietly
    taking the wrong number of rows."""
    import torch
    from types import SimpleNamespace

    import freetoken.models.qwen3_5_moe.model as mod

    IMG = 151655
    req = SimpleNamespace(uid=1, extend_len=3, mm_embeds=torch.zeros(2, 4), mm_scatter=True)
    batch = SimpleNamespace(reqs=[req], has_images=True)
    monkeypatch.setattr(mod, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    with pytest.raises(RuntimeError, match="image-token slots"):
        mod.Qwen3_5Model._merge_multimodal(
            SimpleNamespace(_image_token_id=IMG), torch.tensor([IMG, 5, 5]), torch.zeros(3, 4)
        )


def test_a_chunk_that_does_not_scatter_is_skipped_entirely(monkeypatch):
    """mm_embeds stays set on every chunk (the cache manager reads it as 'keep this out of the
    shared prefix cache'), so mm_scatter is what must gate the scatter."""
    import torch
    from types import SimpleNamespace

    import freetoken.models.qwen3_5_moe.model as mod

    IMG = 151655
    req = SimpleNamespace(uid=1, extend_len=3, mm_embeds=torch.ones(2, 4), mm_scatter=False)
    batch = SimpleNamespace(reqs=[req], has_images=True)
    monkeypatch.setattr(mod, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    out = mod.Qwen3_5Model._merge_multimodal(
        SimpleNamespace(_image_token_id=IMG), torch.tensor([5, 5, 5]), torch.zeros(3, 4)
    )
    assert out.abs().sum() == 0


# ------------------------------------------------------- the other render paths
def test_the_image_path_quantizes_reasoning_effort_like_every_other_path():
    """render_prompt sanitizes reasoning_effort; encode_multimodal must too, or an
    unsupported value renders differently depending on whether the request had an image."""
    from types import SimpleNamespace

    from freetoken.tokenizer.tokenize import TokenizeManager

    seen = {}

    class _Proc:
        def apply_chat_template(self, messages, **kw):
            seen.update(kw)
            return "PROMPT"

        def __call__(self, text, images, return_tensors):
            import torch

            return {"input_ids": torch.zeros(1, 3, dtype=torch.long),
                    "pixel_values": torch.zeros(1, 2, 2)}

    mgr = TokenizeManager.__new__(TokenizeManager)
    mgr._processor_obj = _Proc()
    mgr._processor_lock = __import__("threading").Lock()
    mgr._processor_path = "unused"
    mgr._logged_effort_maps = set()
    from freetoken.tokenizer.effort import EffortProfile

    mgr.effort_profile = lambda: EffortProfile(
        supported=frozenset({"low", "high"}), default="low",
        consumes_effort=True, validates=True,
    )

    import io as _io

    from PIL import Image as _Image

    buf = _io.BytesIO()
    _Image.new("RGB", (4, 4)).save(buf, format="PNG")
    msg = SimpleNamespace(chat_template_kwargs={"reasoning_effort": "ludicrous"}, tools=None)
    mgr.encode_multimodal(msg, [{"role": "user", "content": "x"}], [buf.getvalue()])
    assert seen.get("reasoning_effort") != "ludicrous", seen


def test_the_tokenize_worker_still_carries_inference_mode():
    """The decorator sat on tokenize_worker until a helper was inserted between them, which
    silently moved every worker tokenization out of inference_mode."""
    import freetoken.tokenizer.server as srv

    assert hasattr(srv.tokenize_worker, "__wrapped__"), (
        "tokenize_worker lost @torch.inference_mode()"
    )
    assert not hasattr(srv._image_processor_path, "__wrapped__"), (
        "the decorator landed on the helper instead"
    )


def test_no_image_support_is_a_client_error_not_a_server_fault():
    """count_tokens has no per-request isolation to fall back on, so 'this deployment has no
    vision tower' has to be distinguishable from 'the processor failed to load'."""
    from freetoken.tokenizer.tokenize import ImageInputUnsupported, TokenizeManager

    mgr = TokenizeManager.__new__(TokenizeManager)
    mgr._processor_obj = None
    mgr._processor_lock = __import__("threading").Lock()
    mgr._processor_path = None
    with pytest.raises(ImageInputUnsupported):
        mgr._processor()
    assert issubclass(ImageInputUnsupported, ValueError), "callers still catch ValueError"


def test_a_literal_placeholder_in_the_text_names_itself_in_the_error():
    """transformers raises StopIteration when the text has more placeholders than images, and
    str(StopIteration()) is empty -- the client got 'could not encode request: ' and nothing
    else. Catch it before the processor and say what is wrong."""
    import io as _io
    import threading as _threading
    from types import SimpleNamespace

    from PIL import Image as _Image

    from freetoken.tokenizer.tokenize import TokenizeManager

    class _Proc:
        image_token = "<|image_pad|>"

        def apply_chat_template(self, messages, **kw):
            return "user: <|image_pad|> and a literal <|image_pad|> <|image_pad|>"

        def __call__(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("the pre-check should have fired first")

    mgr = TokenizeManager.__new__(TokenizeManager)
    mgr._processor_obj = _Proc()
    mgr._processor_lock = _threading.Lock()
    mgr._processor_path = "unused"
    mgr._logged_effort_maps = set()
    mgr.effort_profile = lambda: None

    buf = _io.BytesIO()
    _Image.new("RGB", (4, 4)).save(buf, format="PNG")
    msg = SimpleNamespace(chat_template_kwargs=None, tools=None)
    with pytest.raises(ValueError, match=r"2 literal .*image_pad"):
        mgr.encode_multimodal(msg, [{"role": "user", "content": "x"}], [buf.getvalue()])


def test_a_messageless_exception_still_says_something():
    """`f"{exc or exc!r}"` applies !r to the WHOLE expression, so it reads repr() even for a
    normal exception. The client should see the message when there is one, the type when
    there is not."""
    assert (lambda e: str(e) or repr(e))(ValueError("boom")) == "boom"
    assert (lambda e: str(e) or repr(e))(StopIteration()) == "StopIteration()"


# --------------------------------------------------- the same hole in gemma4
def test_gemma4_also_scatters_per_request(monkeypatch):
    """gemma4 carried the identical batch-wide mask and assert. It is a separate model file,
    so the qwen3_5_moe fix does not reach it -- and the failure mode is the same worker kill."""
    import torch
    from types import SimpleNamespace

    import freetoken.models.gemma4.model as g4

    IMG = 262144
    img_req = SimpleNamespace(
        uid=1, extend_len=4, mm_embeds=torch.full((2, 3), 7.0), mm_scatter=True)
    text_req = SimpleNamespace(uid=2, extend_len=2, mm_embeds=None, mm_scatter=True)
    input_ids = torch.tensor([1, IMG, IMG, 1, 2, IMG])
    batch = SimpleNamespace(reqs=[img_req, text_req], has_images=True)
    monkeypatch.setattr(g4, "get_global_ctx", lambda: SimpleNamespace(batch=batch))

    out = g4.Gemma4Model._merge_multimodal(
        SimpleNamespace(_image_token_id=IMG), input_ids, torch.zeros(6, 3)
    )
    assert out[1].tolist() == [7.0] * 3 and out[2].tolist() == [7.0] * 3
    assert out[5].tolist() == [0.0] * 3, "the other request's placeholder is left alone"


def test_a_text_only_batch_never_walks_the_requests(monkeypatch):
    """has_images is the whole point of the flag: every decode step and every text prefill
    must take the early return rather than iterate the batch."""
    import torch
    from types import SimpleNamespace

    import freetoken.models.qwen3_5_moe.model as mod

    class _Boom(list):
        def __iter__(self):  # pragma: no cover - must not be reached
            raise AssertionError("a text-only batch must not walk its requests")

    batch = SimpleNamespace(reqs=_Boom(), has_images=False)
    monkeypatch.setattr(mod, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    x = torch.zeros(3, 4)
    assert mod.Qwen3_5Model._merge_multimodal(
        SimpleNamespace(_image_token_id=5), torch.tensor([1, 2, 3]), x
    ) is x
