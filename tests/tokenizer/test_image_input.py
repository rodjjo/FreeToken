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
