from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.util
import io
import json
import os
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Tuple

import torch
from freetoken.message import TokenizeMsg
from freetoken.utils import init_logger
from transformers import PreTrainedTokenizerBase

from .effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)

logger = init_logger(__name__)

# A Qwen3-VL soft token covers 32x32 pixels (patch 16, merge 2), so an unclamped 4K screenshot
# costs ~8.2k tokens -- more than one prefill chunk, which is a terminal rejection. The
# checkpoint's own processor config only downscales above 16.7 Mpx, i.e. effectively never, so
# the cap is applied here instead: ~1 Mpx, about 1000 tokens for any image. Raise it with
# FREETOKEN_IMAGE_MAX_PIXELS when a prompt genuinely needs the detail.
IMAGE_MIN_PIXELS = 4 * 28 * 28
IMAGE_MAX_PIXELS = int(os.getenv("FREETOKEN_IMAGE_MAX_PIXELS", str(1280 * 28 * 28)))

_IMAGE_KEY_BASE = 1 << 30  # image cache-key ids start above every vocabulary


def _image_cache_ids(
    input_ids: torch.Tensor, image_token_id: int, images: List[bytes]
) -> torch.Tensor | None:
    """Prefix-cache key ids for an image prompt: ``input_ids`` with each placeholder run replaced
    by ids derived from that image's content hash.

    Every image expands to the SAME placeholder token id, so a prefix keyed on the raw ids would
    match a different image's run and serve its KV. Keying the run on the content instead makes
    the prompt safe to share, which is what lets an agent turn that carries a screenshot reuse the
    150k tokens in front of it instead of re-prefilling them.

    The ids are >= 2**30, above any vocabulary, so they can never collide with a real token, and
    they carry the position within the run so two different-length runs of the same image stay
    distinct. The model still reads ``input_ids``; this tensor is only ever a cache key.

    Which run belongs to which image is read off the prompt, not off the processor's geometry:
    runs come out in image order, so a 1:1 count means run i is image i. When the counts disagree
    -- two images expanded into one adjoining run, a template that emits placeholders of its own --
    the whole placeholder set keys on one hash over every image in order. That is coarser (a
    prompt differing only in its LAST image no longer shares the prefix before the first) but it
    is never wrong, which is the property that matters here.
    """
    hits = (input_ids == image_token_id).nonzero().flatten()
    if hits.numel() == 0 or not images:
        return None
    digests = [
        int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") for raw in images
    ]
    # Contiguous runs of placeholder positions, in prompt order.
    breaks = (hits[1:] - hits[:-1] != 1).nonzero().flatten()
    starts = [0] + (breaks + 1).tolist()
    ends = (breaks + 1).tolist() + [hits.numel()]
    if len(starts) != len(digests):
        combined = int.from_bytes(
            hashlib.blake2b(b"".join(digests[i].to_bytes(8, "big") for i in range(len(digests))),
                            digest_size=8).digest(), "big")
        starts, ends, digests = [0], [hits.numel()], [combined]
    ids = input_ids.clone()
    mask = _IMAGE_KEY_BASE - 1
    for lo, hi, h in zip(starts, ends, digests, strict=True):
        run = (torch.arange(hi - lo, dtype=torch.int64) + (h & mask)) & mask
        ids[hits[lo:hi]] = (run + _IMAGE_KEY_BASE).to(ids.dtype)
    return ids


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request.

    The single source of truth for this decision: the encode side
    (``_apply_dsv4_chat_encoder`` below) uses it to pick the prompt the model
    sees, and the frontend parse side (``server/openai_api.py``) imports it to
    decide whether the model's output begins inside a reasoning block. Keeping
    one implementation prevents the two sides from disagreeing. Thinking is on
    when tools are offered (dsv4 only emits well-formed tool calls in thinking
    mode) or when the caller requests it via ``chat_template_kwargs``.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


_EFFORT_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]


@dataclass
class MultimodalInputs:
    """One request's image tensors, in the layout the model's ``encode_images`` wants.

    Produced here (tokenizer worker, CPU) and shipped to the scheduler, which owns the
    model and turns them into soft-token embeddings. Sizes are modest but not tiny --
    gemma-4 gives ``[1, 2520, 768]`` fp32 per image (~7.7 MB) -- so they ride the normal
    backend queue rather than being recomputed on the other side.
    """

    #: The processor's own pixel layout, passed to the model untouched. gemma-4 gives
    #: ``[num_images, num_patches, 3*patch**2]``; Qwen3-VL packs every image's patches into one
    #: ``[total_patches, 3*temporal*patch**2]`` and describes the split in ``image_grid_thw``.
    pixel_values: torch.Tensor
    #: The geometry the model needs to place those patches, in whichever form its processor
    #: emits: gemma-4's ``image_position_ids`` ``[num_images, num_patches, 2]``, or Qwen3-VL's
    #: ``image_grid_thw`` ``[num_images, 3]``. Exactly one is set; the model's
    #: ``encode_images`` knows which one it asked for.
    image_position_ids: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    #: Prefix-cache key ids for this prompt (:func:`_image_cache_ids`), or None when the request
    #: has no usable placeholder run. Not a model input -- it rides the bundle only because that
    #: is what already reaches the scheduler; ``tokenizer.server`` lifts it back out.
    cache_ids: torch.Tensor | None = None

    def as_dict(self) -> dict[str, torch.Tensor]:
        """The same tensors as an opaque bundle for the trip to the scheduler.

        Keys are the processor's own names, so the model reads what it asked for and the
        layers in between carry the dict without knowing what is in it. Unset geometry is
        dropped rather than sent as None: a new modality adds a key here and changes
        nothing on the way."""
        d = {"pixel_values": self.pixel_values}
        if self.image_position_ids is not None:
            d["image_position_ids"] = self.image_position_ids
        if self.image_grid_thw is not None:
            d["image_grid_thw"] = self.image_grid_thw
        if self.cache_ids is not None:
            d["cache_ids"] = self.cache_ids
        return d


def _decode_data_url(url: str) -> bytes:
    """``data:image/png;base64,....`` -> raw bytes. Only data URLs are accepted: fetching
    a remote URL from inside the tokenizer worker would let a request drive outbound
    network traffic, so callers must inline the image."""
    if not url.startswith("data:"):
        raise ValueError(
            "image_url must be a data: URL with inline base64 (remote URLs are not fetched)"
        )
    _, _, payload = url.partition(",")
    if ";base64" not in url.split(",", 1)[0]:
        raise ValueError("image_url data URL must be base64-encoded")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"image_url base64 payload is not decodable: {exc}") from exc


def split_image_parts(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    """Split chat messages into (template-ready messages, image payloads).

    ``{"type": "image_url", "image_url": {"url": "data:..."}}`` blocks become bare
    ``{"type": "image"}`` placeholders -- what HF chat templates expect -- and their bytes
    are returned in the order the template will consume them. Messages without images pass
    through unchanged, so the text-only path keeps its exact behaviour.
    """
    images: List[bytes] = []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                parts.append(part)
                continue
            spec = part.get("image_url") or {}
            url = spec.get("url") if isinstance(spec, dict) else spec
            images.append(_decode_data_url(str(url or "")))
            parts.append({"type": "image"})
        out.append({**msg, "content": parts})
    return out, images


class ImageInputUnsupported(ValueError):
    """This deployment cannot accept images at all -- a CLIENT error (400), unlike a
    processor that exists but fails to load (a server fault, 500). Separated because
    /v1/messages/count_tokens has no per-request isolation to fall back on: without this
    distinction an agent client asking to count an image-bearing turn got a 500."""


class TokenizeManager:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, processor_path: str | None = None
    ) -> None:
        self.tokenizer = tokenizer
        # Only set for checkpoints that ship an image processor AND have vision enabled;
        # loaded on the first image so text-only serving never pays for it.
        self._processor_path = processor_path
        self._processor_obj: Any | None = None
        self._processor_lock = threading.Lock()
        self._dsv4_encoder = _load_dsv4_encoder_if_needed(tokenizer)
        self._effort_profile: EffortProfile | None = None
        self._thinking_profile: ThinkingProfile | None = None
        self._effort_lock = threading.Lock()
        self._logged_effort_maps: set[tuple[Any, str | None]] = set()

    def _processor(self) -> Any:
        """The HF processor for this checkpoint, loaded once, on the first image."""
        with self._processor_lock:
            if self._processor_obj is None:
                if self._processor_path is None:
                    raise ImageInputUnsupported(
                        "this model does not accept image input: either its family has "
                        "no vision tower here, or vision is off (pass --vision), or the "
                        "checkpoint has no processor config"
                    )
                try:
                    from transformers import AutoProcessor
                except ImportError as exc:  # pragma: no cover - import guard
                    raise ValueError(f"image input needs transformers: {exc}") from exc
                try:
                    # Clamp the image processor's resize bounds on the way in. The
                    # checkpoint ships longest_edge=16.7 Mpx, which never fires, so a 4K
                    # screenshot arrives as ~8.2k soft tokens -- past one prefill chunk, where
                    # a multimodal prompt is terminal. size= is forwarded to the nested image
                    # processor by AutoProcessor.
                    self._processor_obj = AutoProcessor.from_pretrained(
                        self._processor_path,
                        size={
                            "shortest_edge": IMAGE_MIN_PIXELS,
                            "longest_edge": IMAGE_MAX_PIXELS,
                        },
                    )
                except Exception as exc:
                    # Pillow/torchvision are optional extras; say so instead of surfacing
                    # transformers' lazy-import message to the API caller.
                    raise ValueError(
                        f"could not load the image processor ({exc}); image input needs "
                        "the 'vision' extra: pip install 'freetoken[vision]'"
                    ) from exc
            return self._processor_obj

    def encode_multimodal(
        self, msg: TokenizeMsg, messages: List[Dict[str, Any]], images: List[bytes]
    ) -> Tuple[torch.Tensor, MultimodalInputs]:
        """Render + tokenize a request that carries images, via the HF processor.

        The processor owns both halves and must run together: it expands each
        ``{"type": "image"}`` placeholder into exactly the number of image tokens its
        patch grid produced, so tokenizing the text separately would desynchronise the
        count the model asserts on.
        """
        from PIL import Image

        proc = self._processor()
        # Same effort quantization every other render path applies (render_prompt ->
        # _sanitize_effort): an unsupported reasoning_effort must not render differently
        # just because the request happened to carry an image.
        chat_template_kwargs = dict(self._sanitize_effort(dict(msg.chat_template_kwargs or {})))
        # ...and the same reasoning_strength broadcast (_render): a template reading only that
        # spelling would otherwise see no effort at all once the request carried an image.
        if "reasoning_effort" in chat_template_kwargs:
            chat_template_kwargs.setdefault(
                "reasoning_strength", chat_template_kwargs["reasoning_effort"]
            )
        if msg.tools is not None:
            chat_template_kwargs["tools"] = msg.tools
        prompt = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        # The template emits exactly one placeholder per image block; any extra one came from
        # the request's own TEXT, because the placeholder has a printable spelling
        # (`<|image_pad|>`). Left alone, the processor runs out of images mid-expansion and
        # raises StopIteration -- whose str() is empty, so the client got
        # "could not encode request: " and nothing else. Say what is actually wrong.
        marker = getattr(proc, "image_token", None)
        if isinstance(marker, str) and marker:
            extra = prompt.count(marker) - len(images)
            if extra > 0:
                raise ValueError(
                    f"prompt text contains {extra} literal {marker!r} token(s) beyond its "
                    f"{len(images)} image(s); remove them (the image placeholder is reserved "
                    "for the processor's own expansion)"
                )
        pil = [Image.open(io.BytesIO(raw)).convert("RGB") for raw in images]
        enc = proc(text=[prompt], images=pil, return_tensors="pt")
        input_ids = enc["input_ids"][0].view(-1).to(torch.int32)
        # The cache key is built from the RAW request bytes, not from the tensors above: two
        # requests that sent the same file must key the same, and the processor's output is the
        # wrong thing to hash (it is float, resize-dependent, and much larger). Failing to build
        # one is not an error -- the request simply keeps the old behaviour of not sharing a
        # prefix, which is what the scheduler does with cache_ids=None.
        cache_ids = None
        if isinstance(marker, str) and marker:
            token_id = self.tokenizer.convert_tokens_to_ids(marker)
            if isinstance(token_id, int) and token_id >= 0:
                cache_ids = _image_cache_ids(input_ids, token_id, images)
        return (
            input_ids,
            MultimodalInputs(
                pixel_values=enc["pixel_values"],
                # Whichever the checkpoint's processor produced. Neither key is universal:
                # gemma-4 has no grid_thw, Qwen3-VL has no position_ids.
                image_position_ids=enc.get("image_position_ids"),
                image_grid_thw=enc.get("image_grid_thw"),
                cache_ids=cache_ids,
            ),
        )

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """Text-only encode. Callers that must support images use :meth:`encode`."""
        return [ids for ids, _ in self.encode(msgs)]

    def encode(self, msgs: List[TokenizeMsg]) -> List[Tuple[torch.Tensor, MultimodalInputs | None]]:
        results: List[Tuple[torch.Tensor, MultimodalInputs | None]] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                messages, images = split_image_parts(msg.text)
                if images:
                    results.append(self.encode_multimodal(msg, messages, images))
                    continue
            prompt = self.render_prompt(msg)
            # A jinja chat template owns every special token (HF's apply_chat_template
            # tokenizes with add_special_tokens=False for the same reason): tokenizers
            # that auto-add bos (muse-glimmer's, llama's) would otherwise double it --
            # the template already rendered one. Raw-string prompts and the dsv4
            # encoder path keep the default.
            templated = isinstance(msg.text, list) and self._dsv4_encoder is None
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=not templated
                )
            )
            results.append((input_ids.view(-1).to(torch.int32), None))
        return results

    def render_prompt(self, msg: TokenizeMsg) -> str:
        """The template/encoder half of ``tokenize``, exposed so the frontend can
        validate a request before committing an SSE stream. Sanitizes
        ``reasoning_effort`` first: every render path (worker, frontend
        validation, count_tokens) must quantize identically."""
        if not isinstance(msg.text, list):
            return msg.text
        return self._render(
            msg.text, msg.tools, self._sanitize_effort(msg.chat_template_kwargs or {})
        )

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any],
    ) -> str:
        """Raw render, no effort sanitation — the probe needs unsupported values
        to actually reach the template so rejection is observable."""
        if self._dsv4_encoder is not None:
            return _apply_dsv4_chat_encoder(
                self._dsv4_encoder, messages, tools, chat_template_kwargs
            )
        # Broadcast the effort in every spelling the ecosystem's templates read
        # (muse-glimmer grades ``reasoning_strength``; Jinja ignores undeclared
        # variables) -- the same rule the thinking toggles use. An explicit
        # caller-provided spelling wins over the broadcast.
        if "reasoning_effort" in chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs.setdefault(
                "reasoning_strength", chat_template_kwargs["reasoning_effort"]
            )
        if tools is not None:
            chat_template_kwargs = {**chat_template_kwargs, "tools": tools}
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        assert isinstance(prompt, str)
        return prompt

    def effort_profile(self) -> EffortProfile:
        """The checkpoint's effort vocabulary, probed on first use and cached
        for the process lifetime."""
        with self._effort_lock:
            if self._effort_profile is None:
                self._effort_profile = probe_effort_profile(self._probe_render)
                logger.info(
                    "reasoning-effort profile: supported=%s default=%s",
                    sorted(self._effort_profile.supported) or "(none)",
                    self._effort_profile.default,
                )
            return self._effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The checkpoint's thinking controls (toggle behavior + effort
        vocabulary), probed on first use and cached for the process lifetime.
        Feeds the /v1/cache/status gear derivation."""
        efforts = self.effort_profile()
        with self._effort_lock:
            if self._thinking_profile is None:
                self._thinking_profile = probe_thinking_profile(self._probe_render, efforts)
            return self._thinking_profile

    def _probe_render(
        self, kwargs: dict[str, Any], tools: list[dict[str, Any]] | None
    ) -> str:
        return self._render(_EFFORT_PROBE_MESSAGES, tools, kwargs)

    def _sanitize_effort(self, chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs.get("reasoning_effort")
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list/dict).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self._logged_effort_maps:
            self._logged_effort_maps.add(key)
            logger.info(
                "reasoning_effort %r is not supported by this checkpoint; using %s",
                raw,
                mapped if mapped is not None else "the template default",
            )
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            del sanitized["reasoning_effort"]
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized


def _load_dsv4_encoder_if_needed(tokenizer: PreTrainedTokenizerBase) -> ModuleType | None:
    if getattr(tokenizer, "chat_template", None):
        return None
    model_path = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "")
    if not model_path:
        return None
    encoder_path = os.path.join(str(model_path), "encoding", "encoding_dsv4.py")
    if not os.path.isfile(encoder_path):
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        return None
    return module


def _apply_dsv4_chat_encoder(
    encoder: ModuleType,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict,
) -> str:
    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("tool_calls"):
            message["tool_calls"] = _dsv4_tool_calls(message["tool_calls"])
    if tools:
        _attach_tools_to_dsv4_messages(rendered_messages, tools)

    # No effort filtering here: the caller sanitized already, and the probe
    # needs raw values to reach the encoder's own validation.
    return encoder.encode_messages(
        rendered_messages,
        thinking_mode=resolve_thinking_mode(chat_template_kwargs, tools),
        reasoning_effort=chat_template_kwargs.get("reasoning_effort"),
    )


def _dsv4_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING
    (it json.loads then iterates .items()); a dict (what ``render_messages``
    produces for Jinja templates) trips its bare-except fallback, which wraps the
    whole payload in a bogus parameter literally named ``arguments``. Re-serialize
    here. Copies each tool-call dict: the outer message copy is shallow, so these
    are shared with the caller."""
    rendered = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        fn["arguments"] = _dsv4_arguments_str(fn.get("arguments"))
        tc["function"] = fn
        rendered.append(tc)
    return rendered


def _dsv4_arguments_str(arguments: Any) -> str:
    """Missing/empty means no arguments (vLLM parity); anything else that is not
    a JSON object is rejected -- ValueError becomes a per-request "could not
    encode request" error, never a worker crash -- matching sglang's
    validate-then-400. A non-object would otherwise raise uncaught in the
    encoder's .items() or be wrapped as garbage."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    shown = f"{arguments!r:.200}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"tool call function.arguments must be valid JSON, got {shown}"
            ) from err
        if isinstance(parsed, dict):
            return arguments
    raise ValueError(f"tool call function.arguments must be a JSON object, got {shown}")


def _attach_tools_to_dsv4_messages(messages: list[dict], tools: list[dict]) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
