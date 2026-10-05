"""Dynamic forward helpers for decoder-only language models.

The main entry point, ``dynamic_forward``, resumes a model forward pass from an
existing hidden state instead of token ids. For Llama, it mirrors the decoder
portion of ``LlamaModel.forward`` from a selected layer: decoder layers plus the
final model norm by default. It intentionally leaves lm_head and logits
post-processing to the caller. Callers pass a normal 2D ``attention_mask``;
model-specific causal masks are created inside this file.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover - useful for clearer runtime errors.
    DynamicCache = None


@dataclass
class DynamicForwardOutput:
    """Model-level output needed by the next decode step."""

    last_hidden_state: torch.Tensor
    past_key_values: Any


def _decoder_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the inner decoder model from a CausalLM wrapper when present."""

    return getattr(model, "model", model)


def _cache_seq_length(cache: Any, layer_idx: int) -> int:
    if cache is None:
        return 0
    get_seq_length = getattr(cache, "get_seq_length", None)
    if get_seq_length is None:
        return 0
    try:
        return int(get_seq_length(layer_idx))
    except TypeError:
        return int(get_seq_length())


def detect_model_type(model: torch.nn.Module) -> str:
    """Return the dynamic-forward backend name for a HF model."""

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        decoder = getattr(model, "model", None)
        decoder_config = getattr(decoder, "config", None)
        model_type = getattr(decoder_config, "model_type", None)

    if model_type is None:
        raise ValueError("Cannot detect model type because `model.config.model_type` is missing.")

    normalized = str(model_type).lower()
    if normalized == "llama":
        return "llama"
    if normalized in {"qwen3_5", "qwen3_5_text"}:
        return "qwen3_5"

    raise ValueError(f"Unsupported model_type for dynamic forward: {model_type!r}")


def dynamic_forward(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    start_layer: int,
    *,
    model_type: str = "auto",
    **kwargs: Any,
) -> DynamicForwardOutput:
    """Dispatch dynamic forward to the implementation for ``model_type``.

    Use ``model_type="auto"`` to read ``model.config.model_type``. You can also
    pass ``"llama"`` or ``"qwen3_5"`` manually. Pass the same ``attention_mask``
    and ``position_ids`` you would pass to Hugging Face ``model(...)``; this
    function builds the layer-level causal mask internally.
    """

    resolved_type = detect_model_type(model) if model_type == "auto" else model_type.lower()
    if resolved_type == "llama":
        return llama_dynamic_forward(model, hidden_states, start_layer, **kwargs)
    if resolved_type == "qwen3_5":
        return qwen3_5_dynamic_forward(model, hidden_states, start_layer, **kwargs)

    raise ValueError(f"Unsupported dynamic forward backend: {model_type!r}")


def _create_llama_causal_mask(
    decoder: torch.nn.Module,
    attention_mask: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    cache_position: torch.LongTensor,
    past_key_values: Any,
    position_ids: torch.LongTensor,
    output_attentions: bool,
    layer_idx: int,
) -> Any:
    update_causal_mask = getattr(decoder, "_update_causal_mask", None)
    if update_causal_mask is not None:
        return update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            past_key_values,
            output_attentions,
        )

    from transformers.masking_utils import create_causal_mask

    return create_causal_mask(
        config=decoder.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        layer_idx=layer_idx,
    )


def _rotary_embeddings(decoder: torch.nn.Module, hidden_states: torch.Tensor, position_ids: torch.LongTensor) -> Any:
    rotary_emb = getattr(decoder, "rotary_emb", None)
    if rotary_emb is None:
        return None
    try:
        return rotary_emb(hidden_states, position_ids=position_ids)
    except TypeError:
        return rotary_emb(hidden_states, position_ids)


def _layer_forward(
    decoder_layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: Any,
    position_ids: torch.LongTensor,
    past_key_values: Any,
    output_attentions: bool,
    use_cache: bool,
    cache_position: torch.LongTensor,
    position_embeddings: Any,
    decoder_kwargs: dict[str, Any],
) -> torch.Tensor:
    signature = inspect.signature(decoder_layer.forward)
    params = signature.parameters
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

    layer_kwargs = {}
    if "attention_mask" in params or has_kwargs:
        layer_kwargs["attention_mask"] = attention_mask
    if "position_ids" in params or has_kwargs:
        layer_kwargs["position_ids"] = position_ids
    if "past_key_values" in params:
        layer_kwargs["past_key_values"] = past_key_values
    elif "past_key_value" in params:
        layer_kwargs["past_key_value"] = past_key_values
    if "output_attentions" in params:
        layer_kwargs["output_attentions"] = output_attentions
    if "use_cache" in params or has_kwargs:
        layer_kwargs["use_cache"] = use_cache
    if "cache_position" in params:
        layer_kwargs["cache_position"] = cache_position
    if "position_embeddings" in params:
        layer_kwargs["position_embeddings"] = position_embeddings

    for key, value in decoder_kwargs.items():
        if key in params or has_kwargs:
            layer_kwargs[key] = value

    layer_outputs = decoder_layer(hidden_states, **layer_kwargs)
    if isinstance(layer_outputs, tuple):
        return layer_outputs[0]
    return layer_outputs


def llama_dynamic_forward(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    start_layer: int,
    *,
    end_layer: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Any = None,
    use_cache: bool = True,
    cache_position: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    apply_final_norm: bool = True,
    **decoder_kwargs: Any,
) -> DynamicForwardOutput:
    """Run Llama decoder layers ``[start_layer, end_layer)`` from an existing hidden state.

    Args:
        model: Usually ``LlamaForCausalLM``. The function also accepts the inner
            ``LlamaModel``.
        hidden_states: Hidden tensor already produced by layer ``start_layer - 1``.
            For ``start_layer == 0`` this should be the token embeddings.
        start_layer: First decoder layer to execute. Layer indices are zero-based.
        end_layer: Exclusive end layer. Defaults to the number of decoder layers.
        attention_mask: The same 2D mask shape used by HF generation, i.e.
            ``[batch, past_length + current_length]``. Do not pass a causal mask;
            it is built internally.
        past_key_values: Cache for the suffix layers. It can be independent from
            the cache used by earlier layers.
        cache_position: Absolute token positions for ``hidden_states``. Pass this
            explicitly when ``start_layer > 0`` during cached decoding.
        apply_final_norm: Apply ``model.norm`` before returning. Keep this true
            when ``end_layer`` reaches the final layer to match ``LlamaModel``.

    Returns:
        ``DynamicForwardOutput`` with output hidden states and the updated suffix
        cache. With the default ``apply_final_norm=True``, ``last_hidden_state``
        matches the normalized ``last_hidden_state`` returned by ``LlamaModel``.
    """

    decoder = _decoder_model(model)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("Expected a Llama-style model with a `.layers` decoder stack.")

    num_layers = len(layers)
    if end_layer is None:
        end_layer = num_layers
    if not 0 <= start_layer <= end_layer <= num_layers:
        raise ValueError(f"Invalid layer range [{start_layer}, {end_layer}) for {num_layers} layers.")

    if use_cache and past_key_values is None:
        past_key_values = _new_dynamic_cache(getattr(decoder, "config", None))

    batch_size, current_length = hidden_states.shape[:2]
    past_length = _cache_seq_length(past_key_values, start_layer)

    if cache_position is None:
        cache_position = torch.arange(
            past_length,
            past_length + current_length,
            device=hidden_states.device,
            dtype=torch.long,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, past_length + current_length),
            device=hidden_states.device,
            dtype=torch.long,
        )

    causal_mask = _create_llama_causal_mask(
        decoder,
        attention_mask,
        hidden_states,
        cache_position,
        past_key_values,
        position_ids,
        output_attentions,
        start_layer,
    )

    position_embeddings = _rotary_embeddings(decoder, hidden_states, position_ids)

    for decoder_layer in layers[start_layer:end_layer]:
        hidden_states = _layer_forward(
            decoder_layer,
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            decoder_kwargs=decoder_kwargs,
        )

    if apply_final_norm:
        hidden_states = decoder.norm(hidden_states)

    return DynamicForwardOutput(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def _text_decoder_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the text decoder for text-only or multimodal Qwen wrappers."""

    decoder = _decoder_model(model)
    if hasattr(decoder, "layers"):
        return decoder
    if hasattr(decoder, "language_model"):
        return decoder.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise TypeError("Expected a model with `.layers` or `.language_model.layers`.")


def _new_dynamic_cache(config: Any = None) -> Any:
    if DynamicCache is None:
        raise ImportError("transformers.cache_utils.DynamicCache is required when use_cache=True.")
    if config is not None:
        try:
            return DynamicCache(config=config)
        except TypeError:
            pass
    return DynamicCache()


def _qwen3_5_position_ids(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    past_key_values: Any,
    start_layer: int,
) -> tuple[torch.LongTensor, Optional[torch.LongTensor]]:
    batch_size, current_length = hidden_states.shape[:2]
    if position_ids is None:
        past_length = _cache_seq_length(past_key_values, start_layer)
        position_ids = torch.arange(
            past_length,
            past_length + current_length,
            device=hidden_states.device,
            dtype=torch.long,
        )
        position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:], position_ids[0]
    return position_ids, None


def qwen3_5_dynamic_forward(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    start_layer: int,
    *,
    end_layer: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Any = None,
    use_cache: bool = True,
    cache_position: Optional[torch.LongTensor] = None,
    apply_final_norm: bool = True,
    **decoder_kwargs: Any,
) -> DynamicForwardOutput:
    """Run Qwen3.5 text decoder layers from an existing hidden state.

    This supports the dense Qwen3.5 family, including Qwen3.5-4B and
    Qwen3.5-9B. ``cache_position`` is accepted for API symmetry with Llama, but
    Qwen3.5 derives positions from ``position_ids``. Pass a normal 2D
    ``attention_mask``; the full/linear causal masks are built internally.
    """

    try:
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 dynamic forward requires a Transformers version with "
            "`transformers.masking_utils.create_recurrent_attention_mask`."
        ) from exc

    del cache_position

    decoder = _text_decoder_model(model)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("Expected Qwen3.5 text decoder with a `.layers` stack.")

    num_layers = len(layers)
    if end_layer is None:
        end_layer = num_layers
    if not 0 <= start_layer <= end_layer <= num_layers:
        raise ValueError(f"Invalid layer range [{start_layer}, {end_layer}) for {num_layers} layers.")

    if use_cache and past_key_values is None:
        past_key_values = _new_dynamic_cache(getattr(decoder, "config", None))

    batch_size, current_length = hidden_states.shape[:2]
    if attention_mask is None:
        past_length = _cache_seq_length(past_key_values, start_layer)
        attention_mask = torch.ones(
            (batch_size, past_length + current_length),
            device=hidden_states.device,
            dtype=torch.long,
        )

    rope_position_ids, text_position_ids = _qwen3_5_position_ids(
        hidden_states,
        position_ids,
        past_key_values,
        start_layer,
    )

    if isinstance(attention_mask, dict):
        causal_mask_mapping = attention_mask
    else:
        mask_kwargs = {
            "config": decoder.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }

    position_embeddings = decoder.rotary_emb(hidden_states, rope_position_ids)

    for layer_idx in range(start_layer, end_layer):
        decoder_layer = layers[layer_idx]
        layer_type = decoder.config.layer_types[layer_idx]
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **decoder_kwargs,
        )

    if apply_final_norm:
        hidden_states = decoder.norm(hidden_states)

    return DynamicForwardOutput(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )
