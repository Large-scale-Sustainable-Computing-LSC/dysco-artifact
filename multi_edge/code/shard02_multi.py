"""Cloud-side server for the heterogeneous-cut prototype.

One loaded model serves every session whatever its cut layer, because a decoder
layer depends only on the hidden state, positions and mask it receives. Four
decode schedulers are selected by flags: FIFO, exact-match batching, round-robin
interleaving, and depth-synchronized batching.
"""

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Iterable, Optional
import argparse
import collections
import contextlib
import gc
import inspect
import os
import queue
import threading
import time
import atexit as _atexit
import signal as _signal
import grpc
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

import llama_multi_pb2
import llama_multi_pb2_grpc

# Most sessions a single decode forward call may cover; every reported fleet is
# eight sessions, so the window is never the binding limit.
MAX_DECODE_BATCH = 8

# Opt-in decode-step profiler (PROFILE_PROGRESSIVE=1). It measures CPU wall time per
# phase, that is kernel-launch and Python cost, not GPU execution time.
_PROF = os.environ.get("PROFILE_PROGRESSIVE") == "1"
_prof_time: dict[str, float] = collections.defaultdict(float)
_prof_calls: dict[str, int] = collections.defaultdict(int)


@contextlib.contextmanager
def _prof(name: str):
    """Accumulate wall time and call count for one phase of a decode step."""

    if not _PROF:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        _prof_time[name] += time.perf_counter() - started
        _prof_calls[name] += 1


def dump_profile(tag: str = "") -> None:
    """Print the accumulated per-phase profile, if profiling is enabled."""

    if not _PROF or not _prof_time:
        return
    total = _prof_time.get("step_total", sum(_prof_time.values()))
    print(f"\n==== decode-step profile {tag} ====", flush=True)
    print(f"{'phase':>22} {'ms_total':>10} {'calls':>8} {'us/call':>9} {'%step':>7}", flush=True)
    if _prof_calls.get("_groupsize_n"):
        print(f"mean sessions merged per progressive-sc step: "
              f"{_prof_time['_groupsize_sum']/_prof_calls['_groupsize_n']:.2f}", flush=True)
    for name in sorted(_prof_calls):
        if name.startswith("_miss"):
            print(f"  fastpath miss cause {name[6:]:>22}: {_prof_calls[name]}", flush=True)
    for name in sorted(_prof_time, key=lambda key: -_prof_time[key]):
        if name.startswith("_groupsize"):
            continue
        ms = _prof_time[name] * 1e3
        calls = _prof_calls[name]
        print(f"{name:>22} {ms:>10.1f} {calls:>8} {ms*1e3/max(calls,1):>9.1f} "
              f"{_prof_time[name]/total*100 if total else 0:>6.1f}%", flush=True)


def _filter_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    """Drop keyword arguments ``func`` does not accept."""

    accepted = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def parse_torch_dtype(name: str) -> torch.dtype:
    """Return the torch dtype for a name such as ``bfloat16`` or ``torch.float32``."""

    if not name:
        return torch.bfloat16
    normalized = name.replace("torch.", "")
    if not hasattr(torch, normalized):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, normalized)


def serialize_tensor(tensor: torch.Tensor) -> Any:
    """Pack a tensor into the wire format shared with the edge client."""

    buffer = BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return llama_multi_pb2.TensorData(
        tensor_bytes=buffer.getvalue(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
    )


def deserialize_tensor(tensor_data: Any, device: torch.device | str) -> torch.Tensor:
    """Unpack a wire-format tensor onto ``device``."""

    tensor = torch.load(BytesIO(tensor_data.tensor_bytes), map_location="cpu", weights_only=True)
    return tensor.to(device)


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Return the highest-scoring next token; decoding is greedy throughout."""

    return logits.argmax(dim=-1, keepdim=True)


def cache_bytes_for(past_key_values: Any) -> int:
    """Return the GPU bytes one session's key/value cache currently holds."""

    if past_key_values is None:
        return 0
    total = 0
    for layer in past_key_values.layers:
        if getattr(layer, "is_initialized", False):
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total


def _left_pad(tensor: torch.Tensor, pad_amount: int) -> torch.Tensor:
    """Pad a ``[batch, heads, seq_len, head_dim]`` cache tensor on the older end."""

    if pad_amount <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[-2] = pad_amount
    zeros = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([zeros, tensor], dim=-2)


def merge_decode_caches(sessions: list[Any], cut_layer: int, config: Any) -> tuple[Any, torch.Tensor, torch.Tensor, list[int], int]:
    """Stack single-session caches that share a cut layer into one batched cache.

    Returns:
        The merged cache, the attention mask, the position ids, each session's
        pre-step history length, and the padded history width.
    """

    past_lengths = [session.current_length - 1 for session in sessions]
    max_past = max(past_lengths)

    merged = DynamicCache(config=config)
    for layer_idx in range(cut_layer, config.num_hidden_layers):
        keys, values = [], []
        for session, past_len in zip(sessions, past_lengths):
            layer = session.past_key_values.layers[layer_idx]
            pad = max_past - past_len
            keys.append(_left_pad(layer.keys, pad))
            values.append(_left_pad(layer.values, pad))
        merged.layers[layer_idx].keys = torch.cat(keys, dim=0)
        merged.layers[layer_idx].values = torch.cat(values, dim=0)
        merged.layers[layer_idx].is_initialized = True

    device = merged.layers[cut_layer].keys.device
    attention_mask = torch.zeros((len(sessions), max_past + 1), dtype=torch.long, device=device)
    position_ids = torch.empty((len(sessions), 1), dtype=torch.long, device=device)
    for row, past_len in enumerate(past_lengths):
        attention_mask[row, max_past - past_len:] = 1
        position_ids[row, 0] = past_len

    return merged, attention_mask, position_ids, past_lengths, max_past


def split_decode_cache(
    merged_cache: Any,
    sessions: list[Any],
    past_lengths: list[int],
    max_past: int,
    cut_layer: int,
    config: Any,
) -> None:
    """Undo ``merge_decode_caches`` after a batched forward pass."""

    for row, (session, past_len) in enumerate(zip(sessions, past_lengths)):
        pad = max_past - past_len
        for layer_idx in range(cut_layer, config.num_hidden_layers):
            merged_layer = merged_cache.layers[layer_idx]
            own_layer = session.past_key_values.layers[layer_idx]
            own_layer.keys = merged_layer.keys[row : row + 1, :, pad:, :].contiguous()
            own_layer.values = merged_layer.values[row : row + 1, :, pad:, :].contiguous()
            own_layer.is_initialized = True


def merge_decode_caches_range(sessions: list[Any], layer_range: tuple[int, int], config: Any) -> tuple[Any, torch.Tensor, torch.Tensor, list[int], int]:
    """Merge caches over an arbitrary ``[start, end)`` layer range."""

    past_lengths = [session.current_length - 1 for session in sessions]
    max_past = max(past_lengths)

    merged = DynamicCache(config=config)
    for layer_idx in range(*layer_range):
        keys, values = [], []
        for session, past_len in zip(sessions, past_lengths):
            layer = session.past_key_values.layers[layer_idx]
            actual_len = layer.keys.shape[-2] if getattr(layer, "is_initialized", False) else -1
            if actual_len != past_len:
                raise RuntimeError(
                    f"DESYNC session={session.session_id} layer_idx={layer_idx} layer_range={layer_range} "
                    f"assumed_past_len={past_len} (from current_length={session.current_length}) "
                    f"actual_cache_len={actual_len}"
                )
            pad = max_past - past_len
            keys.append(_left_pad(layer.keys, pad))
            values.append(_left_pad(layer.values, pad))
        merged.layers[layer_idx].keys = torch.cat(keys, dim=0)
        merged.layers[layer_idx].values = torch.cat(values, dim=0)
        merged.layers[layer_idx].is_initialized = True

    device = merged.layers[layer_range[0]].keys.device
    attention_mask = torch.zeros((len(sessions), max_past + 1), dtype=torch.long, device=device)
    position_ids = torch.empty((len(sessions), 1), dtype=torch.long, device=device)
    for row, past_len in enumerate(past_lengths):
        attention_mask[row, max_past - past_len:] = 1
        position_ids[row, 0] = past_len

    return merged, attention_mask, position_ids, past_lengths, max_past


def split_decode_cache_range(
    merged_cache: Any,
    sessions: list[Any],
    past_lengths: list[int],
    max_past: int,
    layer_range: tuple[int, int],
    config: Any,
) -> None:
    """Undo ``merge_decode_caches_range`` over the same layer range."""

    for row, (session, past_len) in enumerate(zip(sessions, past_lengths)):
        pad = max_past - past_len
        for layer_idx in range(*layer_range):
            merged_layer = merged_cache.layers[layer_idx]
            own_layer = session.past_key_values.layers[layer_idx]
            own_layer.keys = merged_layer.keys[row : row + 1, :, pad:, :].contiguous()
            own_layer.values = merged_layer.values[row : row + 1, :, pad:, :].contiguous()
            own_layer.is_initialized = True


@dataclass
class _PendingDecode:
    """One DecodeStep request waiting for the dispatcher thread to run it."""

    session: Session
    hidden: torch.Tensor
    enqueued_at: float
    event: Optional[threading.Event]
    result: Optional[dict] = None


@dataclass
class BatchGroup:
    """A cut layer's batched cache, kept alive while its membership is stable."""

    cut_layer: int
    session_order: list[Session]
    merged_cache: Any
    current_len: int
    pad_offsets: list[int]


@dataclass
class RangeBatchGroup:
    """A batched cache over an arbitrary ``[start, end)`` layer range."""

    layer_range: tuple[int, int]
    session_order: list[Session]
    merged_cache: Any
    current_len: int
    pad_offsets: list[int]


def _session_key(sessions: Iterable[Session]) -> tuple[str, ...]:
    """Identify a group by its session ids, independent of arrival order."""

    return tuple(session.session_id for session in sessions)


@dataclass
class Session:
    """One connected edge client and the cloud-side state it owns."""

    session_id: str
    cut_layer: int
    past_key_values: Any
    current_length: int = 0


class TailRunner:
    """Runs ``model.layers[cut:]`` for whatever cut a session was assigned."""

    def __init__(self, full_model: torch.nn.Module) -> None:
        self.config = full_model.config
        self.layers = full_model.model.layers
        self.norm = full_model.model.norm
        self.rotary_emb = full_model.model.rotary_emb
        self.lm_head = full_model.lm_head
        self.num_hidden_layers = full_model.config.num_hidden_layers

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        cut_layer: int,
    ) -> torch.Tensor:
        """Run layers ``[cut_layer, end)`` and return next-token logits."""

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_len = inputs_embeds.shape[1]
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device
        )

        causal_mask = create_causal_mask(
            **_filter_supported_kwargs(
                create_causal_mask,
                config=self.config,
                input_embeds=inputs_embeds,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
                # layer_idx must name the first layer this runner populates: sized from
                # layers[0], which a tail session never initializes, no real mask is built
                # and padding would leak into attention.
                layer_idx=cut_layer,
            )
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[cut_layer:]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states[:, -1:, :])

    def forward_range(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        layer_range: tuple[int, int],
        final: bool,
    ) -> torch.Tensor:
        """Run an arbitrary contiguous ``[start, end)`` layer range.

        Args:
            final: Whether this segment reaches the end of the model. If so the
                final norm and language-model head are applied and logits are
                returned; otherwise the raw hidden states are returned, so the
                next segment, with a different set of merged sessions, can
                continue from them.
        """

        start, end = layer_range
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_len = inputs_embeds.shape[1]
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device
        )

        causal_mask = create_causal_mask(
            **_filter_supported_kwargs(
                create_causal_mask,
                config=self.config,
                input_embeds=inputs_embeds,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
                # As in forward(): the first layer this segment touches is the first slot
                # populated for every session in its batch.
                layer_idx=start,
            )
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[start:end]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        if not final:
            return hidden_states
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states[:, -1:, :])


class ShardServicer(llama_multi_pb2_grpc.ShardServiceServicer):
    """Serves every connected edge session from one loaded copy of the model."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        batch_window_s: float = 0.0,
        interleave_group_size: int = 0,
        progressive_batching: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.batch_window_s = batch_window_s
        self.interleave_group_size = interleave_group_size
        # Depth-synchronized batching needs a window to collect a mixed-cut
        # batch from, so it is only meaningful with batch_window_s > 0.
        self.progressive_batching = progressive_batching
        if self.interleave_group_size > 0 and self.batch_window_s > 0:
            # The two schedulers are mutually exclusive. Interleaving is the
            # one explicitly requested here, so it wins over the default window.
            print(f"Note: --interleave-group-size={self.interleave_group_size} given, "
                  f"disabling decode batching (was {self.batch_window_s * 1000:.1f}ms window).")
            self.batch_window_s = 0.0

        print(f"Loading {model_id} once, in full, on {device} ({dtype}) ...")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model.eval()
        model.to(device)
        self.runner = TailRunner(model)
        # Edges send pre-embedded hidden states, so the embedding table is dead weight
        # here. Release it before the baseline below is measured.
        del model
        print(f"Ready. {self.runner.num_hidden_layers} layers total; "
              f"any session may cut anywhere in [1, {self.runner.num_hidden_layers - 1}].")

        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.weight_bytes = torch.cuda.memory_allocated(self.device)
        else:
            self.weight_bytes = 0
        print(f"Baseline (weights + CUDA context): {self.weight_bytes / 1e9:.3f} GB")

        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.Lock()
        # One GPU, so one forward pass at a time.
        self.gpu_lock = threading.Lock()

        # Persistent groups, keyed by cut layer or by ("prefix", c) / ("tail",). They hold
        # live tensors, so they are touched only under gpu_lock.
        self.batch_groups: dict[int, BatchGroup] = {}
        self.progressive_groups: dict[tuple, RangeBatchGroup] = {}

        # Decode requests queue here so arrivals within batch_window_s can share a call.
        # Prefill is never batched: padding would cost more than it saves.
        self.decode_queue: queue.Queue = queue.Queue()
        if self.batch_window_s > 0:
            threading.Thread(target=self._batch_worker, daemon=True).start()
            if self.progressive_batching:
                mode = "depth-synchronized"
            else:
                mode = "exact-cut-match"
            print(f"Decode batching enabled [{mode}]: window={self.batch_window_s * 1000:.1f}ms, "
                  f"max_batch_size={MAX_DECODE_BATCH}")
        elif self.interleave_group_size > 0:
            # Round-robin interleaving: a fixed group takes turns, a session that is not
            # ready is skipped, and the rest wait in a first-come overflow list. No call
            # merges sessions.
            self.interleave_lock = threading.Lock()
            self.interleave_order: list[str] = []
            self.interleave_overflow: list[str] = []
            self.interleave_mailbox: dict[str, _PendingDecode] = {}
            threading.Thread(target=self._interleave_worker, daemon=True).start()
            print(f"Round-robin interleaving enabled: group_size={self.interleave_group_size}")
        else:
            print("Decode batching disabled (--batch-window-ms 0): one session per forward call.")

    def _get_session(self, session_id: str, context: Any) -> Optional[Session]:
        """Return a registered session, or set a gRPC error and return None."""

        with self.sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Unknown session_id {session_id!r}; call InitConnection first.")
        return session

    def InitConnection(self, request: Any, context: Any) -> Any:
        cfg = request.config
        if cfg.model_id and cfg.model_id != self.model_id:
            message = f"Server has {self.model_id} loaded, session asked for {cfg.model_id}"
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(message)
            return llama_multi_pb2.InitResponse(status=412, message=message)

        cut_layer = cfg.layer_split
        if not (1 <= cut_layer < self.runner.num_hidden_layers):
            message = (
                f"layer_split must be in [1, {self.runner.num_hidden_layers - 1}], got {cut_layer}"
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(message)
            return llama_multi_pb2.InitResponse(status=400, message=message)

        session = Session(
            session_id=request.session_id,
            cut_layer=cut_layer,
            past_key_values=DynamicCache(config=self.runner.config),
            current_length=0,
        )
        with self.sessions_lock:
            self.sessions[request.session_id] = session

        print(f"[{request.session_id}] init: cut_layer={cut_layer}")
        return llama_multi_pb2.InitResponse(status=200, message="ok")

    def PrefillStep(self, request: Any, context: Any) -> Any:
        start = time.time()
        session = self._get_session(request.session_id, context)
        if session is None:
            return llama_multi_pb2.PrefillResponse()

        hidden = deserialize_tensor(request.hidden, self.device).to(self.dtype)
        attention_mask = deserialize_tensor(request.attention_mask, self.device)
        position_ids = deserialize_tensor(request.position_ids, self.device)

        wait_start = time.time()
        with self.gpu_lock:
            wait_s = time.time() - wait_start
            infer_start = time.time()
            with torch.no_grad():
                logits = self.runner.forward(
                    inputs_embeds=hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=session.past_key_values,
                    cut_layer=session.cut_layer,
                )
            next_token = greedy_next_token(logits[:, -1, :])
            infer_s = time.time() - infer_start

        session.current_length = attention_mask.shape[1] + 1

        return llama_multi_pb2.PrefillResponse(
            next_token=serialize_tensor(next_token),
            server_infer_s=infer_s,
            server_wait_s=wait_s,
            server_total_s=time.time() - start,
        )

    def _dissolve_group(self, cut_layer: int) -> None:
        """Split a group's merged cache back into per-session caches."""

        group = self.batch_groups.pop(cut_layer, None)
        if group is None:
            return
        for session in group.session_order:
            if session.past_key_values is None:
                # Delegated to the group's merged cache while grouped; needs a real object
                # to split into.
                session.past_key_values = DynamicCache(config=self.runner.config)
        past_lengths = [group.current_len - pad for pad in group.pad_offsets]
        split_decode_cache(
            group.merged_cache, group.session_order, past_lengths, group.current_len, cut_layer, self.runner.config
        )

    def _run_decode_group(self, cut_layer: int, items: list[_PendingDecode]) -> list[torch.Tensor]:
        """Run one forward pass for every request in ``items``, all at ``cut_layer``.

        Returns:
            One next-token tensor per request, in the order of ``items``.
        """

        original_order = [item.session.session_id for item in items]
        sessions = [item.session for item in items]

        existing = self.batch_groups.get(cut_layer)
        if existing is not None and len(sessions) > 1 and set(_session_key(sessions)) == set(_session_key(existing.session_order)):
            order_index = {s.session_id: i for i, s in enumerate(existing.session_order)}
            ordered_items = sorted(items, key=lambda it: order_index[it.session.session_id])
            ordered_sessions = existing.session_order

            hidden = torch.cat([item.hidden for item in ordered_items], dim=0)
            # Left padding is fixed for the group's lifetime, but still has to be masked
            # out on every step.
            attention_mask = torch.zeros((len(ordered_sessions), existing.current_len + 1), device=self.device, dtype=torch.long)
            position_ids = torch.empty((len(ordered_sessions), 1), device=self.device, dtype=torch.long)
            for row, pad in enumerate(existing.pad_offsets):
                attention_mask[row, pad:] = 1
                position_ids[row, 0] = existing.current_len - pad
            logits = self.runner.forward(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=existing.merged_cache,
                cut_layer=cut_layer,
            )
            existing.current_len += 1

            # Return results in the caller's arrival order, not the group's row order.
            next_tokens_by_id = {}
            for row, session in enumerate(ordered_sessions):
                next_tokens_by_id[session.session_id] = greedy_next_token(logits[row : row + 1, -1, :])
                session.current_length += 1
            return [next_tokens_by_id[sid] for sid in original_order]

        # No group matches this membership: dissolve the old one first, so its members get
        # their own caches back.
        if existing is not None:
            self._dissolve_group(cut_layer)

        hidden = torch.cat([item.hidden for item in items], dim=0)

        if len(sessions) == 1:
            session = sessions[0]
            attention_mask = torch.ones((1, session.current_length), device=self.device, dtype=torch.long)
            position_ids = torch.tensor([[session.current_length - 1]], device=self.device, dtype=torch.long)
            logits = self.runner.forward(
                inputs_embeds=hidden, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=session.past_key_values, cut_layer=cut_layer,
            )
            next_tokens = [
                greedy_next_token(logits[:, -1, :])
            ]
            session.current_length += 1
            return next_tokens

        past_key_values, attention_mask, position_ids, past_lengths, max_past = merge_decode_caches(
            sessions, cut_layer, self.runner.config
        )
        logits = self.runner.forward(
            inputs_embeds=hidden, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, cut_layer=cut_layer,
        )
        next_tokens = [
            greedy_next_token(logits[row : row + 1, -1, :])
            for row, session in enumerate(sessions)
        ]
        for session in sessions:
            session.current_length += 1

        # Persist the group for later steps with the same membership. Members' own caches
        # are delegated (None) so nothing reads stale state.
        for session in sessions:
            session.past_key_values = None
        self.batch_groups[cut_layer] = BatchGroup(
            cut_layer=cut_layer,
            session_order=sessions,
            merged_cache=past_key_values,
            current_len=max_past + 1,
            pad_offsets=[max_past - pl for pl in past_lengths],
        )
        return next_tokens

    def _dissolve_range_group(self, key: tuple) -> None:
        with _prof("dissolve_range_group"):
            return self._dissolve_range_group_inner(key)

    def _dissolve_range_group_inner(self, key: tuple) -> None:
        """Inverse of persisting a RangeBatchGroup: write its merged_cache's
        final state back into each member's own past_key_values, scoped to
        exactly group.layer_range (never the whole object - see
        RangeBatchGroup's docstring for why that scoping matters here).
        Caller must hold gpu_lock."""
        group = self.progressive_groups.pop(key, None)
        if group is None:
            return
        past_lengths = [group.current_len - pad for pad in group.pad_offsets]
        split_decode_cache_range(
            group.merged_cache, group.session_order, past_lengths, group.current_len, group.layer_range, self.runner.config
        )

    def _run_range_group(
        self,
        key: tuple,
        layer_range: tuple[int, int],
        items: list[_PendingDecode],
        final: bool,
    ) -> torch.Tensor:
        """Run one forward pass over ``layer_range`` for the requests in ``items``.

        Returns:
            Logits when ``final`` is set, otherwise the segment's hidden states.
        """

        sessions = [item.session for item in items]
        existing = self.progressive_groups.get(key)
        if existing is not None and len(sessions) > 1 and existing.layer_range == layer_range and \
                set(_session_key(sessions)) == set(_session_key(existing.session_order)):
            with _prof("gather_reorder"):
                order_index = {s.session_id: i for i, s in enumerate(existing.session_order)}
                ordered_items = sorted(items, key=lambda it: order_index[it.session.session_id])
                hidden = torch.cat([it.hidden for it in ordered_items], dim=0)
            with _prof("build_mask_posids"):
                attention_mask = torch.zeros((len(existing.session_order), existing.current_len + 1),
                                              device=self.device, dtype=torch.long)
                position_ids = torch.empty((len(existing.session_order), 1), device=self.device, dtype=torch.long)
                for row, pad in enumerate(existing.pad_offsets):
                    attention_mask[row, pad:] = 1
                    position_ids[row, 0] = existing.current_len - pad
            with _prof("forward_launch"):
                out = self.runner.forward_range(hidden, attention_mask, position_ids, existing.merged_cache, layer_range, final)
            existing.current_len += 1
            with _prof("scatter_results"):
                out_by_id = {s.session_id: out[row : row + 1] for row, s in enumerate(existing.session_order)}
                res = torch.cat([out_by_id[s.session_id] for s in sessions], dim=0)
            return res

        if existing is not None:
            if _PROF:
                if existing.layer_range != layer_range:
                    _prof_calls["_miss_range_changed"] += 1
                elif set(_session_key(sessions)).issubset(set(_session_key(existing.session_order))):
                    _prof_calls["_miss_membership_SUBSET"] += 1
                else:
                    _prof_calls["_miss_membership_other"] += 1
            self._dissolve_range_group(key)

        hidden = torch.cat([it.hidden for it in items], dim=0)

        if len(sessions) == 1:
            session = sessions[0]
            attention_mask = torch.ones((1, session.current_length), device=self.device, dtype=torch.long)
            position_ids = torch.tensor([[session.current_length - 1]], device=self.device, dtype=torch.long)
            return self.runner.forward_range(hidden, attention_mask, position_ids, session.past_key_values, layer_range, final)

        with _prof("merge_caches_SLOW"):
            merged_cache, attention_mask, position_ids, past_lengths, max_past = merge_decode_caches_range(
                sessions, layer_range, self.runner.config
            )
        with _prof("forward_launch"):
            out = self.runner.forward_range(hidden, attention_mask, position_ids, merged_cache, layer_range, final)
        self.progressive_groups[key] = RangeBatchGroup(
            layer_range=layer_range,
            session_order=sessions,
            merged_cache=merged_cache,
            current_len=max_past + 1,
            pad_offsets=[max_past - pl for pl in past_lengths],
        )
        return out

    def _run_decode_group_progressive_single_checkpoint(self, items: list[_PendingDecode]) -> list[torch.Tensor]:
        """Run one depth-synchronized decode step for the current window.

        Returns:
            One next-token tensor per request, in the order of ``items``.
        """

        _step_t0 = time.perf_counter()
        deepest_cut = max(item.session.cut_layer for item in items)
        tail_range = (deepest_cut, self.runner.num_hidden_layers)

        # Reconcile the tail group's boundary before prefix processing reads any session
        # cache: when the checkpoint regresses, the current data for the reclaimed range
        # lives in the tail group. The reverse order causes a tensor-size mismatch under
        # real jitter.
        existing_tail = self.progressive_groups.get(("tail",))
        if existing_tail is not None and existing_tail.layer_range != tail_range:
            self._dissolve_range_group(("tail",))

        prefix_groups = {}
        with _prof("orchestration"):
            for item in items:
                if item.session.cut_layer != deepest_cut:
                    prefix_groups.setdefault(item.session.cut_layer, []).append(item)

        # Dissolve prefix groups this round will not touch: an untouched group is never
        # reconciled and would desync from the session's recorded length.
        for key in [k for k in self.progressive_groups if k[0] == "prefix" and k[1] not in prefix_groups]:
            self._dissolve_range_group(key)

        aligned_hidden_by_id = {
            item.session.session_id: item.hidden for item in items if item.session.cut_layer == deepest_cut
        }

        for cut_layer, group in prefix_groups.items():
            prefix_range = (cut_layer, deepest_cut)
            out = self._run_range_group(("prefix", cut_layer), prefix_range, group, final=False)
            for row, it in enumerate(group):
                aligned_hidden_by_id[it.session.session_id] = out[row : row + 1]

        with _prof("orchestration"):
            sessions = [item.session for item in items]
            tail_items = [
                _PendingDecode(session=s, hidden=aligned_hidden_by_id[s.session_id], enqueued_at=0.0, event=None)
                for s in sessions
            ]
        logits = self._run_range_group(("tail",), tail_range, tail_items, final=True)

        with _prof("sampling_loop"):
            next_tokens = [
                greedy_next_token(logits[row : row + 1, -1, :])
                for row, s in enumerate(sessions)
            ]
        for session in sessions:
            session.current_length += 1
        if _PROF:
            _prof_time["step_total"] += time.perf_counter() - _step_t0
            _prof_calls["step_total"] += 1
            _prof_time["_groupsize_sum"] += len(items)
            _prof_calls["_groupsize_n"] += 1
            if _prof_calls["step_total"] % 50 == 0:
                dump_profile(f"(after {_prof_calls['step_total']} progressive-sc steps)")
        return next_tokens

    def _batch_worker(self) -> None:
        """Background thread: drains decode_queue, waiting up to
        batch_window_s for more requests to arrive before dispatching, then
        groups whatever it collected by cut_layer (only same-cut sessions
        can share one forward call) and runs each group under gpu_lock."""
        while True:
            batch = [self.decode_queue.get()]
            deadline = time.time() + self.batch_window_s
            while len(batch) < MAX_DECODE_BATCH:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.decode_queue.get(timeout=remaining))
                except queue.Empty:
                    break

            if self.progressive_batching:
                # One merged pass over the whole window: the depth-synchronized path
                # groups sessions into depth segments itself.
                if len(batch) > 1:
                    cuts = sorted(it.session.cut_layer for it in batch)
                    print(f"progressive batch: size={len(batch)} cut_layers={cuts}", flush=True)
                wait_s_by_item = {id(item): time.time() - item.enqueued_at for item in batch}
                try:
                    with self.gpu_lock:
                        infer_start = time.time()
                        with torch.no_grad():
                            next_tokens = self._run_decode_group_progressive_single_checkpoint(batch)
                        infer_s = time.time() - infer_start
                    for item, next_token in zip(batch, next_tokens):
                        item.result = {"next_token": next_token, "infer_s": infer_s, "wait_s": wait_s_by_item[id(item)]}
                        item.event.set()
                except Exception as exc:
                    for item in batch:
                        item.result = {"error": exc}
                        item.event.set()
                continue

            groups = {}
            for item in batch:
                groups.setdefault(item.session.cut_layer, []).append(item)

            for cut_layer, items in groups.items():
                if len(items) > 1:
                    print(f"decode batch: cut_layer={cut_layer} size={len(items)}", flush=True)
                wait_s_by_item = {id(item): time.time() - item.enqueued_at for item in items}
                try:
                    with self.gpu_lock:
                        infer_start = time.time()
                        with torch.no_grad():
                            next_tokens = self._run_decode_group(cut_layer, items)
                        infer_s = time.time() - infer_start
                    for item, next_token in zip(items, next_tokens):
                        item.result = {"next_token": next_token, "infer_s": infer_s, "wait_s": wait_s_by_item[id(item)]}
                        item.event.set()
                except Exception as exc:  # surface to waiting RPC threads instead of hanging them forever
                    for item in items:
                        item.result = {"error": exc}
                        item.event.set()

    def _interleave_admit(self, session_id: str, pending: _PendingDecode) -> None:
        """Register a pending decode request for round-robin service."""

        with self.interleave_lock:
            already_known = session_id in self.interleave_order or session_id in self.interleave_overflow
            if not already_known:
                if len(self.interleave_order) < self.interleave_group_size:
                    self.interleave_order.append(session_id)
                else:
                    self.interleave_overflow.append(session_id)
            self.interleave_mailbox[session_id] = pending

    def _interleave_remove(self, session_id: str) -> None:
        """Remove a closed session and promote the next waiting one."""

        with self.interleave_lock:
            self.interleave_mailbox.pop(session_id, None)
            if session_id in self.interleave_order:
                self.interleave_order.remove(session_id)
                if self.interleave_overflow:
                    self.interleave_order.append(self.interleave_overflow.pop(0))
            elif session_id in self.interleave_overflow:
                self.interleave_overflow.remove(session_id)

    def _interleave_worker(self) -> None:
        """Serve admitted sessions in rotation, one session per forward call."""

        cursor = 0
        while True:
            with self.interleave_lock:
                order = list(self.interleave_order)
            if not order:
                time.sleep(0.001)
                continue

            serviced = None
            for offset in range(len(order)):
                idx = (cursor + offset) % len(order)
                sid = order[idx]
                with self.interleave_lock:
                    pending = self.interleave_mailbox.get(sid)
                if pending is not None:
                    serviced = (idx, sid, pending)
                    break

            if serviced is None:
                time.sleep(0.001)  # nobody in the group is ready - "cloud waits for one slot"
                continue

            idx, sid, pending = serviced
            with self.interleave_lock:
                self.interleave_mailbox.pop(sid, None)
            cursor = (idx + 1) % len(order)

            wait_s = time.time() - pending.enqueued_at
            try:
                with self.gpu_lock:
                    infer_start = time.time()
                    with torch.no_grad():
                        next_tokens = self._run_decode_group(pending.session.cut_layer, [pending])
                    infer_s = time.time() - infer_start
                pending.result = {"next_token": next_tokens[0], "infer_s": infer_s, "wait_s": wait_s}
            except Exception as exc:  # surface to the waiting RPC thread instead of hanging it forever
                pending.result = {"error": exc}
            pending.event.set()

    def DecodeStep(self, request: Any, context: Any) -> Any:
        start = time.time()
        session = self._get_session(request.session_id, context)
        if session is None:
            return llama_multi_pb2.DecodeResponse()

        hidden = deserialize_tensor(request.hidden, self.device).to(self.dtype)

        if self.interleave_group_size > 0:
            # Round-robin interleaving path: register this request and wait
            # for _interleave_worker to service it in rotation order.
            pending = _PendingDecode(session=session, hidden=hidden, enqueued_at=time.time(), event=threading.Event())
            self._interleave_admit(request.session_id, pending)
            if not pending.event.wait(timeout=30.0):
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details("interleave worker did not respond in time")
                return llama_multi_pb2.DecodeResponse()
            if "error" in pending.result:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(pending.result["error"]))
                return llama_multi_pb2.DecodeResponse()
            return llama_multi_pb2.DecodeResponse(
                next_token=serialize_tensor(pending.result["next_token"]),
                server_infer_s=pending.result["infer_s"],
                server_wait_s=pending.result["wait_s"],
                server_total_s=time.time() - start,
            )

        if self.batch_window_s <= 0:
            # Unbatched path: one session, one forward call, immediately under gpu_lock.
            attention_mask = torch.ones((1, session.current_length), device=self.device, dtype=torch.long)
            position_ids = torch.tensor(
                [[session.current_length - 1]], device=self.device, dtype=torch.long
            )

            wait_start = time.time()
            with self.gpu_lock:
                wait_s = time.time() - wait_start
                infer_start = time.time()
                with torch.no_grad():
                    logits = self.runner.forward(
                        inputs_embeds=hidden,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=session.past_key_values,
                        cut_layer=session.cut_layer,
                    )
                next_token = greedy_next_token(logits[:, -1, :])
                infer_s = time.time() - infer_start

            session.current_length += 1

            return llama_multi_pb2.DecodeResponse(
                next_token=serialize_tensor(next_token),
                server_infer_s=infer_s,
                server_wait_s=wait_s,
                server_total_s=time.time() - start,
            )

        # Batched path: queue behind (and possibly alongside) other sessions
        # that share a cut layer, and wait for the batch worker to run us.
        pending = _PendingDecode(session=session, hidden=hidden, enqueued_at=time.time(), event=threading.Event())
        self.decode_queue.put(pending)
        if not pending.event.wait(timeout=30.0):
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("decode batch worker did not respond in time")
            return llama_multi_pb2.DecodeResponse()
        if "error" in pending.result:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(pending.result["error"]))
            return llama_multi_pb2.DecodeResponse()

        return llama_multi_pb2.DecodeResponse(
            next_token=serialize_tensor(pending.result["next_token"]),
            server_infer_s=pending.result["infer_s"],
            server_wait_s=pending.result["wait_s"],
            server_total_s=time.time() - start,
        )

    def CloseSession(self, request: Any, context: Any) -> Any:
        with self.sessions_lock:
            session = self.sessions.pop(request.session_id, None)
        if session is not None:
            # Dissolve the whole group first, or its other members are left without an
            # authoritative cache.
            with self.gpu_lock:
                for cut_layer, group in list(self.batch_groups.items()):
                    if session in group.session_order:
                        self._dissolve_group(cut_layer)
                        break
                # A session can belong to two range groups at once, its prefix group and
                # the tail group, so check every key.
                for key, group in list(self.progressive_groups.items()):
                    if session in group.session_order:
                        self._dissolve_range_group(key)
                if self.interleave_group_size > 0:
                    self._interleave_remove(request.session_id)
        return llama_multi_pb2.CloseResponse(status=200)

    def GetMemoryStats(self, request: Any, context: Any) -> Any:
        with self.sessions_lock:
            sessions = list(self.sessions.values())

        # A member's own cache is None while grouped, so its bytes are counted through the
        # group instead.
        cache_bytes = sum(cache_bytes_for(s.past_key_values) for s in sessions)
        cache_bytes += sum(cache_bytes_for(g.merged_cache) for g in self.batch_groups.values())
        # Range groups never null out session caches, so this slightly over-counts stale
        # data for a group-owned range.
        cache_bytes += sum(cache_bytes_for(g.merged_cache) for g in self.progressive_groups.values())

        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        else:
            allocated = reserved = 0

        return llama_multi_pb2.MemoryStatsResponse(
            weight_bytes=self.weight_bytes,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            cache_bytes_exact=cache_bytes,
            active_sessions=len(sessions),
        )


_atexit.register(lambda: dump_profile("(atexit)"))


def _prof_sig_handler(signum: int, frame: Any) -> None:
    dump_profile(f"(signal {signum})")
    os._exit(0)


if _PROF:
    _signal.signal(_signal.SIGTERM, _prof_sig_handler)
    _signal.signal(_signal.SIGINT, _prof_sig_handler)


def serve(
    model_id: str,
    host: str,
    port: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_window_s: float = 0.0,
    interleave_group_size: int = 0,
    progressive_batching: bool = False,
) -> None:
    """Start the gRPC server and serve until terminated."""

    options = [
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=options)
    llama_multi_pb2_grpc.add_ShardServiceServicer_to_server(
        ShardServicer(model_id, device, dtype, batch_window_s=batch_window_s,
                      interleave_group_size=interleave_group_size,
                      progressive_batching=progressive_batching),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"shard02_multi listening on {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--host", default="[::]")
    parser.add_argument("--port", type=int, default=5088)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-window-ms", type=float, default=8.0,
                        help="Decode batching window in ms; 0 disables batching (FIFO).")
    parser.add_argument("--interleave-group-size", type=int, default=0,
                        help="Round-robin interleaving over this many sessions; 0 disables it. "
                             "Overrides --batch-window-ms.")
    parser.add_argument("--progressive-batching", action="store_true",
                        help="Depth-synchronized batching instead of exact-cut matching. "
                             "Needs --batch-window-ms > 0.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    serve(
        args.model_id, args.host, args.port, device, parse_torch_dtype(args.dtype),
        batch_window_s=args.batch_window_ms / 1000.0,
        interleave_group_size=args.interleave_group_size,
        progressive_batching=args.progressive_batching,
    )
