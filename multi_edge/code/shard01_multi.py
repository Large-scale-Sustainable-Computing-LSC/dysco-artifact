"""Edge-side client for the heterogeneous-cut prototype.

Loads only the first --cut-layer decoder layers, runs prefill and decode locally
up to that layer, and sends the hidden state to shard02_multi.py over gRPC.
--worker keeps the process resident for a stream of sessions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any, Optional
import argparse
import json
import sys
import time

import grpc
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

import llama_multi_pb2
import llama_multi_pb2_grpc

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class StepResult:
    """One PrefillStep or DecodeStep response, with its server-side timings."""

    next_token: torch.Tensor
    server_infer_s: float
    server_wait_s: float
    server_total_s: float
    rpc_s: float


@dataclass
class SessionResult:
    """Everything one finished session reports back to a benchmark harness."""

    session_id: str
    cut_layer: int
    prompt: str
    output_text: str
    generated_ids: list[int]
    avg_gpu_queue_wait_ms: float
    total_s: float
    gen_start_ts: float
    gen_end_ts: float


def parse_torch_dtype(name: str) -> torch.dtype:
    """Return the torch dtype for a command-line dtype name."""

    return DTYPES[name]


def serialize_tensor(tensor: torch.Tensor) -> Any:
    """Pack a tensor into the wire format used by both shards."""

    tensor = tensor.detach().cpu()
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return llama_multi_pb2.TensorData(
        tensor_bytes=buffer.getvalue(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
    )


def deserialize_tensor(tensor_data: Any, device: torch.device | str) -> torch.Tensor:
    """Unpack a wire-format tensor onto ``device``."""

    tensor = torch.load(BytesIO(tensor_data.tensor_bytes), map_location="cpu", weights_only=True)
    return tensor.to(device)


class ShardClient:
    """gRPC connection for one session's prefill and decode steps."""

    def __init__(self, host: str, port: int, session_id: str) -> None:
        options = [
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
        self.stub = llama_multi_pb2_grpc.ShardServiceStub(self.channel)
        self.session_id = session_id

    def init_connection(self, model_id: str, layer_split: int, dtype: torch.dtype) -> Any:
        """Register this session and its cut layer on the cloud process."""

        request = llama_multi_pb2.InitRequest(
            session_id=self.session_id,
            config=llama_multi_pb2.RuntimeConfig(
                model_id=model_id,
                layer_split=layer_split,
                torch_dtype=str(dtype),
            ),
        )
        response = self.stub.InitConnection(request)
        if response.status != 200:
            raise RuntimeError(f"InitConnection failed: {response.message}")
        return response

    def prefill(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        device: torch.device | str,
    ) -> StepResult:
        """Send the prompt's hidden states and return the first sampled token."""

        request = llama_multi_pb2.PrefillRequest(
            session_id=self.session_id,
            hidden=serialize_tensor(hidden),
            attention_mask=serialize_tensor(attention_mask),
            position_ids=serialize_tensor(position_ids),
        )
        rpc_start = time.time()
        response = self.stub.PrefillStep(request)
        return self._step_result(response, device, rpc_start)

    def decode(self, hidden: torch.Tensor, device: torch.device | str) -> StepResult:
        """Send one token's hidden state and return the next sampled token."""

        request = llama_multi_pb2.DecodeRequest(session_id=self.session_id, hidden=serialize_tensor(hidden))
        rpc_start = time.time()
        response = self.stub.DecodeStep(request)
        return self._step_result(response, device, rpc_start)

    def close(self) -> None:
        """Release this session's cloud-side state, including its cache."""

        self.stub.CloseSession(llama_multi_pb2.CloseRequest(session_id=self.session_id))

    @staticmethod
    def _step_result(response: Any, device: torch.device | str, rpc_start: float) -> StepResult:
        return StepResult(
            next_token=deserialize_tensor(response.next_token, device),
            server_infer_s=response.server_infer_s,
            server_wait_s=response.server_wait_s,
            server_total_s=response.server_total_s,
            rpc_s=time.time() - rpc_start,
        )


def load_edge_model(
    model_path: str,
    cut_layer: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, Any]:
    """Load the embedding table and layers ``[0, cut_layer)`` onto ``device``."""

    model = AutoModel.from_pretrained(model_path, dtype=dtype)
    available = len(model.layers)
    if not 1 <= cut_layer < available:
        raise ValueError(f"--cut-layer must be in [1, {available - 1}], got {cut_layer}")
    model.layers = model.layers[:cut_layer]
    model.norm = nn.Identity()
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_one_session(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | str,
    host: str,
    port: int,
    session_id: str,
    cut_layer: int,
    server_model_id: str,
    dtype: torch.dtype,
    prompt: str,
    max_new_tokens: int,
) -> SessionResult:
    """Run one session's prefill and decode loop against a loaded edge model.

    Returns:
        ``SessionResult`` with the generated text and the timings a benchmark
        harness needs, including the mean time each token spent waiting for the
        cloud GPU.
    """

    client = ShardClient(host, port, session_id)
    client.init_connection(model_id=server_model_id, layer_split=cut_layer, dtype=dtype)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    generated = input_ids.clone()

    token_wait_s: list[float] = []
    gen_start_ts = time.time()

    with torch.no_grad():
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        step = client.prefill(outputs.last_hidden_state, attention_mask, position_ids, device)

        for _ in range(max_new_tokens - 1):
            next_token = step.next_token
            token_wait_s.append(step.server_wait_s)
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            position_ids = torch.tensor([[generated.shape[1] - 1]], device=device, dtype=torch.long)

            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            step = client.decode(outputs.last_hidden_state, device)

        token_wait_s.append(step.server_wait_s)
        generated = torch.cat([generated, step.next_token], dim=-1)

    gen_end_ts = time.time()
    output_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    client.close()

    return SessionResult(
        session_id=session_id,
        cut_layer=cut_layer,
        prompt=prompt,
        output_text=output_text,
        generated_ids=generated[0].tolist(),
        avg_gpu_queue_wait_ms=sum(token_wait_s) / len(token_wait_s) * 1000,
        total_s=gen_end_ts - gen_start_ts,
        gen_start_ts=gen_start_ts,
        gen_end_ts=gen_end_ts,
    )


def run_worker_loop(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    """Serve a stream of sessions read as JSON lines from stdin until it closes."""

    reported = ("session_id", "avg_gpu_queue_wait_ms", "total_s", "gen_start_ts", "gen_end_ts")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        result = run_one_session(
            model, tokenizer, device, args.host, args.port,
            session_id=task["session_id"], cut_layer=args.cut_layer,
            server_model_id=args.server_model_id, dtype=dtype,
            prompt=task["prompt"], max_new_tokens=task.get("max_new_tokens", args.max_new_tokens),
        )
        slim = {key: getattr(result, key) for key in reported}
        sys.stdout.write(json.dumps(slim) + "\n")
        sys.stdout.flush()


def run(args: argparse.Namespace) -> tuple[Optional[str], Optional[list[int]]]:
    """Load the edge model, then either serve a worker stream or one prompt."""

    if not args.worker and not args.session_id:
        raise ValueError("--session-id is required unless --worker is given")

    device = torch.device(args.device)
    dtype = parse_torch_dtype(args.dtype)
    # Worker mode writes a JSON-lines protocol on stdout, so progress messages
    # must go to stderr or they corrupt it.
    print(f"[{args.session_id or 'worker'}] loading edge model, first {args.cut_layer} layers, "
          f"on {device} ({dtype}) ...", file=sys.stderr if args.worker else sys.stdout)
    model, tokenizer = load_edge_model(args.model_path, args.cut_layer, device, dtype)

    if args.worker:
        run_worker_loop(args, model, tokenizer, device, dtype)
        return None, None

    result = run_one_session(
        model, tokenizer, device, args.host, args.port,
        session_id=args.session_id, cut_layer=args.cut_layer, server_model_id=args.server_model_id,
        dtype=dtype, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
    )

    print(f"[{args.session_id}] cut_layer={args.cut_layer}  total={result.total_s:.2f}s  "
          f"avg_gpu_queue_wait={result.avg_gpu_queue_wait_ms:.1f}ms")
    print(f"[{args.session_id}] output: {result.output_text!r}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(asdict(result), handle)

    return result.output_text, result.generated_ids


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for one edge process."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default=None,
                        help="Required unless --worker is given; worker mode gets a session id per task.")
    parser.add_argument("--model-path", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--server-model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--cut-layer", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5088)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=list(DTYPES),
        help="Edge weight dtype; float32 is used for bit-exact checks.",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out", default=None, help="Optional path for a JSON result dump.")
    parser.add_argument("--worker", action="store_true",
                        help="Stay resident and serve sessions as JSON lines on stdin.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
