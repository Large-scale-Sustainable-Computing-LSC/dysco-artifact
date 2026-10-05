from io import BytesIO
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llama_ready_pb2 as pb2
import llama_ready_pb2_grpc as pb2_grpc


def safe_torch_load(buffer, map_location):
    try:
        return torch.load(buffer, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(buffer, map_location=map_location)


def serialize_tensor(tensor):
    tensor = tensor.detach().cpu()
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return pb2.TensorData(
        tensor_bytes=buffer.getvalue(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
    )


def deserialize_tensor(tensor_data, device):
    tensor = safe_torch_load(BytesIO(tensor_data.tensor_bytes), map_location="cpu")
    return tensor.to(device)


def parse_torch_dtype(dtype_name, default=torch.bfloat16):
    if not dtype_name:
        return default

    normalized = dtype_name.replace("torch.", "")
    if not hasattr(torch, normalized):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, normalized)


def sample_next_token(logits, do_sample, temperature, top_p, top_k):
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")

    scores = logits / temperature

    if top_k > 0:
        top_k = min(top_k, scores.shape[-1])
        values, _ = torch.topk(scores, top_k)
        cutoff = values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(sorted_mask, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)
