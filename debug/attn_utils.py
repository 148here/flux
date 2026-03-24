from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from flux.math import apply_rope

from debug import config


def image_token_grid(height: int, width: int) -> tuple[int, int]:
    return math.ceil(height / 16), math.ceil(width / 16)


def compute_image_query_text_key_attention(
    q: Tensor,
    k: Tensor,
    pe: Tensor,
    txt_token_count: int,
    head_mode: str,
    query_chunk_size: int = config.ATTN_QUERY_CHUNK_SIZE,
) -> Tensor:
    if q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("The debug WebUI only supports single-sample inference.")

    q_rope, k_rope = apply_rope(q, k, pe)
    q_rope = q_rope.float()
    k_rope = k_rope.float()

    img_q = q_rope[:, :, txt_token_count:, :]
    scale = q_rope.shape[-1] ** -0.5

    chunks: list[Tensor] = []
    for start in range(0, img_q.shape[2], query_chunk_size):
        end = min(start + query_chunk_size, img_q.shape[2])
        scores = torch.matmul(img_q[:, :, start:end, :], k_rope.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)[..., :txt_token_count]
        chunks.append(probs.cpu())

    img_to_text = torch.cat(chunks, dim=2)[0]
    if head_mode == "mean":
        return img_to_text.mean(dim=0).to(config.TENSOR_DTYPE)
    if head_mode == "all":
        return img_to_text.to(config.TENSOR_DTYPE)
    raise ValueError(f"Unsupported head mode: {head_mode}")


def select_head(attn: Tensor, head_mode: str, head_index: int | None) -> Tensor:
    if head_mode == "mean":
        return attn
    if head_index is None:
        raise ValueError("head_index is required when HEAD_MODE='all'.")
    if head_index < 0 or head_index >= attn.shape[0]:
        raise IndexError(f"Head index {head_index} out of range for {attn.shape[0]} heads.")
    return attn[head_index]


def aggregate_text_tokens(attn_2d: Tensor, token_indices: list[int]) -> Tensor:
    if attn_2d.ndim != 2:
        raise ValueError(f"Expected [img_tokens, text_tokens], got shape {tuple(attn_2d.shape)}")
    if not token_indices:
        raise ValueError("Select at least one T5 token.")
    unique = sorted(set(token_indices))
    return attn_2d[:, unique].sum(dim=-1)


def reshape_image_token_map(token_map: Tensor, grid_height: int, grid_width: int) -> Tensor:
    expected = grid_height * grid_width
    if token_map.numel() != expected:
        raise ValueError(f"Expected {expected} image tokens, got {token_map.numel()}.")
    return token_map.reshape(grid_height, grid_width)


def normalize_map(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    if array.size == 0:
        return array
    min_v = float(array.min())
    max_v = float(array.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_v) / (max_v - min_v)

