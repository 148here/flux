from __future__ import annotations

import numpy as np
from PIL import Image
from torch import Tensor

from debug import config
from debug.attn_utils import aggregate_text_tokens, normalize_map, reshape_image_token_map, select_head


def _colorize_heatmap(norm: np.ndarray) -> np.ndarray:
    r = np.clip(1.7 * norm, 0.0, 1.0)
    g = np.clip(1.7 * norm - 0.45, 0.0, 1.0)
    b = np.clip(1.0 - 1.35 * norm, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def make_heatmap_image(attn_grid: Tensor, image_size: tuple[int, int]) -> Image.Image:
    norm = normalize_map(attn_grid.detach().cpu().numpy())
    heat_rgb = _colorize_heatmap(norm)
    heatmap = Image.fromarray(heat_rgb, mode="RGB")
    return heatmap.resize(image_size, Image.Resampling.BICUBIC)


def make_overlay_image(base_image: Image.Image, heatmap_image: Image.Image, alpha: float = config.VIEW_ALPHA) -> Image.Image:
    return Image.blend(base_image.convert("RGB"), heatmap_image.convert("RGB"), alpha=alpha)


def render_attention_images(
    raw_attention: Tensor,
    head_mode: str,
    head_index: int | None,
    token_indices: list[int],
    token_grid_height: int,
    token_grid_width: int,
    image_size: tuple[int, int],
    base_image: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    attn_2d = select_head(raw_attention, head_mode=head_mode, head_index=head_index)
    token_map = aggregate_text_tokens(attn_2d, token_indices)
    attn_grid = reshape_image_token_map(token_map, token_grid_height, token_grid_width)
    heatmap = make_heatmap_image(attn_grid, image_size=image_size)
    overlay = make_overlay_image(base_image, heatmap)
    return heatmap, overlay
