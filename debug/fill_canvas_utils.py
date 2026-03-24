from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def round_up_to_multiple(value: int, multiple: int = 32) -> int:
    return multiple * ((value + multiple - 1) // multiple)


def load_local_image(path_str: str) -> Image.Image:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension for: {path}")
    return Image.open(path).convert("RGB")


def validate_fill_image_size(image: Image.Image) -> None:
    if image.width % 32 != 0 or image.height % 32 != 0:
        raise ValueError(
            f"FLUX Fill expects input dimensions divisible by 32, got {image.width}x{image.height}."
        )


def build_inpaint_assets(source_image: Image.Image) -> tuple[Image.Image, Image.Image]:
    validate_fill_image_size(source_image)
    return source_image.copy(), Image.new("L", source_image.size, 0)


def build_outpaint_assets(
    source_image: Image.Image,
    expand_left: int,
    expand_right: int,
    expand_top: int,
    expand_bottom: int,
    overlap: int,
) -> tuple[Image.Image, Image.Image]:
    expand_left = max(0, int(expand_left))
    expand_right = max(0, int(expand_right))
    expand_top = max(0, int(expand_top))
    expand_bottom = max(0, int(expand_bottom))
    overlap = max(0, int(overlap))

    raw_width = source_image.width + expand_left + expand_right
    raw_height = source_image.height + expand_top + expand_bottom
    canvas_width = round_up_to_multiple(raw_width, 32)
    canvas_height = round_up_to_multiple(raw_height, 32)

    canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    paste_x = expand_left
    paste_y = expand_top
    canvas.paste(source_image, (paste_x, paste_y))

    base_mask = Image.new("L", canvas.size, 255)
    overlap = min(overlap, source_image.width // 2, source_image.height // 2)
    keep_region = (
        paste_x + overlap,
        paste_y + overlap,
        paste_x + source_image.width - overlap,
        paste_y + source_image.height - overlap,
    )
    base_mask.paste(0, keep_region)
    return canvas, base_mask


def _as_image(value: Any, mode: str, size: tuple[int, int] | None = None) -> Image.Image:
    if value is None:
        image = Image.new(mode, size or (1, 1))
    elif isinstance(value, Image.Image):
        image = value.convert(mode)
    elif isinstance(value, np.ndarray):
        image = Image.fromarray(value).convert(mode)
    elif isinstance(value, str):
        image = Image.open(value).convert(mode)
    else:
        raise TypeError(f"Unsupported image editor payload type: {type(value)!r}")

    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return image


def extract_drawn_mask(editor_value: Any, size: tuple[int, int]) -> Image.Image:
    if editor_value is None:
        return Image.new("L", size, 0)

    layers: list[Any] = []
    if isinstance(editor_value, dict):
        layers = editor_value.get("layers") or []
    elif isinstance(editor_value, Image.Image):
        layers = [editor_value]

    mask = Image.new("L", size, 0)
    for layer in layers:
        rgba = _as_image(layer, "RGBA", size=size)
        mask = ImageChops.lighter(mask, rgba.getchannel("A"))
    return binarize_mask(mask)


def merge_masks(base_mask: Image.Image, drawn_mask: Image.Image) -> Image.Image:
    return binarize_mask(ImageChops.lighter(base_mask.convert("L"), drawn_mask.convert("L")))


def binarize_mask(mask: Image.Image, threshold: int = 1) -> Image.Image:
    array = np.array(mask.convert("L"))
    binary = np.where(array >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")
