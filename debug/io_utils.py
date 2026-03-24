from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from debug import config


@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    inputs_dir: Path
    outputs_dir: Path
    tokens_dir: Path
    attn_dir: Path
    views_dir: Path


def make_run_paths(output_root: Path, save_runs: bool = True) -> RunPaths:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    base_root = output_root if save_runs else output_root / "_tmp"
    run_dir = base_root / run_id
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    tokens_dir = run_dir / "tokens"
    attn_dir = run_dir / "attn"
    views_dir = run_dir / "views"
    for path in [inputs_dir, outputs_dir, tokens_dir, attn_dir, views_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id, run_dir, inputs_dir, outputs_dir, tokens_dir, attn_dir, views_dir)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_image(path: Path, image: Image.Image) -> None:
    image.save(path, format=config.VIEW_IMAGE_FORMAT.upper())


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor, path)


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)


def _timestep_tag(timestep_value: float) -> str:
    return f"{timestep_value:.6f}".replace(".", "p").replace("-", "m")


def capture_dir(attn_root: Path, layer_index: int, step_index: int, timestep_value: float) -> Path:
    path = attn_root / f"layer_{layer_index:02d}" / f"step_{step_index:03d}_t_{_timestep_tag(timestep_value)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_capture_tensor(
    run_paths: RunPaths,
    layer_index: int,
    step_index: int,
    timestep_value: float,
    tensor: torch.Tensor,
) -> str:
    target = capture_dir(run_paths.attn_dir, layer_index, step_index, timestep_value) / "raw.pt"
    save_tensor(target, tensor)
    return str(target)


def token_selection_slug(token_indices: list[int]) -> str:
    unique = sorted(set(token_indices))
    raw = "_".join(str(index) for index in unique)
    if len(raw) <= 80:
        return f"tokens_{raw}" if raw else "tokens_none"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"tokens_{len(unique)}_{digest}"


def save_view_images(
    run_paths: RunPaths,
    layer_index: int,
    step_index: int,
    timestep_value: float,
    head_label: str,
    token_indices: list[int],
    heatmap_image: Image.Image,
    overlay_image: Image.Image,
) -> tuple[str, str]:
    target_dir = (
        run_paths.views_dir
        / f"layer_{layer_index:02d}"
        / f"step_{step_index:03d}_t_{_timestep_tag(timestep_value)}"
        / head_label
        / token_selection_slug(token_indices)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = target_dir / "heatmap.png"
    overlay_path = target_dir / "overlay.png"
    save_image(heatmap_path, heatmap_image)
    save_image(overlay_path, overlay_image)
    return str(heatmap_path), str(overlay_path)
