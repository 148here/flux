from __future__ import annotations

import os
from pathlib import Path

import torch


def _parse_int_list(env_name: str, default: list[int]) -> list[int]:
    raw = os.getenv(env_name)
    if not raw:
        return default
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or default


def _parse_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.getenv("FLUX_DEBUG_OUTPUT_ROOT", PACKAGE_ROOT / "runs"))

DEBUG_LAYERS = _parse_int_list("FLUX_DEBUG_LAYERS", [0])
DEBUG_TIMESTEPS = _parse_int_list("FLUX_DEBUG_TIMESTEPS", [0])
HEAD_MODE = os.getenv("FLUX_DEBUG_HEAD_MODE", "mean").strip().lower()
SAVE_RUNS = _parse_bool("FLUX_DEBUG_SAVE_RUNS", True)

DEFAULT_MODEL_MODE = os.getenv("FLUX_DEBUG_DEFAULT_MODE", "text-to-image")
T2I_MODEL_NAME = "flux-dev"
FILL_MODEL_NAME = "flux-dev-fill"

DEFAULT_DEVICE = os.getenv("FLUX_DEBUG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_OFFLOAD = _parse_bool("FLUX_DEBUG_OFFLOAD", False)

DEFAULT_PROMPT = (
    'a photo of a forest with mist swirling around the tree trunks. The word "FLUX" '
    "is painted over it in big, red brush strokes with visible texture"
)
DEFAULT_WIDTH = 1360
DEFAULT_HEIGHT = 768
DEFAULT_NUM_STEPS = 50
DEFAULT_T2I_GUIDANCE = 2.5
DEFAULT_FILL_GUIDANCE = 30.0
DEFAULT_SEED = -1

DEFAULT_OUTPAINT_LEFT = 256
DEFAULT_OUTPAINT_RIGHT = 256
DEFAULT_OUTPAINT_TOP = 0
DEFAULT_OUTPAINT_BOTTOM = 0
DEFAULT_OUTPAINT_OVERLAP = 32

VIEW_ALPHA = 0.45
VIEW_IMAGE_FORMAT = "png"
TENSOR_DTYPE = torch.float16
ATTN_QUERY_CHUNK_SIZE = 256

LOG_LEVEL = os.getenv("FLUX_DEBUG_LOG_LEVEL", "INFO")

if HEAD_MODE not in {"mean", "all"}:
    raise ValueError(f"HEAD_MODE must be 'mean' or 'all', got: {HEAD_MODE}")

