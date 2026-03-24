# FLUX Debug / Visualization WebUI

This directory adds a minimal-intrusion debug pipeline for the official `black-forest-labs/flux` repository.

## What it supports

- `FLUX.1-dev` text-to-image
- `FLUX.1-Fill-dev` fill-inpaint
- `FLUX.1-Fill-dev` fill-outpaint
- Raw cross-attention visualization for `image-query × text-key`
- Double-stream blocks only
- T5 subtoken display and multi-token aggregation
- Run artifact saving under `debug/runs/<run_id>/`

## What it does not support in v1

- Single-stream attention visualization
- CLIP token-level visualization
- Reverse `text-query × image-key` view
- Batch processing
- TensorRT-based attention debugging

## Install

Follow the official editable install, then ensure Gradio is available:

```bash
pip install -e ".[torch,gradio]"
```

## Start

```bash
python debug/app.py
```

To force fully local text encoders and avoid first-run downloads, you can also set:

```bash
export FLUX_T5_PATH=/path/to/google_t5_v1_1_xxl
export FLUX_CLIP_PATH=/path/to/clip_vit_large_patch14
```

These paths should point to local Hugging Face model directories containing the tokenizer and model files.

If `FLUX_T5_PATH` is not set and your torch version is older than 2.6, the loader automatically uses the `google/t5-v1_1-xxl` safetensors revision `refs/pr/2` to avoid the upstream `.bin` loading restriction.

## Config

Edit `debug/config.py`.

Important options:

- `DEBUG_LAYERS`: 0-based indices into `model.double_blocks`
- `DEBUG_TIMESTEPS`: 0-based denoise step indices
- `HEAD_MODE`: `"mean"` or `"all"`
- `SAVE_RUNS`: persistent save toggle
- `OUTPUT_ROOT`: run artifact root directory
- `DEFAULT_MODEL_MODE`: default UI mode

When `HEAD_MODE="mean"`, the backend saves head-averaged `A[q_img, k_text]`.
When `HEAD_MODE="all"`, the backend saves one tensor per head dimension and the UI exposes head selection.

## Usage

### Text-to-image

1. Choose `text-to-image`.
2. Enter a prompt.
3. Set width, height, steps, guidance, and seed.
4. Click `Generate`.
5. Select T5 subtokens, layer, timestep, and optional head, then click `Render Selection`.

### Fill inpaint

1. Choose `fill-inpaint`.
2. Enter a local image path.
3. Click `Prepare Fill Canvas`.
4. Draw a binary mask in the editor.
5. Click `Generate`.
6. Use the attention controls to render `image-query × text-key` views.

### Fill outpaint

1. Choose `fill-outpaint`.
2. Enter a local image path.
3. Set expansion pixels and overlap.
4. Click `Prepare Fill Canvas`.
5. Optionally add or refine the generated border mask in the editor.
6. Click `Generate`.
7. Use the attention controls to inspect saved results.

## Run artifacts

Each run saves to:

```text
debug/runs/<run_id>/
  metadata.json
  prompt.txt
  tokens/t5_tokens.json
  inputs/
  outputs/generated.png
  attn/layer_*/step_*_t_*/raw.pt
  views/.../heatmap.png
  views/.../overlay.png
```

## Notes

- T5 tokens are taken from the exact tokenizer used by the official model loader.
- CLIP is retained as a global conditioning vector only; there is no CLIP token list in the UI.
- Attention slicing uses the real joint attention softmax over the full key axis before taking the `image-query × text-key` submatrix.
