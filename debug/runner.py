from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, prepare_fill, unpack
from flux.util import embed_watermark, load_ae, load_clip, load_flow_model, load_t5

from debug import config
from debug.attn_utils import image_token_grid
from debug.fill_canvas_utils import (
    build_inpaint_assets,
    build_outpaint_assets,
    extract_drawn_mask,
    load_local_image,
    merge_masks,
)
from debug.hooks import FluxAttentionCapture
from debug.io_utils import RunPaths, load_tensor, make_run_paths, save_capture_tensor, save_image, save_json, save_text, save_view_images
from debug.tokenizer_utils import default_token_indices, parse_token_choices, token_choice, tokenize_prompt_for_display
from debug.vis_utils import render_attention_images


logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model_name: str
    model: Any
    ae: Any
    t5: Any
    clip: Any
    offload: bool
    device: torch.device


@dataclass
class InferenceRequest:
    mode: str
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    image_path: str | None = None
    outpaint_left: int = 0
    outpaint_right: int = 0
    outpaint_top: int = 0
    outpaint_bottom: int = 0
    outpaint_overlap: int = 0


class FluxDebugRunner:
    def __init__(self, device: str | None = None, offload: bool | None = None) -> None:
        self.device = torch.device(device or config.DEFAULT_DEVICE)
        self.offload = config.DEFAULT_OFFLOAD if offload is None else offload
        self._model_cache: dict[tuple[str, str, bool], ModelBundle] = {}
        self._configured_logging = False

    def _ensure_logging(self) -> None:
        if self._configured_logging:
            return
        logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        self._configured_logging = True

    def _cache_key(self, model_name: str) -> tuple[str, str, bool]:
        return model_name, str(self.device), self.offload

    def get_bundle(self, model_name: str) -> ModelBundle:
        key = self._cache_key(model_name)
        if key not in self._model_cache:
            print(f"[debug.runner] Loading model bundle: name={model_name} device={self.device} offload={self.offload}", flush=True)
            logger.info("Loading model bundle name=%s device=%s offload=%s", model_name, self.device, self.offload)
            t5_max_length = 512 if model_name == config.T2I_MODEL_NAME else 128
            bundle = ModelBundle(
                model_name=model_name,
                model=load_flow_model(model_name, device="cpu" if self.offload else self.device),
                ae=load_ae(model_name, device="cpu" if self.offload else self.device),
                t5=load_t5(self.device, max_length=t5_max_length),
                clip=load_clip(self.device),
                offload=self.offload,
                device=self.device,
            )
            self._model_cache[key] = bundle
        return self._model_cache[key]

    def _normalize_seed(self, seed_value: int | None) -> int:
        if seed_value is None or seed_value < 0:
            return torch.Generator(device="cpu").seed()
        return int(seed_value)

    def _sanitize_layers(self, model) -> list[int]:
        max_layers = len(model.double_blocks)
        layers = sorted({index for index in config.DEBUG_LAYERS if 0 <= index < max_layers})
        if not layers:
            raise ValueError(f"No valid DEBUG_LAYERS found for model with {max_layers} double-stream blocks.")
        return layers

    def _sanitize_timesteps(self, num_steps: int) -> list[int]:
        timesteps = sorted({index for index in config.DEBUG_TIMESTEPS if 0 <= index < num_steps})
        if not timesteps:
            raise ValueError(f"No valid DEBUG_TIMESTEPS found for num_steps={num_steps}.")
        return timesteps

    def prepare_fill_preview(
        self,
        mode: str,
        image_path: str,
        expand_left: int,
        expand_right: int,
        expand_top: int,
        expand_bottom: int,
        overlap: int,
    ) -> dict[str, Any]:
        source_image = load_local_image(image_path)
        if mode == "fill-inpaint":
            canvas_image, base_mask = build_inpaint_assets(source_image)
        elif mode == "fill-outpaint":
            canvas_image, base_mask = build_outpaint_assets(
                source_image=source_image,
                expand_left=expand_left,
                expand_right=expand_right,
                expand_top=expand_top,
                expand_bottom=expand_bottom,
                overlap=overlap,
            )
        else:
            raise ValueError(f"Unsupported fill mode: {mode}")
        return {
            "mode": mode,
            "source_image": source_image,
            "canvas_image": canvas_image,
            "base_mask": base_mask,
            "width": canvas_image.width,
            "height": canvas_image.height,
        }

    def _maybe_empty_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decoded_tensor_to_pil(self, decoded: torch.Tensor) -> Image.Image:
        decoded = decoded.clamp(-1, 1)
        decoded = embed_watermark(decoded.float())
        decoded = rearrange(decoded[0], "c h w -> h w c")
        array = (127.5 * (decoded + 1.0)).cpu().byte().numpy()
        return Image.fromarray(array)

    def _layer_label(self, layer_index: int) -> str:
        return f"layer {layer_index:02d}"

    def _step_label(self, step_index: int, timestep_value: float) -> str:
        return f"step {step_index:03d} | t={timestep_value:.6f}"

    def _head_choices(self, run_state: dict[str, Any]) -> list[str]:
        if run_state["head_mode"] == "mean":
            return []
        return [f"head {index:02d}" for index in range(run_state["num_heads"])]

    def _parse_head_label(self, head_label: str | None) -> int | None:
        if not head_label:
            return None
        return int(head_label.split()[1])

    def _find_entry(self, run_state: dict[str, Any], layer_label: str, step_label: str) -> dict[str, Any]:
        key = f"{layer_label}::{step_label}"
        if key not in run_state["entry_map"]:
            raise KeyError(f"Unknown layer/step selection: {key}")
        return run_state["entry_map"][key]

    def render_saved_view(
        self,
        run_state: dict[str, Any],
        layer_label: str,
        step_label: str,
        token_labels: list[str],
        head_label: str | None,
    ) -> dict[str, Any]:
        entry = self._find_entry(run_state, layer_label, step_label)
        raw_attention = load_tensor(Path(entry["raw_path"]))
        token_indices = parse_token_choices(token_labels)
        head_index = self._parse_head_label(head_label) if run_state["head_mode"] == "all" else None
        base_image = Image.open(run_state["generated_image_path"]).convert("RGB")
        heatmap, overlay = render_attention_images(
            raw_attention=raw_attention,
            head_mode=run_state["head_mode"],
            head_index=head_index,
            token_indices=token_indices,
            token_grid_height=run_state["token_grid_height"],
            token_grid_width=run_state["token_grid_width"],
            image_size=tuple(run_state["image_size"]),
            base_image=base_image,
        )
        head_dir = "mean" if run_state["head_mode"] == "mean" else f"head_{head_index:02d}"
        heatmap_path, overlay_path = save_view_images(
            run_paths=RunPaths(
                run_id=run_state["run_id"],
                run_dir=Path(run_state["run_dir"]),
                inputs_dir=Path(run_state["inputs_dir"]),
                outputs_dir=Path(run_state["outputs_dir"]),
                tokens_dir=Path(run_state["tokens_dir"]),
                attn_dir=Path(run_state["attn_dir"]),
                views_dir=Path(run_state["views_dir"]),
            ),
            layer_index=entry["layer_index"],
            step_index=entry["step_index"],
            timestep_value=entry["timestep_value"],
            head_label=head_dir,
            token_indices=token_indices,
            heatmap_image=heatmap,
            overlay_image=overlay,
        )
        logger.info(
            "Rendered run_id=%s layer=%s step=%s head=%s tokens=%s",
            run_state["run_id"],
            entry["layer_index"],
            entry["step_index"],
            head_dir,
            token_indices,
        )
        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
            "summary": (
                f"run_id={run_state['run_id']} | mode={run_state['mode']} | layer={entry['layer_index']} | "
                f"step={entry['step_index']} | head={head_dir} | tokens={token_indices}"
            ),
        }

    @torch.inference_mode()
    def run(
        self,
        request: InferenceRequest,
        fill_state: dict[str, Any] | None = None,
        editor_value: Any | None = None,
    ) -> dict[str, Any]:
        self._ensure_logging()
        print(f"[debug.runner] Starting request: mode={request.mode} steps={request.num_steps} guidance={request.guidance}", flush=True)
        model_name = config.T2I_MODEL_NAME if request.mode == "text-to-image" else config.FILL_MODEL_NAME
        bundle = self.get_bundle(model_name)
        active_layers = self._sanitize_layers(bundle.model)
        active_timesteps = self._sanitize_timesteps(request.num_steps)
        seed = self._normalize_seed(request.seed)
        run_paths = make_run_paths(config.OUTPUT_ROOT, save_runs=config.SAVE_RUNS)

        logger.info(
            "Starting run_id=%s mode=%s model=%s head_mode=%s layers=%s timesteps=%s",
            run_paths.run_id,
            request.mode,
            model_name,
            config.HEAD_MODE,
            active_layers,
            active_timesteps,
        )

        save_text(run_paths.run_dir / "prompt.txt", request.prompt)
        token_info = tokenize_prompt_for_display(bundle.t5, request.prompt)
        save_json(run_paths.tokens_dir / "t5_tokens.json", token_info)

        attention_entries: list[dict[str, Any]] = []

        def capture_callback(layer_index: int, step_index: int, timestep_value: float, raw_attention: torch.Tensor) -> None:
            raw_path = save_capture_tensor(run_paths, layer_index, step_index, timestep_value, raw_attention)
            entry = {
                "layer_index": layer_index,
                "step_index": step_index,
                "timestep_value": timestep_value,
                "raw_path": raw_path,
            }
            if raw_attention.ndim == 3:
                entry["num_heads"] = raw_attention.shape[0]
            attention_entries.append(entry)

        if request.mode == "text-to-image":
            width = 16 * (int(request.width) // 16)
            height = 16 * (int(request.height) // 16)
            x = get_noise(1, height, width, device=self.device, dtype=torch.bfloat16, seed=seed)

            if self.offload:
                bundle.ae = bundle.ae.cpu()
                self._maybe_empty_cache()
                bundle.t5, bundle.clip = bundle.t5.to(self.device), bundle.clip.to(self.device)
            print("[debug.runner] Preparing text-to-image inputs", flush=True)
            inp = prepare(bundle.t5, bundle.clip, x, prompt=request.prompt)
            timesteps = get_schedule(request.num_steps, inp["img"].shape[1], shift=True)
            if self.offload:
                bundle.t5, bundle.clip = bundle.t5.cpu(), bundle.clip.cpu()
                self._maybe_empty_cache()
                bundle.model = bundle.model.to(self.device)

            print("[debug.runner] Entering denoise loop", flush=True)
            with FluxAttentionCapture(bundle.model, active_layers, active_timesteps, config.HEAD_MODE, capture_callback):
                x = denoise(bundle.model, **inp, timesteps=timesteps, guidance=request.guidance)

            if self.offload:
                bundle.model.cpu()
                self._maybe_empty_cache()
                bundle.ae.decoder.to(x.device)

        else:
            if fill_state is None:
                raise ValueError("fill_state is required for fill-inpaint and fill-outpaint modes.")
            source_image = fill_state["source_image"]
            canvas_image = fill_state["canvas_image"]
            base_mask = fill_state["base_mask"]
            drawn_mask = extract_drawn_mask(editor_value, canvas_image.size)
            mask_image = merge_masks(base_mask, drawn_mask)

            width, height = canvas_image.size
            save_image(run_paths.inputs_dir / "source.png", source_image)
            save_image(run_paths.inputs_dir / "canvas.png", canvas_image)
            save_image(run_paths.inputs_dir / "mask.png", mask_image)

            x = get_noise(1, height, width, device=self.device, dtype=torch.bfloat16, seed=seed)

            if self.offload:
                bundle.t5, bundle.clip, bundle.ae = (
                    bundle.t5.to(self.device),
                    bundle.clip.to(self.device),
                    bundle.ae.to(self.device),
                )
            print("[debug.runner] Preparing fill inputs", flush=True)
            inp = prepare_fill(
                bundle.t5,
                bundle.clip,
                x,
                prompt=request.prompt,
                ae=bundle.ae,
                img_cond_path=str(run_paths.inputs_dir / "canvas.png"),
                mask_path=str(run_paths.inputs_dir / "mask.png"),
            )
            timesteps = get_schedule(request.num_steps, inp["img"].shape[1], shift=True)
            if self.offload:
                bundle.t5, bundle.clip, bundle.ae = bundle.t5.cpu(), bundle.clip.cpu(), bundle.ae.cpu()
                self._maybe_empty_cache()
                bundle.model = bundle.model.to(self.device)

            print("[debug.runner] Entering denoise loop", flush=True)
            with FluxAttentionCapture(bundle.model, active_layers, active_timesteps, config.HEAD_MODE, capture_callback):
                x = denoise(bundle.model, **inp, timesteps=timesteps, guidance=request.guidance)

            if self.offload:
                bundle.model.cpu()
                self._maybe_empty_cache()
                bundle.ae.decoder.to(x.device)

        print("[debug.runner] Decoding latents", flush=True)
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            decoded = bundle.ae.decode(x)

        if self.offload:
            bundle.ae.decoder.cpu()
            self._maybe_empty_cache()

        generated_image = self._decoded_tensor_to_pil(decoded)
        generated_path = run_paths.outputs_dir / "generated.png"
        save_image(generated_path, generated_image)

        if not attention_entries:
            raise RuntimeError("No attention tensors were captured. Check DEBUG_LAYERS and DEBUG_TIMESTEPS.")

        attention_entries.sort(key=lambda item: (item["layer_index"], item["step_index"]))
        layer_choices = [self._layer_label(entry["layer_index"]) for entry in attention_entries]
        step_choices = [self._step_label(entry["step_index"], entry["timestep_value"]) for entry in attention_entries]
        visible_token_choices = [token_choice(record) for record in token_info["visible_tokens"]]
        default_token_labels = [token_choice(record) for record in token_info["visible_tokens"]]

        token_grid_height, token_grid_width = image_token_grid(height, width)
        num_heads = attention_entries[0].get("num_heads", 0)
        run_state = {
            "run_id": run_paths.run_id,
            "run_dir": str(run_paths.run_dir),
            "inputs_dir": str(run_paths.inputs_dir),
            "outputs_dir": str(run_paths.outputs_dir),
            "tokens_dir": str(run_paths.tokens_dir),
            "attn_dir": str(run_paths.attn_dir),
            "views_dir": str(run_paths.views_dir),
            "mode": request.mode,
            "model_name": model_name,
            "head_mode": config.HEAD_MODE,
            "num_heads": num_heads,
            "image_size": [generated_image.width, generated_image.height],
            "token_grid_height": token_grid_height,
            "token_grid_width": token_grid_width,
            "generated_image_path": str(generated_path),
            "layer_choices": sorted(set(layer_choices)),
            "step_choices": sorted(set(step_choices), key=lambda label: int(label.split()[1])),
            "token_choices": visible_token_choices,
            "default_token_labels": default_token_labels,
            "entry_map": {},
        }
        for entry in attention_entries:
            layer_label = self._layer_label(entry["layer_index"])
            step_label = self._step_label(entry["step_index"], entry["timestep_value"])
            run_state["entry_map"][f"{layer_label}::{step_label}"] = entry

        metadata = {
            "run_id": run_paths.run_id,
            "mode": request.mode,
            "model_name": model_name,
            "prompt": request.prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "num_steps": request.num_steps,
            "guidance": request.guidance,
            "head_mode": config.HEAD_MODE,
            "debug_layers": active_layers,
            "debug_timesteps": active_timesteps,
            "generated_image_path": str(generated_path),
            "token_grid_height": token_grid_height,
            "token_grid_width": token_grid_width,
            "attention_entries": attention_entries,
        }
        save_json(run_paths.run_dir / "metadata.json", metadata)

        default_layer = run_state["layer_choices"][0]
        default_step = run_state["step_choices"][0]
        default_head = None if config.HEAD_MODE == "mean" else "head 00"
        default_view = self.render_saved_view(
            run_state=run_state,
            layer_label=default_layer,
            step_label=default_step,
            token_labels=default_token_labels,
            head_label=default_head,
        )

        logger.info("Completed run_id=%s output=%s", run_paths.run_id, generated_path)
        return {
            "run_state": run_state,
            "generated_image": generated_image,
            "metadata": metadata,
            "default_layer": default_layer,
            "default_step": default_step,
            "default_head": default_head,
            "default_view": default_view,
            "token_choices": visible_token_choices,
            "default_token_labels": default_token_labels,
            "head_choices": self._head_choices(run_state),
        }


RUNNER = FluxDebugRunner()
