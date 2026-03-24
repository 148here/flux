from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from flux.model import Flux

from debug.attn_utils import compute_image_query_text_key_attention


logger = logging.getLogger(__name__)


class FluxAttentionCapture:
    def __init__(
        self,
        model: Flux,
        debug_layers: list[int],
        debug_timesteps: list[int],
        head_mode: str,
        capture_callback: Callable[[int, int, float, object], None],
    ) -> None:
        self.model = model
        self.debug_layers = sorted(set(debug_layers))
        self.debug_timesteps = set(debug_timesteps)
        self.head_mode = head_mode
        self.capture_callback = capture_callback

        self._current_step = -1
        self._current_timestep = 0.0
        self._seen: set[tuple[int, int]] = set()
        self._forward_handle = None
        self._old_probes: dict[int, object] = {}

    def __enter__(self) -> "FluxAttentionCapture":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def install(self) -> None:
        self._forward_handle = self.model.register_forward_pre_hook(self._on_model_forward, with_kwargs=True)
        for layer_index in self.debug_layers:
            block = self.model.double_blocks[layer_index]
            self._old_probes[layer_index] = getattr(block, "_debug_attn_probe", None)
            block._debug_attn_probe = self._make_probe(layer_index)

    def remove(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        for layer_index, probe in self._old_probes.items():
            self.model.double_blocks[layer_index]._debug_attn_probe = probe
        self._old_probes.clear()

    def _on_model_forward(self, module, args, kwargs) -> None:
        timesteps = kwargs.get("timesteps")
        if timesteps is None and len(args) >= 5:
            timesteps = args[4]
        self._current_step += 1
        self._current_timestep = float(timesteps[0].detach().float().cpu().item()) if timesteps is not None else 0.0

    def _make_probe(self, layer_index: int):
        def probe(*, q, k, pe, txt_token_count: int) -> None:
            if self._current_step not in self.debug_timesteps:
                return
            key = (layer_index, self._current_step)
            if key in self._seen:
                return
            self._seen.add(key)
            logger.info(
                "Capturing layer=%s step=%s timestep=%.6f head_mode=%s",
                layer_index,
                self._current_step,
                self._current_timestep,
                self.head_mode,
            )
            with torch.no_grad():
                raw_attention = compute_image_query_text_key_attention(
                    q=q.detach(),
                    k=k.detach(),
                    pe=pe.detach(),
                    txt_token_count=txt_token_count,
                    head_mode=self.head_mode,
                )
            self.capture_callback(layer_index, self._current_step, self._current_timestep, raw_attention)

        return probe
