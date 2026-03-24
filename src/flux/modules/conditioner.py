from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


def _torch_version_at_least(major: int, minor: int) -> bool:
    version = torch.__version__.split("+", 1)[0]
    parts = version.split(".")
    parsed = []
    for part in parts[:2]:
        digits = "".join(ch for ch in part if ch.isdigit())
        parsed.append(int(digits) if digits else 0)
    while len(parsed) < 2:
        parsed.append(0)
    return tuple(parsed[:2]) >= (major, minor)


def _has_safetensors_weights(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".safetensors"
    return any(path.glob("*.safetensors")) or any(path.glob("*.safetensors.index.json"))


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, is_clip: bool | None = None, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai") if is_clip is None else is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        model_path = Path(version).expanduser()
        if model_path.exists():
            hf_kwargs.setdefault("local_files_only", True)
            if _has_safetensors_weights(model_path):
                hf_kwargs.setdefault("use_safetensors", True)
            elif not _torch_version_at_least(2, 6):
                raise ValueError(
                    f"Local model path '{model_path}' does not contain safetensors weights. "
                    f"Current torch version {torch.__version__} is too old for loading .bin weights via "
                    "transformers. Upgrade torch to >=2.6 or provide a safetensors-based local model directory."
                )

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()
