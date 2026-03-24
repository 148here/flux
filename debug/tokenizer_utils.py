from __future__ import annotations

from typing import Any


def tokenize_prompt_for_display(t5_embedder, prompt: str) -> dict[str, Any]:
    batch_encoding = t5_embedder.tokenizer(
        [prompt],
        truncation=True,
        max_length=t5_embedder.max_length,
        return_length=False,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = batch_encoding["input_ids"][0].tolist()
    attention_mask = batch_encoding["attention_mask"][0].tolist()
    tokens = t5_embedder.tokenizer.convert_ids_to_tokens(input_ids)

    all_tokens = [
        {
            "index": index,
            "id": token_id,
            "text": str(token_text),
            "active": bool(mask_value),
        }
        for index, (token_id, token_text, mask_value) in enumerate(zip(input_ids, tokens, attention_mask))
    ]
    visible_tokens = [record for record in all_tokens if record["active"]]

    return {
        "max_length": t5_embedder.max_length,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "all_tokens": all_tokens,
        "visible_tokens": visible_tokens,
    }


def token_choice(record: dict[str, Any]) -> str:
    return f"[{record['index']:03d}] {record['text']}"


def default_token_indices(token_info: dict[str, Any]) -> list[int]:
    return [record["index"] for record in token_info["visible_tokens"]]


def parse_token_choices(labels: list[str]) -> list[int]:
    indices: list[int] = []
    for label in labels:
        prefix = label.split("]", 1)[0]
        index = int(prefix.replace("[", ""))
        indices.append(index)
    return indices
