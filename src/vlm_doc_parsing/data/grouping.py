"""Exact per-example sequence lengths for length-grouped batching, without decoding images.

The lengths can't live in a dataset column: SFTTrainer drops every column outside its vision
signature before the sampler sees it. They are returned as a list aligned with dataset order
and passed to the sampler directly (docs/design-notes.md#length-grouping).
"""

from __future__ import annotations

import io
from typing import Any

from datasets import Dataset, Image, List
from PIL import Image as PILImage
from transformers import ProcessorMixin
from trl.data_utils import apply_chat_template, prepare_multimodal_messages

from vlm_doc_parsing.data.resolution import estimate_visual_tokens


def _render(example: dict[str, Any], processor: ProcessorMixin) -> tuple[str, str]:
    """Render prompt and completion text exactly as TRL's vision collator does.

    `None` image placeholders suffice: the chat template only needs the image block to exist.
    """
    n_images = len(example.get("images", []) or [])
    prompt = prepare_multimodal_messages(example["prompt"], images=[None] * n_images)
    completion = prepare_multimodal_messages(example["completion"])
    rendered = apply_chat_template({"prompt": prompt, "completion": completion}, processor)
    return rendered["prompt"], rendered["completion"]


def estimate_lengths(
    dataset: Dataset,
    processor: ProcessorMixin,
    *,
    min_pixels: int,
    max_pixels: int,
) -> list[int]:
    """Total token length of every example, aligned with dataset order.

    Exact while nothing is truncated: `len(prompt_ids) - n_images + sum(visual_tokens) +
    len(completion_ids)`, since each `<|image_pad|>` expands to `visual_tokens` copies.
    `scripts/inspect_batch.py` asserts it against a collated batch. The pixel caps are
    explicit so `dataset_stats.py` can sweep them with one processor.
    """
    tokenizer = processor.tokenizer
    # Undecoded bytes: Pillow reads only the header for `.size`.
    undecoded = dataset.cast_column("images", List(Image(decode=False)))

    lengths: list[int] = []
    render_cache: dict[str, int] = {}

    for row in undecoded:
        prompt_str, completion_str = _render(row, processor)

        if prompt_str in render_cache:
            prompt_text_tokens = render_cache[prompt_str]
        else:
            prompt_text_tokens = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
            render_cache[prompt_str] = prompt_text_tokens
        completion_tokens = len(tokenizer(completion_str, add_special_tokens=False)["input_ids"])

        visual = 0
        n_images = 0
        for entry in row["images"]:
            n_images += 1
            with PILImage.open(io.BytesIO(entry["bytes"])) as im:
                width, height = im.size
            visual += estimate_visual_tokens(width, height, min_pixels, max_pixels)

        total = prompt_text_tokens - n_images + visual + completion_tokens
        lengths.append(total)

    return lengths
