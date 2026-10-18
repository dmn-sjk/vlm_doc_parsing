"""Qwen2.5-VL resolution maths and processor construction.

`estimate_visual_tokens` predicts the visual-token count from an image's size alone, with
transformers' own `smart_resize`, so statistics and length grouping never run the image
processor. `scripts/inspect_batch.py` asserts it matches real batches.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoProcessor, ProcessorMixin
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from vlm_doc_parsing.config import DataConfig

# Qwen2.5-VL ViT geometry; one merged visual token covers FACTOR x FACTOR pixels.
PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28


def estimate_visual_tokens(width: int, height: int, min_pixels: int, max_pixels: int) -> int:
    """`<|image_pad|>` tokens the processor emits for one image, excluding the two delimiters.

    Takes `(width, height)` like PIL's `.size`; `smart_resize` takes `(height, width)`.
    """
    resized_h, resized_w = smart_resize(
        height, width, factor=FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
    )
    grid_h = resized_h // PATCH_SIZE
    grid_w = resized_w // PATCH_SIZE
    return (grid_h * grid_w) // (MERGE_SIZE**2)


def estimate_resized_size(width: int, height: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """`(width, height)` the processor resizes this image to."""
    resized_h, resized_w = smart_resize(
        height, width, factor=FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return resized_w, resized_h


def build_processor(cfg: DataConfig, model_id: str, **kwargs: Any) -> ProcessorMixin:
    """Load the processor with the DataConfig pixel caps; `kwargs` go to `from_pretrained`.

    Both caps are passed together (one alone as a per-call kwarg is silently ignored) and read
    back, because the checkpoint's own `max_pixels` is 16384 visual tokens: an instant OOM.
    `padding_side` is left alone: the training collator sets it per call; batched inference
    must set "left".
    """
    min_pixels = int(cfg.min_pixels)
    max_pixels = int(cfg.max_pixels)
    if min_pixels % (FACTOR * FACTOR) or max_pixels % (FACTOR * FACTOR):
        raise ValueError(
            "min_pixels/max_pixels should be multiples of 28*28 (one merged visual token) "
            f"so the caps map to whole tokens; got {min_pixels} / {max_pixels}"
        )
    if min_pixels > max_pixels:
        raise ValueError(f"min_pixels ({min_pixels}) > max_pixels ({max_pixels})")

    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
    )

    size = processor.image_processor.size
    if size.shortest_edge != min_pixels or size.longest_edge != max_pixels:
        raise RuntimeError(
            "resolution caps did not reach the image processor: "
            f"size={size}, wanted shortest_edge={min_pixels}, longest_edge={max_pixels}"
        )
    return processor
