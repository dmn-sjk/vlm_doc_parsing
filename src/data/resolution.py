"""Dynamic-resolution maths and processor construction for Qwen2.5-VL.

Visual tokens dominate sequence length for document images, so the resolution cap is the
single highest-leverage OOM lever (CLAUDE.md gotcha #1). To choose that cap we need to
predict the visual-token count for ~1000 images *without* running the image processor
(which decodes, resizes and normalises every image -- far too slow for a histogram sweep,
and it would blow the login-node memory cgroup).

:func:`estimate_visual_tokens` is therefore a line-for-line replica of the real
``smart_resize`` in
``transformers/models/qwen2_vl/image_processing_qwen2_vl.py`` (transformers 5.16.1), plus
the token count that ``Qwen2_5_VLProcessor.replace_image_token`` derives from
``image_grid_thw``. ``scripts/inspect_batch.py`` asserts the two agree on real batches.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Qwen2.5-VL ViT geometry, from Qwen2VLImageProcessor class attributes.
PATCH_SIZE = 14
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
#: ``factor`` handed to ``smart_resize`` by the image processor:
#: ``factor = patch_size * merge_size`` (image_processing_qwen2_vl.py:226, :280).
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28

#: One image contributes 2 extra text tokens beyond the ``<|image_pad|>`` run:
#: ``<|vision_start|>`` and ``<|vision_end|>`` (see the Qwen2.5-VL chat template).
VISION_DELIMITER_TOKENS = 2


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Replica of ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``.

    Rescales so that (1) both dimensions are divisible by ``factor``, (2) the pixel count
    is within ``[min_pixels, max_pixels]``, and (3) the aspect ratio is preserved as
    closely as possible.

    Kept byte-for-byte equivalent to the installed transformers implementation, including
    the ``round``-then-``floor``/``ceil`` asymmetry -- an "obvious simplification" here
    would desynchronise every estimate from the processor.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def estimate_visual_tokens(
    width: int,
    height: int,
    min_pixels: int,
    max_pixels: int,
) -> int:
    """Number of ``<|image_pad|>`` tokens the processor will emit for one image.

    Args:
        width: original image width in pixels.
        height: original image height in pixels.
        min_pixels: lower bound on post-resize pixel count (``size.shortest_edge``).
        max_pixels: upper bound on post-resize pixel count (``size.longest_edge``).

    Returns:
        The merged visual-token count, i.e. ``grid_h * grid_w // merge_size**2``, which is
        exactly ``(resized_h // 28) * (resized_w // 28)``. This does **not** include the
        two ``<|vision_start|>`` / ``<|vision_end|>`` delimiters -- see
        :data:`VISION_DELIMITER_TOKENS`.

    Note the argument order is ``(width, height)`` here but ``(height, width)`` in
    :func:`smart_resize`, matching the respective upstream conventions. PIL's ``.size`` is
    ``(width, height)``, which is what callers have.
    """
    resized_h, resized_w = smart_resize(
        height, width, FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
    )
    grid_h = resized_h // PATCH_SIZE
    grid_w = resized_w // PATCH_SIZE
    return (grid_h * grid_w) // (MERGE_SIZE**2)


def estimate_resized_size(
    width: int, height: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """``(width, height)`` the processor will resize this image to. Useful for stats."""
    resized_h, resized_w = smart_resize(
        height, width, FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return resized_w, resized_h


def build_processor(cfg: Any, model_id: str | None = None, **kwargs: Any):
    """Load the Qwen2.5-VL processor with the resolution caps from :class:`DataConfig`.

    Args:
        cfg: a ``DataConfig`` (anything with ``min_pixels`` / ``max_pixels``).
        model_id: HF id or local path of the base model. ``DataConfig`` deliberately does
            not carry the model id (it lives on ``ModelConfig``), so Phase 2 should pass
            ``model_cfg.base_model``. Defaults to ``$BASE_MODEL`` and then to the
            ``ModelConfig`` default, which keeps the data-only scripts runnable with just
            a ``DataConfig``.
        **kwargs: forwarded to ``AutoProcessor.from_pretrained``.

    Where the pixel caps go, verified against transformers 5.16.1:

    * ``Qwen2VLImageProcessor.__init__`` pops ``min_pixels`` / ``max_pixels`` and folds
      them into ``size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}``
      (image_processing_qwen2_vl.py:110-118), so passing them straight to
      ``AutoProcessor.from_pretrained`` is the supported path and it does take effect --
      confirmed by reading back ``processor.image_processor.size``.
    * ``_standardize_kwargs`` only rebuilds the ``SizeDict`` when **both** are supplied
      (:120-129). Passing ``max_pixels`` alone as a per-call kwarg is silently ignored.
      Always pass the pair.
    * The checkpoint ships ``min_pixels=3136`` / ``max_pixels=12845056`` (16384 visual
      tokens!). Not overriding these is an instant OOM, hence the assertion below.

    ``tokenizer.padding_side`` is left alone on purpose: TRL's vision collator passes an
    explicit ``padding_side`` on every call (``"left"`` for the prompt half, ``"right"``
    for the completion half, sft_trainer.py:718/725), so the attribute is never consulted
    during training. Setting it here would only create a misleading impression that it
    matters. Inference code that batches generation should set ``"left"`` itself.
    """
    from transformers import AutoProcessor

    if model_id is None:
        model_id = os.environ.get("BASE_MODEL") or "Qwen/Qwen2.5-VL-7B-Instruct"

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
