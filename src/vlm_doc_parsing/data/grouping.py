"""Per-example length estimation for length-grouped batching.

Why this exists
---------------

Project gotcha #2: CORD receipts have wildly different aspect ratios, so visual-token
counts differ by several hundred tokens between neighbouring examples. Under FSDP every
rank waits at the all-reduce for the slowest one, so a randomly-ordered sampler wastes a
large fraction of the cluster. Grouping examples of similar total length into the same
global batch fixes that.

Why the lengths cannot live in the dataset
------------------------------------------

``SFTTrainer`` overrides ``_set_signature_columns_if_needed`` for vision datasets to
``["messages", "prompt", "completion", "image", "images"]``
(``trl/trainer/sft_trainer.py:1683``). With ``remove_unused_columns=True`` -- the
``TrainingArguments`` default -- **any other column is dropped before the sampler ever
sees it**. So an ``est_tokens`` / ``length`` column added here would silently vanish, and
HF ``Trainer``'s built-in ``group_by_length`` path (which looks for
``args.length_column_name``, default ``"length"``) would fall back to tokenising
``input_ids`` that a vision dataset does not have.

The lengths therefore have to be computed *outside* the dataset and handed to the sampler
directly. Phase 2 should do one of:

* ``trainer.args.group_by_length = True`` is **not** enough -- pass the list into a
  ``transformers.trainer_pt_utils.LengthGroupedSampler`` /
  ``DistributedLengthGroupedSampler`` built by overriding
  ``Trainer._get_train_sampler``, using ``lengths=estimate_lengths(...)``; or
* set ``remove_unused_columns=False`` **and** ``length_column_name="est_tokens"``, keeping
  the column -- but then the collator receives the extra key too, and TRL's
  ``DataCollatorForVisionLanguageModeling`` passes ``examples`` straight into
  ``apply_chat_template``, so the extra key must be tolerated. The sampler route is safer.

Either way the list returned here is aligned with dataset order (index ``i`` of the list
is example ``i`` of the dataset), which is the contract both samplers expect.
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
    """Render one prompt-completion example exactly as TRL's vision collator does.

    Mirrors ``DataCollatorForVisionLanguageModeling._collate_prompt_completion``
    (sft_trainer.py:693-712): ``prepare_multimodal_messages`` first (which injects the
    ``{"type": "image"}`` blocks), then ``apply_chat_template``, which splits the rendered
    conversation into the prompt prefix and the completion suffix.

    The images are passed as ``None`` placeholders: nothing downstream of the chat
    template looks at the image *object*, only at the presence of the block, so this keeps
    the estimator from decoding a single pixel.
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
    """Estimated total token length of every example, aligned with dataset order.

    Args:
        dataset: a ``datasets.Dataset`` in the trainer-ready format produced by
            :func:`vlm_doc_parsing.data.cord.build_datasets` -- columns ``images`` / ``prompt`` /
            ``completion``.
        processor: the ``Qwen2_5_VLProcessor`` from
            :func:`vlm_doc_parsing.data.resolution.build_processor`. Only its chat template and
            tokenizer are used; the model weights are never loaded.
        min_pixels / max_pixels: the same caps the processor was built with. Passed
            explicitly rather than read off the processor so the stats script can sweep
            candidate caps against a single processor instance.

    Returns:
        ``list[int]`` of length ``len(dataset)``.

    Accuracy: this is *exact*, not heuristic, as long as the collator is not truncating.
    The collator tokenises the rendered prompt (one ``<|image_pad|>`` per image), the
    processor then expands each of those into ``estimate_visual_tokens`` copies, and the
    completion is tokenised separately and concatenated. So

        total = len(prompt_ids) - n_images + sum(visual_tokens) + len(completion_ids)

    ``scripts/inspect_batch.py`` asserts this against a real collated batch.

    Cost: no image is decoded. The ``images`` column is temporarily cast to
    ``Image(decode=False)`` so only the compressed bytes are read, and Pillow parses just
    the header to get ``.size``.
    """
    tokenizer = processor.tokenizer

    # decode=False -> __getitem__ hands back {"bytes", "path"} instead of a decoded PIL
    # image. Pillow's lazy open() reads the header only.
    undecoded = dataset.cast_column("images", List(Image(decode=False)))

    lengths: list[int] = []
    render_cache: dict[str, int] = {}

    for row in undecoded:
        prompt_str, completion_str = _render(row, processor)

        if prompt_str in render_cache:
            prompt_text_tokens = render_cache[prompt_str]
        else:
            prompt_text_tokens = len(
                tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            )
            render_cache[prompt_str] = prompt_text_tokens
        completion_tokens = len(
            tokenizer(completion_str, add_special_tokens=False)["input_ids"]
        )

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
