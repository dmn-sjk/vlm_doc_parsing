"""CORD-v2 -> TRL prompt-completion dataset.

Output contract (one example)::

    {
      "images":     [PIL.Image.Image],                       # exactly one page image
      "prompt":     [{"role": "system",    "content": SYSTEM_PROMPT},
                     {"role": "user",      "content": USER_INSTRUCTION}],
      "completion": [{"role": "assistant", "content": "<compact target JSON>"}],
    }

Two decisions in there are load-bearing and were taken by reading TRL 1.12.0, not by
recall:

**1. prompt/completion, never messages.** ``SFTTrainer`` raises on
``assistant_only_loss=True`` for any dataset carrying an ``image``/``images`` column
(``trl/trainer/sft_trainer.py:1052``). The only remaining way to train on the answer alone
is the prompt-completion dataset type, where the collator builds a ``completion_mask`` and
sets ``labels[completion_mask == 0] = -100`` (``:774``). Emitting a ``messages``
column instead would train on the system prompt and the instruction as well.

**2. message ``content`` is a plain string, not a list of content blocks.** The collator
calls ``prepare_multimodal_messages(example["prompt"], images=example["images"])``
(``:707``), which -- *for string content only* -- inserts one ``{"type": "image"}`` block
per image ahead of the text and fills in the PIL objects (``trl/data_utils.py:80-93``). So
the image block must **not** be pre-inserted here; doing both would make
``num_placeholders != len(images)`` and raise.

Storing pre-built content blocks would also be actively unsafe. ``datasets`` unifies the
struct type of a list column, so a mixed list of ``{"type": "image"}`` and
``{"type": "text", "text": ...}`` blocks comes back as
``{"type": "text", "text": "...", "image": None}`` -- and the Qwen2.5-VL chat template
branches on ``content['type'] == 'image' or 'image' in content``, so that resurrected
``image: None`` key makes the template render the *text* block as a second
``<|vision_start|><|image_pad|><|vision_end|>``. The result is the classic "Image features
and image tokens do not match" crash. Plain strings sidestep the whole failure mode and
are what ``prepare_multimodal_messages`` is designed for.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, Features, Image, List, Value, load_dataset

from vlm_doc_parsing.data.grouping import estimate_lengths
from vlm_doc_parsing.data.resolution import FACTOR, build_processor
from vlm_doc_parsing.data.schema import SYSTEM_PROMPT, USER_INSTRUCTION, target_json

logger = logging.getLogger(__name__)


def format_example(
    example: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
    user_instruction: str = USER_INSTRUCTION,
) -> dict[str, Any]:
    """Map one raw CORD row to the trainer-ready prompt-completion example.

    ``example["image"]`` is expected to be the *undecoded* ``{"bytes", "path"}`` payload
    (the caller casts the column to ``Image(decode=False)`` first), so this never decodes
    a JPEG/PNG. ``map`` is given :data:`OUTPUT_FEATURES`, whose ``List(Image())`` encoder
    accepts that ``{"bytes", "path"}`` form verbatim, so the bytes are copied straight
    through and the output column decodes lazily to PIL on ``__getitem__``.
    """
    return {
        "images": [example["image"]],
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
        ],
        "completion": [
            {"role": "assistant", "content": target_json(example["ground_truth"])}
        ],
    }


_MESSAGE = {"role": Value("string"), "content": Value("string")}

#: Declared up front so `map` writes the final feature types directly. Letting `datasets`
#: infer them instead produces `List({"bytes": binary, "path": string})` for the images,
#: which then needs a whole extra `cast_column` pass over the table -- a second full copy
#: of every image, and the login node cannot afford it.
OUTPUT_FEATURES = Features(
    {
        "images": List(Image()),
        "prompt": List(_MESSAGE),
        "completion": List(_MESSAGE),
    }
)

#: Rows buffered in RAM before a shard is flushed to disk. `datasets` defaults to 1000,
#: which for CORD means ~1000 full-page PNGs (~800 MB) resident per worker -- enough to get
#: the whole process SIGKILLed by the cluster login-node memory cgroup (observed: exit 137).
WRITER_BATCH_SIZE = 32


def _format_split(split: Dataset, num_proc: int, desc: str) -> Dataset:
    # decode=False keeps map at the bytes level: the arrow cache gets a byte-for-byte copy
    # of the original PNGs instead of a PIL decode + re-encode round trip, and peak RSS
    # stays at one compressed image per worker rather than a decoded RGB buffer.
    undecoded = split.cast_column("image", Image(decode=False))
    workers = num_proc if len(undecoded) >= num_proc > 1 else None
    kwargs = dict(
        remove_columns=undecoded.column_names,
        features=OUTPUT_FEATURES,
        writer_batch_size=WRITER_BATCH_SIZE,
        fn_kwargs={
            "system_prompt": SYSTEM_PROMPT,
            "user_instruction": USER_INSTRUCTION,
        },
        desc=desc,
    )
    try:
        return undecoded.map(format_example, num_proc=workers, **kwargs)
    except RuntimeError as exc:
        # Observed on the cluster login node: the memory cgroup SIGKILLs a `datasets`
        # worker and `iflatmap_unordered` reports "One of the subprocesses has abruptly
        # died". It is a resource limit, not a bug in the map function, so fall back to a
        # single process rather than failing the whole build.
        if workers is None or "subprocess" not in str(exc):
            raise
        logger.warning(
            "%s: multiprocess map with num_proc=%d failed (%s); retrying single-process",
            desc, workers, exc,
        )
        return undecoded.map(format_example, num_proc=None, **kwargs)


def build_datasets(
    cfg: Any,
    processor: Any = None,
    model_id: str | None = None,
) -> DatasetDict:
    """Build the ``train`` / ``validation`` / ``test`` splits for ``SFTTrainer``.

    Args:
        cfg: a :class:`vlm_doc_parsing.config.DataConfig`.
        processor: an already-built ``Qwen2_5_VLProcessor``. Optional; only used to
            estimate sequence lengths for the ``max_length`` filter. Pass the trainer's own
            processor to avoid loading a second copy. If ``None`` one is built via
            :func:`vlm_doc_parsing.data.resolution.build_processor`.
        model_id: forwarded to ``build_processor`` when ``processor`` is ``None``.
            ``DataConfig`` has no model field, so Phase 2 should pass
            ``model_cfg.base_model`` (or rely on ``$BASE_MODEL``).

    Returns:
        A ``DatasetDict`` with keys ``train`` / ``validation`` / ``test``, every example in
        the format documented at the top of this module.

    Pipeline order is: format -> length-filter -> shuffle -> subsample. Filtering before
    subsampling means ``max_train_samples=8`` yields exactly 8 usable examples rather than
    "8 minus however many were too long", which is what a short debug run needs.
    """
    if processor is None:
        processor = build_processor(cfg, model_id=model_id)

    split_map = {
        "train": cfg.train_split,
        "validation": cfg.eval_split,
        "test": cfg.test_split,
    }
    caps = {
        "train": cfg.max_train_samples,
        "validation": cfg.max_eval_samples,
        "test": cfg.max_eval_samples,
    }

    raw = load_dataset(cfg.dataset_id)
    out: dict[str, Dataset] = {}

    for name, source_split in split_map.items():
        if source_split not in raw:
            raise KeyError(
                f"split {source_split!r} not in {cfg.dataset_id} (have {list(raw)})"
            )
        split = _format_split(
            raw[source_split], cfg.num_proc, desc=f"formatting {name}"
        )
        n_before = len(split)

        lengths = estimate_lengths(
            split,
            processor,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
        )
        keep = [i for i, n in enumerate(lengths) if n <= cfg.max_length]
        n_dropped = n_before - len(keep)
        if n_dropped:
            over = sorted((n for n in lengths if n > cfg.max_length), reverse=True)
            logger.warning(
                "%s: dropped %d/%d examples (%.1f%%) over max_length=%d "
                "(longest %d tokens, max_pixels=%d -> %d visual tokens)",
                name, n_dropped, n_before, 100.0 * n_dropped / max(n_before, 1),
                cfg.max_length, over[0], cfg.max_pixels,
                cfg.max_pixels // (FACTOR * FACTOR),
            )
        else:
            logger.info(
                "%s: kept all %d examples (longest %d tokens, max_length=%d)",
                name, n_before, max(lengths) if lengths else 0, cfg.max_length,
            )
        split = split.select(keep)

        cap = caps[name]
        if cap is not None and cap < len(split):
            # Shuffle before truncating so a debug-run subset is a random sample rather
            # than whatever happens to sit at the front of the split.
            split = split.shuffle(seed=cfg.seed).select(range(cap))
            logger.info("%s: subsampled to %d examples (seed=%d)", name, cap, cfg.seed)

        out[name] = split

    return DatasetDict(out)
