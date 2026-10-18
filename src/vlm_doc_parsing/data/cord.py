"""CORD-v2 -> TRL prompt-completion dataset: `{"images", "prompt", "completion"}`.

Prompt-completion (not `messages`) so only the JSON answer is trained on; message `content`
is a plain string because the collator inserts the image block itself. See
docs/adr/0002-prompt-completion-format.md and docs/design-notes.md#example-format.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, Features, Image, List, Value, load_dataset
from transformers import ProcessorMixin

from vlm_doc_parsing.config import DataConfig
from vlm_doc_parsing.data.grouping import estimate_lengths
from vlm_doc_parsing.data.resolution import FACTOR, build_processor
from vlm_doc_parsing.data.schema import SYSTEM_PROMPT, USER_INSTRUCTION, target_json

logger = logging.getLogger(__name__)


def format_example(
    example: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
    user_instruction: str = USER_INSTRUCTION,
) -> dict[str, Any]:
    """Map one raw CORD row (image undecoded, as `{"bytes", "path"}`) to a training example."""
    return {
        "images": [example["image"]],
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
        ],
        "completion": [{"role": "assistant", "content": target_json(example["ground_truth"])}],
    }


_MESSAGE = {"role": Value("string"), "content": Value("string")}

# Declared so `map` writes image features directly; inferred types need an extra cast_column
# pass, a second full copy of every image.
OUTPUT_FEATURES = Features(
    {"images": List(Image()), "prompt": List(_MESSAGE), "completion": List(_MESSAGE)}
)

# The default 1000 rows buffers ~800 MB of PNGs per worker; a login-node cgroup kills that.
WRITER_BATCH_SIZE = 32


def _format_split(split: Dataset, num_proc: int, desc: str) -> Dataset:
    # decode=False copies the PNG bytes through without a decode/re-encode.
    undecoded = split.cast_column("image", Image(decode=False))
    workers = num_proc if len(undecoded) >= num_proc > 1 else None
    kwargs = dict(
        remove_columns=undecoded.column_names,
        features=OUTPUT_FEATURES,
        writer_batch_size=WRITER_BATCH_SIZE,
        fn_kwargs={"system_prompt": SYSTEM_PROMPT, "user_instruction": USER_INSTRUCTION},
        desc=desc,
    )
    try:
        return undecoded.map(format_example, num_proc=workers, **kwargs)
    except RuntimeError as exc:
        # A memory cgroup killing a worker surfaces as "subprocess ... abruptly died".
        if workers is None or "subprocess" not in str(exc):
            raise
        logger.warning(
            "%s: multiprocess map with num_proc=%d failed (%s); retrying single-process",
            desc,
            workers,
            exc,
        )
        return undecoded.map(format_example, num_proc=None, **kwargs)


def build_datasets(cfg: DataConfig, processor: ProcessorMixin | None, model_id: str) -> DatasetDict:
    """Build the `train` / `validation` / `test` splits.

    `processor` measures lengths for the `max_length` filter (pass the trainer's; None loads
    one from `model_id`). Filtering before subsampling makes `max_train_samples=8` yield 8.
    """
    if processor is None:
        processor = build_processor(cfg, model_id=model_id)

    split_map = {"train": cfg.train_split, "validation": cfg.eval_split, "test": cfg.test_split}
    caps = {
        "train": cfg.max_train_samples,
        "validation": cfg.max_eval_samples,
        "test": cfg.max_eval_samples,
    }

    raw = load_dataset(cfg.dataset_id)
    out: dict[str, Dataset] = {}

    for name, source_split in split_map.items():
        if source_split not in raw:
            raise KeyError(f"split {source_split!r} not in {cfg.dataset_id} (have {list(raw)})")
        split = _format_split(raw[source_split], cfg.num_proc, desc=f"formatting {name}")
        n_before = len(split)

        lengths = estimate_lengths(
            split, processor, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels
        )
        keep = [i for i, n in enumerate(lengths) if n <= cfg.max_length]
        n_dropped = n_before - len(keep)
        if n_dropped:
            over = sorted((n for n in lengths if n > cfg.max_length), reverse=True)
            logger.warning(
                "%s: dropped %d/%d examples (%.1f%%) over max_length=%d "
                "(longest %d tokens, max_pixels=%d -> %d visual tokens)",
                name,
                n_dropped,
                n_before,
                100.0 * n_dropped / max(n_before, 1),
                cfg.max_length,
                over[0],
                cfg.max_pixels,
                cfg.max_pixels // (FACTOR * FACTOR),
            )
        else:
            logger.info(
                "%s: kept all %d examples (longest %d tokens, max_length=%d)",
                name,
                n_before,
                max(lengths) if lengths else 0,
                cfg.max_length,
            )
        split = split.select(keep)

        cap = caps[name]
        if cap is not None and cap < len(split):
            # A random subset, not the front of the split.
            split = split.shuffle(seed=cfg.seed).select(range(cap))
            logger.info("%s: subsampled to %d examples (seed=%d)", name, cap, cfg.seed)

        out[name] = split

    return DatasetDict(out)
