#!/usr/bin/env python
"""Verify a real collated batch: image-token count, loss mask, and decoded spans.

CLAUDE.md gotcha #3: "Loss masking must be verified, not assumed." TRL's vision collator
deliberately leaves vision tokens unmasked in ``labels`` and lets the model deal with them
(``trl/trainer/sft_trainer.py:686-688``), so nothing in the stack will tell you if the
prompt is leaking into the loss -- the training curve will just look plausible and wrong.

This script builds a batch with **the same collator class SFTTrainer instantiates for a
vision dataset** (``DataCollatorForVisionLanguageModeling``, chosen at ``:1221``) and
asserts, per example:

1. the number of ``<|image_pad|>`` tokens in ``input_ids`` equals what the image processor
   actually produced (``image_grid_thw.prod() // merge_size**2``), and equals what
   ``src.data.resolution.estimate_visual_tokens`` predicted from the raw image size;
2. the unmasked label span (``labels != -100``) is contiguous and decodes to **exactly**
   the target JSON plus the assistant end-of-turn tokens;
3. every position before that span -- system prompt, instruction, vision tokens, padding --
   is ``-100``;
4. nothing was truncated away by ``max_length``;
5. ``src.data.grouping.estimate_lengths`` -- the length signal Phase 2 feeds to the
   length-grouped sampler -- equals the real unpadded sequence length.

Exits non-zero on any failure.

Usage::

    source scripts/env.sh
    python scripts/inspect_batch.py --split validation --batch-size 2
    python scripts/inspect_batch.py -o data.max_pixels=1280*28*28   # not evaluated; use ints
"""

from __future__ import annotations

import argparse
import io
import logging
import sys

import torch
from PIL import Image as PILImage

from src.config import load_config
from src.data.cord import build_datasets
from src.data.grouping import estimate_lengths
from src.data.resolution import build_processor, estimate_visual_tokens

RED = "\033[31m"
GREEN = "\033[32m"
DIM = "\033[2m"
OFF = "\033[0m"


class CheckFailed(AssertionError):
    pass


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise CheckFailed(message)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="optional YAML config (src.config.load_config)")
    p.add_argument("-o", "--override", action="append", default=[],
                   help="config override, e.g. -o data.max_pixels=802816 (repeatable)")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--batch-size", type=int, default=2,
                   help="examples per batch; >1 exercises the padding / flush_left path")
    p.add_argument("--num-batches", type=int, default=2)
    p.add_argument("--model-id", default=None, help="defaults to $BASE_MODEL")
    p.add_argument("--num-proc", type=int, default=2,
                   help="dataset map workers; kept low because this is meant to run on "
                        "the login node, whose memory cgroup SIGKILLs greedy processes")
    p.add_argument("--quiet", action="store_true", help="skip the human-readable dumps")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config, args.override)

    # Only inspect as many examples as we need; keeps the login-node footprint tiny.
    wanted = args.batch_size * args.num_batches
    cfg.data.max_train_samples = wanted
    cfg.data.max_eval_samples = wanted
    cfg.data.num_proc = args.num_proc

    processor = build_processor(cfg.data, model_id=args.model_id)
    tokenizer = processor.tokenizer
    image_token_id = processor.image_token_id
    merge_length = processor.image_processor.merge_size ** 2

    datasets = build_datasets(cfg.data, processor=processor, model_id=args.model_id)
    ds = datasets[args.split]

    # The length signal Phase 2 will hand to the length-grouped sampler. Asserted against
    # the real collated length below, so a drift in either direction is caught here.
    est_lengths = estimate_lengths(
        ds, processor, min_pixels=cfg.data.min_pixels, max_pixels=cfg.data.max_pixels
    )

    # Exactly what SFTTrainer builds for a vision dataset (sft_trainer.py:1221).
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=cfg.data.max_length,
        completion_only_loss=cfg.train.completion_only_loss,
        pad_to_multiple_of=None,  # raises NotImplementedError for prompt-completion data
        dataset_text_field="text",
    )
    print(f"collator: {type(collator).__name__} "
          f"(completion_only_loss={collator.completion_only_loss}, max_length={collator.max_length})")
    print(f"processor: {type(processor).__name__} size={processor.image_processor.size}")
    print(f"split={args.split} n={len(ds)} batch_size={args.batch_size}\n")

    failures = 0
    checked = 0

    for b in range(args.num_batches):
        lo, hi = b * args.batch_size, min((b + 1) * args.batch_size, len(ds))
        if lo >= hi:
            break

        # Raw image sizes, read from the compressed header, for the independent
        # estimate_visual_tokens cross-check.
        from datasets import Image as HFImage, List as HFList

        undecoded = ds.cast_column("images", HFList(HFImage(decode=False)))
        raw_sizes = []
        for i in range(lo, hi):
            sizes = []
            for entry in undecoded[i]["images"]:
                with PILImage.open(io.BytesIO(entry["bytes"])) as im:
                    sizes.append(im.size)  # (width, height)
            raw_sizes.append(sizes)

        examples = [ds[i] for i in range(lo, hi)]
        # The target string as `src.data.schema.target_json` produced it, read back off the
        # dataset so the assertion compares against what is actually stored.
        targets = [ex["completion"][0]["content"] for ex in examples]

        batch = collator(examples)

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        grid = batch["image_grid_thw"]

        print(f"--- batch {b} --- keys={sorted(batch.keys())}")
        print(f"    input_ids {tuple(input_ids.shape)}  labels {tuple(labels.shape)}  "
              f"pixel_values {tuple(batch['pixel_values'].shape)}  image_grid_thw {grid.tolist()}")

        grid_ptr = 0
        for row in range(input_ids.shape[0]):
            idx = lo + row
            target = targets[row]
            ids = input_ids[row]
            lab = labels[row]
            checked += 1

            try:
                # -- 1. image placeholder accounting -------------------------------------
                n_images = len(raw_sizes[row])
                expected_from_processor = 0
                expected_from_estimator = 0
                for k in range(n_images):
                    thw = grid[grid_ptr + k]
                    expected_from_processor += int(thw.prod()) // merge_length
                    w, h = raw_sizes[row][k]
                    expected_from_estimator += estimate_visual_tokens(
                        w, h, cfg.data.min_pixels, cfg.data.max_pixels
                    )
                grid_ptr += n_images

                actual = int((ids == image_token_id).sum())
                _check(
                    actual == expected_from_processor,
                    f"row {idx}: {actual} <|image_pad|> tokens in input_ids but the image "
                    f"processor produced {expected_from_processor} "
                    f"(image_grid_thw={grid[grid_ptr - n_images:grid_ptr].tolist()})",
                )
                _check(
                    expected_from_estimator == expected_from_processor,
                    f"row {idx}: estimate_visual_tokens said {expected_from_estimator} but "
                    f"the processor produced {expected_from_processor} for sizes "
                    f"{raw_sizes[row]} -- src/data/resolution.py is out of sync with "
                    f"transformers' smart_resize",
                )

                # -- 2. the unmasked span is exactly the completion ----------------------
                unmasked = (lab != -100)
                _check(bool(unmasked.any()), f"row {idx}: no unmasked label positions at all")
                pos = unmasked.nonzero(as_tuple=True)[0]
                start, end = int(pos[0]), int(pos[-1])
                _check(
                    len(pos) == end - start + 1,
                    f"row {idx}: unmasked label positions are not contiguous "
                    f"({len(pos)} positions spanning {start}..{end})",
                )
                _check(
                    torch.equal(lab[start:end + 1], ids[start:end + 1]),
                    f"row {idx}: unmasked labels do not equal the corresponding input_ids",
                )

                decoded_raw = tokenizer.decode(ids[start:end + 1], skip_special_tokens=False)
                eot = f"{tokenizer.eos_token}\n"
                _check(
                    decoded_raw == target + eot,
                    f"row {idx}: unmasked span is not the target JSON.\n"
                    f"        expected: {target + eot!r}\n"
                    f"        got:      {decoded_raw!r}",
                )
                # The end-of-turn token must be inside the loss, or the model never learns
                # to stop generating.
                _check(
                    int(ids[end - 1]) == tokenizer.eos_token_id,
                    f"row {idx}: assistant end-of-turn token is not inside the loss span",
                )

                # -- 3. everything before the completion is masked -----------------------
                _check(
                    bool((lab[:start] == -100).all()),
                    f"row {idx}: {int((lab[:start] != -100).sum())} unmasked positions "
                    f"before the completion (prompt/system/vision tokens are leaking into "
                    f"the loss)",
                )
                _check(
                    bool((lab[end + 1:] == -100).all()),
                    f"row {idx}: unmasked positions after the completion",
                )
                vision_positions = (ids == image_token_id)
                _check(
                    bool((lab[vision_positions] == -100).all()),
                    f"row {idx}: some <|image_pad|> positions are unmasked in labels",
                )
                _check(
                    bool((lab[attention_mask[row] == 0] == -100).all()),
                    f"row {idx}: padding positions are unmasked in labels",
                )

                # -- 4. nothing was truncated -------------------------------------------
                real_len = int(attention_mask[row].sum())
                _check(
                    collator.max_length is None or real_len <= collator.max_length,
                    f"row {idx}: sequence length {real_len} exceeds max_length "
                    f"{collator.max_length}",
                )
                _check(
                    int(ids[-1]) == tokenizer.pad_token_id or end == input_ids.shape[1] - 1,
                    f"row {idx}: sequence appears truncated mid-completion",
                )

                # -- 5. the grouping length signal is exact ------------------------------
                _check(
                    est_lengths[idx] == real_len,
                    f"row {idx}: estimate_lengths said {est_lengths[idx]} but the collated "
                    f"sequence is {real_len} tokens -- the length-grouped sampler would be "
                    f"grouping on a wrong signal",
                )

                print(f"    {GREEN}OK{OFF}  row {idx}: len={real_len} (est {est_lengths[idx]}) "
                      f"visual={expected_from_processor} "
                      f"prompt_masked={start} completion={end - start + 1} "
                      f"image={raw_sizes[row]}")

            except CheckFailed as exc:
                failures += 1
                print(f"    {RED}FAIL{OFF} {exc}")

        if not args.quiet and b == 0:
            row = 0
            ids = input_ids[row]
            lab = labels[row]
            pos = (lab != -100).nonzero(as_tuple=True)[0]
            start, end = int(pos[0]), int(pos[-1])
            prompt_text = tokenizer.decode(ids[:start], skip_special_tokens=False)
            # Collapse the <|image_pad|> run so the dump stays readable.
            pad_tok = processor.image_token
            n_pad = prompt_text.count(pad_tok)
            prompt_text = prompt_text.replace(pad_tok * n_pad, f"{pad_tok} x{n_pad}")
            print(f"\n{DIM}================ decoded PROMPT (all labels == -100) "
                  f"================{OFF}")
            print(prompt_text)
            print(f"{DIM}================ decoded UNMASKED span (labels != -100) "
                  f"=========={OFF}")
            print(tokenizer.decode(ids[start:end + 1], skip_special_tokens=False))
            print(f"{DIM}=========================================================="
                  f"========{OFF}\n")

    print(f"\nchecked {checked} example(s); {failures} failure(s)")
    if failures:
        print(f"{RED}FAILED{OFF}")
        return 1
    print(f"{GREEN}ALL CHECKS PASSED{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
