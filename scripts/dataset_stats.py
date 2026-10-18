#!/usr/bin/env python
"""Dataset statistics that justify the resolution cap.

For each split, prints
  * example count,
  * source image size percentiles,
  * a histogram of estimated visual tokens at the configured ``max_pixels``,
  * target-JSON token-length percentiles,
  * and a sweep over candidate ``max_pixels`` values showing, for each, the visual-token
    budget and what fraction of examples ``build_datasets`` would drop for exceeding
    ``max_length``.

Decodes no image and loads no model weights, so it runs on a login node.

Usage::

    source scripts/env.sh
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --splits train --candidates 256,512,1024,2048
    python scripts/dataset_stats.py -o data.max_length=2048
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from datasets import Image, load_dataset
from PIL import Image as PILImage
from trl.data_utils import apply_chat_template, prepare_multimodal_messages

from vlm_doc_parsing.config import load_config
from vlm_doc_parsing.data.resolution import (
    FACTOR,
    build_processor,
    estimate_resized_size,
    estimate_visual_tokens,
)
from vlm_doc_parsing.data.schema import SYSTEM_PROMPT, USER_INSTRUCTION, target_json

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sft_1node.yaml"


def percentiles(values: list[int], qs=(0, 5, 25, 50, 75, 95, 100)) -> dict[int, int]:
    s = sorted(values)
    n = len(s)
    return {q: s[min(n - 1, max(0, round(q / 100 * (n - 1))))] for q in qs}


def fmt_pct(d: dict[int, int]) -> str:
    return "  ".join(f"p{q}={v}" for q, v in d.items())


def histogram(values: list[int], bins: int = 12, width: int = 46) -> str:
    if not values:
        return "  (empty)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  {lo:>6d}            {'#' * width} {len(values)}"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / step))] += 1
    top = max(counts)
    out = []
    for i, c in enumerate(counts):
        a, b = lo + i * step, lo + (i + 1) * step
        bar = "#" * round(width * c / top) if c else ""
        out.append(f"  {a:7.0f}-{b:<7.0f} |{bar:<{width}} {c}")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="YAML config (see vlm_doc_parsing.config)")
    p.add_argument("-o", "--override", action="append", default=[],
                   help="config override, e.g. -o data.max_pixels=802816 (repeatable)")
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--model-id", default=None, help="defaults to the config's model.base_model")
    p.add_argument(
        "--candidates",
        default="256,512,768,1024,1280,1536,2048,4096",
        help="candidate max_pixels values, expressed in *visual tokens* "
             "(the value is multiplied by 28*28)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    data = cfg.data

    processor = build_processor(data, model_id=args.model_id or cfg.model.base_model)
    tokenizer = processor.tokenizer

    # The prompt is the same for every example: measure it once through the chat template,
    # minus the one <|image_pad|> placeholder the processor expands.
    probe = {
        "prompt": prepare_multimodal_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_INSTRUCTION},
            ],
            images=[None],
        ),
        "completion": prepare_multimodal_messages(
            [{"role": "assistant", "content": ""}]
        ),
    }
    rendered = apply_chat_template(probe, processor)
    prompt_overhead = (
        len(tokenizer(rendered["prompt"], add_special_tokens=False)["input_ids"]) - 1
    )
    # <|im_end|>\n closing every completion.
    completion_overhead = len(
        tokenizer(rendered["completion"], add_special_tokens=False)["input_ids"]
    )

    candidates = [int(c.strip()) for c in args.candidates.split(",") if c.strip()]

    print(f"dataset          : {data.dataset_id}")
    print(f"configured caps  : min_pixels={data.min_pixels} "
          f"({data.min_pixels // (FACTOR * FACTOR)} tokens)  "
          f"max_pixels={data.max_pixels} ({data.max_pixels // (FACTOR * FACTOR)} tokens)")
    print(f"max_length       : {data.max_length}")
    print(f"prompt overhead  : {prompt_overhead} tokens (system + instruction + chat "
          f"template, excluding visual tokens)")
    print(f"completion wrap  : {completion_overhead} tokens (<|im_end|>\\n)")

    raw = load_dataset(data.dataset_id)

    for split in (s.strip() for s in args.splits.split(",")):
        if split not in raw:
            print(f"\n!! split {split!r} not in dataset (have {list(raw)})")
            continue
        ds = raw[split].cast_column("image", Image(decode=False))

        sizes: list[tuple[int, int]] = []
        target_tokens: list[int] = []
        for row in ds:
            with PILImage.open(io.BytesIO(row["image"]["bytes"])) as im:
                sizes.append(im.size)
            tgt = target_json(row["ground_truth"])
            target_tokens.append(
                len(tokenizer(tgt, add_special_tokens=False)["input_ids"])
                + completion_overhead
            )

        n = len(sizes)
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        pixels = [w * h for w, h in sizes]
        aspects = [max(w, h) / min(w, h) for w, h in sizes]

        print(f"\n{'=' * 78}")
        print(f"{split}: {n} examples")
        print(f"{'=' * 78}")
        print(f"  width      {fmt_pct(percentiles(widths))}")
        print(f"  height     {fmt_pct(percentiles(heights))}")
        print(f"  megapixels {fmt_pct({q: v // 1000 for q, v in percentiles(pixels).items()})}  (x1000 px)")
        print(f"  aspect     min={min(aspects):.2f} median={sorted(aspects)[n // 2]:.2f} "
              f"max={max(aspects):.2f}")

        visual = [
            estimate_visual_tokens(w, h, data.min_pixels, data.max_pixels)
            for w, h in sizes
        ]
        print(f"\n  estimated visual tokens @ max_pixels={data.max_pixels} "
              f"({data.max_pixels // (FACTOR * FACTOR)} tokens):")
        print(f"    {fmt_pct(percentiles(visual))}")
        print(histogram(visual))

        rw, rh = zip(*[
            estimate_resized_size(w, h, data.min_pixels, data.max_pixels) for w, h in sizes
        ])
        print(f"  resized width  {fmt_pct(percentiles(list(rw)))}")
        print(f"  resized height {fmt_pct(percentiles(list(rh)))}")

        print(f"\n  target JSON length in tokens (incl. {completion_overhead}-token "
              f"assistant wrapper):")
        print(f"    {fmt_pct(percentiles(target_tokens))}")
        print(histogram(target_tokens))

        print(f"\n  max_pixels sweep (drop = total tokens > max_length={data.max_length}):")
        print(f"    {'tokens':>7} {'max_pixels':>11} | {'visual p50':>10} {'visual p95':>10} "
              f"{'visual max':>10} | {'total p50':>9} {'total p95':>9} {'total max':>9} | dropped")
        for cand in candidates:
            mp = cand * (FACTOR * FACTOR)
            if mp < data.min_pixels:
                continue
            vis = [estimate_visual_tokens(w, h, data.min_pixels, mp) for w, h in sizes]
            totals = [prompt_overhead + v + t for v, t in zip(vis, target_tokens)]
            dropped = sum(1 for t in totals if t > data.max_length)
            vp = percentiles(vis, (50, 95, 100))
            tp = percentiles(totals, (50, 95, 100))
            mark = " <-- configured" if mp == data.max_pixels else ""
            print(f"    {cand:>7} {mp:>11} | {vp[50]:>10} {vp[95]:>10} {vp[100]:>10} | "
                  f"{tp[50]:>9} {tp[95]:>9} {tp[100]:>9} | "
                  f"{dropped:>4}/{n} ({100.0 * dropped / n:5.1f}%){mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
