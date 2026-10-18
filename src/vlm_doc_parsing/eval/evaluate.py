"""Score an adapter on CORD-v2, reporting JSON validity separately from field accuracy.

A trailing comma scores zero downstream, yet `{}` parses 100% of the time, hence three metrics:

  1. parse_rate     -- fraction of generations that `json.loads`: strict (raw) and lenient
                       (after stripping a markdown fence).
  2. field P/R/F1   -- multiset match of flattened `path=value` pairs against the gold
                       target, over parseable predictions and over all (unparseable = 0).
  3. norm_edit_sim  -- 1 - normalised Levenshtein distance of the canonical JSON strings;
                       defined even when the JSON is broken.

    python -m vlm_doc_parsing.eval.evaluate --config configs/sft_multinode.yaml \
        --adapter $SITE_WORK_DIR/runs/.../final_adapter --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import Levenshtein

from vlm_doc_parsing.config import load_config
from vlm_doc_parsing.data.cord import build_datasets
from vlm_doc_parsing.data.schema import SYSTEM_PROMPT, USER_INSTRUCTION
from vlm_doc_parsing.infer.predict import generate_json, load_model_and_processor, parse_json

logger = logging.getLogger(__name__)


def flatten(obj: Any, prefix: str = "") -> Counter:
    """Flatten nested JSON into a multiset of "path=value" strings.

    List indices are dropped (line items are an unordered bag); a Counter keeps duplicates.
    """
    out: Counter = Counter()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for v in obj:
            out += flatten(v, prefix)
    else:
        value = "" if obj is None else str(obj).strip()
        out[f"{prefix}={value}"] += 1
    return out


@dataclass
class FieldCounts:
    """Running multiset-match counts; `metrics()` turns them into P/R/F1."""

    tp: int = 0
    n_pred: int = 0
    n_gold: int = 0

    def add(self, pred: Counter, gold: Counter) -> None:
        self.tp += sum((pred & gold).values())
        self.n_pred += sum(pred.values())
        self.n_gold += sum(gold.values())

    def metrics(self) -> dict[str, float | int]:
        p = self.tp / self.n_pred if self.n_pred else 0.0
        r = self.tp / self.n_gold if self.n_gold else 0.0
        return {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": self.tp,
            "n_pred": self.n_pred,
            "n_gold": self.n_gold,
        }


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def norm_edit_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def gold_from_example(example: dict) -> Any:
    """The training target itself, so eval can't drift from what the model was taught."""
    return json.loads(example["completion"][-1]["content"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="The YAML the adapter was trained with")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    p.add_argument(
        "--adapter", default=None, help="LoRA adapter dir (omit to score the base model)"
    )
    p.add_argument("--base-model", default=None)
    p.add_argument("--processor", default=None)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--limit", type=int, default=None, help="Score only the first N examples")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--dump-predictions", default=None, help="JSONL of every raw generation")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config, args.overrides)

    split = {
        "train": cfg.data.train_split,
        "validation": cfg.data.eval_split,
        "test": cfg.data.test_split,
    }[args.split]
    ds = build_datasets(cfg.data, processor=None, model_id=cfg.model.base_model)[split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    logger.info("split=%s n=%d", split, len(ds))

    # predict's loader, with the resolution caps the run trained with.
    args.min_pixels = cfg.data.min_pixels
    args.max_pixels = cfg.data.max_pixels
    model, processor = load_model_and_processor(args)

    n = len(ds)
    strict_ok = lenient_ok = 0
    parseable = FieldCounts()  # parseable predictions only
    overall = FieldCounts()  # all predictions
    exact = 0
    edit_sims: list[float] = []
    records = []
    t0 = time.time()

    for i, example in enumerate(ds):
        image = example["images"][0]
        raw = generate_json(
            model, processor, image, SYSTEM_PROMPT, USER_INSTRUCTION, args.max_new_tokens
        )
        parsed = parse_json(raw)
        gold = gold_from_example(example)
        gold_flat = flatten(gold)
        gold_str = canonical(gold)

        # Strict: no repair. Lenient: after stripping a markdown fence.
        strict = False
        try:
            json.loads(raw)
            strict = True
        except json.JSONDecodeError:
            pass
        strict_ok += int(strict)
        lenient_ok += int(parsed["parse_ok"])

        if parsed["parse_ok"]:
            pred_flat = flatten(parsed["json"])
            pred_str = canonical(parsed["json"])
            parseable.add(pred_flat, gold_flat)
            overall.add(pred_flat, gold_flat)
            exact += int(pred_str == gold_str)
            edit_sims.append(norm_edit_sim(pred_str, gold_str))
        else:
            overall.add(Counter(), gold_flat)  # zero credit, full recall penalty
            edit_sims.append(norm_edit_sim(raw.strip(), gold_str))

        if args.dump_predictions:
            records.append({"index": i, "raw": raw, "parse_ok": parsed["parse_ok"], "gold": gold})
        if (i + 1) % 10 == 0 or i + 1 == n:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            logger.info(
                "%d/%d  lenient_parse=%.3f  %.2f ex/s", i + 1, n, lenient_ok / (i + 1), rate
            )

    results = {
        "n": n,
        "split": split,
        "adapter": args.adapter,
        "base_model": args.base_model,
        "merged": bool(args.merge),
        "json_validity": {
            "parse_rate_strict": strict_ok / n if n else 0.0,
            "parse_rate_lenient": lenient_ok / n if n else 0.0,
            "n_strict_ok": strict_ok,
            "n_lenient_ok": lenient_ok,
            "n_fence_repaired": lenient_ok - strict_ok,
        },
        "field_metrics_parseable_only": parseable.metrics(),
        "field_metrics_all": overall.metrics(),
        "exact_match": exact / n if n else 0.0,
        "norm_edit_similarity": sum(edit_sims) / len(edit_sims) if edit_sims else 0.0,
        "wall_seconds": time.time() - t0,
    }

    print(json.dumps(results, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("results written to %s", args.out)
    if args.dump_predictions:
        with open(args.dump_predictions, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("predictions written to %s", args.dump_predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
