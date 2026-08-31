"""Score a fine-tuned adapter on CORD-v2, reporting JSON validity separately from content.

Gotcha #5: a model that extracts every field correctly but emits a trailing comma is worth
zero downstream, and a model that emits perfect `{}` scores 100% on "valid JSON". So we
report three independent numbers:

  1. parse_rate      -- fraction of generations that `json.loads` cleanly.
                        Also reported *strictly* (no markdown-fence repair) and leniently.
  2. field P/R/F1    -- multiset comparison of flattened (path, value) pairs against the
                        normalised ground truth. Computed over parseable predictions and,
                        separately, over all predictions (unparseable == zero credit),
                        because those two numbers answer different questions.
  3. norm_edit_sim   -- 1 - normalised Levenshtein distance between the canonicalised JSON
                        strings. Cheap, and it degrades gracefully when the JSON is broken,
                        so it is the only one of the three that is defined for every sample.

    python -m src.eval.evaluate --config configs/sft_multinode.yaml \
        --adapter $SCRATCH/vlm_finetuning/runs/.../final_adapter --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config, parse_override  # noqa: E402
from src.infer.predict import (  # noqa: E402
    generate_json,
    load_model_and_processor,
    load_prompt_texts,
    parse_json,
)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def flatten(obj: Any, prefix: str = "") -> Counter:
    """Flatten nested JSON into a multiset of "path=value" strings.

    List indices are dropped on purpose: CORD receipts are an unordered bag of line items,
    so `menu[0]` vs `menu[3]` should not count as a miss. Using a Counter rather than a set
    keeps duplicate line items (two identical drinks) meaningful.
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


def prf(pred: Counter, gold: Counter) -> tuple[int, int, int]:
    tp = sum((pred & gold).values())
    return tp, sum(pred.values()), sum(gold.values())


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def norm_edit_sim(a: str, b: str) -> float:
    import Levenshtein

    if not a and not b:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def gold_from_example(example: dict) -> Any:
    """Recover the target JSON from a prompt-completion row.

    The completion's assistant turn *is* the training target, so scoring against it is
    exactly scoring against what the model was taught -- no second normalisation path that
    could silently drift from Phase 1's.
    """
    completion = example["completion"]
    msg = completion[-1] if isinstance(completion, list) else completion
    content = msg["content"] if isinstance(msg, dict) else msg
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return json.loads(content)


# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="The YAML the adapter was trained with")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (omit to score the base model)")
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
    args = build_parser().parse_args(argv)

    merged: dict[str, Any] = {}
    for item in args.overrides:
        merged.update(parse_override(item))
    cfg = load_config(args.config, merged or None)

    from src.data.cord import build_datasets  # Phase 1

    split = {"train": cfg.data.train_split, "validation": cfg.data.eval_split, "test": cfg.data.test_split}[args.split]
    ds = build_datasets(cfg.data)[split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[eval] split={split} n={len(ds)}", flush=True)

    # Reuse the *inference* loader so eval cannot silently diverge from what predict.py
    # does; the resolution caps come from the same config the run trained with.
    args.min_pixels = cfg.data.min_pixels
    args.max_pixels = cfg.data.max_pixels
    model, processor = load_model_and_processor(args)
    sys_prompt, instruction = load_prompt_texts()

    n = len(ds)
    strict_ok = lenient_ok = 0
    tp_p = pred_p = gold_p = 0            # over parseable predictions only
    tp_a = pred_a = gold_a = 0            # over all predictions
    exact = 0
    edit_sims: list[float] = []
    records = []
    t0 = time.time()

    for i, example in enumerate(ds):
        raw = generate_json(
            model, processor, example["images"][0], sys_prompt, instruction, args.max_new_tokens
        )
        parsed = parse_json(raw)
        gold = gold_from_example(example)
        gold_flat = flatten(gold)
        gold_str = canonical(gold)

        # Strict = json.loads(raw) with no repair at all. Lenient = after stripping a
        # markdown fence. The gap between the two is the "model is right but formats it in
        # a code block" failure, which is trivially fixable in a serving layer.
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
            tp, np_, ng = prf(pred_flat, gold_flat)
            tp_p += tp; pred_p += np_; gold_p += ng
            tp_a += tp; pred_a += np_; gold_a += ng
            exact += int(pred_str == gold_str)
            edit_sims.append(norm_edit_sim(pred_str, gold_str))
        else:
            gold_a += sum(gold_flat.values())     # zero credit, full recall penalty
            edit_sims.append(norm_edit_sim(raw.strip(), gold_str))

        if args.dump_predictions:
            records.append({"index": i, "raw": raw, "parse_ok": parsed["parse_ok"], "gold": gold})
        if (i + 1) % 10 == 0 or i + 1 == n:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"[eval] {i + 1}/{n}  lenient_parse={lenient_ok / (i + 1):.3f}  {rate:.2f} ex/s", flush=True)

    def _f1(tp, np_, ng):
        p = tp / np_ if np_ else 0.0
        r = tp / ng if ng else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "tp": tp, "n_pred": np_, "n_gold": ng}

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
        "field_metrics_parseable_only": _f1(tp_p, pred_p, gold_p),
        "field_metrics_all": _f1(tp_a, pred_a, gold_a),
        "exact_match": exact / n if n else 0.0,
        "norm_edit_similarity": sum(edit_sims) / len(edit_sims) if edit_sims else 0.0,
        "wall_seconds": time.time() - t0,
    }

    print(json.dumps(results, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[eval] results written to {args.out}", flush=True)
    if args.dump_predictions:
        with open(args.dump_predictions, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[eval] predictions written to {args.dump_predictions}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
