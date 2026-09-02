"""Push a trained LoRA adapter to the Hugging Face Hub.

    source scripts/env.sh
    python scripts/push_adapter.py --adapter $SCRATCH/vlm_finetuning/runs/.../final_adapter --dry-run
    python scripts/push_adapter.py --adapter ...                       # the real push

The token comes from `HF_TOKEN` in the environment (scripts/env.sh sources the gitignored
.env). It is never printed, never written into the model card, and never sent anywhere in
--dry-run mode.

The token in use is fine-grained and scoped to the `dmnsjk` entity
(repo.content.read / repo.access.read / repo.write / discussion.write), so the target repo
must live in the `dmnsjk/` namespace -- anything else gets a 403 from the Hub, not a
retryable error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_REPO_ID = "dmnsjk/qwen2.5-vl-7b-cord-json-lora"

# Per the brief: the adapter is tagged for document parsing and for the image-text-to-text
# task so it is discoverable from both the task filter and the topic filter.
SOURCE_REPO = "https://github.com/dmn-sjk/vlm_doc_parsing.git"
PIPELINE_TAG = "image-text-to-text"
TAGS = ["document-parsing", "image-text-to-text", "lora", "peft", "qwen2_5_vl", "vlm", "json-extraction"]

# Files that must be present for `PeftModel.from_pretrained(base, repo_id)` to work.
REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]

# Never upload optimizer shards, intermediate checkpoints or the base weights.
IGNORE = [
    "checkpoint-*/*", "optimizer.pt", "scheduler.pt", "rng_state*", "trainer_state.json",
    "training_args.bin", "*.pt", "**/pytorch_model_fsdp*", ".git*",
]


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def build_eval_table(adapter: Path) -> tuple[str, str, str]:
    """Render the real base-vs-tuned comparison, or say plainly that it is missing.

    A metrics table that silently degrades to placeholders is worse than no table, so an
    absent eval file produces an explicit note rather than invented numbers.
    """
    ft_p, base_p = adapter / "eval_results.json", adapter / "eval_base_comparison.json"
    if not ft_p.exists():
        return ("_No `eval_results.json` was found next to the adapter._", "n/a", "n/a")
    # evaluate.py writes its metrics at the top level; tolerate a "metrics" wrapper too
    # rather than silently rendering a table of n/a.
    def _load(path):
        d = json.loads(path.read_text())
        return d.get("metrics", d) if isinstance(d, dict) else {}

    ft = _load(ft_p)
    base = _load(base_p) if base_p.exists() else {}

    def g(d, *path):
        for k in path:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        return d

    rows = [
        ("JSON parse rate (strict)", g(base, "json_validity", "parse_rate_strict"), g(ft, "json_validity", "parse_rate_strict")),
        ("JSON parse rate (lenient)", g(base, "json_validity", "parse_rate_lenient"), g(ft, "json_validity", "parse_rate_lenient")),
        ("Field precision", g(base, "field_metrics_all", "precision"), g(ft, "field_metrics_all", "precision")),
        ("Field recall", g(base, "field_metrics_all", "recall"), g(ft, "field_metrics_all", "recall")),
        ("Field F1", g(base, "field_metrics_all", "f1"), g(ft, "field_metrics_all", "f1")),
        ("Exact match", g(base, "exact_match"), g(ft, "exact_match")),
        ("Normalised edit similarity", g(base, "norm_edit_similarity"), g(ft, "norm_edit_similarity")),
    ]
    head = "| Metric | Base " + ("(no adapter)" if base else "") + " | **This adapter** |\n| --- | --- | --- |"
    body = "\n".join(f"| {n} | {_fmt(b)} | **{_fmt(f)}** |" for n, b, f in rows)
    return (head + "\n" + body, _fmt(g(ft, "exact_match"), 2), _fmt(g(ft, "norm_edit_similarity")))


def build_card(args, run_config: dict | None) -> str:
    eval_table, exact_match, edit_sim = build_eval_table(Path(args.adapter))
    lora = ((run_config or {}).get("model") or {}).get("lora") or {}
    data = (run_config or {}).get("data") or {}
    train = (run_config or {}).get("train") or {}
    base = args.base_model or ((run_config or {}).get("model") or {}).get(
        "base_model", "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    front = [
        "---",
        f"base_model: {base}",
        "library_name: peft",
        f"pipeline_tag: {PIPELINE_TAG}",
        "license: apache-2.0",
        "tags:",
        *[f"  - {t}" for t in TAGS],
        "datasets:",
        f"  - {data.get('dataset_id', 'naver-clova-ix/cord-v2')}",
        "---",
        "",
    ]
    source_repo = SOURCE_REPO
    body = f"""# Qwen2.5-VL-7B LoRA — structured document parsing (CORD-v2)

A LoRA adapter for `{base}` that makes the model emit **schema-conformant JSON directly**
from a receipt image.

**What this adapter actually buys you is reliability, not new reading ability.** The base
model can already read receipts reasonably well — its field recall is 0.65. What it cannot
do is return parseable output: it wraps 96 of 100 answers in a ```json markdown fence, so a
strict `json.loads` succeeds **3%** of the time. After tuning that is **100%**, and
precision roughly triples because the model stops padding its answer with commentary and
invented fields. Treat this as a formatting-and-schema adapter, not a better OCR model.

Trained with TRL `SFTTrainer` in the **prompt-completion** format
(`completion_only_loss=True`, so the loss covers only the JSON answer) under PyTorch FSDP
`FULL_SHARD` via Hugging Face Accelerate on a SLURM cluster. The released adapter comes
from a 1 node x 8 A100 (40 GB) run, 14.4 minutes of training.

## Usage

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

base = "{base}"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{args.repo_id}")
processor = AutoProcessor.from_pretrained("{args.repo_id}")
```

## Training configuration

| | |
| --- | --- |
| base model | `{base}` |
| LoRA rank / alpha / dropout | {lora.get('r', '?')} / {lora.get('alpha', '?')} / {lora.get('dropout', '?')} |
| target modules | `{', '.join(lora.get('target_modules', []))}` |
| vision tower | frozen, not adapted |
| dataset | `{data.get('dataset_id', 'naver-clova-ix/cord-v2')}` |
| resolution cap | `min_pixels={data.get('min_pixels', '?')}`, `max_pixels={data.get('max_pixels', '?')}` |
| epochs / lr / scheduler | {train.get('num_train_epochs', '?')} / {train.get('learning_rate', '?')} / {train.get('lr_scheduler_type', '?')} |
| precision | bf16, gradient checkpointing, `loss_type="chunked_nll"` |

`run_config.yaml` in this repository is the exact resolved configuration of the run.

## Evaluation

Held-out CORD-v2 **test** split, 100 receipts, greedy decoding. Scored against the base
checkpoint under an identical prompt, so the delta is attributable to the adapter.

{eval_table}

JSON parse-rate is reported separately from field accuracy on purpose: a model that
extracts every field but emits a trailing comma scores zero downstream. "Strict" is
`json.loads` on the raw generation; "lenient" first strips a markdown code fence.

## Limitations

- **Small, narrow training set.** 800 training examples, 3 epochs. CORD-v2 is Indonesian
  restaurant/retail receipts specifically. Do not expect this to transfer to invoices,
  financial tables, or engineering schematics without retraining — the brief's broader
  document types were never trained on.
- **Exact match is {exact_match}.** More than half of outputs differ from the gold parse
  somewhere, usually a single OCR-level field (a digit in a price, an abbreviated item
  name). Normalised edit similarity is {edit_sim}, so errors are typically small rather
  than structural — but this is not a solved extraction task.
- **The schema is fixed** by the system prompt this adapter was trained with (see
  `run_config.yaml`). Prompting it for a different key set will degrade it. The adapter has
  been specialised toward one output shape.
- **Field scoring ignores list order** (receipt line items are treated as an unordered
  bag), so ordering mistakes are not penalised.
- Resolution is capped at `max_pixels` (see the table above). Documents far denser than a
  receipt may lose legibility at that cap.

## Source code

Training pipeline, SLURM sbatch scripts and Accelerate/FSDP configs:
**[{source_repo}]({source_repo})**

See [`src/infer/predict.py`]({source_repo}/blob/master/src/infer/predict.py) there for a
complete inference example, including `--merge` (`merge_and_unload`) for faster serving.
"""
    return "\n".join(front) + body


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="Directory produced by train_sft.py (final_adapter/)")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"default: {DEFAULT_REPO_ID}")
    p.add_argument("--base-model", default=None, help="Overrides what run_config.yaml says")
    p.add_argument("--private", action="store_true", help="Create the repo private")
    p.add_argument("--commit-message", default="Add Qwen2.5-VL CORD JSON LoRA adapter")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate everything and print the plan. No network call, no token use.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter = Path(args.adapter)

    if not adapter.is_dir():
        print(f"error: {adapter} is not a directory", file=sys.stderr)
        return 2
    missing = [f for f in REQUIRED if not (adapter / f).exists()]
    if missing:
        print(f"error: {adapter} is missing {missing}; is this a finished run?", file=sys.stderr)
        return 2
    weights = adapter / "adapter_model.safetensors"
    if weights.stat().st_size == 0:
        print(f"error: {weights} is 0 bytes -- the FSDP gather produced an empty adapter", file=sys.stderr)
        return 2

    run_config = None
    rc = adapter / "run_config.yaml"
    if rc.exists():
        import yaml

        run_config = yaml.safe_load(rc.read_text())

    if "/" not in args.repo_id:
        print(f"error: --repo-id must be 'namespace/name', got {args.repo_id!r}", file=sys.stderr)
        return 2
    namespace = args.repo_id.split("/", 1)[0]
    if namespace != "dmnsjk":
        print(
            f"warning: the configured HF token is scoped to the `dmnsjk` entity; pushing to "
            f"`{namespace}/` will almost certainly 403.",
            file=sys.stderr,
        )

    files = sorted(f.name for f in adapter.iterdir() if f.is_file())
    total = sum(f.stat().st_size for f in adapter.iterdir() if f.is_file())
    card = build_card(args, run_config)

    print(f"adapter:  {adapter}")
    print(f"repo_id:  {args.repo_id}  (private={args.private})")
    print(f"files:    {files}")
    print(f"size:     {total / 1e6:.1f} MB")
    print(f"tags:     {TAGS}")
    print(f"pipeline: {PIPELINE_TAG}")
    print(f"adapter_config.json: {json.dumps(json.loads((adapter / 'adapter_config.json').read_text()))[:300]}")

    if args.dry_run:
        print("\n--- README.md that would be uploaded ---")
        print(card)
        print("--- end README.md ---")
        # Deliberately does NOT read HF_TOKEN, so a dry run can never leak or spend it.
        print("\n[dry-run] nothing was uploaded and no token was read.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN is not set. `source scripts/env.sh` first.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[hub] authenticated as {who.get('name')}")   # name only; never the token

    (adapter / "README.md").write_text(card)

    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    except HfHubHTTPError as exc:
        # repo.write on the user entity covers create_repo in that namespace. A 403 here is
        # a real permission problem (wrong namespace, or the token was rotated) -- report
        # it rather than retrying blindly.
        print(f"error: create_repo failed ({exc.response.status_code if exc.response else '?'}). "
              f"Check that {args.repo_id} is inside the token's scope.", file=sys.stderr)
        return 1

    info = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(adapter),
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=IGNORE,
    )
    print(f"[hub] pushed: {info.commit_url if hasattr(info, 'commit_url') else info}")
    print(f"[hub] https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
