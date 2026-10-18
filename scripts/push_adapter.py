"""Push the trained LoRA adapter to the Hugging Face Hub, with model_card/model_card.md as its README.

    source scripts/env.sh
    python scripts/push_adapter.py --adapter $SITE_WORK_DIR/runs/.../final_adapter --dry-run
    python scripts/push_adapter.py --adapter ...                       # the real push

The token comes from `HF_TOKEN` (scripts/env.sh sources the gitignored .env); it is never
printed, and --dry-run never reads it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

DEFAULT_REPO_ID = "dmnsjk/qwen2.5-vl-7b-cord-json-lora"
MODEL_CARD = Path(__file__).resolve().parent.parent / "model_card" / "model_card.md"

# Files that must be present for `PeftModel.from_pretrained(base, repo_id)` to work.
REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]

# Never upload optimizer shards, intermediate checkpoints or the base weights.
IGNORE = [
    "checkpoint-*/*",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*",
    "trainer_state.json",
    "training_args.bin",
    "*.pt",
    "**/pytorch_model_fsdp*",
    ".git*",
]
# PEFT's save_pretrained writes its own README.md; the model card replaces it.
UPLOAD_IGNORE = IGNORE + ["README.md"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--adapter", required=True, help="Directory produced by train_sft.py (final_adapter/)"
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target Hub model repo")
    p.add_argument("--private", action="store_true", help="Create the repo private")
    p.add_argument("--commit-message", default="Add Qwen2.5-VL CORD JSON LoRA adapter")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate everything and print the plan. No network call, no token use.",
    )
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
        print(
            f"error: {weights} is 0 bytes -- the FSDP gather produced an empty adapter",
            file=sys.stderr,
        )
        return 2
    if not MODEL_CARD.is_file():
        print(f"error: model card not found at {MODEL_CARD}", file=sys.stderr)
        return 2

    uploaded = [
        f
        for f in adapter.iterdir()
        if f.is_file() and not any(fnmatch.fnmatch(f.name, pat) for pat in UPLOAD_IGNORE)
    ]
    files = sorted(f.name for f in uploaded)
    total = sum(f.stat().st_size for f in uploaded)

    print(f"adapter:  {adapter}")
    print(f"repo_id:  {args.repo_id}  (private={args.private})")
    print(f"card:     {MODEL_CARD} -> README.md")
    print(f"files:    {files}")
    print(f"size:     {total / 1e6:.1f} MB")
    print(
        f"adapter_config.json: {json.dumps(json.loads((adapter / 'adapter_config.json').read_text()))[:300]}"
    )

    if args.dry_run:
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
    print(f"[hub] authenticated as {who.get('name')}")  # name only; never the token

    try:
        api.create_repo(
            repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True
        )
    except HfHubHTTPError as exc:
        print(
            f"error: create_repo failed ({exc.response.status_code if exc.response else '?'}). "
            f"Check that {args.repo_id} is inside the token's scope.",
            file=sys.stderr,
        )
        return 1

    info = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(adapter),
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=UPLOAD_IGNORE,
    )
    print(f"[hub] pushed: {info.commit_url if hasattr(info, 'commit_url') else info}")
    api.upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )
    print(f"[hub] https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
