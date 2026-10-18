"""Deliverable 2: load the base Qwen2.5-VL, attach (or merge) the LoRA adapter, feed it a
document image, print structured JSON, and say whether that JSON actually parses.

    python -m src.infer.predict --adapter $SITE_WORK_DIR/runs/.../final_adapter \
        --image receipt.png --merge

`--merge` calls `merge_and_unload()` (faster inference, one dense model, no adapter
bookkeeping); without it the adapter stays attached as a PeftModel, which is what you want
when you are comparing several adapters or want `disable_adapter()` available.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.data.schema import SYSTEM_PROMPT, USER_INSTRUCTION

# ```json ... ``` fences are the single most common way a VLM breaks json.loads while
# being otherwise correct. Strip them, but report that we had to (gotcha #5).
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def resolve_base_model(adapter: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if adapter:
        cfg = Path(adapter) / "adapter_config.json"
        if cfg.exists():
            name = json.loads(cfg.read_text()).get("base_model_name_or_path")
            if name:
                return name
    return "Qwen/Qwen2.5-VL-7B-Instruct"


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


def load_model_and_processor(args):
    base = resolve_base_model(args.adapter, args.base_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[load] base={base} dtype={args.dtype} attn={args.attn_implementation}", flush=True)
    # transformers 5.x: the kwarg is `dtype`; `torch_dtype` is deprecated
    # (modeling_utils.py:1490).
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )

    if args.adapter:
        print(f"[load] attaching adapter {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge:
            print("[load] merge_and_unload()", flush=True)
            model = model.merge_and_unload()
    else:
        print("[load] no adapter -- running the base model", flush=True)

    model.eval()
    # The adapter directory carries the processor that was used for training (Trainer._save
    # writes processing_class alongside it), including the min_pixels/max_pixels caps.
    # Prefer it so inference resolution matches training resolution.
    proc_src = args.processor or (
        args.adapter if args.adapter and (Path(args.adapter) / "preprocessor_config.json").exists() else base
    )
    proc_kwargs: dict[str, Any] = {}
    if args.min_pixels:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels:
        proc_kwargs["max_pixels"] = args.max_pixels
    print(f"[load] processor from {proc_src} {proc_kwargs or ''}", flush=True)
    processor = AutoProcessor.from_pretrained(proc_src, **proc_kwargs)
    return model, processor


def build_inputs(processor, image, system_prompt: str, instruction: str):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(images=[image], text=[text], return_tensors="pt")


@torch.no_grad()
def generate_json(model, processor, image, system_prompt, instruction, max_new_tokens=1024):
    inputs = build_inputs(processor, image, system_prompt, instruction)
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # extraction is not a creative task: greedy
        temperature=None,
        top_p=None,
        top_k=None,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def parse_json(raw: str) -> dict:
    """json.loads our own output and report exactly how it went."""
    result: dict[str, Any] = {"raw_chars": len(raw), "fenced": False}
    text = raw.strip()
    m = _FENCE.match(text)
    if m:
        result["fenced"] = True
        text = m.group(1)
    try:
        result["json"] = json.loads(text)
        result["parse_ok"] = True
    except json.JSONDecodeError as exc:
        result["parse_ok"] = False
        result["error"] = f"{exc.msg} at line {exc.lineno} col {exc.colno}"
    return result


# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="Path to a document image file (PNG, JPEG, ...)")
    p.add_argument("--adapter", default=None, help="LoRA adapter directory (omit for the base model)")
    p.add_argument("--base-model", default=None, help="Overrides adapter_config.json")
    p.add_argument("--processor", default=None, help="Overrides where the processor is loaded from")
    p.add_argument("--merge", action="store_true", help="merge_and_unload() the adapter into the base weights")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", default="sdpa", help="flash-attn is not installed here")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--instruction", default=None)
    p.add_argument("--out", default=None, help="Write the parsed JSON to this file")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.image.lower().endswith(".pdf"):
        # No PDF rasteriser is a dependency; fail before minutes of model loading.
        parser.error(f"{args.image}: PDFs are not supported; convert the page to PNG first")
    sys_prompt = args.system_prompt or SYSTEM_PROMPT
    instruction = args.instruction or USER_INSTRUCTION

    image = Image.open(args.image).convert("RGB")
    print(f"[input] {args.image} {image.size[0]}x{image.size[1]}", flush=True)

    model, processor = load_model_and_processor(args)
    raw = generate_json(model, processor, image, sys_prompt, instruction, args.max_new_tokens)

    result = parse_json(raw)
    print("=" * 72, flush=True)
    if result["parse_ok"]:
        print(json.dumps(result["json"], indent=2, ensure_ascii=False), flush=True)
        print("=" * 72, flush=True)
        print(f"[json] parse_ok=True fenced={result['fenced']} chars={result['raw_chars']}", flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(result["json"], indent=2, ensure_ascii=False))
            print(f"[json] written to {args.out}", flush=True)
        return 0

    print(raw, flush=True)
    print("=" * 72, flush=True)
    print(f"[json] parse_ok=False error={result['error']}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
