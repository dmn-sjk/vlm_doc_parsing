"""Load Qwen2.5-VL with the LoRA adapter, extract JSON from a document image, print it.

    python -m vlm_doc_parsing.infer.predict --adapter $SITE_WORK_DIR/runs/.../final_adapter \
        --image receipt.png --merge

Prints the JSON on stdout and exits 0; if the output doesn't parse, prints the raw text and
exits 1. `--merge` folds the adapter into the base weights for faster inference; without it
the adapter stays a PeftModel (`disable_adapter()`, comparing adapters).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vlm_doc_parsing.data.schema import SYSTEM_PROMPT, USER_INSTRUCTION

logger = logging.getLogger(__name__)

# A ```json fence is the most common way otherwise-correct output fails json.loads.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


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


def load_model_and_processor(args: argparse.Namespace):
    base = resolve_base_model(args.adapter, args.base_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]

    logger.info("base=%s dtype=%s attn=%s", base, args.dtype, args.attn_implementation)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, dtype=dtype, attn_implementation=args.attn_implementation, device_map=args.device_map
    )

    if args.adapter:
        logger.info("attaching adapter %s", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge:
            logger.info("merge_and_unload()")
            model = model.merge_and_unload()
    else:
        logger.info("no adapter: running the base model")

    model.eval()
    # The adapter dir holds the training processor (with its pixel caps): prefer it.
    proc_src = args.processor or (
        args.adapter
        if args.adapter and (Path(args.adapter) / "preprocessor_config.json").exists()
        else base
    )
    proc_kwargs = {
        k: v for k, v in (("min_pixels", args.min_pixels), ("max_pixels", args.max_pixels)) if v
    }
    logger.info("processor from %s %s", proc_src, proc_kwargs or "")
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
    out = model.generate(  # greedy
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def parse_json(raw: str) -> dict:
    """`json.loads` the generation after stripping a fence; report how it went."""
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image", required=True, help="Path to a document image file (PNG, JPEG, ...)")
    p.add_argument(
        "--adapter", default=None, help="LoRA adapter directory (omit for the base model)"
    )
    p.add_argument("--base-model", default=None, help="Overrides adapter_config.json")
    p.add_argument("--processor", default=None, help="Overrides where the processor is loaded from")
    p.add_argument(
        "--merge", action="store_true", help="merge_and_unload() the adapter into the base weights"
    )
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
    # Logs go to stderr; stdout carries only the JSON or the raw text.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.image.lower().endswith(".pdf"):
        # Fail before minutes of model loading.
        parser.error(f"{args.image}: PDFs are not supported; convert the page to PNG first")
    sys_prompt = args.system_prompt or SYSTEM_PROMPT
    instruction = args.instruction or USER_INSTRUCTION

    image = Image.open(args.image).convert("RGB")
    logger.info("input %s %dx%d", args.image, image.size[0], image.size[1])

    model, processor = load_model_and_processor(args)
    raw = generate_json(model, processor, image, sys_prompt, instruction, args.max_new_tokens)

    result = parse_json(raw)
    if result["parse_ok"]:
        print(json.dumps(result["json"], indent=2, ensure_ascii=False))
        logger.info("parse_ok=True fenced=%s chars=%d", result["fenced"], result["raw_chars"])
        if args.out:
            Path(args.out).write_text(json.dumps(result["json"], indent=2, ensure_ascii=False))
            logger.info("written to %s", args.out)
        return 0

    print(raw)
    logger.error("parse_ok=False error=%s", result["error"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
