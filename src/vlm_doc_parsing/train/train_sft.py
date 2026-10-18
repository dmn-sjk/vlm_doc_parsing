"""Config-driven SFT entrypoint: Qwen2.5-VL + LoRA + FSDP via TRL's SFTTrainer.

Run it through `accelerate launch` (see `configs/accelerate_*.yaml` and `slurm/*.sbatch`);
never directly on a login node.

    accelerate launch --config_file configs/accelerate_single.yaml \
        -m vlm_doc_parsing.train.train_sft --config configs/sft_1node.yaml \
        --set train.max_steps=2 --set 'train.output_dir=${SITE_WORK_DIR}/runs/short' \
        --set train.report_to=none

Everything that defines an experiment lives in the YAML (see `src/vlm_doc_parsing/config.py`);
`--set` applies dotted overrides on top so a committed config is never edited in place.

Design reasoning and library citations live in `docs/design-notes.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import PartialState
from peft import LoraConfig as PeftLoraConfig
from safetensors.torch import load_file
from torch.utils.data import SequentialSampler
from transformers import Qwen2_5_VLForConditionalGeneration, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler
from trl import SFTConfig, SFTTrainer

from vlm_doc_parsing.config import Config, expand_path, load_config
from vlm_doc_parsing.data.cord import build_datasets
from vlm_doc_parsing.data.grouping import estimate_lengths
from vlm_doc_parsing.data.resolution import build_processor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Dotted override applied on top of the YAML, repeatable. "
        "e.g. --set train.learning_rate=2e-4 --set data.max_train_samples=16",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config, build the model and the trainer, print the trainable "
        "parameter count, then exit without training.",
    )
    return p


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Read out of transformers 5.16.1:
#   models/qwen2_5_vl/modeling_qwen2_5_vl.py:1268  Qwen2_5_VLForConditionalGeneration
#   models/qwen2_5_vl/modeling_qwen2_5_vl.py:330   _no_split_modules =
#       ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
# The vision tower lives at `model.model.visual` (a Qwen2_5_VisionTransformerPretrainedModel);
# `model.visual` does NOT exist on the ForConditionalGeneration wrapper in v5.
VISION_TOWER_PATH = "model.visual"
VISION_TOWER_EXCLUDE_RE = r"model\.visual\..*"


def load_model(cfg: Config) -> Qwen2_5_VLForConditionalGeneration:
    dtype = _DTYPES[cfg.model.dtype]
    # transformers 5.x renamed `torch_dtype` -> `dtype`; the old name still works but logs a
    # deprecation warning (modeling_utils.py:1490, :4131).
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model.base_model,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        # Distributed training requires device_map=None; accelerate/FSDP place the model.
        device_map=None,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if cfg.model.freeze_vision_tower:
        tower = model.get_submodule(VISION_TOWER_PATH)
        n = 0
        for p in tower.parameters():
            p.requires_grad_(False)
            n += p.numel()
        if PartialState().is_main_process:
            logger.info("froze vision tower (%s): %.1fM params", VISION_TOWER_PATH, n / 1e6)
    return model


def build_peft_config(cfg: Config) -> PeftLoraConfig:
    """LoRA config for the text decoder only (unless data says otherwise).

    Two traps, both verified against the real module tree:

    1. `Qwen2_5_VLVisionBlock.mlp` is a `Qwen2_5_VLMLP` and therefore also has
       `gate_proj`/`up_proj`/`down_proj` (modeling_qwen2_5_vl.py:85, :294). PEFT matches
       `target_modules` by name suffix, so the default target list silently adapts 96
       modules inside the frozen ViT. `exclude_modules` must be the *regex* form -- the
       list form only matches an exact key or a `.suffix`
       (peft/tuners/tuners_utils.py:1907-1914), which cannot exclude a subtree.
    2. `lm_head` must never be adapted: TRL refuses `loss_type="chunked_nll"` when the
       output embedding is a PEFT tuner layer (trl/trainer/sft_trainer.py:1333).
    """
    lora = cfg.model.lora
    bad = [m for m in list(lora.target_modules) + list(lora.modules_to_save) if "lm_head" in m]
    if bad:
        raise ValueError(
            f"LoRA config targets {bad}; `lm_head` must not be adapted or saved because "
            "loss_type='chunked_nll' rejects it (trl/trainer/sft_trainer.py:1333)."
        )

    kwargs: dict[str, Any] = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=list(lora.target_modules),
        modules_to_save=list(lora.modules_to_save) or None,
        task_type="CAUSAL_LM",
    )
    if not lora.adapt_vision_tower:
        kwargs["exclude_modules"] = VISION_TOWER_EXCLUDE_RE
    return PeftLoraConfig(**kwargs)


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class LengthGroupedVisionSFTTrainer(SFTTrainer):
    """SFTTrainer with a length-grouped train sampler driven by explicit lengths.

    `TrainingArguments.train_sampling_strategy="group_by_length"` reads the lengths out
    of `args.length_column_name` on the dataset (transformers/trainer.py:1036-1044) --
    but `Trainer._get_dataloader` strips the dataset to the signature columns *before*
    handing it to the sampler factory (trainer.py:985 then :1003), and TRL's vision
    signature columns are only ["messages","prompt","completion","image","images"]
    (trl/trainer/sft_trainer.py:1683). A precomputed length column therefore cannot
    survive, so we pass the lengths in from the outside instead.

    Row order is stable: vision datasets take the `_skip_prepare_dataset` path
    (sft_trainer.py:1270-1275), so `self.train_dataset` is exactly the dataset we were
    handed, and `_remove_unused_columns` only drops columns. We still assert on length.

    The megabatch is `per_device_bs * grad_accum * world_size` rather than
    transformers' `train_batch_size * grad_accum` (which is per-device under DDP/FSDP).
    Grouping only helps if every rank in the *same* optimizer step draws from the same
    length bucket -- otherwise the ranks still stall against each other at the
    all-reduce, which is the actual failure mode in gotcha #2.
    """

    def __init__(self, *args, train_lengths: list[int] | None = None, **kwargs):
        self._train_lengths = train_lengths
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if self._train_lengths is None or self.args.train_sampling_strategy != "group_by_length":
            return super()._get_train_sampler(train_dataset)
        if len(self._train_lengths) != len(train_dataset):
            raise ValueError(
                f"length signal has {len(self._train_lengths)} entries but the training "
                f"dataset has {len(train_dataset)}; refusing to mis-group batches."
            )
        megabatch = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * max(1, self.args.world_size)
        )
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        # Explicit generator: every rank iterates the *same* sampler and accelerate's
        # BatchSamplerShard slices it, so the permutation must be identical everywhere.
        return LengthGroupedSampler(
            megabatch,
            lengths=self._train_lengths,
            generator=torch.Generator().manual_seed(seed),
        )

    def _get_eval_sampler(self, eval_dataset):
        # The inherited implementation branches on `train_sampling_strategy` too
        # (trainer.py:1064) and would try to infer lengths from an "input_ids" key that
        # a vision dataset does not have, raising in LengthGroupedSampler.__init__
        # (trainer_pt_utils.py:540). Evaluation is small; plain sequential order is fine.
        if eval_dataset is None:
            return None
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        return None


def build_sft_config(cfg: Config, output_dir: str) -> SFTConfig:
    t = cfg.train
    return SFTConfig(
        output_dir=output_dir,
        run_name=t.run_name,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        # transformers 5.x dropped `warmup_ratio`; `warmup_steps` now doubles as a ratio
        # when < 1 (training_args.py:2119-2121).
        warmup_steps=t.warmup_ratio,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        bf16=t.bf16,
        seed=t.seed,
        report_to=t.report_to,
        # --- SFT / VLM specifics -------------------------------------------------------
        loss_type=t.loss_type,            # chunked_nll keeps peak activations off the vocab dim
        completion_only_loss=t.completion_only_loss,
        packing=t.packing,                # both rejected for vision datasets
        padding_free=t.padding_free,
        max_length=cfg.data.max_length,   # collator truncates to this (keep_start only)
        dataset_num_proc=cfg.data.num_proc,
        # --- memory --------------------------------------------------------------------
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        # transformers 5.16.1 already defaults to non-reentrant when the kwargs dict is
        # None (modeling_utils.py:3205), but TRL only injects it for transformers < 5
        # (sft_trainer.py:1350), so state it explicitly.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # --- batching ------------------------------------------------------------------
        train_sampling_strategy="group_by_length" if cfg.data.group_by_length else "random",
        # --- distributed ---------------------------------------------------------------
        ddp_timeout=t.ddp_timeout,
        dataloader_num_workers=t.dataloader_num_workers,
        dataloader_pin_memory=True,
    )


# --------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------


def save_adapter(trainer: SFTTrainer, cfg: Config, output_dir: str, resolved: dict) -> None:
    """Save the LoRA adapter correctly under FSDP, then prove it is not empty.

    `Trainer.save_model` is *collective* under FSDP: every rank must enter it, because
    `accelerator.get_state_dict` runs `FSDP.state_dict_type(..., FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True))` which is an all-gather
    (accelerate/accelerator.py:4060-4067). Only rank 0 then writes
    (transformers/trainer.py:3826-3830). Calling `save_pretrained` yourself on rank 0 only
    is the classic footgun: the other ranks never join the gather, so rank 0 either hangs
    or writes zero-sized shards.

    It also means SHARDED_STATE_DICT would silently write *nothing at all* here -- the
    `"FULL_STATE_DICT" in str(...)` guard at trainer.py:3827 has no else branch. That is
    why `configs/accelerate_fsdp.yaml` pins `fsdp_state_dict_type: FULL_STATE_DICT`.
    """
    final_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_dir)          # all ranks
    trainer.accelerator.wait_for_everyone()

    if not trainer.accelerator.is_main_process:
        return

    Path(final_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(final_dir, "run_config.yaml"), "w") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)

    report = verify_adapter(final_dir)
    logger.info("adapter check:\n%s", json.dumps({"adapter_check": report}, indent=2))
    if not report["ok"]:
        raise RuntimeError(f"adapter at {final_dir} looks empty/degenerate: {report}")
    logger.info("adapter written to %s", final_dir)


def verify_adapter(path: str) -> dict:
    """Open the saved adapter and check it actually contains trained weights."""
    p = Path(path) / "adapter_model.safetensors"
    report: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        report["ok"] = False
        return report
    report["bytes"] = p.stat().st_size
    tensors = load_file(str(p))
    report["n_tensors"] = len(tensors)
    report["n_elements"] = int(sum(t.numel() for t in tensors.values()))
    # lora_B is zero-initialised; lora_A is not. If *every* tensor is all-zero the FSDP
    # gather produced empty shards rather than real weights.
    nonzero = sum(1 for t in tensors.values() if torch.count_nonzero(t).item() > 0)
    report["n_nonzero_tensors"] = nonzero
    report["ok"] = bool(report["bytes"] > 0 and len(tensors) > 0 and nonzero > 0)
    return report


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    output_dir = expand_path(cfg.train.output_dir)

    # Build the training args *first* and touch `.device`: that constructs accelerate's
    # PartialState (with ddp_timeout, which a bare `PartialState()` here would drop) and
    # initialises torch.distributed. `from_pretrained` only takes the
    # rank-0-reads-the-checkpoint path when `is_fsdp_enabled()` is true, and that helper
    # requires torch.distributed to already be initialised
    # (transformers/distributed/fsdp.py:41-47 -> modeling_utils.py:4790). Loading the
    # model before this point makes all 16 ranks read 16 GB of safetensors each.
    sft_args = build_sft_config(cfg, output_dir)
    device = sft_args.device
    state = PartialState()

    resolved = dataclasses.asdict(cfg)
    resolved["_provenance"] = {
        "config_file": str(Path(args.config).resolve()),
        "overrides": list(args.overrides),
        "output_dir": output_dir,
        "world_size": state.num_processes,
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    # Global rank 0 only: `state.print` would repeat these once per node.
    if state.is_main_process:
        logger.info("resolved config:\n%s", yaml.safe_dump(resolved, sort_keys=False).rstrip())
        logger.info(
            "world_size=%d device=%s fsdp=%s cpu_ram_efficient_loading=%s",
            state.num_processes,
            device,
            os.environ.get("ACCELERATE_USE_FSDP", "false"),
            os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "false"),
        )

    processor = build_processor(cfg.data, model_id=cfg.model.base_model)

    # Every rank builds the dataset, and both `build_datasets` and `estimate_lengths` go
    # through `datasets.map`/`cast_column`, which write Arrow cache files derived from a
    # content fingerprint. All 8 ranks therefore race for the *same* path: one finishes and
    # renames its temp file out from under the others, and the losers die with
    #   FileNotFoundError: .../cache-1ca3ca1e8f18eeab.arrow
    # (observed in job 3108496). `main_process_first` lets rank 0 build and cache, then
    # releases the rest to hit the completed cache instead of regenerating it.
    with state.main_process_first():
        # The trainer's own processor, so the max_length filter measures the collator's
        # tokenisation and no second copy is loaded.
        ds = build_datasets(cfg.data, processor=processor, model_id=cfg.model.base_model)
        train_ds = ds[cfg.data.train_split]
        eval_ds = ds.get(cfg.data.eval_split) if cfg.train.eval_strategy != "no" else None
        lengths = (
            estimate_lengths(
                train_ds,
                processor,
                min_pixels=cfg.data.min_pixels,
                max_pixels=cfg.data.max_pixels,
            )
            if cfg.data.group_by_length
            else None
        )

    if state.is_main_process:
        logger.info("train=%d eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else "-")
        if lengths is not None:
            logger.info(
                "length signal: min=%d p50=%d max=%d",
                min(lengths), sorted(lengths)[len(lengths) // 2], max(lengths),
            )

    # SFTTrainer initialises LoRA before Trainer calls set_seed, so seed here or the adapter
    # init depends on whatever consumed the RNG earlier.
    set_seed(cfg.train.seed)
    model = load_model(cfg)
    peft_config = build_peft_config(cfg)

    trainer = LengthGroupedVisionSFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config,
        train_lengths=lengths,
    )

    if state.is_main_process:
        trainer.model.print_trainable_parameters()
        adapted = [n for n, _ in trainer.model.named_modules() if n.endswith(".lora_A")]
        in_vision = [n for n in adapted if ".visual." in n]
        logger.info("adapted modules: %d (inside vision tower: %d)", len(adapted), len(in_vision))

    if args.dry_run:
        if state.is_main_process:
            logger.info("dry run: trainer built; exiting before train()")
        return 0

    trainer.train()

    # Peak VRAM per rank. The cluster's A100s are the 40 GB SXM4 part, not the 80 GB one, so
    # headroom is the number that decides whether max_pixels / batch size are safe to
    # scale. Reported per rank because FSDP shards unevenly when the wrap policy splits
    # a frozen ViT across ranks.
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(
            "vram rank=%d peak_allocated=%.2fGiB peak_reserved=%.2fGiB device_total=%.2fGiB "
            "headroom=%.2fGiB",
            state.process_index, peak_gib, reserved_gib, total_gib, total_gib - reserved_gib,
        )

    if eval_ds is not None:
        metrics = trainer.evaluate()
        if state.is_main_process:
            logger.info("eval %s", metrics)

    save_adapter(trainer, cfg, output_dir, resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
