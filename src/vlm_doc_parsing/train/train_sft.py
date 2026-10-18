"""Config-driven SFT entrypoint: Qwen2.5-VL + LoRA + FSDP via TRL's SFTTrainer.

Run it through `accelerate launch` (`slurm/*.sbatch`), never on a login node.

    accelerate launch --config_file configs/accelerate_single.yaml \
        -m vlm_doc_parsing.train.train_sft --config configs/sft_1node.yaml \
        --set train.max_steps=2 --set 'train.output_dir=${SITE_WORK_DIR}/runs/short' \
        --set train.report_to=none

`--set` applies dotted overrides on top of the YAML. Design notes: `docs/design-notes.md`.
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Dotted override on top of the YAML, repeatable, e.g. --set train.learning_rate=2e-4",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the model and the trainer, print the trainable parameters, exit.",
    )
    return p


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# From the ForConditionalGeneration wrapper (wrapper.model.visual); transformers 5 has no
# wrapper.visual.
VISION_TOWER_PATH = "model.visual"
VISION_TOWER_EXCLUDE_RE = r"model\.visual\..*"


def load_model(cfg: Config) -> Qwen2_5_VLForConditionalGeneration:
    dtype = _DTYPES[cfg.model.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model.base_model,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        device_map=None,  # accelerate/FSDP place the model
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
    """LoRA on the text decoder only, unless `adapt_vision_tower` is set.

    The ViT's MLP also has `gate/up/down_proj` and PEFT matches `target_modules` by suffix, so
    the vision tower is excluded with a regex (the list form can't exclude a subtree).
    """
    lora = cfg.model.lora
    bad = [m for m in list(lora.target_modules) + list(lora.modules_to_save) if "lm_head" in m]
    if bad:
        raise ValueError(
            f"LoRA config targets {bad}; `lm_head` must not be adapted or saved because "
            "TRL rejects loss_type='chunked_nll' with a PEFT lm_head."
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


class LengthGroupedVisionSFTTrainer(SFTTrainer):
    """SFTTrainer whose train sampler groups by externally supplied lengths.

    Trainer strips the dataset to TRL's signature columns before building the sampler, so a
    length column can't survive. The megabatch spans all ranks, so every rank in one optimizer
    step draws from the same length bucket. See docs/design-notes.md#length-grouping.
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
        a = self.args
        megabatch = (
            a.per_device_train_batch_size * a.gradient_accumulation_steps * max(1, a.world_size)
        )
        seed = a.data_seed if a.data_seed is not None else a.seed
        # accelerate slices one sampler on every rank, so the permutation must match everywhere.
        generator = torch.Generator().manual_seed(seed)
        return LengthGroupedSampler(megabatch, lengths=self._train_lengths, generator=generator)

    def _get_eval_sampler(self, eval_dataset):
        # The inherited one also groups by length and looks for `input_ids`, which a vision
        # dataset lacks.
        if eval_dataset is None:
            return None
        return SequentialSampler(eval_dataset) if self.args.world_size <= 1 else None


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
        # transformers 5 dropped `warmup_ratio`; `warmup_steps` < 1 is read as a ratio.
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
        loss_type=t.loss_type,
        completion_only_loss=t.completion_only_loss,
        packing=t.packing,
        padding_free=t.padding_free,
        # Vision truncation is keep_start: too short a max_length cuts off the answer.
        max_length=cfg.data.max_length,
        dataset_num_proc=cfg.data.num_proc,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # TRL sets it only for transformers < 5
        train_sampling_strategy="group_by_length" if cfg.data.group_by_length else "random",
        ddp_timeout=t.ddp_timeout,
        dataloader_num_workers=t.dataloader_num_workers,
        dataloader_pin_memory=True,
    )


def save_adapter(trainer: SFTTrainer, cfg: Config, output_dir: str, resolved: dict) -> None:
    """Save the LoRA adapter under FSDP, then check it is not empty.

    `save_model` all-gathers a FULL_STATE_DICT, so every rank must enter it; rank 0 alone
    would hang or write empty shards. See docs/adr/0003-fsdp1-full-state-dict.md.
    """
    final_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_dir)  # all ranks
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
        return {**report, "ok": False}
    report["bytes"] = p.stat().st_size
    tensors = load_file(str(p))
    report["n_tensors"] = len(tensors)
    report["n_elements"] = int(sum(t.numel() for t in tensors.values()))
    # lora_B starts at zero, lora_A doesn't: all-zero tensors mean the gather saved empty shards.
    nonzero = sum(1 for t in tensors.values() if torch.count_nonzero(t).item() > 0)
    report["n_nonzero_tensors"] = nonzero
    report["ok"] = bool(report["bytes"] > 0 and len(tensors) > 0 and nonzero > 0)
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    output_dir = expand_path(cfg.train.output_dir)

    # `.device` initialises torch.distributed (with ddp_timeout) before `from_pretrained`;
    # otherwise every rank reads the full checkpoint instead of rank 0 only.
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

    # Arrow cache files are named by content fingerprint, so concurrent ranks race for one
    # path: rank 0 builds the cache, the rest read it.
    with state.main_process_first():
        ds = build_datasets(cfg.data, processor=processor, model_id=cfg.model.base_model)
        train_ds = ds[cfg.data.train_split]
        eval_ds = ds.get(cfg.data.eval_split) if cfg.train.eval_strategy != "no" else None
        lengths = None
        if cfg.data.group_by_length:
            lengths = estimate_lengths(
                train_ds, processor, min_pixels=cfg.data.min_pixels, max_pixels=cfg.data.max_pixels
            )

    if state.is_main_process:
        logger.info("train=%d eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else "-")
        if lengths is not None:
            logger.info(
                "length signal: min=%d p50=%d max=%d",
                min(lengths),
                sorted(lengths)[len(lengths) // 2],
                max(lengths),
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

    # Per rank: FSDP shards unevenly when the frozen ViT is split across ranks.
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(
            "vram rank=%d peak_allocated=%.2fGiB peak_reserved=%.2fGiB device_total=%.2fGiB "
            "headroom=%.2fGiB",
            state.process_index,
            peak_gib,
            reserved_gib,
            total_gib,
            total_gib - reserved_gib,
        )

    if eval_ds is not None:
        metrics = trainer.evaluate()
        if state.is_main_process:
            logger.info("eval %s", metrics)

    save_adapter(trainer, cfg, output_dir, resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
