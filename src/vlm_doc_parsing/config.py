"""Shared config schema.

This module is the contract between the data pipeline (Phase 1) and the training /
inference entrypoints (Phase 2). Experiments are reproducible from a committed YAML +
sbatch pair, so every knob lives here rather than in an edited-in-place script.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset_id: str = "naver-clova-ix/cord-v2"
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"

    # Cap on the number of examples per split (None = all). For short debug runs.
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Dynamic-resolution caps handed to Qwen2_5_VLProcessor. Visual tokens are the
    # dominant term in sequence length, so this is the primary OOM lever.
    # 28*28 = one visual patch after the 2x2 merge; keep both multiples of that.
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1024 * 28 * 28

    # Drop examples whose estimated visual+text token count exceeds this.
    max_length: int = 4096

    # Group samples of similar estimated visual-token count into the same batch so
    # ranks stay balanced at the all-reduce.
    group_by_length: bool = True

    num_proc: int = 8
    seed: int = 42


@dataclass
class LoraConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    # Attention + MLP projections of the text decoder. NOTE: never include `lm_head`
    # here -- TRL refuses loss_type="chunked_nll" when lm_head is wrapped by a PEFT
    # adapter (trl/trainer/sft_trainer.py:1333).
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    # When False, the vision tower is excluded from adaptation and stays frozen.
    adapt_vision_tower: bool = False
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    attn_implementation: str = "sdpa"  # "flash_attention_2" if the wheel is available
    dtype: str = "bfloat16"
    freeze_vision_tower: bool = True
    gradient_checkpointing: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class TrainConfig:
    output_dir: str = "${SITE_WORK_DIR}/runs/default"
    run_name: str = "qwen25vl-cord-lora"

    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    logging_steps: int = 1
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3

    bf16: bool = True
    seed: int = 42
    report_to: str = "wandb"

    # SFTConfig specifics. chunked_nll drops -100 positions before the lm_head matmul,
    # which is what keeps peak activation memory off the vocab dimension.
    loss_type: str = "chunked_nll"
    completion_only_loss: bool = True
    packing: bool = False        # unsupported for vision datasets
    padding_free: bool = False   # unsupported for vision datasets


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _from_dict(cls, d: dict[str, Any]):
    """Recursively build a dataclass, rejecting unknown keys loudly."""
    kwargs = {}
    # `from __future__ import annotations` makes `f.type` a string; resolve the real types.
    types = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name, value in d.items():
        ftype = types[name]
        if is_dataclass(ftype):
            if not isinstance(value, dict):
                raise ValueError(
                    f"{cls.__name__}.{name} is a section; expected a mapping, got {value!r}"
                )
            value = _from_dict(ftype, value)
        elif ftype is bool and isinstance(value, str):
            # `parse_override` deliberately keeps `no`/`yes`/`on`/`off` as strings so that
            # string-valued options like `eval_strategy=no` survive. For a field genuinely
            # declared `bool`, convert back -- otherwise `group_by_length=no` would arrive
            # as the non-empty (and therefore truthy) string "no" and silently mean its
            # opposite.
            token = value.strip().lower()
            if token in ("no", "off", "false", "0"):
                value = False
            elif token in ("yes", "on", "true", "1"):
                value = True
            else:
                raise ValueError(f"{name}: cannot read {value!r} as a boolean")
        kwargs[name] = value
    return cls(**kwargs)


def _deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    """Read a YAML config, resolving a top-level `extends: <path relative to this file>`
    by deep-merging this file over its parent."""
    path = path.resolve()
    if path in chain:
        raise ValueError(f"config extends cycle: {' -> '.join(map(str, chain + (path,)))}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {raw!r}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ValueError(f"{path}: `extends` must be a path, got {parent!r}")
    return _deep_update(_load_yaml(path.parent / parent, chain + (path,)), raw)


def load_config(path: str | Path | None, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config (or the defaults when `path` is None), then deep-merge each
    `KEY.PATH=VALUE` override on top, in order (CLI wins)."""
    raw = {}
    if path is not None:
        raw = _load_yaml(Path(path))
    for item in overrides:
        raw = _deep_update(raw, parse_override(item))
    return _from_dict(Config, raw)


def expand_path(path: str) -> str:
    """Expand `${VAR}` in a config path, refusing to leave any unexpanded.

    `os.path.expandvars` leaves an unset `${VAR}` in place, which silently creates a
    directory literally named `${SITE_WORK_DIR}` under the cwd. Only `$VAR` / `${VAR}`
    are supported; shell forms such as `${VAR:-default}` are rejected.
    """
    expanded = os.path.expandvars(path)
    leftover = re.findall(r"\$\{?(\w+)", expanded)
    if leftover:
        raise ValueError(
            f"{path!r}: environment variable(s) {', '.join(leftover)} not set; "
            "`source scripts/env.sh` first"
        )
    return expanded


def parse_override(item: str) -> dict[str, Any]:
    """Turn `train.learning_rate=2e-4` into `{"train": {"learning_rate": 2e-4}}`."""
    key, _, value = item.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got {item!r}")
    parsed = yaml.safe_load(value)
    # YAML 1.1 does not recognise `2e-4` as a float (it wants `2.0e-4`), and learning
    # rates are typed that way constantly. Recover the numeric intent.
    if isinstance(parsed, str):
        try:
            parsed = float(parsed)
        except ValueError:
            pass
    # YAML 1.1 also reads `no`/`yes`/`on`/`off` as booleans, but several of our options are
    # strings that happen to spell one of those -- `eval_strategy=no`, `save_strategy=no`,
    # `report_to=no`. Left alone this yields `False is not a valid IntervalStrategy` at
    # trainer construction, minutes into a queued multi-node job. `true`/`false` still
    # parse as real booleans, which is what anyone setting an actual flag would type.
    if isinstance(parsed, bool) and value.strip().lower() in ("no", "yes", "on", "off"):
        parsed = value.strip()
    out: dict[str, Any] = {}
    cursor = out
    parts = key.split(".")
    for p in parts[:-1]:
        cursor = cursor.setdefault(p, {})
    cursor[parts[-1]] = parsed
    return out
