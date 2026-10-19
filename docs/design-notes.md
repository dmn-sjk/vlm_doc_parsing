# Design notes

Library versions for every citation below: trl 1.12.0, transformers 5.16.1, accelerate 1.14.0,
peft 0.20.0. Line numbers rot on upgrade; re-check them against the installed sources.

## Example format

Every example is exactly:

```python
{
  "images":     [PIL.Image.Image],                                  # one page image
  "prompt":     [{"role": "system",    "content": SYSTEM_PROMPT},
                 {"role": "user",      "content": USER_INSTRUCTION}],
  "completion": [{"role": "assistant", "content": "<compact target JSON>"}],
}
```

- **Prompt-completion, not `messages`.** The model must learn only the JSON answer. For a
  `messages` dataset that means `assistant_only_loss=True`, which `SFTTrainer` rejects
  whenever the dataset has an `image` or `images` column; without it the 615-token system
  prompt is trained on too. With prompt-completion, TRL's vision collator builds a
  `completion_mask` and sets every label outside the completion to `-100`
  (`completion_only_loss=True`, passed explicitly rather than relying on TRL's default).
  Packing and padding-free batching are unavailable (both rejected for vision datasets).
- **`content` is a plain string.** The collator's `prepare_multimodal_messages` inserts one
  `{"type": "image"}` block per image ahead of string content. Pre-inserted blocks break:
  `datasets` unifies the struct type of a list column, so a text block comes back as
  `{"type": "text", "text": ..., "image": None}`, the Qwen2.5-VL chat template treats any
  block with an `image` key as an image, and training crashes with "Image features and image
  tokens do not match".
- **No extra columns.** Only `messages`, `prompt`, `completion`, `image`, `images` survive
  `SFTTrainer`'s signature-column filter; anything else is silently dropped.
- **Batched inference needs `padding_side="left"`.** The training collator passes
  `padding_side` per call, so the tokenizer attribute is irrelevant there; code that
  batches `generate` must set it.
- **`mm_token_type_ids`** is in every collated batch (TRL extends it with zeros over the
  completion) and must be forwarded to the model.
- **`inspect_batch.py` is the mask check.** Per row it asserts: `<|image_pad|>` count equals
  `image_grid_thw` and the estimate; the unmasked span is exactly the target JSON plus
  `<|im_end|>\n` (without `<|im_end|>` in the loss the model never learns to stop); prompt,
  vision tokens and padding are `-100`; nothing truncated; estimated length equals the real
  one. TRL leaves vision tokens unmasked in `labels` itself, so nothing else would notice.
- **Images stay lazy.** The formatting `map` runs on `Image(decode=False)` payloads, so PNG
  bytes pass through without a decode/re-encode; the collator decodes one batch at a time.
- **`writer_batch_size=32`** in the formatting `map`: the default 1000 buffers ~800 MB of PNGs
  per worker and a login node's memory cgroup SIGKILLs it.

## Target schema

`normalize_gt()` turns CORD's `gt_parse` into a fixed schema (embedded verbatim in the system
prompt as `JSON_SCHEMA_DESCRIPTION`); `target_json()` serialises it with
`json.dumps(obj, ensure_ascii=False, separators=(",", ":"))`.

```
{ "menu": [<item>…], "void_menu": [<item>…],
  "subtotal": {subtotal_price, discount_price, service_price,
               other_service_price, tax_price, etc},
  "total":    {total_price, etc, menu_type_count, menu_qty_count,
               cash_price, change_price, credit_card_price, emoney_price} }

<item> = {name, item_no, count, unit_price, price, discount_price,
          item_subtotal, vat, etc, sub: [<item>…]}
```

Raw CORD is not a usable target: `menu` is a dict in 431/1000 receipts and a list in the
rest, `sub_total` is a list once, and any leaf may be a list of strings where the annotator
split a value across OCR lines.

| Rule | Why |
|---|---|
| Absent keys are omitted, never `null` | 319/1000 receipts have no subtotal; padding would ~triple the target and teach a hallucinated skeleton |
| All leaves are verbatim strings | receipts mix `.` and `,` as thousands separators; numeric coercion corrupts values and exact-match accuracy |
| Canonical key order, not `sort_keys` | matches the system prompt and the receipt's reading order |
| Compact separators, `ensure_ascii=False` | ~40 tokens saved per receipt; an escaped CJK char costs ~6 tokens vs 1-2 literal |
| Unknown keys raise `SchemaError` | a silent drop would poison every target for that field |

All 1000 examples normalise and every target round-trips through `json.loads`.

## Sizing

Measured on the CORD-v2 train split (800 rows) with `scripts/dataset_stats.py`:

| Quantity | p50 | p95 | max |
|---|---|---|---|
| Image pixels | 1.12 Mpx | 9.44 Mpx | 12.19 Mpx |
| Uncapped visual tokens (pixels / 28²) | 1428 | 12037 | |
| Target JSON tokens (incl. `<\|im_end\|>\n`) | 101 | 257 | 572 |

Prompt overhead (system + instruction + chat template, excluding visual tokens) is 615 tokens.
One visual token covers a 28×28 block (14×14 patches merged 2×2), so the visual budget is
exactly `max_pixels / 784`. The checkpoint's own `max_pixels=12845056` (16384 tokens) would
turn the p95 receipt into 12k visual tokens: the cap is the main OOM lever.

**The Published run uses `max_pixels = 1280 * 28 * 28 = 1003520`** and
`min_pixels = 256 * 28 * 28 = 200704`. The sweep that informed it (train split,
`max_length=4096`):

| tokens | max_pixels | visual p50 / p95 / max | total p50 / p95 / max | dropped |
|---|---|---|---|---|
| 256 | 200704 | 247 / 252 / 288 | 963 / 1119 / 1439 | 0 |
| 512 | 401408 | 486 / 494 / 506 | 1199 / 1354 / 1667 | 0 |
| 768 | 602112 | 726 / 768 / 768 | 1428 / 1594 / 1907 | 0 |
| 1024 | 802816 | 1008 / 1014 / 1024 | 1705 / 1857 / 2160 | 0 |
| **1280** | **1003520** | **1230 / 1247 / 1271** | **1935 / 2089 / 2393** | **0** |
| 1536 | 1204224 | 1426 / 1508 / 1530 | 2137 / 2307 / 2620 | 0 |
| 2048 | 1605632 | 1426 / 2028 / 2028 | 2137 / 2788 / 3163 | 0 |
| 4096 | 3211264 | 1426 / 4080 / 4080 | 2137 / 4813 / 5185 | 120 (15%) |

- The cap is chosen for VRAM and step time, not to avoid drops: nothing is dropped below
  2048 tokens. 512 resizes the median receipt to 532×798, where small print suffers; 1024
  (728×1092) and 1280 are ~220 visual tokens apart at p50. At 1280 the measured peak is
  22.6 GiB on one unsharded 40 GB A100 (see [Scaling](#scaling)), so it fits with headroom.
- `min_pixels` upscales the smallest images (204×364 → 264 tokens) so a thumbnail never
  becomes a 30-token image the ViT can't read.
- The longest train sequence is 2393 tokens, under `max_length=4096`, so nothing is
  truncated. That matters: `truncation_mode` is forced to `"keep_start"` for vision
  datasets, so a too-small `max_length` cuts the JSON answer off the end of the label.
  `SFTConfig.max_length` defaults to 1024, which masks *every* label with no warning and
  still yields a falling loss; `train_sft` sets it from `data.max_length`.
- The distribution is bimodal (651 vs ~1010 visual tokens at 1024), which is why batches are
  length-grouped.
- Processor kwargs: `AutoProcessor.from_pretrained(model_id, min_pixels=…, max_pixels=…)`
  folds both into `image_processor.size`; `build_processor` asserts they landed. Passing
  only one as a per-call kwarg is silently ignored.
- The 615-token prompt is ~35% of the median sequence; it is masked from the loss but not
  the forward pass. If step time turns out prompt-bound, shortening `JSON_SCHEMA_DESCRIPTION`
  is the lever. It was left as is because JSON validity is a headline metric.
- Multi-image examples are supported by `estimate_lengths` and `inspect_batch`, but CORD is
  single-image, so that path has never run.

## Length grouping

`grouping.estimate_lengths` returns one exact length per row,
`len(prompt_ids) - n_images + sum(visual_tokens) + len(completion_ids)`, read from image
headers without decoding. `inspect_batch.py` asserts it equals the collated
`attention_mask.sum()`.

The built-in `train_sampling_strategy="group_by_length"` can't work here: `Trainer` strips
the dataset to TRL's signature columns before building the sampler, so a length column is
gone, and `LengthGroupedSampler` then looks for `input_ids`, which a vision dataset lacks
(a crash). `LengthGroupedVisionSFTTrainer` passes the lengths in explicitly:

- **Row order is stable.** Vision datasets skip TRL's dataset preparation and
  `shuffle_dataset` defaults to false, so the lengths stay aligned; the trainer still
  asserts `len(lengths) == len(dataset)`.
- **Same permutation on every rank.** accelerate's `BatchSamplerShard` slices one sampler
  on all ranks, so the sampler gets a `torch.Generator` seeded from `data_seed or seed`.
- **World-aware megabatch.** `per_device_bs * grad_accum * world_size`, not transformers'
  per-device value: grouping only removes the all-reduce stall if every rank in one
  optimizer step draws from the same length bucket.
- `_get_eval_sampler` is overridden too: the inherited one branches on
  `train_sampling_strategy` and hits the same crash.

Multi-rank dataset building runs under `PartialState().main_process_first()`: the Arrow
cache file is named by content fingerprint, so concurrent ranks race for the same path.

## FSDP wrap policy

The wrap classes are the model's own `_no_split_modules`: `Qwen2_5_VLDecoderLayer` (28 text
blocks) and `Qwen2_5_VLVisionBlock` (32 ViT blocks). Naming the vision block matters: without
it the 677M-parameter ViT falls into the root unit and is all-gathered in one 1.35 GB shot on
every forward.

Under PEFT, transformers replaces the YAML policy with `peft.utils.other.fsdp_auto_wrap_policy`:
an OR of "leaf module whose weight requires grad" (every LoRA A/B linear) and the transformer
classes from `FSDP_TRANSFORMER_CLS_TO_WRAP`, the env var `accelerate launch` exports from the
YAML. So the class names still reach FSDP. accelerate's later `set_auto_wrap_policy` is a no-op
on that OR policy. Net: 28 decoder layers + 32 vision blocks + 392 LoRA linears, each its own
unit.

**Activation checkpointing** is off in the FSDP config and on in `SFTConfig`
(`gradient_checkpointing=True`, `use_reentrant=False` stated explicitly). The two are
mutually exclusive. FSDP's version would use the PEFT OR policy as its check function and
checkpoint every LoRA linear (pure overhead); transformers checkpoints exactly the
`GradientCheckpointingLayer` subclasses, i.e. the two block classes. TRL calls
`enable_input_require_grads()` for PEFT + checkpointing so gradients reach the adapters.

| `accelerate_fsdp.yaml` key | value | why |
|---|---|---|
| `fsdp_version` | `1` | see [FSDP1 and full state dict](#fsdp1-and-full-state-dict) |
| `fsdp_sharding_strategy` | `FULL_SHARD` | 16.6 GB bf16 base → ~2 GB of weights per rank at 8 ranks (~1 GB at 16) |
| `fsdp_use_orig_params` | `true` | frozen ViT + trainable LoRA in one unit; optimizer (created after wrapping) sees original params; PEFT's state-dict filter sees real names |
| `fsdp_cpu_ram_efficient_loading` | `true` | only local rank 0 reads the 16 GB checkpoint; otherwise 8 × 16 GB of CPU RAM per node |
| `fsdp_sync_module_states` | `true` | required by the above; all ranks start from rank 0's LoRA init |
| `fsdp_state_dict_type` | `FULL_STATE_DICT` | `SHARDED_STATE_DICT` saves nothing (see below) |
| `fsdp_backward_prefetch` | `BACKWARD_PRE` | overlap the next all-gather with the current backward |
| `mixed_precision` | `bf16` | `MixedPrecision(param/reduce/buffer=bf16)`; only units whose flat param requires grad are upcast to fp32, i.e. the ~81M adapter params |

`cpu_ram_efficient_loading` only takes effect if `torch.distributed` is initialised before
`from_pretrained`, so `train_sft` builds `SFTConfig` and creates `PartialState` first. Load
the model earlier and all 16 ranks read 16 GB each.

`attn_implementation` is `sdpa`: flash-attn is not installed.

### FSDP1 and full state dict

FSDP1 (`fsdp_version: 1`) because `use_orig_params` exists only there, and it is what makes
units mixing frozen and trainable parameters legal, lets the optimizer (created after
wrapping) see the original parameters, and keeps the names PEFT's state-dict filter matches
on. `Trainer.save_model` under FSDP saves only when the state-dict type is
`FULL_STATE_DICT`, with no else branch: with `SHARDED_STATE_DICT` the adapter is silently
never written and the job exits 0.

Moving to FSDP2 means replacing `use_orig_params` and the save path together.

## LoRA targets

`r=32, alpha=64, dropout=0.05` on `q/k/v/o_proj, gate/up/down_proj` of the text decoder only;
the vision tower is frozen and unadapted.

- **Vision-tower trap.** `Qwen2_5_VLVisionBlock.mlp` also has `gate/up/down_proj`, and PEFT
  matches `target_modules` by suffix, so the list hits 96 modules in the frozen ViT (292
  total: 196 decoder + 96 vision). `build_peft_config` adds
  `exclude_modules=r"model\.visual\..*"`. It must be the regex form: the list form matches
  only an exact key or `.suffix` and can't exclude a subtree. Result: 196 adapted modules,
  0 in the vision tower.
- **No `lm_head`.** TRL rejects `loss_type="chunked_nll"` when the output embedding is a PEFT
  layer, and `chunked_nll` keeps peak activations off the 152064-wide vocab dimension.
  `build_peft_config` refuses it.

Trainable parameters (hidden 3584, intermediate 18944, KV 4×128 = 512, 28 layers): 2,883,584
per layer, 80,740,352 total (0.97% of 8.29B): ~161 MB bf16 weights, ~646 MB fp32 AdamW state.

## Adapter saving

`save_adapter` calls `trainer.save_model()` on **all** ranks, then `wait_for_everyone()`, and
only then does rank 0 write `run_config.yaml`. The full-state-dict gather is a collective:
guarding the save with `if is_main_process` hangs, or writes empty shards. Under FSDP `Trainer._save` takes the
`unwrap_model(...).save_pretrained(state_dict=...)` path, so `PeftModel` filters the gathered
state dict down to LoRA tensors, and the processor is saved next to the adapter (inference
then reproduces the training resolution).

`verify_adapter` reopens `adapter_model.safetensors` and requires it to be non-empty with at
least one non-zero tensor (`lora_B` starts at zero, so an all-zero file means empty shards).
It logs the check as JSON and raises on failure.

## Multi-node

`train_multinode.sbatch` requests one SLURM task per node (`--ntasks-per-node=1`):
`accelerate launch` spawns the node's ranks itself, so more tasks would oversubscribe the
GPUs. The batch script derives `MASTER_ADDR` (first name from `scontrol show hostnames`) and
`MASTER_PORT` (`20000 + SLURM_JOB_ID % 20000`, so concurrent jobs don't collide), exports
them with `NPROC`, `NNODES` and `TRAIN_CFG`, and runs `srun slurm/launch_node.sh "$@"`.
`launch_node.sh` passes `SLURM_PROCID` as `--machine_rank` and forwards the CLI arguments as
an argv, never re-parsed by a shell: forwarding through `bash -c "... $*"` stripped the
quotes from `--set` overrides (job 3108801). `rdzv_backend: static` is why an explicit master
address/port is the right mechanism.

| var | where | why |
|---|---|---|
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | Site profile | left unset NCCL may bootstrap over a non-IB interface and the nodes never handshake |
| `NCCL_IB_DISABLE=0` | sbatch | IB verbs, not TCP over IPoIB |
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET` | sbatch | the log shows `NET/IB` was chosen (NCCL writes it to stdout, not stderr) |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | sbatch | a hung collective aborts instead of running to the wall-clock limit |
| `OMP_NUM_THREADS` | sbatch | CPUs per task / GPUs per node |

`ddp_timeout=3600` covers the slow start while rank 0 reads and broadcasts the checkpoint.

## Scaling

Same config, only the rank count changing (A100 40 GB):

| Shape | Job | peak reserved / rank | samples/s | vs 1 GPU | efficiency |
|---|---|---|---|---|---|
| 1 GPU, unsharded | 3108487 | 22.62 GiB | 0.556 | 1.0× | — |
| 2 GPU, `FULL_SHARD` | 3108780 | 18.13 GiB | 0.874 | 1.57× | 79% |
| 8 GPU, `FULL_SHARD` (Published run) | 3108796 | 12.73 GiB (rank 0: 14.76) | 2.783 | 5.0× | 63% |
| 16 GPU / 2 nodes, `FULL_SHARD` | 3109090 | 11.29 GiB | 4.503 | 8.1× | 51% |

- Peaks are identical across ranks (except rank 0 at 8 GPU, which also writes), so the wrap
  policy shards evenly, within and across nodes.
- The speedup comes from data parallelism, not sharding. `FULL_SHARD` all-gathers parameters
  on every forward and backward, so plain DDP would likely be slightly faster (not measured).
  The model already fits on one 40 GB GPU; what FSDP buys is memory (22.6 → 12.7 GiB per
  rank), which a larger model, a higher resolution cap or full fine-tuning would need.
- Efficiency falls as ranks are added because 800 examples at batch 32 leave little work per
  rank per step. Only the 8-GPU row is a full run; the others are 4-10 step probes dominated
  by model load and NCCL init, so the efficiency column is rough.
- Hyperparameters were first sized for 80 GB A100s; the 1-GPU probe at Published run
  settings fit in 40 GB, so none changed.

## Published run

Job 3108796, 1 node × 8, global batch 32 (8 × 2 × accum 2), 3 epochs = 75 optimizer steps:
`train_runtime` 862 s (15 m 55 s wall), train loss 0.215 → 0.032, adapter check 392/392
tensors non-zero (80.7M elements).

| Epoch | eval_loss | eval token accuracy |
|---|---|---|
| 1 | 0.0533 | 0.9854 |
| 2 | 0.0404 | 0.9886 |
| 3 | 0.0381 | 0.9897 |

A 5-step Short run proves plumbing, not learning: job 3108434 (16 samples, batch 1) logged
0.297, 0.346, 0.061, 0.135, 0.457, which is per-example noise. The VRAM probe (job 3108487,
8 examples, 2 epochs) fell monotonically, 0.348 → 0.110.

## Site profiles

Everything that differs between machines (SLURM account and partition, storage roots, NCCL
interfaces, local workarounds) lives in one Site profile, a shell file of
`: "${VAR:=value}"` lines sourced by `scripts/env.sh`; the experiment (YAML configs, sbatch
job shapes, `src/`) names no site. A shell profile rather than Hydra or similar, because the
site facts are consumed by shell first (sbatch flags, `uv`, NCCL, HF caches).

- Precedence is environment > profile > `env.sh` defaults, so a one-off override is an
  environment variable, never an edit. `:=` treats an empty variable as unset, so a profile
  value cannot be cleared from the environment.
- Account and partition stay out of `#SBATCH` lines and are passed by `scripts/submit.sh`.
  Project variables use the `SITE_` prefix; `SLURM_*` names would leak into the `srun`
  inside the multi-node job.

## Library citations

| Fact | Source |
|---|---|
| `assistant_only_loss` rejected for vision datasets | `trl/trainer/sft_trainer.py:1052` |
| `packing`, `padding_free` rejected for vision | `sft_trainer.py:1043`, `:1047` |
| `truncation_mode="keep_end"` rejected for vision | `sft_trainer.py:1057` |
| `max_length` default 1024, `truncation_mode` default `keep_start` | `trl/trainer/sft_config.py:201`, `:209` |
| Vision collator chosen | `sft_trainer.py:1221-1228` |
| Prompt-completion collation, `completion_mask` → `-100` | `sft_trainer.py:693-712`, `:774` |
| `pad_to_multiple_of` must be `None` | `sft_trainer.py:693-696` |
| Vision tokens left unmasked in `labels` by the collator | `sft_trainer.py:686-688` |
| `completion_only_loss=None` resolves to `True` for prompt-completion | `sft_trainer.py:1200-1203` |
| `shuffle_dataset` defaults to `False` | `trl/trainer/sft_config.py:217` |
| Collator passes `padding_side` per call | `sft_trainer.py:718`, `:725` |
| `mm_token_type_ids` concatenated over the completion | `sft_trainer.py:742` |
| `prepare_multimodal_messages` image-block insertion | `trl/data_utils.py:80-93` |
| Signature columns | `sft_trainer.py:1683` |
| Vision datasets skip dataset preparation | `sft_trainer.py:1270-1275` |
| `chunked_nll` rejects a PEFT `lm_head` | `sft_trainer.py:1333` |
| `enable_input_require_grads()` for PEFT + checkpointing | `sft_trainer.py:1148` |
| Non-reentrant default injected only for transformers < 5 | `sft_trainer.py:1350` |
| Non-reentrant default when kwargs are `None` | `transformers/modeling_utils.py:3205` |
| Checkpointing applies to `GradientCheckpointingLayer` | `modeling_utils.py:3234-3260` |
| `torch_dtype` → `dtype` deprecation | `modeling_utils.py:1490`, `:4131` |
| Rank-0 checkpoint read under FSDP | `modeling_utils.py:4790-4797` |
| `is_fsdp_enabled()` requires initialised `torch.distributed` | `transformers/distributed/fsdp.py:41-47` |
| PEFT replaces the wrap policy | `transformers/distributed/fsdp.py:257-269`, called from `trainer.py:1621` |
| Gradient checkpointing and FSDP activation checkpointing exclusive | `transformers/trainer.py:844` |
| Sampler built from the stripped dataset | `trainer.py:985`, `:1003` |
| Length column read for `group_by_length` | `trainer.py:1036-1044` |
| Eval sampler branches on `train_sampling_strategy` | `trainer.py:1064` |
| `LengthGroupedSampler` falls back to `input_ids` | `trainer_pt_utils.py:540` |
| Optimizer created after wrapping | `trainer.py:1591` |
| `self.model = self.model_wrapped` under FSDP | `trainer.py:1646` |
| FULL_STATE_DICT save branch, no else | `trainer.py:3826-3830` |
| `_save` via `unwrap_model(...).save_pretrained` | `trainer.py:3877` |
| `warmup_steps` < 1 is a ratio | `transformers/training_args.py:2119-2121` |
| `group_by_length` renamed to `train_sampling_strategy` | `training_args.py:1355-1366` |
| Wrap class definitions | `models/qwen2_5_vl/modeling_qwen2_5_vl.py:692`, `:294`; `_no_split_modules` `:330`, `:881`; `ForConditionalGeneration` `:1268` |
| Vision MLP has `gate/up/down_proj` | `modeling_qwen2_5_vl.py:85`, `:294` |
| `_supports_sdpa = True` | `modeling_qwen2_5_vl.py:333` |
| Processor folds `min_pixels`/`max_pixels` into `size` | `image_processing_qwen2_vl.py:110-118`; both required `:120-129` |
| `factor = patch_size * merge_size` | `image_processing_qwen2_vl.py:226`, `:280` |
| PEFT OR wrap policy | `peft/utils/other.py:1259-1296` |
| `exclude_modules` list form: exact key or `.suffix` | `peft/tuners/tuners_utils.py:1907-1914`; suffix matching `:1933` |
| `use_orig_params` FSDP1 only | `accelerate/utils/dataclasses.py:1756-1759` |
| bf16 `MixedPrecision` | `dataclasses.py:2117-2124` |
| FULL_STATE_DICT gathers to CPU on rank 0 | `dataclasses.py:2039-2041` |
| `set_auto_wrap_policy` rewrites only the two built-in policies | `accelerate/accelerator.py:1925`; `dataclasses.py:2069`, `:2083` |
| FSDP activation checkpointing uses the wrap policy | `accelerator.py:1972-1979` |
| Trainable units upcast to fp32 | `accelerator.py:1990-2029` |
| `get_state_dict` all-gather (`rank0_only=True`) | `accelerator.py:4060-4067` |
| `sync_module_states` required with RAM-efficient loading | `accelerate/utils/launch.py:306` |
| `FSDP_TRANSFORMER_CLS_TO_WRAP` exported | `launch.py:321` |
