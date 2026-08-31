# Phase 2 — Distributed fine-tuning (SFTTrainer + FSDP + LoRA + SLURM)

Design note for the training half of the project. Everything below was read out of the
installed sources in `$SCRATCH/vlm_finetuning/venvs/vlm/lib/python3.12/site-packages`
(torch 2.11.0+cu128, transformers 5.16.1, trl 1.12.0, peft 0.20.0, accelerate 1.14.0) or
measured on the cluster. Nothing here is recalled from a different version.

## Files

| Path | What it is |
| --- | --- |
| `src/train/train_sft.py` | The single config-driven entrypoint. |
| `configs/sft_smoke.yaml` | 1 GPU, 5 steps, 16 samples, tiny images, no W&B. |
| `configs/sft_multinode.yaml` | 2 x 8 A100, 3 epochs on CORD-v2, W&B on. |
| `configs/accelerate_fsdp.yaml` | FSDP1 FULL_SHARD plugin config. |
| `configs/accelerate_single.yaml` | `distributed_type: NO`, for the smoke test. |
| `slurm/smoke_1gpu.sbatch` | `gpu-short`, 1 GPU, 20 min. |
| `slurm/train_multinode.sbatch` | `gpu-a100`, 2 nodes x 8 GPU, 4 h. |
| `slurm/eval.sbatch` | 1 GPU evaluation. |
| `src/infer/predict.py` | Deliverable 2 — image/PDF in, JSON out, parse-checked. |
| `src/eval/evaluate.py` | Parse-rate, field P/R/F1, normalised edit similarity. |
| `scripts/push_adapter.py` | Hub push with `--dry-run`; token never printed. |

---

## 1. The FSDP auto-wrap policy

### The class names

```
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:692   class Qwen2_5_VLDecoderLayer(GradientCheckpointingLayer)
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:294   class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer)
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:330   _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1268  class Qwen2_5_VLForConditionalGeneration(...)   <- the class to load
```

Verified against a real (meta-device) instantiation of the 7B config: 735 modules,
`model.visual` is a `Qwen2_5_VisionTransformerPretrainedModel` with 676.6 M params,
`model.language_model.layers[0]` is a `Qwen2_5_VLDecoderLayer`, and
`model.visual.blocks[0]` is a `Qwen2_5_VLVisionBlock`. Total 8.292 B params.

Naming **both** classes matters. With only the decoder layer named, the 677 M-parameter
ViT falls into the root FSDP unit: it is all-gathered in one 1.35 GB shot on every forward
and never resharded between blocks, which is precisely the case FULL_SHARD is supposed to
avoid.

### What actually happens once PEFT is in the picture

The policy in `accelerate_fsdp.yaml` is **not** the policy FSDP ends up using. In
`Trainer._inner_training_loop`:

```
transformers/trainer.py:1621         if self.is_fsdp_enabled and _is_peft_model(self.model):
                                         update_fsdp_plugin_peft(self.model, self.accelerator)
transformers/distributed/fsdp.py:269     accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
```

and `peft.utils.other.fsdp_auto_wrap_policy` (peft/utils/other.py:1259-1296) builds

```
_or_policy(policies=[
    lambda_auto_wrap_policy(fn = leaf module with weight.requires_grad),   # every LoRA A/B linear
    transformer_auto_wrap_policy(transformer_layer_cls = classes named in
                                 os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"]),
])
```

Two consequences:

* our `fsdp_transformer_layer_cls_to_wrap` value still reaches FSDP — peft reads the same
  env var that `accelerate launch` exports (`accelerate/utils/launch.py:321`), so keeping
  the exact class names in the YAML is not cosmetic;
* accelerate's later `set_auto_wrap_policy(model)` (accelerator.py:1925) is a no-op here,
  because it only rewrites the policy when it is still literally
  `transformer_auto_wrap_policy` or `size_based_auto_wrap_policy`
  (utils/dataclasses.py:2069, :2083) — a `functools.partial(_or_policy, ...)` is neither.

Net wrapping: 28 decoder layers + 32 vision blocks + 392 LoRA linears, each its own FSDP
unit. The LoRA units are what make a frozen/trainable mix safe, and `use_orig_params=True`
is the belt to that suspenders — plus it is what lets the optimizer, which `Trainer`
creates *after* wrapping (`delay_optimizer_creation`, trainer.py:1591), see the original
parameters, and what keeps `named_parameters()` returning the names PEFT's state-dict
filter matches on.

### Why activation checkpointing is off in the FSDP config

`fsdp_activation_checkpointing: false`, and `gradient_checkpointing: true` in `SFTConfig`
instead. They are mutually exclusive — `transformers/trainer.py:844` raises if both are
set — so this is a real choice:

* accelerate applies FSDP activation checkpointing with `fsdp_plugin.auto_wrap_policy` as
  the check function (`accelerator.py:1972-1979`). Under PEFT that policy is the OR-policy
  above, so every individual LoRA `nn.Linear` would get its own checkpoint wrapper: pure
  overhead, zero memory saved.
* transformers instead checkpoints exactly the `GradientCheckpointingLayer` subclasses
  (`modeling_utils.py:3234-3260`), which for this model are precisely
  `Qwen2_5_VLDecoderLayer` and `Qwen2_5_VLVisionBlock` — the granularity we want.

`gradient_checkpointing_kwargs={"use_reentrant": False}` is passed explicitly. transformers
5.16.1 already defaults to non-reentrant when the dict is `None` (`modeling_utils.py:3205`),
and TRL only injects the default for transformers < 5.0.0 (`sft_trainer.py:1350`), so
stating it removes the dependence on either default. TRL calls
`model.enable_input_require_grads()` for PEFT + checkpointing on our behalf
(`sft_trainer.py:1148`), which is what lets gradients reach the adapters through frozen
checkpointed blocks.

### `accelerate_fsdp.yaml`, annotated

The file itself carries the reasoning inline; the short version:

| key | value | why |
| --- | --- | --- |
| `fsdp_version` | `1` | `use_orig_params` exists only in FSDP1 (dataclasses.py:1756-1759), and `Trainer.save_model`'s FULL_STATE_DICT branch is the FSDP1 path. |
| `fsdp_sharding_strategy` | `FULL_SHARD` | 16.6 GB bf16 base -> ~1.04 GB of weights per rank across 16 GPUs. |
| `fsdp_auto_wrap_policy` | `TRANSFORMER_BASED_WRAP` | with the two class names above. |
| `fsdp_use_orig_params` | `true` | gotcha #4: frozen ViT + trainable LoRA under FULL_SHARD. |
| `fsdp_cpu_ram_efficient_loading` | `true` | only local rank 0 reads the 16 GB checkpoint (modeling_utils.py:4790); otherwise 8 x 16 GB of host RAM per node. |
| `fsdp_sync_module_states` | `true` | mandatory with the above (launch.py:306); also guarantees all ranks share rank 0's random LoRA init. |
| `fsdp_state_dict_type` | `FULL_STATE_DICT` | see §4 — `SHARDED_STATE_DICT` silently saves nothing. |
| `fsdp_backward_prefetch` | `BACKWARD_PRE` | overlap the next all-gather with the current backward. |
| `fsdp_activation_checkpointing` | `false` | see above. |
| `mixed_precision` (top level) | `bf16` | becomes `MixedPrecision(param/reduce/buffer = bf16)` (dataclasses.py:2117-2124). Only FSDP units whose flat param requires grad are then upcast to fp32 (accelerator.py:1990-2029) — i.e. the ~81 M adapter params, not the 8.2 B base. |

---

## 2. LoRA target choice

`r=32, alpha=64, dropout=0.05`, on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj` of the **text decoder only**. Vision tower frozen and unadapted.

**The trap.** `Qwen2_5_VLVisionBlock.mlp` is a `Qwen2_5_VLMLP`
(modeling_qwen2_5_vl.py:294, :85) and therefore also has `gate_proj` / `up_proj` /
`down_proj`. PEFT matches `target_modules` by name suffix
(`peft/tuners/tuners_utils.py:1933`), so that target list silently matches **96 modules
inside the frozen ViT** — measured on the real module tree: 292 suffix hits total, 196 in
the text decoder (28 x 7) and 96 in the vision tower (32 x 3).

`train_sft.py` therefore adds `exclude_modules=r"model\.visual\..*"` whenever
`lora.adapt_vision_tower` is false. It has to be the **regex** form: the list form only
matches an exact key or a `.suffix` (`tuners_utils.py:1907-1914`) and cannot exclude a
subtree. Verified: the regex matches 362 module names, and zero vision-tower target hits
survive it.

`lm_head` is refused outright by a guard in `build_peft_config` — TRL raises when
`loss_type="chunked_nll"` and the output embedding is a PEFT tuner layer
(`sft_trainer.py:1333`), and `chunked_nll` is exactly what keeps peak activation memory off
the 152 064-wide vocab dimension.

Trainable parameter arithmetic (`hidden=3584`, `intermediate=18944`, 4 KV heads x 128 =
512, 28 layers):

```
q_proj  32*(3584+3584) = 229,376     gate_proj 32*(3584+18944) = 720,896
k_proj  32*(3584+ 512) = 131,072     up_proj   32*(3584+18944) = 720,896
v_proj  32*(3584+ 512) = 131,072     down_proj 32*(18944+3584) = 720,896
o_proj  32*(3584+3584) = 229,376
                                     per layer = 2,883,584
                                     x 28      = 80,740,352  (0.97 % of 8.292 B)
```

-> ~161 MB of bf16 adapter weights, ~646 MB of fp32 AdamW state, both sharded 16 ways.

---

## 3. Batch size and `max_pixels` arithmetic

Measured on the real `naver-clova-ix/cord-v2` train split (800 rows, every 8th sampled):

```
width    p25 576   p50  864   p75  960   p95 2304   max 2304
height   p25 1080  p50 1296   p75 1296   p95 4096   max 4096
pixels   p25 0.64M p50 1.12M  p75 1.23M  p95 9.44M  max 9.44M
uncapped visual tokens (pixels / 28^2):  p50 1428   p95 12037
gt_parse JSON tokens (Qwen tokenizer):   p50  123   p90 236  p95 300  max 682
```

One Qwen2.5-VL visual token covers a 28x28 pixel block (14x14 ViT patches merged 2x2), so
the visual-token budget is exactly `max_pixels / 784`. Without a cap the p95 receipt alone
contributes 12 k visual tokens — this is gotcha #1 in numbers, and it is why `max_pixels`
is the highest-leverage OOM lever in the whole project.

**Production choice: `max_pixels = 1280 * 28 * 28 = 1,003,520`** (<= 1280 visual tokens),
`min_pixels = 256 * 28 * 28 = 200,704` (>= 256). The floor almost never binds — the
smallest CORD image is 0.28 Mpx = 357 tokens. 1280 tokens is roughly an 896x1120 render,
which for a 1.5-aspect receipt is ~818x1227 — still legible text, while cutting the p95
image by 9.4x in area.

Worst-case sequence: `1280 visual + ~150 prompt + 682 completion ~= 2.1 k tokens`, against
`max_length = 4096`. Nothing is truncated, which matters: `truncation_mode` is forced to
`keep_start` for vision datasets (`sft_trainer.py:1057` rejects `keep_end`), so a too-small
`max_length` would chop the JSON answer off the end of the label and train the model to
emit unterminated JSON.

**Batch:** `per_device_train_batch_size=2`, `gradient_accumulation_steps=1`, 16 ranks
-> global batch **32**. `800 / 32 = 25` optimizer steps per epoch, `x3 epochs = 75 steps`;
`eval_steps = save_steps = 25` is therefore exactly once per epoch, and
`warmup_ratio = 0.05` is ~4 steps. Memory at bs=2 and ~2 k tokens: sharded weights ~1.04 GB
per rank, checkpointed decoder activations ~0.7 GB, ViT activations ~0.8 GB, plus
all-gather buffers for the largest unit (one decoder layer, ~466 MB bf16) — comfortably
inside the available VRAM, deliberately so, because the p95 image is the one that decides
whether the job survives.

> **Correction (measured, 2026-08-30).** This section was originally written against 80 GB
> A100s. The cluster's are the **40 GB SXM4** part (`nvidia-smi`: `40960 MiB`, confirmed on
> t0006/t0011). The sizing survives the correction, but by measurement rather than by the
> arithmetic above. Peak reserved per rank, same config:
>
> | Shape | peak reserved / rank | headroom |
> | --- | --- | --- |
> | 1 GPU, unsharded | 22.62 GiB | 17.05 GiB |
> | 2 GPU, `FULL_SHARD` | 18.13 GiB | 21.55 GiB |
> | 8 GPU, `FULL_SHARD` | 12.73 GiB | 26.94 GiB |
> | 16 GPU (2 nodes), `FULL_SHARD` | 11.29 GiB | 28.38 GiB |

CORD is small; 2400 sample-passes over 16 A100s is a short job. The 4 h SLURM request is an
upper bound covering queueing, the rank-0 checkpoint read + broadcast, and eval — not an
estimate of the gradient math.

**Smoke config** is the same shape with `max_pixels = 256*784` (5x cheaper), 16 samples and
`max_steps=5`, so a data or wiring bug costs three minutes on `gpu-short` instead of a
40-minute queue wait plus a 2-node OOM.

---

## 4. Length-grouped batching, and why the built-in switch cannot work here

transformers 5.16.1 renamed the knob: there is no `group_by_length` any more, it is
`train_sampling_strategy="group_by_length"` plus `length_column_name`
(`training_args.py:1355-1366`).

That built-in path is unusable for a TRL vision dataset:

```
transformers/trainer.py:985   dataset = self._remove_unused_columns(dataset, ...)   # returns a NEW dataset
transformers/trainer.py:1003      dataloader_params["sampler"] = sampler_fn(dataset) # ...the stripped one
trl/sft_trainer.py:1683       self._signature_columns = ["messages","prompt","completion","image","images"]
```

so a precomputed `length` column is dropped *before* `_get_train_sampler` ever sees the
dataset. `Trainer._get_train_sampler` then finds no column, passes `lengths=None`, and
`LengthGroupedSampler` tries to infer lengths from an `input_ids` key that a vision dataset
does not have (`trainer_pt_utils.py:540`) — a crash, not a silent fallback.

`src/train/train_sft.py` therefore subclasses `SFTTrainer` and overrides
`_get_train_sampler` to build `LengthGroupedSampler(lengths=<passed in explicitly>)`.
Two details make that safe and one makes it actually useful:

* **Row order is stable.** Vision datasets take the `_skip_prepare_dataset` branch
  (`sft_trainer.py:1270-1275`), so TRL never maps, filters or shuffles the dataset;
  `SFTConfig.shuffle_dataset` also defaults to `False` (`sft_config.py:217`). The lengths
  list therefore stays aligned with `self.train_dataset`, and the trainer asserts
  `len(lengths) == len(dataset)` rather than trusting it.
* **The permutation must be identical on every rank.** accelerate's `BatchSamplerShard`
  iterates the same sampler on all ranks and slices it, so the sampler is given an explicit
  `torch.Generator` seeded from `data_seed or seed` instead of relying on the global RNG
  staying in lockstep.
* **The megabatch is world-aware.** transformers uses `train_batch_size * grad_accum`,
  which under FSDP is per-device. Grouping only removes the stall if all 16 ranks in the
  *same* optimizer step draw from the same length bucket, so the override uses
  `per_device_bs * grad_accum * world_size`. That is the actual content of gotcha #2:
  otherwise rank 3 gets a 12 k-token blueprint while rank 11 gets a 400-token receipt and
  everybody waits at the all-reduce.

`_get_eval_sampler` is overridden too — the inherited one branches on
`train_sampling_strategy` as well (`trainer.py:1064`) and would hit the same
`LengthGroupedSampler` crash on the eval split.

The length signal itself comes from Phase 1's `src/data/grouping.py`. `train_sft.py`
resolves it by trying the documented callables and falls back to estimating visual tokens
straight from the stored image sizes (`clamp(w*h, min_pixels, max_pixels) / 784`), which is
the dominant term anyway.

---

## 5. Rank-0 adapter saving under FSDP

The failure mode called out in the brief — a 0-byte or empty adapter — has two distinct
causes, and both are closed deliberately.

**Cause A: the wrong state-dict type.** `Trainer.save_model` under FSDP is:

```
transformers/trainer.py:3826   elif self.is_fsdp_enabled:
                    :3827          if "FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type):
                    :3828              state_dict = self.accelerator.get_state_dict(self.model)
                    :3829              if self.args.should_save:
                    :3830                  self._save(output_dir, state_dict=state_dict)
```

There is **no else branch**. With `SHARDED_STATE_DICT` the final adapter is never written
and the job exits 0. Hence `fsdp_state_dict_type: FULL_STATE_DICT` in the accelerate
config, and a post-save assertion in the code.

**Cause B: saving on rank 0 only.** `accelerator.get_state_dict` for FSDP1 is

```
accelerate/accelerator.py:4060   full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                       :4066     with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                       :4067         state_dict = model.state_dict()
```

which is a **collective**: every rank must enter it or the all-gather never completes.
Guarding the save with `if accelerator.is_main_process: model.save_pretrained(...)` — the
obvious-looking thing to write — either hangs the job or, on the ranks that do run, yields
the empty local shards. So `save_adapter()` in `train_sft.py` calls `trainer.save_model()`
on **all** ranks, then `wait_for_everyone()`, and only afterwards does rank 0 write
`run_config.yaml`.

`_save` then dispatches correctly on its own: under FSDP `Trainer` reassigns
`self.model = self.model_wrapped` (trainer.py:1646), so `self.model` is the FSDP wrapper,
`isinstance(self.model, (PreTrainedModel, PeftModel))` is False, and it takes the
`accelerator.unwrap_model(...).save_pretrained(output_dir, state_dict=state_dict)` path
(trainer.py:3877) — i.e. `PeftModel.save_pretrained` filters the full gathered state dict
down to the LoRA tensors. `_save` also writes the processor next to the adapter, which is
what makes `AutoProcessor.from_pretrained(<adapter dir>)` reproduce the training resolution
at inference time.

**Then it is proved, not assumed.** `verify_adapter()` reopens
`adapter_model.safetensors`, and asserts it exists, is non-zero, contains tensors, and that
at least one tensor has a non-zero element (`lora_B` is zero-initialised, so an all-zero
file means the gather produced empty shards, not a trained adapter). The check is printed
as JSON and raises on failure.

---

## 6. Multi-node rendezvous

`slurm/train_multinode.sbatch` requests `--nodes=2 --gres=gpu:8 --ntasks-per-node=1` — one
SLURM task per node, because `accelerate launch` itself spawns the 8 local ranks. Two tasks
per node would give 32 processes fighting over 16 GPUs.

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

LAUNCHER="accelerate launch --config_file configs/accelerate_fsdp.yaml \
  --num_processes 16 --num_machines 2 --machine_rank \$SLURM_PROCID \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"

srun --ntasks=2 --ntasks-per-node=1 bash -c "$LAUNCHER -m src.train.train_sft --config ..."
```

`\$SLURM_PROCID` is escaped so it is expanded inside each `srun` task (0 on the first node,
1 on the second) rather than by the batch shell, where it would be wrong or empty. The
flag names were checked against `accelerate launch --help` on the installed 1.14.0:
`--num_processes`, `--num_machines`, `--machine_rank`, `--main_process_ip`,
`--main_process_port` all exist. `accelerate_fsdp.yaml` pins `rdzv_backend: static`, which
is why an explicit master address/port is the right mechanism here rather than an elastic
c10d rendezvous — SLURM already tells us who the master is.

NCCL, for the cluster specifically (read off the cluster, not assumed: `/sys/class/infiniband`
lists exactly `mlx5_0`, and `ip link` shows `lo`, `ext`, `svc-eth`, `ib0`):

| var | value | why |
| --- | --- | --- |
| `NCCL_SOCKET_IFNAME` | `ib0` | bootstrap over IPoIB. Left unset, NCCL can pick `ext` or `svc-eth` and the two nodes never complete the handshake. |
| `NCCL_IB_HCA` | `mlx5_0` | the node's only IB HCA; pinning it skips probing. |
| `NCCL_IB_DISABLE` | `0` | force the IB verbs transport rather than TCP over IPoIB. |
| `NCCL_DEBUG` / `_SUBSYS` | `INFO` / `INIT,NET` | the log then contains the `NET/IB` line that proves IB was chosen over `NET/Socket`, without the per-collective firehose. |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `1` | a hung collective aborts the job instead of wedging it until the wall-clock limit. Note the `TORCH_` prefix — the old `NCCL_ASYNC_ERROR_HANDLING` name is deprecated in torch 2.11. |
| `OMP_NUM_THREADS` | `8` | 64 requested cores / 8 ranks per node. |

Intra-node traffic uses NVLink and needs no configuration; `SFTConfig(ddp_timeout=3600)`
covers the slow start where one rank per node reads 16 GB of safetensors and broadcasts it.

One more ordering constraint, easy to get wrong: `train_sft.py` builds the `SFTConfig` and
touches `.device` **before** loading the model. `from_pretrained` only takes the
rank-0-reads-the-checkpoint path when `is_fsdp_enabled()` is true, and that helper requires
`torch.distributed` to already be initialised
(`transformers/distributed/fsdp.py:41-47` -> `modeling_utils.py:4790`). Load the model
first and all 16 ranks read 16 GB each.

---

## 7. Cluster facts and job records

Verified on the cluster:

* `gpu-short` QOS is capped at **`cpu=16, gres/gpu=1, mem=125G, 1 job per user`**
  (`sacctmgr show qos`). Multi-GPU and multi-node validation is therefore **impossible**
  on that partition — the 1-GPU smoke test is the only thing that fits.
* Nodes: `CPUTot=128` (2 sockets x 64 cores), `RealMemory=1024000` MB, `Gres=gpu:a100:8`.
* `sbatch --test-only` accepts all three committed sbatch scripts; the `gpu-a100`
  worst-case start estimate for the 2-node job was ~4 weeks out at the time of writing
  (that is the no-backfill bound, not the expected wait).

flash-attn is **not** installed and no wheel for torch 2.11 / cu128 was assumed to exist,
so `attn_implementation` is `sdpa` everywhere. Qwen2.5-VL declares `_supports_sdpa = True`
(modeling_qwen2_5_vl.py:333).

### Loss-mask verification (gotcha #3), run on CPU

Before spending a GPU, the TRL vision collator was driven directly with a two-example
batch built from the real CORD split and the production message shape:

```
batch keys:      ['attention_mask', 'image_grid_thw', 'input_ids', 'labels',
                  'mm_token_type_ids', 'pixel_values']
input_ids:       (2, 975)      pixel_values: (1976, 1176)
image_grid_thw:  [[1, 38, 26], [1, 38, 26]]   -> 38*26 = 988 patches / 4 = 247 visual tokens
                                                 (max_pixels cap = 256, so the cap bound)
unmasked label span: 303 -> 974 of 975
decoded unmasked:  '{"menu": [{"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"}, ...'
```

The unmasked span is exactly the target JSON: `completion_only_loss=True` on a
prompt-completion dataset masks the system prompt, the instruction and all 247 image
tokens to `-100`, which is the property CLAUDE.md asks to be decoded rather than assumed.

### Job log

| job | shape | partition | purpose | state |
| --- | --- | --- | --- | --- |
| 3106249 | 1 node, 1 GPU, 40 min | `gpu-short` | `sft_smoke.yaml` end to end | see below |
| 3106253 | 1 node, 8 GPU, 25 min | `gpu-a100` | `accelerate_fsdp.yaml` on a GPU: wrap policy, `use_orig_params`, FULL_STATE_DICT adapter save | see below |
| 3106254 | 2 nodes, 8 GPU, 25 min | `gpu-a100` | the above plus the `scontrol`/`SLURM_PROCID` rendezvous and NCCL over IB | see below |

3106253 / 3106254 are 5-step validation runs against the production configs
(`--set train.max_steps=5 --set data.max_train_samples=64`), **not** the production job.

