# Implementation Progress

Living log for the Qwen2.5-VL structured-extraction build. Newest status at the top of
each section. Facts here are things that were *run and observed*, not intentions.

---

## Decisions (locked 2026-08-29)

| Question | Decision | Why |
| --- | --- | --- |
| Dataset | `naver-clova-ix/cord-v2` | The dataset named in `PROJECT.md` (`Glazkov/qwen2.5-vl-table-extraction`) **does not exist on the Hub** (HTTP 401 / not found). CORD-v2 is ungated, ~1K receipts, each with a real nested `ground_truth` JSON parse — so JSON-validity is a meaningful metric rather than a synthetic one. |
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` | Per brief. Public, ungated, 5 safetensors shards (~16 GB). |
| Hub push | Token supplied via gitignored `.env` | No token existed on the account. `scripts/env.sh` sources `.env`; `.gitignore` covers it. |
| Final run shape | 2 nodes x 8 A100, ~3-4 h, `gpu-a100` | Genuinely exercises multi-node FSDP. ~50-65 GPU-h of the 50 000 available. |
| Grant | `<grant>` (`-A <account>`) | Active to 2027-08-24. The other grant, `<other-grant>`, expires 2026-08-31. |

## Environment (verified, not assumed)

- Cluster: `<login-host>`, 8x A100 per node.
- Login node **has** outbound internet (`curl https://huggingface.co` -> 200). Compute
  nodes are assumed offline until proven otherwise; jobs run with `HF_HUB_OFFLINE=1`.
- Quotas: `$HOME` 9.54/10 GiB (**95% full** — nothing large may land here),
  `$SCRATCH` 303 GiB/12 TiB.
- Venv: `$SCRATCH/vlm_finetuning/venvs/vlm`, Python 3.12.12, built by
  `scripts/setup_env.sh`.

  ```
  torch 2.11.0+cu128   transformers 5.16.1   trl 1.12.0
  peft 0.20.0          accelerate 1.14.0     datasets 4.8.5
  ```

- W&B credentials already present in `~/.netrc`. `WANDB_PROJECT` is set in the sbatch
  scripts, never in code.

## Library facts read out of the installed source (not recalled)

These come from `trl/trainer/sft_trainer.py` @ 1.12.0 and drive the design:

1. A dataset is treated as vision if it has an `image` or `images` column
   (`sft_trainer.py:1036`). Signature columns are
   `["messages", "prompt", "completion", "image", "images"]` (`:1683`).
2. **`assistant_only_loss` is rejected for vision datasets** (`:1052`). The only way to
   get loss on the answer alone is the **prompt-completion** format, where
   `completion_only_loss=True` masks everything before the completion (`:772`).
   -> Phase 1 must emit `{"images", "prompt", "completion"}`, *not* `{"images", "messages"}`.
3. `packing` and `padding_free` are both rejected for vision datasets (`:1043`, `:1047`).
4. `truncation_mode="keep_end"` is rejected for vision datasets — image tokens live at
   the front (`:1057`).
5. `loss_type="chunked_nll"` **raises if `lm_head` is wrapped by a PEFT adapter**
   (`:1333`). -> LoRA `target_modules` must never include `lm_head`.
6. The collator masks only padding in labels; **vision tokens are deliberately left
   unmasked** and the model handles them. Loss masking therefore has to be verified by
   decoding, per the project gotcha.

## Status

| Item | State |
| --- | --- |
| Repo skeleton, `.gitignore`, `.env.example` | done |
| `scripts/env.sh` (caches off `$HOME`, `.env` sourcing) | done |
| `scripts/setup_env.sh` + venv built | done |
| `src/config.py` — shared config contract | done |
| Model + dataset prefetch | done (16 GB model, 800/100/100 CORD) |
| Phase 1 — data pipeline | **done, independently verified** |
| Phase 2 — FSDP/LoRA training + SLURM | agent cut off by session limit; I finished the integration |
| Smoke test on `gpu-short` | pending |
| 2-node production run | pending |

## Log

- **2026-08-29 14:50** — Surveyed cluster. Found `PROJECT.md`'s dataset missing from the
  Hub; confirmed replacements; confirmed no HF token on the account.
- **2026-08-29 14:55** — Venv built on `$SCRATCH`.
- **2026-08-29 14:58** — First prefetch attempt died mid-download (backgrounded shell was
  reaped, leaving `.incomplete` blobs). Relaunched; `snapshot_download` resumes.
- **2026-08-29 15:05** — `src/config.py` contract fixed in place so both phases can be
  built in parallel without colliding.
- **2026-08-29 15:10** — Prefetch kept dying with **exit 137 (SIGKILL)**. Cause: the Xet
  transfer backend buffers aggressively in RAM and the login-node cgroup kills it. Fixed by
  setting `HF_HUB_DISABLE_XET=1` in `scripts/env.sh` and using `max_workers=1`. All 5
  shards (16 GB) then downloaded cleanly. **This will bite anyone re-running the prefetch
  on a login node — that is why the workaround is committed, not just applied.**
- **2026-08-29 15:15** — CORD-v2 verified: `train` 800 / `validation` 100 / `test` 100,
  columns `image` (PIL RGB) + `ground_truth` (JSON string with a `gt_parse` key). `menu`
  is sometimes a dict rather than a list — the normaliser has to handle both.
- **2026-08-29 15:20** — Phase 1 and Phase 2 agents dispatched in parallel against the
  `src/config.py` contract, with disjoint file ownership.

## 2026-08-30 — Phase 1 verified, Phase 2 recovered

### Phase 1: verified, not just reported
I re-ran `scripts/inspect_batch.py` myself rather than trusting the agent's summary:
`ALL CHECKS PASSED`, 0 failures over 4 examples. The decoded unmasked span is byte-exactly
the target JSON; everything before it (615-token system prompt, instruction, 1008 image
tokens) is `-100`. The negative control (`-o train.completion_only_loss=false`) exits 1
with `unmasked span is not the target JSON`, so the assertions genuinely bind rather than
passing vacuously.

**The headline Phase 1 finding, confirmed at the source:** `SFTConfig.max_length` defaults
to **1024** (`trl/trainer/sft_config.py:201`) and `truncation_mode` defaults to
`"keep_start"` (`:209`). Our sequences run 1300-2200 tokens behind a ~1620-token prompt, so
at the default the trainer keeps the prompt and truncates *exactly the answer*, leaving
**every label `-100`**. TRL emits no warning and the run still produces a falling loss
curve, so "loss went down" would not have caught it. `train_sft.py:350` now sets
`max_length=cfg.data.max_length`.

### Phase 2: what the failed jobs actually proved
The Phase 2 agent hit a session limit mid-rewire. Three jobs existed:

| Job | Shape | State | Real cause |
| --- | --- | --- | --- |
| 3106249 | 1 GPU, `gpu-short` | FAILED | pointed at a scratchpad stub |
| 3106253 | 1 node x 8, FSDP | FAILED | same stub |
| 3106254 | **2 nodes x 8** | cancelled by me | same stub, would have burned a 16-GPU slot |

**Root cause: `/tmp` is node-local on the cluster.** The agent's scratchpad lives under
`/tmp/...`, so scripts it wrote there are simply absent on the compute node —
`[Errno 2] No such file or directory`. The *committed* `slurm/*.sbatch` are fine: they
invoke `-m src.train.train_sft` from the repo on shared storage. Only the agent's ad-hoc
validation scripts had the defect. **Anything a job touches must live on shared storage.**

Neither failure reached FSDP, model loading, or a training step — they died at process
spawn. So nothing about the FSDP config was validated, despite two jobs having "run".
The 8-rank launch itself did work (all 8 ranks spawned and rendezvoused), so the launcher
plumbing is sound.

### Two facts the jobs did establish, both of which change the plan
1. **Compute nodes HAVE outbound internet** — the probe returned `huggingface.co -> HTTP 200`
   from `t0006`. `CLAUDE.md` assumes they may not. The offline flags stay (they turn a
   silent DNS hang into a fast error) but they are belt-and-braces, not load-bearing.
2. **The GPUs are A100-SXM4-40GB, not 80GB** (`40960 MiB`, from the smoke job's
   `nvidia-smi`). `configs/sft_multinode.yaml` and `md_files/PHASE2.md:200` were both sized
   against 80 GB. This is a live OOM risk and is being re-measured on a real GPU rather
   than re-guessed.

### Integration bugs I fixed in `train_sft.py`
The interrupted rewire left two call sites disagreeing with their own shims:
- `build_datasets(cfg.data)` -> `build_datasets(cfg, processor)`
- `train_lengths(train_ds, cfg)` -> `train_lengths(train_ds, cfg, processor)`
Both would have raised `TypeError` on the first line of real work.

### Smoke test PASSED — job 3108434, `gpu-short`, t0011, 1 GPU, 63 s

First end-to-end run of the integrated pipeline. Real output:

```
[dist] world_size=1 device=cuda:0 fsdp=false
train: kept all 800 examples (longest 1439 tokens, max_length=2048)
[data] train=16 eval=8
[data] length signal: min=936 p50=964 max=1083
[model] froze vision tower (model.visual): 676.6M params
trainable params: 80,740,352 || all params: 8,372,907,008 || trainable%: 0.9643
[peft] adapted modules: 196 (inside vision tower: 0)
{'loss': '0.2972', ... 'mean_token_accuracy': '0.9213', 'epoch': '0.0625'}
{'loss': '0.3464', ... 'epoch': '0.125'}
{'loss': '0.06139', ... 'mean_token_accuracy': '1', 'epoch': '0.1875'}
{'loss': '0.1347', ... 'epoch': '0.25'}
{'loss': '0.4573', ... 'epoch': '0.3125'}
{'eval_loss': '0.1453', 'eval_mean_token_accuracy': '0.9524'}
"adapter_check": {"bytes": 323020048, "n_tensors": 392,
                  "n_elements": 80740352, "n_nonzero_tensors": 392, "ok": true}
```

What this actually establishes:
- The vision-tower exclusion works: **196 adapted modules, 0 inside `model.visual`**. The
  naive `target_modules` list would have matched 96 modules in the frozen ViT.
- The adapter is real: 392/392 tensors non-zero, 80.7 M elements — matching the trainable
  count exactly. Not the empty-adapter failure the brief warns about.
- Loss masking survives into training: `num_tokens` climbs 1083 -> 4993 across 5 steps,
  i.e. ~200 label tokens per step, consistent with the JSON answers alone.

Honest caveat: **the loss does not monotonically decrease** (0.297, 0.346, 0.061, 0.135,
0.457). On 16 samples at batch size 1 over 5 steps that is per-example difficulty noise,
not a trend — the run is far too short to read as learning. `mean_token_accuracy` of
0.90-1.00 from step 1 says the base model is already strong on this format. The smoke test
proves the *plumbing*, not the *learning*.

### Corrections to Phase 2's sizing, from real hardware
`nvidia-smi` on both t0006 and t0011: **`NVIDIA A100-SXM4-40GB, 40960 MiB`**.
`configs/sft_multinode.yaml` and `md_files/PHASE2.md:200` were both written against 80 GB.
Added `[vram]` peak-allocated/reserved reporting to `train_sft.py` and am measuring at
production settings on 8 GPUs before committing 16.

### VRAM settled on real hardware — job 3108487, 1 GPU, production settings

The 80 GB assumption was the last live risk. Rather than re-guess, I ran the *production*
config (`max_pixels=1003520`, batch 2, accum 2) on a **single** GPU with the model
unsharded — a strictly harsher test than FSDP, which shards parameters across 8 ranks:

```
[peft] adapted modules: 196 (inside vision tower: 0)
{'loss': '0.3483', 'mean_token_accuracy': '0.9226', 'epoch': '0.5'}
{'loss': '0.3305', 'mean_token_accuracy': '0.912',  'epoch': '1'}
{'loss': '0.2033', 'mean_token_accuracy': '0.9294', 'epoch': '1.5'}
{'loss': '0.1095', 'mean_token_accuracy': '0.9699', 'epoch': '2'}
[vram] rank=0 peak_allocated=21.85GiB peak_reserved=22.62GiB
       device_total=39.67GiB headroom=17.05GiB
```

**22.6 of 39.7 GiB, 17 GiB spare.** No resizing was needed for the 40 GB part after all —
`max_pixels` and batch size stand as Phase 2 chose them. And unlike the 5-step smoke, the
loss here falls monotonically (0.348 -> 0.110) over 2 epochs of 8 examples, which is the
first evidence of actual learning rather than working plumbing.

A footgun I hit myself, worth recording: `--set train.eval_strategy=no` fails with
`ValueError: False is not a valid IntervalStrategy`, because YAML parses bare `no` as
boolean false. **Quote it: `--set 'train.eval_strategy="no"'`.** The committed configs
already quote `"no"`; only ad-hoc CLI overrides are exposed.

### Scale decision revised (user's call, and they were right to push)

I originally recommended 2 nodes x 8 for ~3-4 h. That estimate predated any measurement
and was wrong: CORD-v2 is 800 examples, so 3 epochs at global batch 32 is **75 optimizer
steps**, roughly 20 minutes of compute. Sixteen GPUs would halve an already-short run while
doubling the queue wait.

Final plan:
- **`slurm/train_1node.sbatch` + `configs/sft_1node.yaml`** — the real adapter. 1 node x 8
  A100, 8 ranks x batch 2 x accum 2 = global batch 32, identical optimisation trajectory to
  the 2-node shape. Job **3108496**.
- **`slurm/train_multinode.sbatch`** — a short 10-step 2-node run purely to prove the
  16-rank FSDP path and capture real NCCL/IB logs. Job **3108500**, submitted with
  `--dependency=afterok:3108496` so it neither competes for nodes nor runs if FSDP turns
  out broken.

### Cluster constraints discovered the hard way
- **`gpu-short` QOS caps at `cpu=16, gres/gpu=1, mem=125G`, `MaxJobsPU=1`.** It cannot run
  a multi-GPU job at all, and one job per user at a time. Unrelated jobs of the user's
  (`pi0-libero-*`) were holding that slot, which is why the 8-GPU validation had to go to
  `gpu-a100` and wait.
- 1-GPU jobs on `gpu-a100` schedule in seconds; 8-GPU jobs were quoted ~6.5 h.
  Cheap 1-GPU probes are therefore the right debugging instrument on this cluster.

### Production attempt 1 FAILED — job 3108496, 8 ranks, 58 s: distributed dataset-cache race

FSDP itself came up correctly (`[dist] world_size=8 device=cuda:0 fsdp=true
cpu_ram_efficient_loading=true`) and the full dataset loaded (`[data] train=800 eval=100`).
Then ranks died with:

```
File "src/data/grouping.py", line 120, in estimate_lengths
File "datasets/arrow_dataset.py", line 2486, in cast_column
FileNotFoundError: [Errno 2] No such file or directory:
  '.../cord-v2/default/0.0.0/7f01.../cache-1ca3ca1e8f18eeab.arrow'
```

**Cause.** `build_datasets` and `estimate_lengths` both go through
`datasets.map`/`cast_column`, which write an Arrow cache file whose name is a *content
fingerprint* — identical across ranks by construction. All 8 ranks raced for the same path;
whichever finished first renamed its temp file out from under the others.

This is invisible on 1 GPU, which is exactly why the single-GPU smoke test passed twice
while the real run could not survive a minute. **A green smoke test on one rank says
nothing about rank-coordination bugs.**

**Fix** (`train_sft.py`): wrap dataset construction and length estimation in
`PartialState().main_process_first()`, so rank 0 builds and caches while the others wait,
then all of them read the completed cache instead of regenerating it.

**Also fixed:** `slurm/train_1node.sbatch` hardcoded `--num_processes 8`, so it could not be
run on a reduced debug allocation. It now derives the rank count from
`${SLURM_GPUS_ON_NODE}`.

### Requeued
- **3108780** — 2 GPU, 4 steps: validates the cache fix and FSDP on the cheapest allocation
  that can reproduce a multi-rank race. Est. start 04:40.
- **3108796** — the production 1 node x 8 run. Est. after the above.

Queue note: 8-GPU allocations on `gpu-a100` were quoted ~6.5 h out, 2-GPU ~2 h,
1-GPU seconds. Debugging strategy on this cluster follows that gradient — reproduce on the
smallest allocation that can exhibit the bug.

### Cache fix VALIDATED — job 3108780, 2 GPU, FSDP, 105 s

```
launching 2 ranks
[dist] world_size=2 device=cuda:0 fsdp=true cpu_ram_efficient_loading=true
[data] train=800
[data] length signal: min=948 p50=1935 max=2393
trainable params: 80,740,352 || all params: 8,372,907,008 || trainable%: 0.9643
[peft] adapted modules: 196 (inside vision tower: 0)
{'loss': '0.1782', 'mean_token_accuracy': '0.9585'}
{'loss': '0.1824', 'mean_token_accuracy': '0.9557'}
{'loss': '0.128',  'mean_token_accuracy': '0.9653'}
{'loss': '0.1271', 'mean_token_accuracy': '0.969'}
[vram] rank=0 peak_allocated=15.89GiB peak_reserved=18.13GiB headroom=21.55GiB
[vram] rank=1 peak_allocated=15.89GiB peak_reserved=18.13GiB headroom=21.55GiB
"n_tensors": 392, "n_nonzero_tensors": 392, "ok": true
[save] adapter written to $SCRATCH/vlm_finetuning/runs/fsdp2g/final_adapter
```

Everything the production run depends on is now proven under real multi-rank FSDP:

- **The cache race is gone.** Same code path that killed 8 ranks at 58 s now completes.
- **FSDP sharding is real, not nominal.** Peak reserved drops from **22.62 GiB on one
  unsharded GPU to 18.13 GiB per rank under `FULL_SHARD`** — parameters are genuinely
  distributed, which is the headroom argument that made 40 GB safe.
- **Both ranks report identical peaks**, so the wrap policy is not splitting the frozen ViT
  unevenly across ranks.
- **The collective adapter save works under FSDP** — 392/392 non-zero tensors. This is the
  "adapter saves empty" failure mode called out in the brief's gotcha #4, and it is the one
  thing a single-GPU test structurally cannot check.
- **Length grouping is active**: p50 1935 tokens against max 2393, versus the ~964 median
  of the low-resolution smoke config.

### Queued
- **3108796** — production, 1 node x 8, 3 epochs, W&B on. Est. start 10:10.
- **3108801** — 2-node 16-rank proof, `afterok:3108796`.

---

## RESULTS — production run 3108796, 1 node x 8 A100, COMPLETED in 15 m 55 s

```
launching 8 ranks
[dist] world_size=8 device=cuda:0 fsdp=true cpu_ram_efficient_loading=true
[data] train=800 eval=100
[data] length signal: min=948 p50=1935 max=2393
trainable params: 80,740,352 || all params: 8,372,907,008 || trainable%: 0.9643
[peft] adapted modules: 196 (inside vision tower: 0)

train loss   0.2147 -> 0.2793 -> 0.3369  ...  0.0374 -> 0.0498 -> 0.0317
eval_loss    epoch 1: 0.0533   epoch 2: 0.04042   epoch 3: 0.03812
eval_token_acc         0.9854            0.9886            0.9897
train_runtime 862.4 s (14.4 min), 2.783 samples/s

[vram] ranks 1-7: peak_reserved=12.73GiB  headroom=26.94GiB
[vram] rank 0:    peak_reserved=14.76GiB  headroom=24.91GiB   (rank 0 also writes)
"n_tensors": 392, "n_nonzero_tensors": 392, "ok": true
[save] adapter -> $SCRATCH/vlm_finetuning/runs/cord-lora-1n8g/final_adapter
```

The VRAM series across the three shapes shows FSDP doing exactly what it claims:

| Shape | peak reserved / rank |
| --- | --- |
| 1 GPU, unsharded | 22.62 GiB |
| 2 GPU, `FULL_SHARD` | 18.13 GiB |
| 8 GPU, `FULL_SHARD` | 12.73 GiB |

My earlier runtime estimate of ~20 min was right: 14.4 min of training. The original
"3-4 h on 2 nodes" figure was wrong by an order of magnitude.

## Evaluation on the held-out test split (100 receipts, jobs 3108958 / 3108959)

| Metric | Base Qwen2.5-VL-7B | **Fine-tuned (LoRA)** |
| --- | --- | --- |
| **JSON parse rate (strict)** | **0.03** | **1.00** |
| JSON parse rate (lenient, fences stripped) | 0.99 | 1.00 |
| outputs needing fence repair | 96 / 100 | **0 / 100** |
| Field precision | 0.288 | **0.860** |
| Field recall | 0.651 | **0.839** |
| **Field F1** | **0.400** | **0.849** |
| Exact match | 0.00 | **0.42** |
| Normalised edit similarity | 0.496 | **0.946** |

The parse-rate split is the whole argument for measuring it separately (brief gotcha #5).
The base model is not incompetent at reading receipts — its *recall* is 0.651, so it finds
most fields. It is simply unusable downstream: it wraps 96 of 100 answers in a ```json
markdown fence, so a strict `json.loads` succeeds **3%** of the time. Fine-tuning moves
strict parse rate 0.03 -> 1.00 and precision 0.288 -> 0.860; the precision gap is the base
model padding output with commentary and invented fields (2918 predicted fields against
1291 gold, versus 1259 after tuning). A single headline "accuracy" number would have hidden
all of this.

## Inference deliverable verified — job 3108989

`src/infer/predict.py`, merged adapter, on a real 960x1280 test receipt:

```
[load] base=Qwen/Qwen2.5-VL-7B-Instruct dtype=bfloat16 attn=sdpa
[load] merge_and_unload()
{"menu": [{"name": "J.STB PROMO", "price": "17500"},
          {"name": "Y.B.BAT", "price": "46000"},
          {"name": "Y.BASO PROM", "price": "27500"}],
 "total": {"total_price": "91000", "cash_price": "91000"}}
[json] parse_ok=True fenced=False chars=176
```

## Outstanding

- **Hub push is NOT done.** `scripts/push_adapter.py` is written and its `--dry-run`
  verifies clean (repo `dmnsjk/qwen2.5-vl-7b-cord-json-lora`, 334.5 MB, correct tags and
  model card). The actual upload was blocked by the sandbox permission classifier because
  it publishes to the internet. **It needs the user to run it or grant permission.**
- **Job 3108801** (2-node, 16-rank proof) is still queued behind an 8-GPU-class allocation.
  The multi-node path is therefore **written and dependency-released but not yet executed**.
  Single-node 8-rank FSDP is proven; 2-node NCCL/IB is not.
- Nothing is committed to git yet.

### 2-node proof attempt 1 FAILED — job 3108801, 24 s: shell quoting, not FSDP

```
job=3108801 nodes=2 nproc=16 master=t0012:28801
ValueError: False is not a valid IntervalStrategy, please select one of ['no','steps','epoch']
  at trl/trainer/sft_config.py:306 <- train_sft.py:466 build_sft_config
```

The 16-rank launch and rendezvous were fine; it died constructing `SFTConfig`, before any
distributed work. **Two independent bugs stacked:**

1. **`train_multinode.sbatch` forwarded arguments through `bash -c "... $*"`.** `bash -c`
   re-parses its argument, so `$*` strips one layer of quoting: the
   `--set 'train.eval_strategy="no"'` I passed arrived as `--set train.eval_strategy=no`.
   `slurm/train_1node.sbatch` passes `"$@"` straight to `accelerate` and never had the
   problem, which is why the identical override worked at 1 node and failed at 2.
   Fixed with `FWD_ARGS=$(printf '%q ' "$@")`.
2. **YAML 1.1 reads bare `no` as boolean false.** `eval_strategy`, `save_strategy` and
   `report_to` are string-valued options that happen to spell YAML booleans. I hit this
   once before from the CLI (job 3108482) and worked around it with quotes instead of
   fixing it — so it came back through a different path and cost a queued multi-node slot.

**Fixed properly this time**, in `src/config.py` rather than at the call site:
- `parse_override` keeps `no`/`yes`/`on`/`off` as strings, so string options survive.
- `_from_dict` then coerces back to a real boolean when the dataclass field is *declared*
  `bool` — otherwise `data.group_by_length=no` would arrive as the non-empty, therefore
  **truthy**, string `"no"` and silently mean its opposite. That would have been a worse
  bug than the one being fixed.
- `true`/`false` still parse as genuine booleans.

Verified: `group_by_length=no -> False`, `group_by_length=yes -> True`,
`eval_strategy=no -> 'no'`, `bf16=false -> False`, `learning_rate=2e-4 -> 0.0002`, and all
three committed configs still load unchanged.

Requeued as **3109090**.

### 2-node proof PASSED — job 3109090, 2 nodes x 8 A100 = 16 ranks, 2 m 43 s

```
job=3109090 nodes=2 nproc=16 master=t0012:29090
[dist] world_size=16 device=cuda:0 fsdp=true cpu_ram_efficient_loading=true
[peft] adapted modules: 196 (inside vision tower: 0)
loss 0.2147 -> 0.2793 -> 0.1861 -> 0.1291 ... 0.09656 -> 0.07366 -> 0.1095
train_runtime 71.06 s, 4.503 samples/s
[vram] all 16 ranks: peak_reserved=11.29GiB  headroom=28.38GiB
"n_tensors": 392, "n_nonzero_tensors": 392, "ok": true
```

**NCCL is genuinely on InfiniBand, not silently falling back to TCP** — the reason
`NCCL_DEBUG=INFO` was set in the sbatch in the first place:

```
Using network IB
NET/IB : GPU Direct RDMA Enabled for HCA 0 'mlx5_0'
NET/IB : Made virtual device [0] name=mlx5_0 speed=200000 ndevs=1
NET/IB : Using [0]mlx5_0:1/IB [RO]; OOB ib0:172.23.16.12<0>
NET/IB : Using [0]mlx5_0:1/IB [RO]; OOB ib0:172.23.16.13<0>
```

200 Gb/s HCA, GPUDirect RDMA enabled, two distinct nodes on the IPoIB fabric. (Note NCCL
writes its INFO stream to **stdout**, not stderr — grepping the `.err` file finds nothing
and would make you think the debug flag was ignored.)

All 16 ranks report an identical 11.29 GiB peak, so the wrap policy shards evenly across
nodes as well as within one. The VRAM series across all four shapes:

| Shape | peak reserved / rank | headroom on 40 GB |
| --- | --- | --- |
| 1 GPU, unsharded | 22.62 GiB | 17.05 GiB |
| 2 GPU, `FULL_SHARD` | 18.13 GiB | 21.55 GiB |
| 8 GPU, `FULL_SHARD` | 12.73 GiB | 26.94 GiB |
| **16 GPU / 2 nodes, `FULL_SHARD`** | **11.29 GiB** | **28.38 GiB** |

Also corrected the stale "comfortably inside 80 GB" claim in `md_files/PHASE2.md`, which
was written against hardware this cluster does not have.

---

## FINAL STATE

| Deliverable | Status |
| --- | --- |
| Phase 1 — multimodal data pipeline | done, loss masking verified by decode + negative control |
| Phase 2 — FSDP + LoRA + SFTTrainer | done, proven at 1 / 2 / 8 / 16 ranks |
| Trained LoRA adapter | done — `$SCRATCH/vlm_finetuning/runs/cord-lora-1n8g/final_adapter` |
| Evaluation vs. baseline | done — strict JSON parse rate 0.03 -> 1.00, field F1 0.400 -> 0.849 |
| Inference script | done, verified end-to-end on a real receipt |
| SLURM + Accelerate configs | done — 1-GPU, 1-node, 2-node, eval |
| **Adapter on the HF Hub** | **NOT DONE — upload blocked by the sandbox permission classifier** |
| Git commit | **NOT DONE — not authorised** |

Jobs run, in order: 3108434 smoke PASS, 3108482 FAIL (my YAML `no` bug), 3108487 VRAM PASS,
3108496 FAIL (dataset cache race), 3108780 2-GPU PASS, 3108796 **production PASS**,
3108958/3108959 eval PASS, 3108989 inference PASS, 3108801 FAIL (shell quoting),
3109090 2-node PASS.

---

## Did FSDP make training faster? No — data parallelism did.

Worth recording explicitly, because "we used FSDP and it trained in 14 minutes" is an easy
sentence to write and a misleading one.

Measured `train_samples_per_second`, same config, varying only the rank count:

| Shape | samples/s | vs 1 GPU | scaling efficiency |
| --- | --- | --- | --- |
| 1 GPU, unsharded (no FSDP), job 3108487 | 0.556 | 1.0x | — |
| 2 GPU `FULL_SHARD`, job 3108780 | 0.874 | 1.57x | 79% |
| **8 GPU `FULL_SHARD`, job 3108796 (production)** | **2.783** | **5.0x** | 63% |
| 16 GPU / 2 nodes `FULL_SHARD`, job 3109090 | 4.503 | 8.1x | 51% |

The production run was **~5x faster** than a single GPU: 14.4 minutes instead of roughly 72.

**But the speedup comes from splitting the batch across 8 GPUs, not from sharding.** Plain
DDP would deliver the same parallel speedup and would very likely be *slightly faster* than
FSDP here: `FULL_SHARD` all-gathers parameters before every forward and backward, whereas
DDP keeps weights resident and only all-reduces gradients. **FSDP trades throughput for
memory.** We never benchmarked DDP, so the size of that penalty is unmeasured here — the
claim is directional, from what the algorithms do, not from a measurement.

For this particular job **FSDP bought nothing that was needed**: at 22.62 GiB unsharded the
model already fits on one 40 GB A100, so there was no memory wall to break. The
configuration is real, correct, and exercised — and it is what `PROJECT.md` asked for — but
the honest claim is the memory one, not a speed one:

> ~5x wall-clock speedup from 8-way data parallelism; FSDP `FULL_SHARD` cut per-rank memory
> from 22.62 GiB to 12.73 GiB, which is what would make a larger model, a higher resolution
> cap, or full-parameter fine-tuning feasible on this hardware.

### Caveats on these numbers

- **Only the 8-GPU row is a full 3-epoch run.** The others are 4-10 step probes whose
  30-70 s runtimes are substantially model load, first-step warmup and NCCL init. That
  inflates the apparent penalty at small scale, so the efficiency column is rougher than the
  two significant figures suggest.
- **The falling efficiency is expected, not a fabric problem.** With 800 examples at global
  batch 32, each rank sees very little work per step, so communication dominates as ranks
  are added. The 2-node run was confirmed running on InfiniBand with GPUDirect RDMA
  (`Using network IB`, `mlx5_0`, 200 Gb/s), not a TCP fallback.
- Comparing the 1-GPU row to the rest also changes one thing besides FSDP: the single-GPU
  probe held the whole model resident. It is a throughput baseline, not a controlled
  FSDP-vs-DDP experiment.
