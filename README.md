# Distributed Fine-Tuning of Qwen2.5-VL for Structured Document Parsing

LoRA fine-tuning of **Qwen2.5-VL-7B** to extract **schema-conformant JSON** from document
images, trained with TRL's `SFTTrainer` under **PyTorch FSDP (`FULL_SHARD`)** via Hugging
Face Accelerate on a multi-node **SLURM** cluster (8× A100 40 GB per node).

**Adapter:** [`dmnsjk/qwen2.5-vl-7b-cord-json-lora`](https://huggingface.co/dmnsjk/qwen2.5-vl-7b-cord-json-lora)
· **Dataset:** [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

---

## Result

Held-out test split, 100 receipts, greedy decoding, scored against the base checkpoint
under an identical prompt.

| Metric | Base Qwen2.5-VL-7B | **Fine-tuned (LoRA)** |
| --- | --- | --- |
| **JSON parse rate (strict)** | 0.03 | **1.00** |
| JSON parse rate (lenient) | 0.99 | **1.00** |
| Field precision | 0.288 | **0.860** |
| Field recall | 0.651 | **0.839** |
| **Field F1** | 0.400 | **0.849** |
| Exact match | 0.00 | **0.42** |
| Normalised edit similarity | 0.496 | **0.946** |

**What the adapter buys is reliability, not new reading ability.** The base model already
reads receipts adequately — recall 0.65. What it cannot do is return parseable output: it
wraps 96 of 100 answers in a ` ```json ` markdown fence, so strict `json.loads` succeeds
**3 %** of the time. Precision triples because the tuned model stops padding answers with
commentary and invented fields (2 918 predicted fields vs 1 291 gold, against 1 259 after
tuning).

This is exactly why **parse rate is measured separately from field accuracy** — a single
"accuracy" number would have hidden the entire finding.

## FSDP actually shards

Same config, only the rank count changing:

| Shape | peak reserved / rank | headroom on 40 GB |
| --- | --- | --- |
| 1 GPU, unsharded | 22.62 GiB | 17.05 GiB |
| 2 GPU, `FULL_SHARD` | 18.13 GiB | 21.55 GiB |
| **8 GPU, `FULL_SHARD`** (released adapter) | **12.73 GiB** | 26.94 GiB |
| 16 GPU / 2 nodes, `FULL_SHARD` | 11.29 GiB | 28.38 GiB |

The 2-node run reached 16 ranks over InfiniBand with GPUDirect RDMA (`Using network IB`,
`NET/IB : GPU Direct RDMA Enabled for HCA 0 'mlx5_0'`, 200 Gb/s), not a silent TCP
fallback.

Training the released adapter took **14.4 minutes** on 1 node × 8 A100: 800 examples,
3 epochs, global batch 32, **80 740 352 trainable params (0.96 % of 8.37 B)**.

## Layout

```
pyproject.toml, uv.lock   pinned, reproducible environment (uv sync --frozen)
configs/     accelerate FSDP + single-GPU yaml, training configs (1-node / 2-node)
  site/      site profiles: cluster.env (the published run), example.env (template)
slurm/       one sbatch per job shape: train_1node, train_multinode, eval
src/
  config.py  dataclass+YAML contract shared by the data and training halves
  data/      CORD loading, JSON schema, resolution capping, length grouping
  train/     SFTTrainer entrypoint (LoRA, FSDP-safe adapter save)
  infer/     base + adapter -> JSON
  eval/      parse-rate and field-F1 scoring
scripts/     env.sh, submit.sh, setup_env.sh, prefetch.sh, dataset_stats.py, inspect_batch.py, push_adapter.py
model_card/  model_card.md, uploaded verbatim as the Hub README by push_adapter.py
md_files/    PROJECT brief, PHASE1/PHASE2 design notes, PROGRESS engineering log
```

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and a Linux x86_64 machine with CUDA 12.8-capable
GPUs (built and run on A100 40 GB).

```bash
cp configs/site/example.env site.env  # set account/partition/storage (see below)
bash scripts/setup_env.sh          # uv sync --frozen from uv.lock (login node, needs network)
bash scripts/prefetch.sh           # model + dataset into $HF_HOME (login node, needs network)

source scripts/env.sh
python scripts/inspect_batch.py --split validation --batch-size 2   # verify loss masking
# short run, check loss falls; own output_dir and no W&B so the real run's adapter is untouched
scripts/submit.sh --time=00:15:00 train1n --set train.max_steps=5 --set train.eval_steps=5 \
  --set 'train.output_dir=${SITE_WORK_DIR}/runs/short' --set train.report_to=none
scripts/submit.sh train1n                          # the real run, ~15 min
scripts/submit.sh eval <adapter_dir> --split test  # `base` scores the baseline
```

`scripts/submit.sh` only adds `-A`/`-p` from the site profile; plain
`mkdir -p logs; sbatch -A <acct> -p <part> slurm/train_1node.sbatch` from the repo root
works too.

Every run is reproducible from a committed config + sbatch pair; the entrypoint is
config-driven with `--set key.path=value` overrides and never edited in place.

## Running on your cluster

Everything that differs between machines lives in one small **site profile**; the
configs and sbatch scripts are the same everywhere. `scripts/env.sh` loads
`$SITE_ENV` (default: `site.env` in the repo root) and fills anything unset with the
defaults below. Values already in your environment always win, so
`SITE_PARTITION=debug scripts/submit.sh train1n` overrides one value for one command.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SITE_ACCOUNT` | SLURM account (`sbatch -A`) | none (cluster default) |
| `SITE_PARTITION` | partition for `train1n`, `train2n`, `eval` | none (cluster default) |
| `SITE_WORK_DIR` | root for runs, adapters and caches | `<repo>/.work` |
| `HF_HOME` | Hugging Face cache | `$SITE_WORK_DIR/huggingface` |
| `UV_PROJECT_ENVIRONMENT` | the uv venv (~7 GB) | `<repo>/.venv` |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | multi-node network interface / IB HCA | unset (NCCL auto-detects) |

GPUs per node come from the allocation. The job shapes (`--gres=gpu:8`, memory, wall time)
are part of the experiment and stay as committed. Hyperparameters assume **A100 40 GB**;
on smaller GPUs, lower `data.max_pixels` or the per-device batch with `--set`.
`configs/site/cluster.env` is a complete worked profile.

## Engineering notes

The genuinely hard parts, each verified rather than assumed:

- **Loss masking is verified, not trusted.** TRL *rejects* `assistant_only_loss` for vision
  datasets, so answer-only loss requires the **prompt-completion** format with
  `completion_only_loss=True`. `scripts/inspect_batch.py` decodes a real batch and asserts
  the unmasked span is byte-exactly the target JSON, that image/prompt tokens are `-100`,
  and that the image-placeholder count matches `image_grid_thw`. It exits non-zero
  otherwise, and a negative control confirms the assertions bind.
- **`SFTConfig.max_length` defaults to 1024** with `truncation_mode="keep_start"`. Our
  sequences run 1300–2200 tokens behind a ~1620-token prompt, so the default keeps the
  prompt and truncates *exactly the answer*, leaving every label `-100` — with no warning
  and a perfectly plausible falling loss curve.
- **`loss_type="chunked_nll"` raises if `lm_head` is wrapped by a PEFT adapter**, which
  constrains the LoRA target list.
- **FSDP + LoRA + a frozen ViT** needs `use_orig_params=True`, and the auto-wrap policy must
  name the real `Qwen2_5_VLDecoderLayer` *and* `Qwen2_5_VLVisionBlock` — read out of the
  installed `modeling_qwen2_5_vl.py`, not recalled. An `exclude_modules` regex keeps
  `gate/up/down_proj` from matching 96 modules inside the frozen vision tower.
- **Adapter saving under FSDP is collective.** `save_model` must be entered by every rank
  (it all-gathers a `FULL_STATE_DICT`); calling `save_pretrained` on rank 0 alone hangs or
  writes empty shards. The save is followed by an assertion that the adapter contains
  non-zero weights.
- **Multi-rank dataset cache race.** `datasets.map`/`cast_column` name their Arrow cache by
  content fingerprint, so all ranks race for one path. Fixed with
  `PartialState().main_process_first()`. Invisible to single-GPU testing — the smoke test
  passed twice while the 8-rank run could not survive a minute.

`md_files/PROGRESS.md` is the full log, including the runs that failed and why.

## Environment

`pyproject.toml` + `uv.lock` pin the environment exactly; `scripts/setup_env.sh` is a thin
wrapper around `uv sync --frozen`.

```
torch 2.11.0+cu128   transformers 5.16.1   trl 1.12.0
peft 0.20.0          accelerate 1.14.0     datasets 4.8.5     Python 3.12
```

Two deliberate choices in `pyproject.toml`:

- **`torch`/`torchvision` come from the pinned `download.pytorch.org/whl/cu128` index**, not
  PyPI, so a sync cannot silently install a build with a different CUDA runtime.
- **`exclude-newer` freezes the resolution horizon** to the date the trained environment was
  built. Without it, re-locking pulls newer transitive patch releases and the lock stops
  reproducing the environment that produced the published adapter. Raise it deliberately
  when upgrading, and re-run the 5-step short run afterwards.

Direct dependency versions are pinned rather than floated on purpose: several behaviours
this project relies on are version-specific (prompt-completion masking for vision datasets,
`chunked_nll` rejecting a PEFT-wrapped `lm_head`, the vision collator's key set), so a
resolver free to pick newer wheels would change training semantics silently.

**The environment is rebuilt from the lock, not kept.** It lives wherever
`UV_PROJECT_ENVIRONMENT` points (default `.venv`, ~7 GB). A profile may put it on purged
scratch storage; when it disappears, `scripts/setup_env.sh` rebuilds it from `uv.lock`.
That is why the lock is committed.

### Worked example: reference cluster

The published adapter was trained on the cluster (8× A100 40 GB per node) with
`configs/site/cluster.env` (`ln -s configs/site/cluster.env site.env`). What that profile
handles and why:

- `$HOME` is a 10 GiB quota, so `SITE_WORK_DIR`, `HF_HOME` and the venv go onto `$SCRATCH`
  (purged; `scripts/setup_env.sh` rebuilds the venv from `uv.lock`).
- The Xet transfer backend is OOM-killed by the login-node cgroup mid-download
  (exit 137); the profile sets `HF_HUB_DISABLE_XET=1`.
- NCCL is pinned to `ib0` / `mlx5_0`; without the interface NCCL can pick a management
  NIC and the two nodes never rendezvous.
- `gpu-short` caps at **1 GPU, 1 job per user**, so it cannot validate multi-GPU runs;
  all jobs use `gpu-a100`.
- Secrets come from a gitignored `.env` (see `.env.example`) or `~/.netrc`, never from code.
