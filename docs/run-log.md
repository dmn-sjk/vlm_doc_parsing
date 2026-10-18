# Run log

Full original: `git show d601e32:md_files/PROGRESS.md`.

All jobs ran on the cluster (8× A100-SXM4 **40 GB** per node), 2026-08-29 to 2026-08-30.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Dataset | `naver-clova-ix/cord-v2` (800 / 100 / 100) | The brief's `Glazkov/qwen2.5-vl-table-extraction` does not exist on the Hub. CORD-v2 is ungated and each receipt has a real nested JSON parse, so JSON validity is a meaningful metric |
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` | Per the brief. Public, ungated, ~16 GB in 5 shards |
| Resolution cap | `max_pixels = 1280·28·28` | See `design-notes.md#sizing` |
| Published run shape | 1 node × 8 A100, global batch 32 (8 × 2 × accum 2) | 800 examples × 3 epochs at batch 32 is 75 optimizer steps, ~15 min. 16 GPUs would halve a short job and double the queue wait. The 2 × 8 shape (16 × 2 × accum 1 = 32, same trajectory) was proved separately (job 3109090) |
| Attention | `sdpa` | flash-attn not installed for torch 2.11 / cu128 |
| Hub token | gitignored `.env`, sourced by `scripts/env.sh` | never committed or logged |

## Results

**Published run, job 3108796** (1 × 8, 3 epochs): 15 m 55 s wall, `train_runtime` 862 s,
2.78 samples/s. Train loss 0.215 → 0.032. Adapter check: 392/392 tensors non-zero, 80.7M
elements.

| Epoch | eval_loss | eval token accuracy |
|---|---|---|
| 1 | 0.0533 | 0.9854 |
| 2 | 0.0404 | 0.9886 |
| 3 | 0.0381 | 0.9897 |

**Evaluation on the test split** (100 receipts, jobs 3108958 / 3108959):

| Metric | Base Qwen2.5-VL-7B | Fine-tuned (LoRA) |
|---|---|---|
| JSON parse rate (strict) | 0.03 | **1.00** |
| JSON parse rate (lenient, fences stripped) | 0.99 | 1.00 |
| Outputs needing fence repair | 96 / 100 | 0 / 100 |
| Field precision | 0.288 | **0.860** |
| Field recall | 0.651 | **0.839** |
| Field F1 | 0.400 | **0.849** |
| Exact match | 0.00 | 0.42 |
| Normalised edit similarity | 0.496 | 0.946 |

The base model finds most fields (recall 0.651) but wraps 96/100 answers in a markdown fence
and pads with invented fields (2918 predicted vs 1291 gold; 1259 after tuning). Strict
parse rate is why the metric is reported separately from field accuracy.

**Inference, job 3108989:** `predict.py` with the merged adapter on a 960×1280 test receipt
printed valid JSON (`parse_ok=True fenced=False`).

**Peak VRAM per rank** (same config):

| Shape | Job | peak reserved | headroom on 40 GB |
|---|---|---|---|
| 1 GPU, unsharded | 3108487 | 22.62 GiB | 17.05 GiB |
| 2 GPU, `FULL_SHARD` | 3108780 | 18.13 GiB | 21.55 GiB |
| 8 GPU, `FULL_SHARD` | 3108796 | 12.73 GiB (rank 0: 14.76) | 26.94 GiB |
| 16 GPU / 2 nodes, `FULL_SHARD` | 3109090 | 11.29 GiB | 28.38 GiB |

Peaks are identical across ranks (except rank 0 at 8 GPU, which also writes), so the wrap
policy shards evenly, within and across nodes.

**Throughput** (`train_samples_per_second`):

| Shape | samples/s | vs 1 GPU | scaling efficiency |
|---|---|---|---|
| 1 GPU, unsharded | 0.556 | 1.0× | — |
| 2 GPU `FULL_SHARD` | 0.874 | 1.57× | 79% |
| 8 GPU `FULL_SHARD` (Published run) | 2.783 | 5.0× | 63% |
| 16 GPU / 2 nodes `FULL_SHARD` | 4.503 | 8.1× | 51% |

The speedup comes from data parallelism, not sharding. `FULL_SHARD` all-gathers parameters
on every forward and backward, so plain DDP would likely be slightly faster (not measured).
At 22.6 GiB the model already fits on one 40 GB GPU; what FSDP bought is memory (22.6 →
12.7 GiB per rank), which is what a larger model, a higher resolution cap or full fine-tuning
would need. Only the 8-GPU row is a full run; the others are 4-10 step probes dominated by
model load and NCCL init, so the efficiency column is rough. Efficiency falls as ranks are
added because 800 examples at batch 32 leave little work per rank per step.

**2-node proof, job 3109090** (16 ranks, 2 m 43 s): NCCL used InfiniBand
(`Using network IB`, GPUDirect RDMA on `mlx5_0`, 200 Gb/s), not a TCP fallback; adapter
check `ok`.

## Notes

- **Cluster compute nodes have outbound internet** (`huggingface.co` → HTTP 200 from a
  compute node). Jobs still set `HF_HUB_OFFLINE=1`: it turns a silent network hang into a
  fast error.
- **Queue wait scales with shape** on `gpu-a100`: 1 GPU starts in seconds, 2 GPU
  ~2 h, 8 GPU ~6.5 h. Reproduce a bug on the smallest shape that shows it.
- **A 5-step Short run proves plumbing, not learning.** Job 3108434 (16 samples, batch 1)
  logged 0.297, 0.346, 0.061, 0.135, 0.457: per-example noise. The VRAM probe (3108487,
  8 examples, 2 epochs) fell monotonically, 0.348 → 0.110.

## Failures

**Login-node prefetch killed (no job id).** Symptom: `snapshot_download` exit 137. Cause:
the Xet transfer backend buffers in RAM and the login-node memory cgroup kills it. Fix:
`HF_HUB_DISABLE_XET=1` in the cluster Site profile.

**`max_length` default (caught before any job).** Symptom: none; loss still falls. Cause:
`SFTConfig.max_length` defaults to 1024 with `keep_start` truncation, and our sequences are
up to 2393 tokens with the answer last, so every label is masked (measured: 0 unmasked
label tokens per row at 1024). Fix: `train_sft` sets `max_length` from `data.max_length` (4096).

**Jobs 3106249, 3106253** (3106254 cancelled before it ran). Symptom: `[Errno 2] No such
file or directory` at process spawn. Cause: ad-hoc validation scripts lived in node-local `/tmp`, absent on compute
nodes. Fix: anything a job touches lives on shared storage. These jobs proved nothing about
FSDP; only that 8 ranks spawn and rendezvous.

**Job 3108482.** Symptom: `ValueError: False is not a valid IntervalStrategy`. Cause: YAML
1.1 reads `--set train.eval_strategy=no` as boolean false. Fix at the time: quote it
(`'train.eval_strategy="no"'`); fixed properly after job 3108801.

**Job 3108496** (first 8-rank Published run attempt, 58 s). Symptom: `FileNotFoundError` on
`cache-<fingerprint>.arrow` in `estimate_lengths`. Cause: `datasets.map`/`cast_column` name
the cache file by content fingerprint, identical on all ranks, so 8 ranks raced for one path.
Invisible on one GPU, where the Short run and the VRAM probe had passed. Fix: dataset building and length
estimation run under `PartialState().main_process_first()`. Validated on 2 GPUs by job
3108780.

**Job 3108801** (first 2-node attempt, 24 s). Symptom: the same `IntervalStrategy` error
despite the quoted override. Cause: two bugs. `train_multinode.sbatch` forwarded arguments
through `bash -c "... $*"`, which stripped the quotes; and `parse_override` still parsed bare
`no` as a boolean. Fix: `parse_override` keeps `no`/`yes`/`on`/`off` as strings, and
`_from_dict` coerces them to `bool` only for fields declared `bool` (otherwise
`group_by_length=no` would be the truthy string `"no"`), and the sbatch forwarded
`$(printf '%q ' "$@")`. Rerun as job 3109090 (passed). `slurm/launch_node.sh`, which passes
the arguments as an argv, replaced the `printf %q` forwarding later and has not yet run on
the cluster.

**80 GB sizing assumption.** Symptom: none (caught by `nvidia-smi` in job 3108434). Cause: the
first sizing was written for 80 GB A100s; the cluster's are 40 GB. Fix: job 3108487 ran the
Published run's settings on one unsharded GPU (22.6 GiB peak), so no hyperparameter changed.

## Job index

| Job | Shape | Purpose | Result |
|---|---|---|---|
| 3108434 | 1 GPU | Short run (16 samples, 5 steps) | pass; adapter 392/392 non-zero |
| 3108482 | 1 GPU | probe with `--set train.eval_strategy=no` | fail (YAML `no`) |
| 3108487 | 1 GPU | VRAM probe at Published run settings | pass, 22.62 GiB |
| 3108496 | 1 × 8 | Published run | fail (dataset cache race) |
| 3108780 | 2 GPU | cache-race fix, FSDP | pass |
| 3108796 | 1 × 8 | **Published run** | pass |
| 3108958 / 3108959 | 1 GPU | evaluation, base and fine-tuned | pass |
| 3108989 | 1 GPU | inference | pass |
| 3108801 | 2 × 8 | 2-node proof | fail (quoting + YAML `no`) |
| 3109090 | 2 × 8 | 2-node proof | pass |
