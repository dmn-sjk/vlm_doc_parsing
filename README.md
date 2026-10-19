# Distributed Fine-Tuning of Qwen2.5-VL for Structured Document Parsing

LoRA fine-tuning of **Qwen2.5-VL-7B** to extract **schema-conformant JSON** from document images, trained with TRL's `SFTTrainer` under **PyTorch FSDP (`FULL_SHARD`)** via Hugging Face Accelerate on a multi-node **SLURM** cluster (8× A100 40 GB per node).

**Adapter:** [`dmnsjk/qwen2.5-vl-7b-cord-json-lora`](https://huggingface.co/dmnsjk/qwen2.5-vl-7b-cord-json-lora) · **Dataset:** [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

______________________________________________________________________

## Result

Held-out test split, 100 receipts, greedy decoding, scored against the base checkpoint under an identical prompt.

| Metric | Base Qwen2.5-VL-7B | **Fine-tuned (LoRA)** |
| --- | --- | --- |
| **JSON parse rate (strict)** | 0.03 | **1.00** |
| JSON parse rate (lenient) | 0.99 | **1.00** |
| Field precision | 0.288 | **0.860** |
| Field recall | 0.651 | **0.839** |
| **Field F1** | 0.400 | **0.849** |
| Exact match | 0.00 | **0.42** |
| Normalised edit similarity | 0.496 | **0.946** |

*Strict* is `json.loads` on the raw generation. *Lenient* first strips a markdown fence. It does no other repair and does not check the schema. Field, exact-match and edit metrics are computed on the lenient parse, so the base model's field scores already include the fence fix.

**What the adapter buys is reliability, not new reading ability.** The base model already reads receipts adequately - recall 0.65. What it cannot do is return parseable output: it wraps 96 of 100 answers in a ```` ```json ```` markdown fence, so strict `json.loads` succeeds **3 %** of the time. Precision triples because the tuned model stops padding answers with commentary and invented fields (2 918 predicted fields vs 1 291 gold, against 1 259 after tuning).

This is exactly why **parse rate is measured separately from field accuracy** — a single "accuracy" number would have hidden the entire finding.

## FSDP

Same config, only the rank count changing:

| Shape | peak reserved / rank |
| --- | --- 
| 1 GPU, unsharded | 22.62 GiB 
| 2 GPU, `FULL_SHARD` | 18.13 GiB 
| **8 GPU, `FULL_SHARD`** (released adapter) | **12.73 GiB** 
| 16 GPU / 2 nodes, `FULL_SHARD` | 11.29 GiB 


Training the released adapter took **14.4 minutes** on 1 node × 8 A100: 800 examples, 3 epochs, global batch 32, **80 740 352 trainable params (0.97 % of the 8.29 B base)**.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Linux x86_64 machine with CUDA 12.8-capable GPUs.

```bash
cp configs/site/example.env site.env  # set account/partition/storage (see below)
cp .env.example .env                # HF_TOKEN; WANDB_API_KEY or ~/.netrc
bash scripts/setup_env.sh          # uv sync --frozen from uv.lock
bash scripts/prefetch.sh           # model + dataset into $HF_HOME

source scripts/env.sh
python scripts/inspect_batch.py --split validation --batch-size 2   # verify loss masking
# short run, check loss falls; own output_dir and no W&B so the real run's adapter is untouched
scripts/submit.sh --time=00:15:00 train1n --set train.max_steps=5 --set train.eval_steps=5 \
  --set 'train.output_dir=${SITE_WORK_DIR}/runs/short' --set train.report_to=none
scripts/submit.sh train1n                          # the real run, ~15 min
scripts/submit.sh eval <adapter_dir> --split test  # `base` scores the baseline
```

`scripts/submit.sh` only adds `-A`/`-p` from the site profile; plain `mkdir -p logs; sbatch -A <acct> -p <part> slurm/train_1node.sbatch` from the repo root works too.

Every run is reproducible from a committed config + sbatch pair. The entrypoint is config-driven with `--set key.path=value` overrides and never edited in place.

### Cluster configuration

Everything that differs between machines lives in one small **site profile**. The configs and sbatch scripts are the same everywhere. `scripts/env.sh` loads `$SITE_ENV` (default: `site.env` in the repo root) and fills anything unset with the defaults below. Values already in your environment always win, so `SITE_PARTITION=debug scripts/submit.sh train1n` overrides one value for one command.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SITE_ACCOUNT` | SLURM account (`sbatch -A`) | none (cluster default) |
| `SITE_PARTITION` | partition for `train1n`, `train2n`, `eval` | none (cluster default) |
| `SITE_WORK_DIR` | root for runs, adapters and caches | `<repo>/.work` |
| `HF_HOME` | Hugging Face cache | `$SITE_WORK_DIR/huggingface` |
| `UV_PROJECT_ENVIRONMENT` | the uv venv (~7 GB) | `<repo>/.venv` |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | multi-node network interface / IB HCA | unset (NCCL auto-detects) |

GPUs per node come from the allocation. The job shapes (`--gres=gpu:8`, memory, wall time) are part of the experiment and stay as committed. Hyperparameters assume **A100 40 GB**. On smaller GPUs, lower `data.max_pixels` or the per-device batch with `--set`.


## Layout

```
configs/     accelerate FSDP + single-GPU yaml, training configs (1-node / 2-node)
  site/      site profiles: example.env (template)
slurm/       one sbatch per job shape: train_1node, train_multinode, eval; launch_node.sh (per-node launcher)
src/vlm_doc_parsing/
  config.py  dataclass+YAML contract shared by the data and training halves
  data/      CORD loading, JSON schema, resolution capping, length grouping
  train/     SFTTrainer entrypoint (LoRA, FSDP-safe adapter save)
  infer/     base + adapter -> JSON
  eval/      parse-rate and field-F1 scoring
scripts/     env.sh, submit.sh, setup_env.sh, prefetch.sh, dataset_stats.py, inspect_batch.py, push_adapter.py
model_card/  model_card.md, uploaded verbatim as the Hub README by push_adapter.py
docs/        design notes
```




## License

Code: [MIT](LICENSE). The released LoRA adapter is Apache-2.0, matching its base model `Qwen/Qwen2.5-VL-7B-Instruct`. Training data, [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), is CC BY 4.0; see the [model card](model_card/model_card.md) for its citation.
