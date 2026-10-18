# CLAUDE.md

You are the expert AI engineer.

## What this project is

Distributed fine-tuning of **Qwen2.5-VL (7B)** to extract **structured JSON** from visual
documents (tables, invoices, schematics). Training runs on a **multi-node SLURM cluster**
using TRL's `SFTTrainer` with **FSDP** (via Accelerate) and **LoRA** adapters.

The full brief lives in `docs/project-brief.md` — read it before making architectural
decisions. This file covers *how to work here*; the brief covers *what we're building*.

Deliverables: a LoRA adapter on the HF Hub, a clean inference script, and the SLURM +
Accelerate configs that made it reproducible. The HPC orchestration is a first-class
artifact, not scaffolding — keep `slurm/` and `configs/` readable and committed.

## Repository layout

When creating files, follow this layout so the config/orchestration story stays legible:

```
configs/          # accelerate FSDP yaml, training yaml/json configs
  site/           # site profiles: cluster.env (reference), example.env (template)
slurm/            # sbatch scripts (one per job shape: single-node, multi-node, eval)
src/vlm_doc_parsing/  # the installed package (import vlm_doc_parsing.…)
  data/           # dataset loading, chat-template formatting, collators
  train/          # SFTTrainer entrypoint
  infer/          # load base + merge LoRA -> JSON out
  eval/           # parse-rate and field-accuracy scoring
scripts/          # env.sh, submit.sh, setup/prefetch, one-off utilities
docs/             # project brief, design notes, run log, ADRs (docs/adr/)
```

## Environment (any SLURM cluster or GPU workstation)

- Site-specific values (account, partition, storage roots, NCCL interfaces, workarounds)
  live in a **site profile**, `configs/site/*.env`, loaded by `scripts/env.sh`; notes about
  the machine this checkout runs on live in `CLAUDE.local.md`. Committed code, configs and
  sbatch scripts stay site-free: a value that differs between machines goes in the profile.
- Run Python from the project venv (`source scripts/env.sh`), never the system interpreter.
- Keep caches off small home dirs: everything large goes under `$SITE_WORK_DIR`, and the
  profile moves `SITE_WORK_DIR` / `UV_PROJECT_ENVIRONMENT` when the repo sits on a quota.
- **Never run training on a login node.** Submit with `scripts/submit.sh`, or `srun --pty`
  for interactive debugging. Long-running commands issued by Claude must go through SLURM.
- Compute nodes may have no outbound internet. Pre-download model weights and datasets on
  the login node (`scripts/prefetch.sh`), then run with `HF_HUB_OFFLINE=1` /
  `HF_DATASETS_OFFLINE=1`.

## Working conventions

- `configs/sft_1node.yaml` is the Published run: never change its values; a new
  experiment gets a new YAML.
- **Short run before scale.** Any change to data pipeline, trainer config or pinned
  dependencies gets a 5-step single-node run before a multi-node job is submitted.
  Multi-node OOMs 40 minutes into a queue wait are the expensive failure mode. A short run
  gets its own `output_dir` and no W&B, so it never overwrites the published adapter:
  `scripts/submit.sh --time=00:15:00 train1n --set train.max_steps=5 --set train.eval_steps=5 --set 'train.output_dir=${SITE_WORK_DIR}/runs/short' --set train.report_to=none`
- **Don't guess library APIs.** TRL, PEFT, and Accelerate change fast and the VLM paths in
  `SFTTrainer` especially so. Check the installed version (`pip show trl transformers`) and
  read the local source in site-packages rather than recalling an older signature.
- Keep the training entrypoint config-driven (yaml/CLI args), not edited in place —
  experiments must be reproducible from a committed config + sbatch script pair.
- Log to W&B; set `WANDB_PROJECT` in the sbatch script, not in code.
- Secrets (`HF_TOKEN`, `WANDB_API_KEY`) come from the environment or `~/.netrc`. Never
  commit them, never echo them into logs.

## Domain gotchas (these are the actual hard parts)

1. **Visual tokens explode sequence length.** One high-res page can add 4,000+ tokens.
   Always cap resolution at the processor (`min_pixels`/`max_pixels` on
   `Qwen2_5_VLProcessor`) — this is the single highest-leverage OOM lever. Pair with
   gradient checkpointing and `loss_type="chunked_nll"` in `SFTConfig`.
2. **Ragged batches stall the cluster.** Variable document aspect ratios mean one rank can
   be 4x slower than another and everyone waits at the all-reduce. Group samples by
   approximate visual-token count when building batches; use dynamic padding.
3. **Loss masking must be verified, not assumed.** The model should learn only the JSON
   answer — image tokens and the prompt/system instruction must be masked to `-100`.
   Decode a formatted batch and assert the unmasked span is exactly the target JSON before
   trusting a training curve.
4. **FSDP + LoRA + frozen ViT interact badly if configured carelessly.** Mixing frozen and
   trainable params under `FULL_SHARD` needs `use_orig_params=True`; the auto-wrap policy
   must name the real Qwen2.5-VL decoder *and* vision blocks. If loss is `nan` or the
   adapter saves empty, suspect the wrap policy first.
5. **JSON validity is a metric.** Evaluate parse-rate separately from field accuracy — a
   model that extracts correctly but emits a trailing comma scores zero downstream.

## Checks before saying something works

- Training change: the 5-step short run completes and loss decreases over those steps.
- Data change: decode and eyeball a formatted sample (image placeholder count, chat
  template, masked label span).
- Inference change: run the script end-to-end on a real document and `json.loads()` the
  output.

Report actual command output. If a job is still queued, say it's queued — don't imply it
passed.
