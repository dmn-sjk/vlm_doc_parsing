# Site facts live in a sourced shell profile, not a config framework

Everything that differs between machines (SLURM account and partition, storage roots,
NCCL interfaces, local workarounds) lives in one **Site profile**, a shell file of
`: "${VAR:=value}"` lines sourced by `scripts/env.sh`. The **Experiment** (YAML configs,
sbatch job shapes, `src/`) names no site. We chose this over Hydra or a similar framework
because the site facts are consumed by shell first (sbatch flags, `uv`, NCCL, HF caches
are all environment variables), and `src/vlm_doc_parsing/config.py` already separates the
experiment from the site, apart from one `${VAR}` in `output_dir`.

## Consequences

- Precedence is environment > profile > `env.sh` defaults. A profile only fills gaps, so a
  one-off override is an environment variable, never an edit. Only non-empty values
  override: `:=` treats an empty variable as unset, so a profile value cannot be cleared
  from the environment.
- Account and partition are kept out of `#SBATCH` lines and passed by `scripts/submit.sh`.
  Project variables use the `SITE_` prefix; `SLURM_*` names would leak into the `srun`
  inside the multi-node job.
- The cluster's profile is committed as the record of how the published adapter was produced.
