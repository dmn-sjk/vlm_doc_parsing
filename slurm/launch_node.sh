#!/bin/bash
# Run once per node by srun; accelerate spawns that node's ranks.
set -euo pipefail
exec accelerate launch --config_file configs/accelerate_fsdp.yaml \
  --num_processes "${NPROC}" --num_machines "${NNODES}" --machine_rank "${SLURM_PROCID}" \
  --main_process_ip "${MASTER_ADDR}" --main_process_port "${MASTER_PORT}" \
  -m vlm_doc_parsing.train.train_sft --config "${TRAIN_CFG}" "$@"
