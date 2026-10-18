# FSDP1 with `use_orig_params` and `FULL_STATE_DICT`

Training shards a frozen 8.2B base (including a frozen ViT) plus ~81M trainable LoRA
parameters. We use FSDP1 (`fsdp_version: 1`) with `fsdp_use_orig_params: true` and
`fsdp_state_dict_type: FULL_STATE_DICT`. `use_orig_params` exists only in FSDP1, and it is what
makes FSDP units mixing frozen and trainable parameters legal, lets the optimizer (created
after wrapping) see the original parameters, and keeps the parameter names PEFT's state-dict
filter matches on. `Trainer.save_model` under FSDP saves only when the state-dict type is
`FULL_STATE_DICT`, with no else branch.

## Consequences

- With `SHARDED_STATE_DICT` the final adapter is silently never written and the job exits 0.
- The full-state-dict gather is a collective: every rank calls `trainer.save_model()`. Guarding
  the save with `if is_main_process` hangs, or writes empty shards.
- `verify_adapter` asserts the saved adapter has non-zero tensors; an empty adapter fails the
  job instead of being published. If loss is `nan` or the adapter is empty, suspect the wrap
  policy first.
- Moving to FSDP2 means replacing `use_orig_params` and the save path together; re-run a
  2-rank Short run and check the adapter before any multi-node job.
