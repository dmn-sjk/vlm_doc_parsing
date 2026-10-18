# Reproducibility

How a published result is split into the part that is identical on every machine and the
part that describes one machine, so the same commands reproduce it anywhere.

## Language

**Experiment**:
Everything that defines a result and must be identical wherever it is reproduced: model,
data, hyperparameters, launcher logic and **Job shape**.
_Avoid_: run config, recipe

**Site**:
One machine or cluster the **Experiment** is run on, described only by the facts that differ
between machines (account, partitions, storage roots, network interfaces, local workarounds).
_Avoid_: cluster, environment, host

**Site profile**:
The single file that describes one **Site**. A **Site profile** never changes what the
**Experiment** computes.
_Avoid_: site config, env file, machine config

**Job shape**:
The resources one job requests: nodes, GPUs per node, wall time, memory. Part of the
**Experiment**, because changing it changes world size and therefore the result.
_Avoid_: allocation, resources

**Work dir**:
The root under which a **Site** keeps everything large the project writes: runs, adapters,
caches, the environment.
_Avoid_: scratch dir, project scratch

**Published run**:
The one **Experiment** whose adapter is released on the Hub. Frozen: none of its resolved
values ever change, so a new idea is a new **Experiment**, never an edit to this one.
_Avoid_: production run, final run, release run

**Short run**:
A few-step, single-node run of an **Experiment** with its own output location and no
experiment tracking, required before any multi-node job. It never replaces the
**Published run**'s adapter.
_Avoid_: smoke test, smoke job, dry run
