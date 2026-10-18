---
base_model: Qwen/Qwen2.5-VL-7B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - document-parsing
  - image-text-to-text
  - lora
  - peft
  - qwen2_5_vl
  - vlm
  - json-extraction
datasets:
  - naver-clova-ix/cord-v2
---
# Qwen2.5-VL-7B LoRA — structured document parsing (CORD-v2)

A LoRA adapter for `Qwen/Qwen2.5-VL-7B-Instruct` that makes the model emit **schema-conformant JSON directly**
from a receipt image.

**What this adapter buys you is reliability, not new reading ability.** The base
model can already read receipts reasonably well — its field recall is 0.65. What it cannot
do is return parseable output: it wraps 96 of 100 answers in a ```json markdown fence, so a
strict `json.loads` succeeds **3%** of the time. After tuning that is **100%**, and
precision roughly triples because the model stops padding its answer with commentary and
invented fields. Treat this as a formatting-and-schema adapter, not a better OCR model.

Trained with TRL `SFTTrainer` in the **prompt-completion** format
(`completion_only_loss=True`, so the loss covers only the JSON answer) under PyTorch FSDP
`FULL_SHARD` via Hugging Face Accelerate on a SLURM cluster. The released adapter comes
from a 1 node x 8 A100 (40 GB) run, 14.4 minutes of training.

## Usage

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

base = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "dmnsjk/qwen2.5-vl-7b-cord-json-lora")
processor = AutoProcessor.from_pretrained("dmnsjk/qwen2.5-vl-7b-cord-json-lora")
```

## Training configuration

| | |
| --- | --- |
| base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| vision tower | frozen, not adapted |
| dataset | `naver-clova-ix/cord-v2` |
| resolution cap | `min_pixels=200704`, `max_pixels=1003520` |
| epochs / lr / scheduler | 3.0 / 0.0001 / cosine |
| precision | bf16, gradient checkpointing, `loss_type="chunked_nll"` |

`run_config.yaml` in this repository is the exact resolved configuration of the run.

## Evaluation

Held-out CORD-v2 **test** split, 100 receipts, greedy decoding. Scored against the base
checkpoint under an identical prompt, so the delta is attributable to the adapter.

| Metric | Base (no adapter) | **This adapter** |
| --- | --- | --- |
| JSON parse rate (strict) | 0.030 | **1.000** |
| JSON parse rate (lenient) | 0.990 | **1.000** |
| Field precision | 0.288 | **0.860** |
| Field recall | 0.651 | **0.839** |
| Field F1 | 0.400 | **0.849** |
| Exact match | 0.000 | **0.420** |
| Normalised edit similarity | 0.496 | **0.946** |

JSON parse rate is reported separately from field accuracy because a model that
extracts every field but emits a trailing comma scores zero downstream. "Strict" is
`json.loads` on the raw generation; "lenient" first strips a markdown code fence.

## Limitations

- **Small, narrow training set.** 800 training examples, 3 epochs. CORD-v2 is Indonesian
  restaurant/retail receipts specifically. Do not expect this to transfer to invoices,
  financial tables, or engineering schematics without retraining — the brief's broader
  document types were never trained on.
- **Exact match is 0.42.** More than half of outputs differ from the gold parse
  somewhere, usually a single OCR-level field (a digit in a price, an abbreviated item
  name). Normalised edit similarity is 0.946, so errors are typically small rather
  than structural — but this is not a solved extraction task.
- **The schema is fixed** by the system prompt this adapter was trained with (see
  `run_config.yaml`). Prompting it for a different key set will degrade it. The adapter has
  been specialised toward one output shape.
- **Field scoring ignores list order** (receipt line items are treated as an unordered
  bag), so ordering mistakes are not penalised.
- Resolution is capped at `max_pixels` (see the table above). Documents far denser than a
  receipt may lose legibility at that cap.

## Source code

Training pipeline, SLURM sbatch scripts and Accelerate/FSDP configs:
**[https://github.com/dmn-sjk/vlm_doc_parsing](https://github.com/dmn-sjk/vlm_doc_parsing)**

See
[`src/vlm_doc_parsing/infer/predict.py`](https://github.com/dmn-sjk/vlm_doc_parsing/blob/master/src/vlm_doc_parsing/infer/predict.py)
there for a complete inference example, including `--merge` (`merge_and_unload`) for faster serving.
