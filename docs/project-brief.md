> **Note:** the dataset named below (`Glazkov/qwen2.5-vl-table-extraction`) does not exist on
> the Hub. This project uses `naver-clova-ix/cord-v2` (receipts) instead.

# Distributed Fine-Tuning of Qwen2.5-VL for Structured Document Parsing

**Project Status:** Active / Portfolio Template

**Domain:** Multimodal Deep Learning, Information Extraction, High-Performance Computing (HPC)

**Primary Tools:** PyTorch, SLURM, Hugging Face (TRL, Accelerate, PEFT), FSDP / DeepSpeed, Qwen2.5-VL

---

## Executive Summary

This project establishes an end-to-end distributed training pipeline to fine-tune a state-of-the-art Vision-Language Model (VLM)—specifically **Qwen2.5-VL (7B)**—to extract structured JSON data from complex visual documents such as financial tables, invoices, blueprints, or engineering schematics. Operating on a multi-node SLURM cluster, the pipeline leverages Hugging Face's `TRL` (Transformer Reinforcement Learning) library and PyTorch FSDP (Fully Sharded Data Parallel) to manage the massive memory footprint of visual tokens.

> **CV Highlight:** Proves your ability to adapt SOTA multimodal architectures to solve real-world enterprise bottlenecks (unstructured document parsing) while independently navigating the complexities of scaling heavy Vision Transformer (ViT) workloads across GPU clusters.

---

## Technical Stack & Infrastructure

| Category | Technologies Used |
| --- | --- |
| **Model Architecture** | Qwen2.5-VL (7B), dynamic resolution Vision Transformers (ViT) |
| **Distributed Orchestration** | SLURM Workload Manager, Hugging Face Accelerate |
| **Memory Optimization** | PyTorch FSDP (`FULL_SHARD`), PEFT (LoRA), Flash Attention |
| **Training Framework** | Hugging Face `TRL` (`SFTTrainer` with VLM support) |
| **Telemetry & MLOps** | Weights & Biases (W&B), Hugging Face Hub (`huggingface_hub`) |

---

## Project Architecture & Phased Implementation

Training a VLM is fundamentally different from a pure text LLM because high-resolution images exponentially increase the sequence length. This project is structured to carefully handle the intersection of visual inputs, JSON generation, and multi-GPU memory limitations.

### Phase 1: Multimodal Data Pipeline & Formatting

The biggest hurdle in VLM fine-tuning is correctly mapping images and bounding boxes to the model's expected conversational format.

* **Dataset Selection:** Utilize an existing high-quality dataset like **DocVQA** or a synthetic table-to-JSON dataset (e.g., Hugging Face's `Glazkov/qwen2.5-vl-table-extraction` data).
* **Chat Template Formatting:** You will use Hugging Face `datasets` to map raw images and OCR targets into the strict multimodal chat schema required by `SFTTrainer`. Every sequence must properly interleave image tokens and system prompts defining the JSON extraction schema:
```json
[
  {
    "role": "user",
    "content": [
      {"type": "image", "image": "<PIL.Image>"},
      {"type": "text", "text": "Extract the data from this table and return it as a JSON array with keys: 'parameter', 'value'."}
    ]
  }
]

```


* **Dynamic Resolution Handling:** Qwen2.5-VL utilizes a dynamic FPS and dynamic resolution ViT. You will configure the processor to cap the maximum resolution to prevent a single high-definition document from triggering an Out-Of-Memory (OOM) error across the cluster.

### Phase 2: Distributed VLM Fine-Tuning (SFTTrainer + FSDP)

Once the data is tokenized, the focus shifts to infrastructure and memory optimization.

* **LoRA Injection:** Instead of full-parameter fine-tuning, you will use `peft` to inject Low-Rank Adaptation (LoRA) adapters into the attention modules of both the text backbone and, optionally, the visual encoder.
* **FSDP via Accelerate:** Because even a 7B VLM with long visual contexts exceeds the VRAM of a single 80GB GPU, you will configure `accelerate config` to use **FSDP (Fully Sharded Data Parallel)**. This shards the model weights, gradients, and optimizer states across your SLURM cluster's GPUs (e.g., `ShardingStrategy.FULL_SHARD`).
* **VLM SFTTrainer:** You will utilize TRL's `SFTTrainer` (Supervised Fine-Tuning Trainer) tailored for VLMs. The trainer automatically handles the cross-entropy loss computation over the generated JSON tokens while masking the visual inputs and prompt instructions so the model only learns to generate the answer.

---

## Expected Challenges & Solutions

1. **Exploding Sequence Lengths & OOM Crashes:**
* *The Problem:* A standard text prompt might be 500 tokens. A single high-res document image can easily add 4,000+ visual tokens, immediately blowing up the attention matrix memory.
* *The Solution:* Implement **Gradient Checkpointing** (which trades compute for memory by recomputing activations during the backward pass) and aggressively downscale the maximum image resolution during the processor initialization. Use `loss_type="chunked_nll"` in `SFTConfig` to prevent peak activation memory from scaling with the full vocabulary.


2. **Imbalanced Batching Across GPUs:**
* *The Problem:* Documents have different aspect ratios and resolutions, resulting in highly variable sequence lengths. If one GPU processes a massive blueprint while another processes a small receipt, the cluster will stall waiting for the slowest GPU to finish its backward pass.
* *The Solution:* Use dynamic padding, ensure uniform batch sizes, or group documents of similar resolutions together during dataloading.


3. **JSON Hallucinations:**
* *The Problem:* The model extracts the right data but breaks the JSON syntax (e.g., missing commas, unescaped quotes).
* *The Solution:* Incorporate a strong System Message enforcing the strict schema, and ensure your target dataset is perfectly formatted.



---

## Deliverables & Artifacts

1. **Open-Source Weights:** A fine-tuned Qwen2.5-VL LoRA adapter uploaded to the Hugging Face Hub, specifically tagged for `document-parsing` and `image-text-to-text`.
2. **Inference Script (`vllm` or `transformers`):** A clean Python script demonstrating how to load the base VLM, merge the LoRA weights, and pass it a raw PDF/Image to receive structured JSON out.
3. **SLURM & Accelerate Configurations:** A GitHub repository showcasing your `sbatch` scripts, FSDP `yaml` configs, and multimodal data prep code—proving to recruiters that you can manage the HPC orchestration, not just the model math.
