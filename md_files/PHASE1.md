# Phase 1 — Multimodal Data Pipeline & Formatting

Status: **done and verified.** Everything below was run on the cluster login node (CPU only,
no model weights loaded) and the pasted output is real.

| File | Role |
| --- | --- |
| `src/data/schema.py` | system/user prompts, target schema, CORD ground-truth normalisation |
| `src/data/resolution.py` | `smart_resize` replica, `estimate_visual_tokens`, `build_processor` |
| `src/data/cord.py` | `build_datasets(cfg) -> DatasetDict` in TRL prompt-completion format |
| `src/data/grouping.py` | `estimate_lengths(...) -> list[int]` for the length-grouped sampler |
| `scripts/dataset_stats.py` | resolution / target-length statistics, `max_pixels` sweep |
| `scripts/inspect_batch.py` | asserts image-token count and loss masking on a real collated batch |

---

## 1. Example format: prompt-completion, string content

Every example is exactly:

```python
{
  "images":     [PIL.Image.Image],                                  # one page image
  "prompt":     [{"role": "system",    "content": SYSTEM_PROMPT},
                 {"role": "user",      "content": USER_INSTRUCTION}],
  "completion": [{"role": "assistant", "content": "<compact target JSON>"}],
}
```

**Why prompt-completion and not `messages`.** `SFTTrainer` raises on
`assistant_only_loss=True` whenever the dataset has an `image`/`images` column
(`trl/trainer/sft_trainer.py:1052`). The only remaining way to put loss on the answer alone
is the prompt-completion dataset type, where the vision collator builds a `completion_mask`
and applies `labels[completion_mask == 0] = -100`
(`DataCollatorForVisionLanguageModeling._collate_prompt_completion`, `:774`). A
`messages` column would train on the 615-token system prompt as well.

**Why `content` is a plain string and not a list of content blocks.** The collator calls
`prepare_multimodal_messages(example["prompt"], images=example["images"])` (`:707`). For
*string* content that function inserts one `{"type": "image"}` block per image ahead of the
text and fills in the PIL objects (`trl/data_utils.py:80-93`); if the blocks were already
present it would count them and require the counts to match. Pre-inserting them is
therefore redundant — and actively dangerous:

> `datasets` unifies the struct type of a list column. A mixed list of `{"type": "image"}`
> and `{"type": "text", "text": ...}` comes back as
> `{"type": "text", "text": "...", "image": None}`. The Qwen2.5-VL chat template branches on
> `content['type'] == 'image' or 'image' in content`, so that resurrected `image: None` key
> makes the template render the **text** block as a second
> `<|vision_start|><|image_pad|><|vision_end|>` — the classic "Image features and image
> tokens do not match" crash.

Plain strings sidestep the whole failure mode, and are what `prepare_multimodal_messages`
was designed for.

**Column names.** Only `["messages", "prompt", "completion", "image", "images"]` survive
`SFTTrainer._set_signature_columns_if_needed` (`:1683`) when `remove_unused_columns=True`.
No metadata column is emitted — anything extra would be silently dropped.

---

## 2. Target schema

`normalize_gt()` turns CORD's `gt_parse` into a fixed schema; `target_json()` serialises it
with `json.dumps(obj, ensure_ascii=False, separators=(",", ":"))`.

Raw CORD is not usable as a target as-is: `menu` is a bare dict for 431/1000 receipts and a
list for the other 569, `sub_total` is a list in exactly one example, and any leaf may be a
*list of strings* when the annotator split a value across two OCR lines (observed on `nm`,
`cnt`, `price`, `subtotal_price`, `tax_price`, `etc`, …). Training on that shape teaches the
annotation tool's quirks, not the extraction task.

Canonical schema (also embedded verbatim in the system prompt as
`JSON_SCHEMA_DESCRIPTION`):

```
{ "menu": [<item>…], "void_menu": [<item>…],
  "subtotal": {subtotal_price, discount_price, service_price,
               other_service_price, tax_price, etc},
  "total":    {total_price, etc, menu_type_count, menu_qty_count,
               cash_price, change_price, credit_card_price, emoney_price} }

<item> = {name, item_no, count, unit_price, price, discount_price,
          item_subtotal, vat, etc, sub: [<item>…]}
```

Rules and their reasons:

* **Absent means absent.** 319/1000 receipts have no subtotal block; no receipt carries more
  than a few of the eight `total` sub-fields. Emitting the full key set with `null` padding
  would roughly triple the target and teach the model to hallucinate a skeleton.
* **All leaves are verbatim strings.** The receipts mix `.` and `,` as thousands separators;
  any numeric coercion silently corrupts values and would make exact-match field accuracy
  meaningless.
* **Canonical key order, not `sort_keys=True`.** Equally deterministic, but it matches the
  order the keys are listed in the system prompt and the natural reading order of a receipt.
* **Compact separators + `ensure_ascii=False`.** ~1 token saved per field from dropping the
  `", "`/`": "` spaces (~40/receipt), and literal UTF-8 instead of `\uXXXX` for the
  Indonesian/Korean characters (an escaped CJK char costs ~6 tokens vs ~1–2 literal, and is
  much harder to emit syntactically correctly).
* **Unknown keys raise `SchemaError`** rather than being dropped — a silent drop would poison
  every training target for that field.

Verified over the full dataset: **1000/1000 examples normalise without a `SchemaError` and
every `target_json()` output round-trips through `json.loads`.** Target string lengths:
min 78 chars, p50 258, p95 627, max 1375.

---

## 3. Resolution cap

`estimate_visual_tokens(w, h, min_pixels, max_pixels)` is a line-for-line replica of
`transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize` (transformers 5.16.1;
`factor = patch_size * merge_size = 28`, tokens `= (h/28) * (w/28)`).

Cross-checked against the transformers oracle `Qwen2VLImageProcessor.get_number_of_image_patches`
over **2074 unique sizes (all 1000 CORD images plus 2000 random sizes): 0 mismatches**, and
against real `image_grid_thw` tensors inside `inspect_batch.py`.

### Chosen: `max_pixels = 1024 * 28 * 28 = 802816` (the `DataConfig` default — unchanged)

`scripts/dataset_stats.py` (train split, 800 examples), abbreviated:

```
configured caps  : min_pixels=200704 (256 tokens)  max_pixels=802816 (1024 tokens)
max_length       : 4096
prompt overhead  : 615 tokens (system + instruction + chat template, excluding visual tokens)
completion wrap  : 2 tokens (<|im_end|>\n)

train: 800 examples
  width      p0=204  p5=549  p25=576  p50=864  p75=1080  p95=2304  p100=3024
  height     p0=364  p5=864  p25=864  p50=1296  p75=1478  p95=4096  p100=4224
  megapixels p0=85  p5=497  p25=497  p50=1119  p75=1637  p95=9437  p100=12192  (x1000 px)
  aspect     min=1.00 median=1.50 max=2.96

  estimated visual tokens @ max_pixels=802816 (1024 tokens):
    p0=264  p5=651  p25=651  p50=1008  p75=1014  p95=1014  p100=1024
      264-327     |#                                              11
      327-391     |##                                             25
      391-454     |                                               1
      454-517     |                                               1
      517-581     |                                               2
      581-644     |                                               0
      644-707     |################                               192
      707-771     |                                               0
      771-834     |                                               0
      834-897     |                                               1
      897-961     |                                               3
      961-1024    |############################################## 564
  resized width  p0=280  p5=532  p25=588  p50=700  p75=728  p95=756  p100=896
  resized height p0=504  p5=868  p25=868  p50=1092  p75=1092  p95=1176  p100=1540

  target JSON length in tokens (incl. 2-token assistant wrapper):
    p0=33  p5=57  p25=80  p50=101  p75=142  p95=257  p100=572

  max_pixels sweep (drop = total tokens > max_length=4096):
     tokens  max_pixels | visual p50 visual p95 visual max | total p50 total p95 total max | dropped
        256      200704 |        247        252        288 |       963      1119      1439 |    0/800 (  0.0%)
        512      401408 |        486        494        506 |      1199      1354      1667 |    0/800 (  0.0%)
        768      602112 |        726        768        768 |      1428      1594      1907 |    0/800 (  0.0%)
       1024      802816 |       1008       1014       1024 |      1705      1857      2160 |    0/800 (  0.0%) <-- configured
       1280     1003520 |       1230       1247       1271 |      1935      2089      2393 |    0/800 (  0.0%)
       1536     1204224 |       1426       1508       1530 |      2137      2307      2620 |    0/800 (  0.0%)
       2048     1605632 |       1426       2028       2028 |      2137      2788      3163 |    0/800 (  0.0%)
       4096     3211264 |       1426       4080       4080 |      2137      4813      5185 |  120/800 ( 15.0%)
```

(`validation` and `test` behave identically: 0 dropped up to 2048 tokens, 15 % / 17 % dropped
at 4096.)

Reading of the sweep:

1. **The cap is not chosen to avoid drops.** Nothing is dropped anywhere below 2048 visual
   tokens; the `max_length=4096` filter is a safety net, not a live constraint. The cap is
   chosen for VRAM and step time.
2. **1024 barely degrades the median receipt.** The median source image is 864×1296; at this
   cap it is resized to 728×1092 — 0.84× linear, well inside the range where receipt text
   stays legible. What the cap actually bites is the 9–12 megapixel tail (2304×4096,
   3024×4224), which the checkpoint's own default (`max_pixels=12845056`, 16384 tokens) would
   turn into 4000+ visual tokens each.
3. **Going to 1280 buys 22 % more sequence length** (p50 total 1705 → 1935) for images that
   are already above native resolution for half the dataset. Going down to 512 halves the
   sequence but resizes the median receipt to 532×798, at which point small-print line items
   start to matter. 1024 is the knee.
4. **Do not forget `min_pixels`.** At `min_pixels = 256*28*28` the smallest images
   (204×364) are *upscaled* to 280×504 = 264 tokens. That is deliberate: it stops a tiny
   thumbnail from producing a 30-token image the ViT cannot read.

Note the distribution is strongly **bimodal** (651 vs 1008–1014 tokens, plus a small
264–390 tail). That is exactly the ragged-batch problem from CLAUDE.md gotcha #2 and is why
`src/data/grouping.py` exists.

---

## 4. Length signal for grouped batching

`src/data/grouping.py::estimate_lengths(dataset, processor, min_pixels=…, max_pixels=…)`
returns a `list[int]` aligned with dataset order:

```
total = len(prompt_ids) - n_images + sum(visual_tokens) + len(completion_ids)
```

It is **exact**, not heuristic — `inspect_batch.py` asserts it equals the real collated
`attention_mask.sum()` on every row it checks, and it does (see §5: `len=1867 (est 1867)`,
`1795/1795`, `1331/1331`, `1373/1373`).

It cannot live as a dataset column: `SFTTrainer` overrides the signature columns to
`["messages", "prompt", "completion", "image", "images"]` (`:1683`), so with the default
`remove_unused_columns=True` any `est_tokens` column is dropped before the sampler sees it,
and HF `Trainer`'s built-in `group_by_length` path would then try to fall back to
`input_ids`, which a vision dataset does not have. Phase 2 must pass the list into a
`LengthGroupedSampler` / `DistributedLengthGroupedSampler` directly (override
`Trainer._get_train_sampler`).

No image is decoded to compute it: the `images` column is temporarily cast to
`Image(decode=False)` and Pillow reads only the header for `.size`.

---

## 5. Verified output of `inspect_batch.py`

`inspect_batch.py` builds a batch with **the same collator class `SFTTrainer` instantiates
for a vision dataset** (`DataCollatorForVisionLanguageModeling`, chosen at `:1221-1228`) and
asserts per example:

1. `<|image_pad|>` count in `input_ids` == `image_grid_thw.prod() // merge_size**2` == what
   `estimate_visual_tokens` predicted from the raw image size;
2. the unmasked span (`labels != -100`) is contiguous and decodes to **exactly** the target
   JSON plus `<|im_end|>\n`, and `<|im_end|>` is inside the loss (or the model never learns
   to stop);
3. everything before it — system prompt, instruction, vision tokens — and all padding is
   `-100`;
4. nothing was truncated by `max_length`;
5. `estimate_lengths` equals the real sequence length.

Non-zero exit on any failure.

```
$ source scripts/env.sh
$ HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python scripts/inspect_batch.py --split validation --batch-size 2 --num-batches 2 --num-proc 2

INFO src.data.cord: train: kept all 800 examples (longest 2160 tokens, max_length=4096)
INFO src.data.cord: train: subsampled to 4 examples (seed=42)
INFO src.data.cord: validation: kept all 100 examples (longest 1868 tokens, max_length=4096)
INFO src.data.cord: validation: subsampled to 4 examples (seed=42)
INFO src.data.cord: test: kept all 100 examples (longest 1958 tokens, max_length=4096)
INFO src.data.cord: test: subsampled to 4 examples (seed=42)
collator: DataCollatorForVisionLanguageModeling (completion_only_loss=True, max_length=4096)
processor: Qwen2_5_VLProcessor size=SizeDict(height=None, width=None, longest_edge=802816, shortest_edge=200704, max_height=None, max_width=None)
split=validation n=4 batch_size=2

--- batch 0 --- keys=['attention_mask', 'image_grid_thw', 'input_ids', 'labels', 'mm_token_type_ids', 'pixel_values']
    input_ids (2, 1867)  labels (2, 1867)  pixel_values (8088, 1176)  image_grid_thw [[1, 84, 48], [1, 78, 52]]
    OK  row 0: len=1867 (est 1867) visual=1008 prompt_masked=1623 completion=244 image=[(2304, 4096)]
    OK  row 1: len=1795 (est 1795) visual=1014 prompt_masked=1629 completion=166 image=[(864, 1296)]

================ decoded PROMPT (all labels == -100) ================
<|im_start|>system
You are a document information-extraction engine. You are given a photograph of a retail receipt and you return its contents as a single JSON object.

OUTPUT SCHEMA
{
  "menu": [                      // one entry per line item, in the order printed
    {
      "name":           string,  // item name as printed
      ...
    }
  ],
  ...
}

RULES
1. Respond with the JSON object and NOTHING else. No prose, no explanation, no apology, no markdown code fences, no ```json marker, no leading or trailing text or whitespace. Your entire response must start with `{` and end with `}`.
2. Emit only keys that are actually present on the receipt. Omit any key whose value is absent; never emit `null`, `""`, `"N/A"`, or a placeholder.
3. Every value is a string, transcribed verbatim from the receipt including its thousands separators and currency punctuation (e.g. "75,000", "1 x"). Do not convert to numbers, do not reformat, do not compute values that are not printed.
4. Preserve the printed order of the line items in "menu".
5. The output must be strictly valid JSON: double-quoted keys and values, no trailing commas, no comments, all inner quotes and backslashes escaped.
6. If the receipt is unreadable, still return a syntactically valid JSON object containing only the fields you can read.
<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|> x1008<|vision_end|>Extract this receipt as JSON using the schema above. Return only the JSON object.<|im_end|>
<|im_start|>assistant

================ decoded UNMASKED span (labels != -100) ==========
{"menu":[{"name":"XLB Org Pork 10x","count":"1","price":"78,000"},{"name":"Sicito Babi","count":"1","price":"65,000"},{"name":"Hotplate Tahu","count":"1","price":"58,000"},{"name":"DF Fish Fillet Garlc","count":"1","price":"108,000"},{"name":"LM Pork Belly","count":"1","price":"88,000"},{"name":"Siew Mai","count":"1","price":"38,800"},{"name":"Kwan Yin Cup","count":"5","price":"50,000"},{"name":"Herbal Jelly","count":"1","price":"38,000"},{"name":"Onde-Onde","count":"1","price":"38,000"}],"subtotal":{"subtotal_price":"561,800","service_price":"42,135","tax_price":"60,394"},"total":{"total_price":"664,329","menu_qty_count":"13"}}<|im_end|>

==================================================================

--- batch 1 --- keys=['attention_mask', 'image_grid_thw', 'input_ids', 'labels', 'mm_token_type_ids', 'pixel_values']
    input_ids (2, 1373)  labels (2, 1373)  pixel_values (5208, 1176)  image_grid_thw [[1, 62, 42], [1, 62, 42]]
    OK  row 2: len=1331 (est 1331) visual=651 prompt_masked=1266 completion=65 image=[(576, 864)]
    OK  row 3: len=1373 (est 1373) visual=651 prompt_masked=1266 completion=107 image=[(576, 864)]

checked 4 example(s); 0 failure(s)
ALL CHECKS PASSED
```

(The `OUTPUT SCHEMA` and `RULES` blocks are elided with `...` above; the script prints them
in full.)

**Negative control.** The same command with `-o train.completion_only_loss=false` exits `1`
and reports `unmasked span is not the target JSON` — so the assertions genuinely bind.

---

## 6. Things Phase 2 must know

1. **`SFTConfig.max_length` defaults to `1024` and that silently destroys training here.**
   Our sequences are 1300–2200 tokens and the ~1600-token prompt sits in front of the
   completion, so truncation removes the entire answer. Measured directly:

   ```
   SFTConfig.max_length=1024: input_ids(2, 1024) unmasked label tokens per row = [0, 0]
   SFTConfig.max_length=4096: input_ids(2, 1867) unmasked label tokens per row = [244, 166]
   ```

   Nothing in TRL warns about this. **Set `SFTConfig.max_length = DataConfig.max_length`
   (4096).**

2. **`completion_only_loss=True` must actually be passed.** `SFTConfig.completion_only_loss`
   defaults to `None`, which TRL resolves to `True` for a prompt-completion dataset
   (`:1200-1203`), so the default is fine — but `TrainConfig.completion_only_loss = True` in
   `src/config.py` should be forwarded explicitly rather than relied on implicitly.

3. **`pad_to_multiple_of` must stay `None`.** `_collate_prompt_completion` raises
   `NotImplementedError` for any other value (`:693-696`).

4. **Length grouping needs a sampler, not a column** — see §4. `args.group_by_length = True`
   on its own is not sufficient for a vision dataset.

5. **`build_datasets` signature.** `build_datasets(cfg: DataConfig, processor=None,
   model_id=None)`. `DataConfig` has no model field, so pass `model_cfg.base_model` (or set
   `$BASE_MODEL`, which `scripts/env.sh` already exports). Pass the trainer's own processor
   to avoid loading a second copy. Likewise `build_processor(cfg: DataConfig, model_id=None)`.

6. **Processor kwargs.** `AutoProcessor.from_pretrained(model_id, min_pixels=…, max_pixels=…)`
   is the correct path in transformers 5.16.1 — the kwargs are folded into
   `image_processor.size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)`
   (`image_processing_qwen2_vl.py:110-118`), and `build_processor` asserts it landed.
   Caveat: `_standardize_kwargs` only rebuilds the `SizeDict` when **both** are supplied
   (`:120-129`) — passing `max_pixels` alone as a per-call kwarg is silently ignored.
   The checkpoint's own `preprocessor_config.json` ships `max_pixels=12845056`
   (16 384 visual tokens), so *not* overriding is an instant OOM.

7. **`processor.tokenizer.padding_side` is irrelevant during training.** The vision collator
   passes an explicit `padding_side` on every call — `"left"` for the prompt half, `"right"`
   for the completion half (`:718`, `:725`). `build_processor` deliberately leaves the
   attribute alone. Inference code that batches `generate` must set `"left"` itself.

8. **Batch keys produced by the collator** (Qwen2.5-VL specific):
   `input_ids, attention_mask, labels, pixel_values, image_grid_thw, mm_token_type_ids`.
   `mm_token_type_ids` is a Qwen2.5-VL special case that TRL concatenates with zeros over
   the completion (`:742`); it must be forwarded to the model.

9. **Login-node fragility.** `datasets.map` with `num_proc=8` gets a worker SIGKILLed by the
   the cluster memory cgroup roughly half the time. Mitigations already in `src/data/cord.py`:
   `writer_batch_size=32` (the default 1000 buffers ~800 MB of PNGs per worker) and an
   automatic single-process retry. On compute nodes `num_proc=8` is fine. First build takes
   ~40 s for all three splits; afterwards it is served from the `datasets` cache under
   `$SCRATCH/huggingface/datasets`.

10. **Images stay lazy.** The formatting `map` runs on `Image(decode=False)` payloads and
    declares `List(Image())` output features, so PNG bytes are copied through without a
    PIL decode/re-encode round trip and decode lazily on `__getitem__`. The collator is what
    decodes, one batch at a time.

---

## 7. Not verified

* **Nothing has been run on a GPU.** No forward/backward pass, no check that the model
  accepts these batches — only that the collator produces them. That is the Phase 2 smoke
  test.
* **`num_proc=8` on a compute node** was not exercised; only the login node, where it is
  unreliable for the reason above.
* **Multiple images per example** is supported by the code path but CORD is single-image, so
  the multi-image branch of `estimate_lengths` / `inspect_batch` has never actually run with
  `n_images > 1`.
* **Prompt cost.** The system prompt is 615 tokens on every example — ~35 % of the median
  sequence and ~50 % of the shortest. It is masked out of the loss but not out of the
  forward pass. If step time turns out to be prompt-bound, shortening
  `JSON_SCHEMA_DESCRIPTION` is the obvious lever; it was not tuned because JSON validity is
  a headline metric and the brief calls for a forceful schema prompt.
