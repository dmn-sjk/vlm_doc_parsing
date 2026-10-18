# Examples use the prompt-completion format, not `messages`

The model must learn only the JSON answer: image tokens, the system prompt and the
instruction are masked to `-100`. For a `messages` dataset that would mean
`assistant_only_loss=True`, which `SFTTrainer` rejects whenever the dataset has an `image` or
`images` column. A `messages` dataset without it trains on the 615-token system prompt as
well. So `cord.format_example` emits `{"images", "prompt", "completion"}`: TRL's vision
collator then builds a `completion_mask` and sets every label outside the completion to
`-100` (`completion_only_loss=True`, passed explicitly rather than relying on TRL's default).

## Consequences

- Message `content` is a plain string; the collator inserts the image block itself. Content
  block lists are unsafe here (see [design notes](../design-notes.md#example-format)).
- The mask is checked by decoding, not trusted: `scripts/inspect_batch.py` asserts the
  unmasked span is exactly the target JSON plus `<|im_end|>\n`, and with
  `completion_only_loss=false` it exits 1.
- `SFTConfig.max_length` must cover the whole sequence: truncation is `keep_start` for vision
  datasets, so a short limit cuts the completion and masks every label.
- Packing and padding-free batching are unavailable (both rejected for vision datasets).
