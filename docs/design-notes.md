# Design notes

## Sizing

Measured on the real CORD-v2 train split (800 rows) with `scripts/dataset_stats.py`:

| Quantity | p50 | p95 | max |
|---|---|---|---|
| Image pixels | 1.12 Mpx | 9.44 Mpx | 9.44 Mpx |
| Uncapped visual tokens | 1428 | 12037 | |
| `gt_parse` JSON tokens | 123 | 300 | 682 |

The uncapped p95 of 12037 visual tokens is why `max_pixels` is not optional. With
`max_pixels = 1280 * 28 * 28` the visual span is capped at 1280 tokens, so the longest
sequence is 1280 + ~150 prompt + 682 completion ~= 2.1 k tokens, under `max_length=4096`.
Nothing gets truncated. That matters because `truncation_mode` is forced to `"keep_start"`
for vision datasets: a too-small `max_length` would silently cut the JSON answer off the
end of the label.
