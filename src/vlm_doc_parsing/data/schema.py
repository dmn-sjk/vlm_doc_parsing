"""Target schema, prompts, and CORD ground-truth normalisation.

The model emits one compact JSON object per receipt. Key names, key order and omission rules
live only here, so the system prompt and the training target can't drift apart.

- CORD's shorthand keys are renamed, and its inconsistent shapes (dict-or-list blocks,
  leaves split into lists across OCR lines) collapse into one fixed schema.
- Absent fields are omitted, never `null`; key order is canonical, not sorted.
- Leaves are verbatim strings: receipts mix `.` and `,` thousands separators.
- Serialised compactly with `ensure_ascii=False` to save tokens.

Rationale and the schema diagram: docs/design-notes.md#target-schema.
"""

from __future__ import annotations

import json
from typing import Any

# CORD key -> canonical key, in output order. An unlisted CORD key raises rather than being
# silently dropped from every target.

_ITEM_FIELDS: tuple[tuple[str, str], ...] = (
    ("nm", "name"),
    ("num", "item_no"),
    ("cnt", "count"),
    ("unitprice", "unit_price"),
    ("price", "price"),
    ("discountprice", "discount_price"),
    ("itemsubtotal", "item_subtotal"),
    ("vatyn", "vat"),
    ("etc", "etc"),
    ("sub", "sub"),  # nested list of <item>; handled recursively
)

_SUBTOTAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("subtotal_price", "subtotal_price"),
    ("discount_price", "discount_price"),
    ("service_price", "service_price"),
    ("othersvc_price", "other_service_price"),
    ("tax_price", "tax_price"),
    ("etc", "etc"),
)

_TOTAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("total_price", "total_price"),
    ("total_etc", "etc"),
    ("menutype_cnt", "menu_type_count"),
    ("menuqty_cnt", "menu_qty_count"),
    ("cashprice", "cash_price"),
    ("changeprice", "change_price"),
    ("creditcardprice", "credit_card_price"),
    ("emoneyprice", "emoney_price"),
)

# Top-level blocks, in canonical order.
_TOP_LEVEL: tuple[str, ...] = ("menu", "void_menu", "subtotal", "total")


JSON_SCHEMA_DESCRIPTION = """\
{
  "menu": [                      // one entry per line item, in the order printed
    {
      "name":           string,  // item name as printed
      "item_no":        string,  // item/product code, if printed
      "count":          string,  // quantity as printed, e.g. "2 x"
      "unit_price":     string,  // price of one unit
      "price":          string,  // line total for this item
      "discount_price": string,  // discount applied to this line
      "item_subtotal":  string,
      "vat":            string,
      "etc":            string,
      "sub": [ ... ]             // nested sub-items, same fields as above
    }
  ],
  "void_menu": [ ... ],          // cancelled / voided line items, same fields as "menu"
  "subtotal": {
    "subtotal_price":      string,
    "discount_price":      string,
    "service_price":       string,
    "other_service_price": string,
    "tax_price":           string,
    "etc":                 string
  },
  "total": {
    "total_price":       string,
    "etc":               string,
    "menu_type_count":   string,  // number of distinct items
    "menu_qty_count":    string,  // total number of units
    "cash_price":        string,
    "change_price":      string,
    "credit_card_price": string,
    "emoney_price":      string
  }
}"""


SYSTEM_PROMPT = f"""\
You are a document information-extraction engine. You are given a photograph of a retail \
receipt and you return its contents as a single JSON object.

OUTPUT SCHEMA
{JSON_SCHEMA_DESCRIPTION}

RULES
1. Respond with the JSON object and NOTHING else. No prose, no explanation, no apology, \
no markdown code fences, no ```json marker, no leading or trailing text or whitespace. \
Your entire response must start with `{{` and end with `}}`.
2. Emit only keys that are actually present on the receipt. Omit any key whose value is \
absent; never emit `null`, `""`, `"N/A"`, or a placeholder.
3. Every value is a string, transcribed verbatim from the receipt including its thousands \
separators and currency punctuation (e.g. "75,000", "1 x"). Do not convert to numbers, do \
not reformat, do not compute values that are not printed.
4. Preserve the printed order of the line items in "menu".
5. The output must be strictly valid JSON: double-quoted keys and values, no trailing \
commas, no comments, all inner quotes and backslashes escaped.
6. If the receipt is unreadable, still return a syntactically valid JSON object containing \
only the fields you can read.
"""


USER_INSTRUCTION = (
    "Extract this receipt as JSON using the schema above. Return only the JSON object."
)


class SchemaError(ValueError):
    """Raised when CORD ground truth contains a key or shape this schema does not model."""


def _scalar(value: Any) -> str | None:
    """Collapse a CORD leaf to one trimmed string, or `None` if empty.

    A value split across OCR lines arrives as a list of strings; re-join it with spaces.
    """
    if value is None:
        return None
    if isinstance(value, list):
        parts = [p for p in (_scalar(v) for v in value) if p]
        return " ".join(parts) or None
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def _as_list(value: Any) -> list[Any]:
    """CORD stores a single-element block as a bare dict; normalise to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_dicts(values: list[Any]) -> dict[str, Any]:
    """Merge a list of dicts left-to-right (`sub_total` is a list in one example)."""
    out: dict[str, Any] = {}
    for v in values:
        if not isinstance(v, dict):
            raise SchemaError(f"expected a dict block, got {type(v).__name__}: {v!r}")
        out.update(v)
    return out


def _normalize_item(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SchemaError(f"menu entry must be a dict, got {type(raw).__name__}: {raw!r}")
    known = {src for src, _ in _ITEM_FIELDS}
    unknown = set(raw) - known
    if unknown:
        raise SchemaError(f"unknown menu-item keys: {sorted(unknown)}")

    item: dict[str, Any] = {}
    for src, dst in _ITEM_FIELDS:
        if src not in raw:
            continue
        if src == "sub":
            sub = [i for i in (_normalize_item(s) for s in _as_list(raw[src])) if i]
            if sub:
                item[dst] = sub
            continue
        val = _scalar(raw[src])
        if val is not None:
            item[dst] = val
    return item


def _normalize_block(raw: Any, fields: tuple[tuple[str, str], ...], name: str) -> dict[str, str]:
    merged = _merge_dicts(_as_list(raw))
    known = {src for src, _ in fields}
    unknown = set(merged) - known
    if unknown:
        raise SchemaError(f"unknown {name} keys: {sorted(unknown)}")
    out: dict[str, str] = {}
    for src, dst in fields:
        if src not in merged:
            continue
        val = _scalar(merged[src])
        if val is not None:
            out[dst] = val
    return out


def normalize_gt(ground_truth: str) -> dict[str, Any]:
    """Parse CORD's `ground_truth` JSON string (its `gt_parse` key) into the canonical schema.

    Empty blocks are omitted, so `{}` is a legal result. An unmodelled key raises SchemaError.
    """
    parsed = json.loads(ground_truth)
    if "gt_parse" not in parsed:
        raise SchemaError("ground_truth has no 'gt_parse' key")
    gt = parsed["gt_parse"]
    if not isinstance(gt, dict):
        raise SchemaError(f"gt_parse must be a dict, got {type(gt).__name__}")

    known_top = {"menu", "void_menu", "sub_total", "total"}
    unknown = set(gt) - known_top
    if unknown:
        raise SchemaError(f"unknown gt_parse keys: {sorted(unknown)}")

    blocks: dict[str, Any] = {}

    for src, dst in (("menu", "menu"), ("void_menu", "void_menu")):
        if src in gt:
            items = [i for i in (_normalize_item(x) for x in _as_list(gt[src])) if i]
            if items:
                blocks[dst] = items

    if "sub_total" in gt:
        sub = _normalize_block(gt["sub_total"], _SUBTOTAL_FIELDS, "sub_total")
        if sub:
            blocks["subtotal"] = sub

    if "total" in gt:
        tot = _normalize_block(gt["total"], _TOTAL_FIELDS, "total")
        if tot:
            blocks["total"] = tot

    return {k: blocks[k] for k in _TOP_LEVEL if k in blocks}


def dumps(obj: Any) -> str:
    """The one serialisation used for targets, and the one eval should compare against."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def target_json(ground_truth: str) -> str:
    """The exact assistant target string for one CORD example."""
    return dumps(normalize_gt(ground_truth))
