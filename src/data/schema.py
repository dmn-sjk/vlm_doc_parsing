"""Target schema, prompts, and CORD ground-truth normalisation.

The model is trained to emit **one compact JSON object** describing a receipt. Everything
about that object -- key names, key order, value types, what is omitted -- is decided here
and nowhere else, so that the string in the system prompt and the string in the training
target can never drift apart.

Design decisions
----------------

*Why rename CORD's keys.* Raw CORD ``gt_parse`` uses annotation-tool shorthand (``nm``,
``cnt``, ``unitprice``, ``menutype_cnt``) and is structurally inconsistent: ``menu`` is a
bare dict when the receipt has a single line item and a list otherwise, ``sub_total`` is a
list in exactly one example, and any leaf may be either a string or a list of strings
(the annotator split a value across two OCR lines). Training on that raw shape teaches the
model the annotation tool's quirks rather than the extraction task. :func:`normalize_gt`
collapses all of it into one fixed schema.

*Why omit absent fields instead of emitting ``null``.* Receipts genuinely differ -- 319 of
the 1000 CORD receipts have no subtotal block at all, and no single receipt carries more
than a handful of the eight ``total`` sub-fields. Emitting the full key set with ``null``
padding would roughly triple the target length for no information gain, and would teach
the model to hallucinate a fixed skeleton. Absent means absent.

*Why canonical key order rather than sorted keys.* Order is fixed by the tables below, so
the target string is fully deterministic (the requirement), and it matches the order the
keys are listed in the system prompt, so the model sees prompt and target agree. Alphabetic
sorting would be equally deterministic but would scramble the natural reading order of a
receipt (``name`` before ``price``, ``subtotal`` before ``total``).

*Why ``separators=(",", ":")`` and ``ensure_ascii=False``.* Both are pure token savings on
a target the model must emit at every step of training and inference:

* Compact separators drop the ``", "`` / ``": "`` spaces json.dumps adds by default. On
  CORD that is ~1 token saved per field, ~40 tokens per receipt.
* ``ensure_ascii=False`` keeps Indonesian/Korean characters as literal UTF-8 instead of
  ``\\uXXXX`` escapes. A single escaped CJK character costs ~6 tokens versus ~1-2 for the
  literal, and the escaped form is far harder for the model to get syntactically right.

No indentation and no trailing newline: the target is exactly the JSON and nothing else,
so ``json.loads(prediction)`` is a fair pass/fail metric at eval time.

Schema (canonical)
------------------

.. code-block:: text

    {
      "menu":      [ <item>, ... ],     # line items, in receipt order
      "void_menu": [ <item>, ... ],     # voided/cancelled line items (rare)
      "subtotal":  { subtotal_price, discount_price, service_price,
                     other_service_price, tax_price, etc },
      "total":     { total_price, etc, menu_type_count, menu_qty_count,
                     cash_price, change_price, credit_card_price, emoney_price }
    }

    <item> = { name, item_no, count, unit_price, price, discount_price,
               item_subtotal, vat, etc, sub: [ <item>, ... ] }

All leaf values are **strings copied verbatim from the receipt** (``"75,000"``, ``"1 x"``)
-- no numeric parsing, no currency normalisation. The receipts use ``.``/``,`` as thousands
separators inconsistently, so any numeric coercion would silently corrupt values; keeping
strings makes exact-match field accuracy a meaningful metric.
"""

from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------------------
# CORD key -> canonical key. Order of each tuple list *is* the key order in the output.
# Any CORD key not listed here raises, so a schema change in the data fails loudly rather
# than silently dropping a field from every training target.
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when CORD ground truth contains a key or shape this schema does not model."""


def _scalar(value: Any) -> str | None:
    """Collapse a CORD leaf into a single trimmed string, or ``None`` if empty.

    Leaves are usually strings, but the annotation tool emits a *list* of strings when a
    value was split across OCR lines (observed on ``nm``, ``cnt``, ``price``,
    ``subtotal_price``, ``tax_price``, ``etc``, ...). Those fragments are re-joined with a
    single space, which is how they read on the receipt.
    """
    if value is None:
        return None
    if isinstance(value, list):
        parts = [p for p in (_scalar(v) for v in value) if p]
        return " ".join(parts) or None
    if isinstance(value, str):
        return value.strip() or None
    # ints/floats do not occur in CORD today, but do not lose data if they appear.
    return str(value)


def _as_list(value: Any) -> list[Any]:
    """CORD stores a single-element block as a bare dict; normalise to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_dicts(values: list[Any]) -> dict[str, Any]:
    """Merge a list of dicts left-to-right (``sub_total`` is a list in one example)."""
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
    """Parse a CORD ``ground_truth`` string into the canonical extraction schema.

    Args:
        ground_truth: the raw ``ground_truth`` column of ``naver-clova-ix/cord-v2``, a JSON
            **string** whose ``"gt_parse"`` key holds the nested receipt annotation. The
            sibling keys (``meta``, ``valid_line``, ``roi``, ``repeating_symbol``,
            ``dontcare``) are annotation bookkeeping and are ignored.

    Returns:
        A dict in canonical key order (see the module docstring). Empty blocks and absent
        fields are omitted, so ``{}`` is a legal -- if degenerate -- result.

    Raises:
        SchemaError: if the annotation carries a key this schema does not model. This is
            deliberate: a silent drop would poison every training target for that field.
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

    # Re-emit in canonical order so the serialised string is stable.
    return {k: blocks[k] for k in _TOP_LEVEL if k in blocks}


def dumps(obj: Any) -> str:
    """The one serialisation used for targets, and the one eval should compare against."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def target_json(ground_truth: str) -> str:
    """The exact assistant target string for one CORD example.

    Compact, UTF-8 literal, canonical key order, no trailing newline -- see the module
    docstring for why each of those was chosen.
    """
    return dumps(normalize_gt(ground_truth))
