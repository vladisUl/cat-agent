"""Lossless content blocks plus non-duplicated structured MCP result data.

Only machine-checkable equivalence is used: JSON values (not JSON spelling)
and the SDK's single-key ``result`` wrapper around an exact text/JSON value.
No prose summarization, substring matching, or media interpretation is done.
This adapter has no MCP SDK dependency.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json

_ABSENT = object()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous JSON object")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("not a JSON number")


def _parse_text(text):
    try:
        return json.loads(text, parse_float=Decimal, object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (ValueError, RecursionError):
        return _ABSENT


def _same(left, right):
    # Python considers True == 1; JSON does not. Arrays also retain order/count.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    numbers = (int, float, Decimal)
    if isinstance(left, numbers) and isinstance(right, numbers):
        return Decimal(str(left)) == Decimal(str(right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _additional(value, candidates):
    """Subtract equal object fields at the SAME path; arrays are indivisible.

    Conflicting representations are retained rather than choosing an authority.
    Empty objects/arrays, false, zero and null are data, not missing values.
    """
    if candidates and all(_same(value, other) for other in candidates):
        return _ABSENT
    if isinstance(value, dict) and value:
        objects = [other for other in candidates if isinstance(other, dict)]
        result = {}
        for key, item in value.items():
            rest = _additional(item, [other[key] for other in objects if key in other])
            if rest is not _ABSENT:
                result[key] = rest
        return result if result else _ABSENT
    return deepcopy(value)


def normalize_mcp_result(result):
    """Return the common manager/agent payload, without dumping CallToolResult."""
    content = []
    candidates = []
    texts = []
    for block in result.content:
        # Dump blocks individually so their type-specific fields, annotations,
        # URI, MIME type and media data survive. Unknown blocks are preserved too.
        item = (deepcopy(block) if isinstance(block, dict) else
                block.model_dump(mode="json", by_alias=True, exclude_none=True))
        content.append(item)
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            text = item["text"]
            texts.append(text)
            parsed = _parse_text(text)
            if parsed is not _ABSENT:
                candidates.append(parsed)

    payload = {"isError": bool(result.is_error), "content": content}
    structured = result.structured_content
    if structured is not None:
        rest = _additional(structured, candidates)
        # Official SDK wraps scalar/list returns in {"result": ...}. Do not
        # apply this to arbitrary keys: {"status": "ok"} adds a named meaning.
        if isinstance(structured, dict) and set(structured) == {"result"}:
            value = structured["result"]
            if (len(texts) == 1 and isinstance(value, str) and value == texts[0]) or (
                candidates and all(_same(value, other) for other in candidates)
            ):
                rest = _ABSENT
            elif (isinstance(value, list) and value and len(value) == len(texts)
                  and len(texts) == len(content)):
                # Some SDK servers render list items as separate text blocks.
                # Preserve order/multiplicity and require a match for every item.
                if all(_same(item, text if isinstance(item, str) else _parse_text(text))
                       for item, text in zip(value, texts)):
                    rest = _ABSENT
        if rest is not _ABSENT:
            payload["structuredContent"] = rest
    return payload


def format_mcp_result(result):
    return "MCP_RESULT\n" + json.dumps(normalize_mcp_result(result), ensure_ascii=False,
                                       separators=(",", ":"), allow_nan=False)
