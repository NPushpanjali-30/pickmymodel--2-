"""scoring.py - simple, deterministic quality checks for model answers.

Every check function takes the model's answer text and returns True or False.
Use `score_answer(test, answer)` as the single entry point.
"""
import json
import re


def strip_think(text):
    """Remove <think>...</think> blocks that reasoning models may print."""
    if not text:
        return ""
    # Closed blocks first, then an unclosed block that runs to the end.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_json(text):
    """Find and parse JSON inside text. Tolerates ```json fences and extra prose.

    Returns the parsed object, or None if nothing valid is found.
    """
    if not text:
        return None
    # 1) Try the content of a markdown code fence first.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            pass
    # 2) Scan for the first balanced {...} or [...] that parses.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except ValueError:
                continue
    return None


def check_json_valid(answer, expected):
    """Pass if the answer has valid JSON containing every required key."""
    data = extract_json(answer)
    if data is None:
        return False
    required = expected or []
    if not required:
        return True
    if isinstance(data, list):  # accept a list of objects; check the first one
        data = data[0] if data and isinstance(data[0], dict) else {}
    if not isinstance(data, dict):
        return False
    return all(key in data for key in required)


def check_contains_keywords(answer, expected, min_matches=None):
    """Pass if at least `min_matches` keywords appear (default: all of them)."""
    words = expected or []
    needed = len(words) if min_matches is None else min_matches
    lowered = answer.lower()
    hits = sum(1 for w in words if str(w).lower() in lowered)
    return hits >= needed


def _normalize(text):
    """Lowercase, trim, drop quotes and trailing . or ! for exact matching."""
    text = text.strip().lower()
    text = text.strip("\"'`“”‘’ \n\t")
    text = text.rstrip(".!")
    return text.strip("\"'`“”‘’ \n\t")


def check_exact_match(answer, expected):
    """Pass if the answer equals `expected` ignoring case, quotes, trailing . or !"""
    return _normalize(answer) == _normalize(str(expected))


def check_tool_call(answer, expected):
    """Pass if the answer is a JSON tool call with the right tool and argument names.

    expected = {"tool": "get_weather", "arguments": ["city"]}
    """
    data = extract_json(answer)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return False
    name = data.get("tool") or data.get("name") or data.get("function")
    args = data.get("arguments", data.get("parameters", data.get("args")))
    if name != expected.get("tool"):
        return False
    if not isinstance(args, dict):
        return False
    return all(key in args and args[key] not in ("", None) for key in expected.get("arguments", []))


def score_answer(test, answer):
    """Strip <think> blocks, then run the check named in test["check"]."""
    clean = strip_think(answer)
    kind = test.get("check")
    expected = test.get("expected")
    if kind == "json_valid":
        return check_json_valid(clean, expected)
    if kind == "contains_keywords":
        return check_contains_keywords(clean, expected, test.get("min_matches"))
    if kind == "exact_match":
        return check_exact_match(clean, expected)
    if kind == "tool_call":
        return check_tool_call(clean, expected)
    raise ValueError("Unknown check type: %r" % kind)
