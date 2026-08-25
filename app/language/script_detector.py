import unicodedata

# Perso-Arabic script range covers Urdu's script (Arabic block + Arabic Presentation Forms)
ARABIC_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]


def _char_script(ch: str) -> str:
    if ch.isspace() or unicodedata.category(ch).startswith("P") or ch.isdigit():
        return "neutral"
    code = ord(ch)
    if any(lo <= code <= hi for lo, hi in ARABIC_RANGES):
        return "arabic"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return "other"


def analyze_script(text: str) -> dict:
    """Return script ratios and the dominant script for a piece of text."""
    counts = {"arabic": 0, "latin": 0, "other": 0}
    for ch in text:
        script = _char_script(ch)
        if script == "neutral":
            continue
        counts[script] = counts.get(script, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {"dominant": "unknown", "ratios": {"arabic": 0.0, "latin": 0.0, "other": 0.0}}

    ratios = {k: v / total for k, v in counts.items()}

    # "mixed" script only if both major scripts appear meaningfully (not just stray chars)
    if ratios["arabic"] > 0.15 and ratios["latin"] > 0.15:
        dominant = "mixed"
    else:
        dominant = max(ratios, key=ratios.get)

    return {"dominant": dominant, "ratios": ratios}
