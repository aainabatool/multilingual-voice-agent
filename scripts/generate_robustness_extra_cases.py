import asyncio
import json
from pathlib import Path

import edge_tts

# Accent variation: same English sentence, different English accent voice
ACCENT_CASE = {
    "id": "english_accent_en_IN",
    "text": "Hi, I want to track my order. The order number is 4589.",
    "voice": "en-IN-NeerjaNeural",
    "reference_text": "Hi, I want to track my order. The order number is 4589.",
    "language": "en",
    "script": "latin",
    "category": "order_tracking",
    "is_code_switched": False,
    "condition": "accent_en_IN",
}

# Spelling variation: same Roman Urdu meaning, informal/alternate spellings
# ("hy" for "hai", "kia" for "karna", "nai" for "nahi" -- common digital-Urdu variants)
SPELLING_CASE = {
    "id": "roman_urdu_spelling_variant",
    "text": "Mujhe apna order track kia hai. Order number 4589 hy.",
    "voice": "ur-PK-UzmaNeural",
    "reference_text": "Mujhe apna order track kia hai. Order number 4589 hy.",
    "language": "ur-Latn",
    "script": "latin",
    "category": "order_tracking",
    "is_code_switched": True,
    "condition": "spelling_variant",
}

# Mixed-script: spec's own reference example (section 8 table)
MIXED_SCRIPT_CASE = {
    "id": "mixed_script_001",
    "text": "Can you check میرا order?",
    "voice": "ur-PK-UzmaNeural",
    "reference_text": "Can you check میرا order?",
    "language": "mixed",
    "script": "mixed",
    "category": "order_tracking",
    "is_code_switched": True,
    "condition": "mixed_script",
}

CASES = [ACCENT_CASE, SPELLING_CASE, MIXED_SCRIPT_CASE]


async def generate():
    Path("data/audio/robustness").mkdir(parents=True, exist_ok=True)
    entries = []

    for case in CASES:
        audio_path = f"data/audio/robustness/{case['id']}.mp3"
        communicate = edge_tts.Communicate(case["text"], case["voice"])
        await communicate.save(audio_path)
        print(f"Saved {audio_path}")

        entry = {k: v for k, v in case.items() if k not in ("text", "voice")}
        entry["audio"] = audio_path
        entries.append(entry)

    # Merge into the existing robustness manifest
    existing_path = Path("benchmark/datasets/manifest_robustness.json")
    with open(existing_path, encoding="utf-8-sig") as f:
        existing = json.load(f)

    combined = existing + entries
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\nAdded {len(entries)} cases -> {existing_path} (total: {len(combined)})")


if __name__ == "__main__":
    asyncio.run(generate())
