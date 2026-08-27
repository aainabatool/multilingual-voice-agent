import soundfile as sf

from app.audio.noise import inject_white_noise
from app.audio.speed import change_speaking_rate

# Base cases: (id_prefix, source_mp3, reference_text, language, script)
BASE_CASES = [
    ("english", "data/audio/sample_en.mp3", "Hi, I want to track my order. The order number is 4589.", "en", "latin"),
    ("urdu", "data/audio/sample_ur.mp3", "میرا اوڈر کہا ہے؟ اوڈر نمبر چار پانچ آٹھ نو ہے۔", "ur", "urdu"),
    ("roman_urdu", "data/audio/sample_ur_latn.mp3", "Mujhe apna order track karna hai. Order number 4589 hai.", "ur-Latn", "latin"),
]

NOISE_LEVELS_DB = [10, 0]
SPEED_FACTORS = [1.3, 0.75]


def convert_to_wav(mp3_path: str, wav_path: str) -> str:
    """faster-whisper's decoder handles mp3 fine, but numpy transforms need a clean wav first."""
    data, sr = sf.read(mp3_path)
    sf.write(wav_path, data, sr)
    return wav_path


def generate():
    manifest_entries = []

    for prefix, mp3_path, ref_text, language, script in BASE_CASES:
        base_wav = f"data/audio/robustness/{prefix}_clean.wav"
        convert_to_wav(mp3_path, base_wav)

        manifest_entries.append({
            "id": f"{prefix}_clean",
            "audio": base_wav,
            "reference_text": ref_text,
            "language": language,
            "script": script,
            "category": "order_tracking",
            "is_code_switched": prefix == "roman_urdu",
            "condition": "clean",
        })

        for snr in NOISE_LEVELS_DB:
            out_path = f"data/audio/robustness/{prefix}_noise_{snr}db.wav"
            inject_white_noise(base_wav, out_path, target_snr_db=snr)
            manifest_entries.append({
                "id": f"{prefix}_noise_{snr}db",
                "audio": out_path,
                "reference_text": ref_text,
                "language": language,
                "script": script,
                "category": "order_tracking",
                "is_code_switched": prefix == "roman_urdu",
                "condition": f"noise_{snr}db",
            })

        for factor in SPEED_FACTORS:
            label = f"speed_{str(factor).replace('.', '')}x"
            out_path = f"data/audio/robustness/{prefix}_{label}.wav"
            change_speaking_rate(base_wav, out_path, rate_factor=factor)
            manifest_entries.append({
                "id": f"{prefix}_{label}",
                "audio": out_path,
                "reference_text": ref_text,
                "language": language,
                "script": script,
                "category": "order_tracking",
                "is_code_switched": prefix == "roman_urdu",
                "condition": label,
            })

    return manifest_entries


if __name__ == "__main__":
    import json
    from pathlib import Path

    Path("data/audio/robustness").mkdir(parents=True, exist_ok=True)
    entries = generate()

    with open("benchmark/datasets/manifest_robustness.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(entries)} robustness test cases -> benchmark/datasets/manifest_robustness.json")
