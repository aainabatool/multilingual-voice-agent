import asyncio
import edge_tts

SAMPLES = [
    {
        "id": "sample_en",
        "text": "Hi, I want to track my order. The order number is 4589.",
        "voice": "en-US-AriaNeural",
    },
    {
        "id": "sample_ur",
        "text": "میرا آرڈر کہاں ہے؟ آرڈر نمبر چار پانچ آٹھ نو ہے۔",
        "voice": "ur-PK-UzmaNeural",
    },
    {
        "id": "sample_ur_latn",
        "text": "Mujhe apna order track karna hai. Order number 4589 hai.",
        "voice": "ur-PK-UzmaNeural",
    },
]

async def generate():
    for sample in SAMPLES:
        out_path = f"data/audio/{sample['id']}.mp3"
        communicate = edge_tts.Communicate(sample["text"], sample["voice"])
        await communicate.save(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    asyncio.run(generate())
