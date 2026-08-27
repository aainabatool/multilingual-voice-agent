import json
from datetime import datetime, timezone
from pathlib import Path

from app.language.detector import detect_language_state
from app.language.router import route_transcription
from app.stt.whisper_runner import WhisperRunner
from benchmark.metrics.language_metrics import code_switch_f1, language_accuracy
from benchmark.metrics.wer_cer import character_error_rate, word_error_rate


def run_benchmark(manifest_path: str = "benchmark/datasets/manifest.json", model_size: str = "small") -> dict:
    with open(manifest_path, encoding="utf-8-sig") as f:
        manifest = json.load(f)

    print(f"Loading faster-whisper ({model_size})...")
    stt = WhisperRunner(model_size=model_size)

    per_case_results = []
    metric_rows = []

    for case in manifest:
        print(f"\nRunning case: {case['id']}")
        transcription = route_transcription(stt, case["audio"])
        lang_state = detect_language_state(
            transcription.text,
            stt_language=transcription.language,
            stt_confidence=transcription.confidence,
        )

        wer = word_error_rate(case["reference_text"], transcription.text)
        cer = character_error_rate(case["reference_text"], transcription.text)

        row = {
            "id": case["id"],
            "category": case["category"],
            "reference_text": case["reference_text"],
            "hypothesis_text": transcription.text,
            "wer": wer,
            "cer": cer,
            "reference_language": case["language"],
            "detected_language": lang_state.primary_language,
            "code_switch_score": lang_state.code_switch_score,
            "is_code_switched": case["is_code_switched"],
            "inference_time_s": transcription.inference_time_s,
        }
        per_case_results.append(row)
        metric_rows.append(row)

        print(f"  WER: {wer:.3f}, CER: {cer:.3f}")
        print(f"  Reference language: {case['language']}, Detected: {lang_state.primary_language}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_size": model_size,
        "num_cases": len(manifest),
        "avg_wer": round(sum(r["wer"] for r in metric_rows) / len(metric_rows), 4),
        "avg_cer": round(sum(r["cer"] for r in metric_rows) / len(metric_rows), 4),
        "avg_inference_time_s": round(sum(r["inference_time_s"] for r in metric_rows) / len(metric_rows), 4),
        "language_accuracy": language_accuracy(metric_rows),
        "code_switch_f1": code_switch_f1(metric_rows),
        "per_case": per_case_results,
    }

    report_path = Path("benchmark/reports") / f"benchmark_{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to {report_path}")
    print(f"\nSummary: avg WER={summary['avg_wer']}, avg CER={summary['avg_cer']}, "
          f"language accuracy={summary['language_accuracy']}, "
          f"code-switch F1={summary['code_switch_f1']['f1']}")

    return summary


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "small"
    run_benchmark(model_size=model)
