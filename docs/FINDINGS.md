# Findings

Answers to the 8 research questions from the project specification (section 17), grounded in the actual benchmark data produced in Phases 1, 6, 7, and 8. Sample sizes are small (1-3 cases per condition) since this is a hand-built demonstration dataset, not a large-scale evaluation -- results here are **illustrative and directional**, not statistically robust. See [LIMITATIONS.md](LIMITATIONS.md) for what that means concretely.

## 1. How much does STT accuracy degrade from English to Urdu?
Substantial. On the clean benchmark (faster-whisper `small`): **English WER = 0.000**, **Urdu WER = 0.455-0.545** across repeated runs. Character-level error (CER) tells a similar story: 0.000 vs 0.35-0.38. This matches the expectation that Whisper's training data skews heavily English, with far less Urdu coverage.

## 2. How much worse is Roman Urdu than native Urdu?
**Not reliably answerable with the current methodology.** Our STT router (Phase 5) forces `language="ur"` whenever Urdu is suspected, which correctly transcribes Roman Urdu speech's *content* but converts it into **Urdu script**. Comparing that against a Latin-script reference text inflates WER to ~1.0 regardless of actual transcription quality -- a measurement artifact, not a real accuracy signal. A fair comparison would need script normalization (e.g. transliterating both sides to a common script) before scoring, which is flagged as future work.

## 3. Does code-switching increase WER/CER?
**Confounded by the same script-mismatch issue as (2).** Every code-switched (Roman Urdu) test case in our benchmark hit the same forced-Urdu-script transcription, so its WER/CER numbers aren't measuring code-switching difficulty in isolation. This needs to be re-run with script-aware scoring before it can answer the question.

## 4. Does explicit language routing improve intent accuracy?
We didn't run a formal A/B test on intent accuracy specifically, but the routing logic (Phase 5) demonstrably fixed a real, measured problem: Whisper's auto-detect frequently mislabels Urdu speech as Hindi (confidence 0.72-0.93, wrong script). Forcing a retry to `ur` when this happens raised confidence to 1.00 and produced the correct script every time it was tested (Phase 1). A dedicated intent-accuracy comparison (routing on vs off) is open future work.

## 5. Does conversation context improve ambiguous language detection?
Not formally tested. `SessionState` tracks `recent_languages` (Phase 3) but no controlled experiment has isolated its effect on resolving ambiguous turns. Open future work.

## 6. Which TTS model gives acceptable Urdu/Roman Urdu output?
Of the two candidates considered, **Kokoro was excluded early** (English-only, no Urdu support) and **Piper** was used throughout. Piper's native Urdu voice (`ur_PK-fasih-medium`) works and is intelligible, but is noticeably less natural than Piper's mature English voice (subjective listening test during Phase 5 pipeline verification). There is currently **no dedicated Roman Urdu TTS voice** -- the system falls back to the English voice, which mispronounces Urdu words read as English text (a logged, not silent, fallback per spec 16).

## 7. How does background noise affect each language differently?
On TTS-synthesized audio with injected white Gaussian noise (Phase 8): **English stayed perfectly robust** (WER = 0.000 at both 10dB and 0dB SNR). **Urdu WER stayed roughly flat** across noise conditions (0.455-0.545), not clearly worse at 0dB than at 10dB in this small sample -- likely because clean TTS-synthesized speech leaves enough phonetic signal even under noise, and/or the sample size (n=3) is too small to see a clear trend. **Inconclusive** with real recorded speech, which would likely show a starker effect.

## 8. Which configuration gives the best accuracy/latency trade-off?
Based on the Phase 7 model comparison, **faster-whisper `small` beat `tiny` on both axes**: avg WER 0.485 vs 0.606, and avg inference time 3.82s vs 5.88s. This is a bit counterintuitive (smaller models are usually faster) -- likely because `tiny` produced longer, more repetitive/garbled output on Urdu content, which took longer to decode. **Recommendation: use `small`, not `tiny`, for this multilingual use case** given current findings.

## Findings not tied to a specific research question
- **Piper Windows stdin encoding bug** (Phase 4): Piper's CLI reads stdin using the OS console codepage by default on Windows, silently corrupting non-ASCII (Urdu/Arabic-script) text into mojibake and causing phonemization to fail with zero audio frames. Fixed by forcing `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` in the subprocess environment.
- **Mixed-script input** (`"Can you check میرا order؟"`, Phase 8) caused STT to lock into pure English detection and drop/mangle the embedded Urdu word, rather than correctly flagging the utterance as mixed. This matches the spec's own stress-test example and is a genuine open problem.
- **Live microphone testing** (Phase 5) surfaced a real robustness gap not visible in the TTS-only benchmark dataset: one live Roman Urdu utterance was badly mis-transcribed into pseudo-Spanish gibberish by `small`, while the agent still responded sensibly (asked for clarification rather than hallucinating).
