# Known Limitations

## Dataset
- The benchmark dataset has only 3 base sentences (English, Urdu, Roman Urdu), expanded to 18 cases via noise/speed/accent/spelling/mixed-script variants. Every per-condition number is based on **1-3 samples**. Treat all reported metrics as illustrative, not statistically significant.
- All benchmark audio is **TTS-synthesized** (edge-tts), not real human speech. It lacks the disfluency, real accent variation, and background conditions of genuine recordings. Live microphone testing (Phase 5) already showed real speech is meaningfully harder than the synthetic benchmark suggests.

## Measurement methodology
- **Script mismatch confounds WER/CER for Roman Urdu.** The STT router forces `language="ur"` when Urdu is suspected, which transcribes Roman Urdu speech into Urdu script. Scoring that against a Latin-script reference inflates WER to near 1.0 regardless of actual accuracy. This affects the Roman Urdu, code-switching, and spelling-variation results specifically -- they measure a script mismatch, not real transcription quality.
- The code-switch detector (`app/language/code_switch.py`) only tokenizes **Latin-script** text. It cannot detect code-switching in Urdu-script output, which is a direct consequence of the script-mismatch issue above.

## TTS
- Only one native Urdu voice is available (Piper's `ur_PK-fasih-medium`), and it is a newer/less mature voice than Piper's English voices -- audibly less natural on subjective listening.
- There is no Roman Urdu TTS voice. The system falls back to the English voice for `ur-Latn`/`mixed` text, which mispronounces Urdu words read as English (a logged fallback, not a silent failure -- but still a real gap).

## Language/code-switch detection
- The Roman Urdu lexicon (`ROMAN_URDU_LEXICON`) is a small, hand-picked list of ~50 common words. It will not recognize novel vocabulary or spelling variants outside that list -- confirmed by the spelling-variation test, though that result is also confounded by the script-mismatch issue above.
- Mixed-script input (Urdu word embedded in an English sentence) is not reliably detected; STT tends to lock into one language and drop or mangle the embedded word.

## Untested claims
- No formal experiment isolates whether explicit language routing improves intent accuracy, or whether conversation context improves ambiguous-language resolution (spec research questions 4 and 5). Both are plausible based on design but unverified.
- The "accent" robustness test uses a synthetic TTS voice (edge-tts `en-IN-NeerjaNeural`) as a proxy for Indian-accented English, not a real speaker. It is a weak proxy at best.

## Platform
- Development and all testing was done on Windows (PowerShell). Two real Windows-specific bugs were found and fixed along the way (JSON BOM handling, Piper stdin codepage) -- there may be others not yet surfaced, and behavior on Linux/Mac is unverified.
