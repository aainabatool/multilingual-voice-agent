import re

# Common Roman Urdu function/content words -- deliberately small and transparent,
# per spec 5.4: "Begin with transparent heuristics, then add a classifier if needed."
ROMAN_URDU_LEXICON = {
    "mera", "meri", "mere", "mujhe", "hum", "hamara", "tum", "aap",
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "ho",
    "kya", "kyun", "kaise", "kahan", "kab", "kaun", "nahi", "nahin", "haan",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "aur", "ya",
    "karna", "kar", "karo", "kiya", "karta", "karti", "raha", "rahi", "rahe",
    "hua", "hui", "chahiye", "chahta", "chahti", "wala", "wali",
    "aaj", "kal", "abhi", "phir", "bhi", "sirf",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def detect_code_switch(text: str) -> dict:
    """Heuristic word-level language tagging for Latin-script text.

    Classifies each Latin-script token as Roman Urdu (lexicon match) or
    English/other (no match), then scores how mixed the utterance is.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"code_switch_score": 0.0, "ur_ratio": 0.0, "en_ratio": 0.0, "tagged": []}

    tagged = []
    ur_count = 0
    en_count = 0
    for tok in tokens:
        if tok in ROMAN_URDU_LEXICON:
            tagged.append((tok, "ur-Latn"))
            ur_count += 1
        else:
            tagged.append((tok, "en"))
            en_count += 1

    total = len(tokens)
    ur_ratio = ur_count / total
    en_ratio = en_count / total

    # Balanced presence of both = higher code-switch score.
    # 2 * min(ratio) is 0 when monolingual, approaches 1.0 as the split approaches 50/50.
    code_switch_score = round(2 * min(ur_ratio, en_ratio), 3)

    return {
        "code_switch_score": code_switch_score,
        "ur_ratio": round(ur_ratio, 3),
        "en_ratio": round(en_ratio, 3),
        "tagged": tagged,
    }
