def language_accuracy(results: list[dict]) -> float:
    """Fraction of test cases where detected primary language matched the reference label."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["detected_language"] == r["reference_language"])
    return round(correct / len(results), 4)


def code_switch_f1(results: list[dict], threshold: float = 0.3) -> dict:
    """Precision/recall/F1 for binary code-switch detection against reference labels."""
    tp = fp = fn = tn = 0
    for r in results:
        predicted = r["code_switch_score"] >= threshold
        actual = r["is_code_switched"]
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
