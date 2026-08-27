from benchmark.metrics.language_metrics import code_switch_f1, language_accuracy
from benchmark.metrics.wer_cer import character_error_rate, word_error_rate


def test_wer_perfect_match():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_one_substitution():
    assert word_error_rate("hello world", "hello there") == 0.5


def test_cer_perfect_match():
    assert character_error_rate("hello", "hello") == 0.0


def test_language_accuracy():
    results = [
        {"detected_language": "en", "reference_language": "en"},
        {"detected_language": "ur", "reference_language": "ur"},
        {"detected_language": "en", "reference_language": "ur"},
    ]
    assert language_accuracy(results) == round(2 / 3, 4)


def test_code_switch_f1_perfect():
    results = [
        {"code_switch_score": 0.8, "is_code_switched": True},
        {"code_switch_score": 0.0, "is_code_switched": False},
    ]
    result = code_switch_f1(results)
    assert result["f1"] == 1.0
    assert result["tp"] == 1
    assert result["tn"] == 1
