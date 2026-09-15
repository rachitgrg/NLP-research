from __future__ import annotations

# evaluation/metrics.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# Text-normalisation and speech-recognition metric helpers.
#
# All three metric functions (WER, CER, accuracy) delegate to
# the `jiwer` library, which must be installed:
#     pip install jiwer
# ─────────────────────────────────────────────────────────────

import re
import string


def normalize_text(text: str) -> str:
    """
    Normalise a transcript string before metric computation.

    Steps applied (in order):
    1. Strip leading/trailing whitespace.
    2. Convert to lowercase.
    3. Remove all punctuation characters (``string.punctuation``).
    4. Collapse multiple consecutive spaces into a single space.
    5. Strip again.

    Meaningful words are preserved; only punctuation and case are changed.

    Args:
        text: Raw transcript string from a model or reference CSV.

    Returns:
        Normalised string ready for WER/CER computation.

    Examples::

        >>> normalize_text("Where is my BLACK bottle?")
        'where is my black bottle'
        >>> normalize_text("  Hello,  World!  ")
        'hello  world'  # punctuation removed, then collapsed
        >>> normalize_text("Can you see the green box?")
        'can you see the green box'
    """
    if not text:
        return ""
    text = text.strip()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute the Word Error Rate (WER) between a reference and hypothesis.

    Both strings are normalised with :func:`normalize_text` before scoring.

    Args:
        reference:  Ground-truth transcript (the expected English text).
        hypothesis: Model output to evaluate.

    Returns:
        WER as a float in the range [0.0, ∞).  0.0 = perfect match.

    Raises:
        ImportError: If ``jiwer`` is not installed.
        ValueError:  If ``reference`` is empty after normalisation.
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError(
            "jiwer is required for WER computation.\n"
            "Install it with:  pip install jiwer"
        ) from exc

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        raise ValueError(
            f"Reference text is empty after normalisation: {reference!r}"
        )

    return float(jiwer.wer(ref_norm, hyp_norm))


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute the Character Error Rate (CER) between a reference and hypothesis.

    Both strings are normalised with :func:`normalize_text` before scoring.

    Args:
        reference:  Ground-truth transcript.
        hypothesis: Model output to evaluate.

    Returns:
        CER as a float in the range [0.0, ∞).  0.0 = perfect match.

    Raises:
        ImportError: If ``jiwer`` is not installed.
        ValueError:  If ``reference`` is empty after normalisation.
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError(
            "jiwer is required for CER computation.\n"
            "Install it with:  pip install jiwer"
        ) from exc

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        raise ValueError(
            f"Reference text is empty after normalisation: {reference!r}"
        )

    return float(jiwer.cer(ref_norm, hyp_norm))


def compute_accuracy(wer: float) -> float:
    """
    Convert a WER value to a percentage accuracy score.

    accuracy = max(0.0, (1.0 - wer) * 100.0)

    A WER of 0.0 gives 100 % accuracy.
    A WER ≥ 1.0 is clamped to 0 % accuracy (never negative).

    Args:
        wer: Word Error Rate as returned by :func:`compute_wer`.

    Returns:
        Accuracy in the range [0.0, 100.0].
    """
    return max(0.0, 100.0 - wer * 100.0)
