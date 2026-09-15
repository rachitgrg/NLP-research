# nlp/keyword_extractor.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# English rule-based keyword extraction using spaCy.
# No LLM or external API is used — 100% local.
# ─────────────────────────────────────────────────────────────

import spacy


# Part-of-speech tags we want to keep as keywords
# NOUN  → common nouns:  bottle, table, door, book
# PROPN → proper nouns:  London, John, Google
# ADJ   → adjectives:    black, large (kept only when next to a noun)
_KEEP_POS = {"NOUN", "PROPN"}
_OPTIONAL_ADJ = "ADJ"


def load_nlp_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """
    Load a spaCy language model.

    Args:
        model_name (str): The spaCy model to load.
                          Default is "en_core_web_sm".
                          Install it with:
                              python -m spacy download en_core_web_sm

    Returns:
        spacy.Language: A ready-to-use spaCy NLP pipeline.

    Raises:
        OSError: If the model is not installed.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' not found.\n"
            f"Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc
    return nlp


def extract_keywords(nlp_model: spacy.Language, text: str) -> dict:
    """
    Extract meaningful keywords from a transcribed English sentence.

    Strategy:
        1. Parse the text with spaCy.
        2. Keep tokens whose POS tag is NOUN or PROPN.
        3. Also keep ADJ tokens that appear immediately before a kept noun
           (e.g. "black" in "black bottle").
        4. Lower-case, de-duplicate, and preserve original word order.

    Args:
        nlp_model (spacy.Language): Loaded spaCy pipeline.
        text (str): Transcribed text string (e.g. "Where is my black bottle?").

    Returns:
        dict: {
            "keywords"     (list[str]): All retained keywords in order.
            "main_keyword" (str | None): The last/most prominent noun, or None.
        }

    Examples:
        "Where is my bottle?"          → keywords=["bottle"],    main="bottle"
        "Where is my black bottle?"    → keywords=["black","bottle"], main="bottle"
        ""                             → keywords=[],            main=None
    """
    if not text or not text.strip():
        return {"keywords": [], "main_keyword": None}

    doc = nlp_model(text)
    tokens = list(doc)

    kept = []  # indices of tokens we will keep

    # Pass 1: collect nouns and proper nouns
    noun_indices = {
        i for i, tok in enumerate(tokens)
        if tok.pos_ in _KEEP_POS and not tok.is_stop and not tok.is_punct
    }

    # Pass 2: include adjectives that appear directly before a noun
    adj_indices = set()
    for i, tok in enumerate(tokens):
        if tok.pos_ == _OPTIONAL_ADJ and not tok.is_stop and not tok.is_punct:
            # Check if the very next non-space token is a noun we already kept
            for j in range(i + 1, len(tokens)):
                if tokens[j].is_space:
                    continue
                if j in noun_indices:
                    adj_indices.add(i)
                break  # Stop after first non-space neighbour

    # Merge, sort by position in sentence, lower-case, deduplicate
    all_indices = noun_indices | adj_indices
    seen = set()
    keywords = []
    for i in sorted(all_indices):
        word = tokens[i].text.lower()
        if word not in seen:
            seen.add(word)
            keywords.append(word)

    # The "main keyword" is the last pure noun in the sentence.
    # "Where is my black bottle?" → "bottle" (the object being searched for).
    main_keyword = None
    for i in sorted(noun_indices, reverse=True):
        candidate = tokens[i].text.lower()
        if candidate in seen:
            main_keyword = candidate
            break

    return {
        "keywords": keywords,
        "main_keyword": main_keyword,
    }
