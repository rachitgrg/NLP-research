# guidance/phrases.py
# ─────────────────────────────────────────────────────────────
# Module 2 – Spoken guidance phrases per language (roadmap R9, phase 3)
#
# Every sentence the assistant speaks during a search, in one table.
# The vocabulary is small and fixed (~15 templates), so replies in the
# user's language need no machine translation at runtime: each language
# is a reviewed template table, and object / obstacle names come from a
# small per-language dictionary (English word if missing).
#
#   phrase("target_left", "en", obj="bottle")  → "Bottle, left."
#   phrase("target_left", "hi", obj="bottle")  → "बोतल, बाईं ओर।"
#
# English strings are unchanged from before the table existed.
# The Hindi table is a DRAFT: have it checked by a native speaker before
# reporting results with it (research/decisions.md D-17).
# Unknown languages fall back to English.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

DEFAULT_LANGUAGE = "en"

PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "target_reached":   "{Obj} is directly in front of you. Stop and reach out.",
        "target_ahead":     "{Obj}, straight ahead.",
        "target_far_left":  "{Obj}, far left.",
        "target_left":      "{Obj}, left.",
        "target_right":     "{Obj}, right.",
        "target_far_right": "{Obj}, far right.",
        "target_not_seen":  "{Obj} not seen. Turn slowly.",
        "target_lost":      "{Obj} lost. Turn back slowly.",
        "unsupported":      "{Obj} is not detectable by the vision model.",
        "timeout":          "Could not find the {obj}. Stopping the search.",
        "cancelled":        "Search cancelled.",
        "obstacle_ahead":   "Stop. {Obs} ahead.",
        "obstacle_left":    "{Obs} on left. Step slightly right.",
        "obstacle_right":   "{Obs} on right. Step slightly left.",
        "ask_target":       "Please say what object you are looking for.",
        "looking_for":      "Looking for {obj}. Please hold camera steady and scan slowly.",
    },
    # DRAFT — needs native-speaker review (D-17).  Phrased so that no verb
    # agrees with the object: Hindi nouns have gender (बोतल f., फ़ोन m.)
    # and one template serves every object.
    "hi": {
        "target_reached":   "{Obj} ठीक आपके सामने है। रुकिए और हाथ बढ़ाइए।",
        "target_ahead":     "{Obj}, सीधे सामने।",
        "target_far_left":  "{Obj}, काफ़ी बाईं ओर।",
        "target_left":      "{Obj}, बाईं ओर।",
        "target_right":     "{Obj}, दाईं ओर।",
        "target_far_right": "{Obj}, काफ़ी दाईं ओर।",
        "target_not_seen":  "{Obj} नज़र में नहीं। धीरे से घूमिए।",
        "target_lost":      "{Obj} नज़र से बाहर। धीरे से वापस मुड़िए।",
        "unsupported":      "कैमरा {obj} नहीं पहचान सकता।",
        "timeout":          "{obj} का पता नहीं चला। खोज बंद की जा रही है।",
        "cancelled":        "खोज रद्द की गई।",
        "obstacle_ahead":   "रुकिए। सामने {obs} है।",
        "obstacle_left":    "बाईं ओर {obs} है। थोड़ा दाईं ओर हटिए।",
        "obstacle_right":   "दाईं ओर {obs} है। थोड़ा बाईं ओर हटिए।",
        "ask_target":       "बताइए, आप क्या ढूँढ रहे हैं?",
        "looking_for":      "{obj} ढूँढ रहे हैं। कैमरा स्थिर रखिए और धीरे-धीरे घुमाइए।",
    },
}

# Object and obstacle names per language (English class name → word).
OBJECT_NAMES: dict[str, dict[str, str]] = {
    "hi": {
        "bottle": "बोतल", "cup": "कप", "bowl": "कटोरी", "cell phone": "फ़ोन", "laptop": "लैपटॉप",
        "book": "किताब", "tv": "टीवी", "backpack": "बैग", "bag": "बैग", "umbrella": "छाता",
        "suitcase": "सूटकेस", "watch": "घड़ी", "keys": "चाबी", "key": "चाबी", "slippers": "चप्पल",
        "glasses": "चश्मा", "wallet": "पर्स", "medicine": "दवाई", "charger": "चार्जर",
        "person": "कोई व्यक्ति", "chair": "कुर्सी", "dining table": "मेज़", "couch": "सोफ़ा",
        "bed": "बिस्तर", "bench": "बेंच", "refrigerator": "फ्रिज", "toilet": "टॉयलेट",
    },
}


def base_language(language: str | None) -> str:
    """'hi-IN' → 'hi'; unknown / empty → DEFAULT_LANGUAGE."""
    code = (language or "").split("-")[0].strip().lower()
    return code if code in PHRASES else DEFAULT_LANGUAGE


def object_name(name: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Spoken word for an English class name (the English word if unknown)."""
    name = (name or "").strip().lower()
    return OBJECT_NAMES.get(base_language(language), {}).get(name, name)


def choose_reply_language(spoken: str | None, voices: set[str], setting: str = "auto") -> str:
    """
    Language for spoken replies.

    ``setting`` "auto" follows ``spoken`` (the recogniser's language code);
    any other code is used as is.  Either way the language needs a phrase
    table here and an installed voice (``voices``), else English.
    """
    wanted = spoken if setting in ("", "auto") else setting
    lang = base_language(wanted)
    return lang if lang in voices else DEFAULT_LANGUAGE


def phrase(key: str, language: str = DEFAULT_LANGUAGE, obj: str = "", obs: str = "") -> str:
    """
    The sentence ``key`` in ``language``.

    ``obj`` (target) and ``obs`` (obstacle) are English class names; the
    templates use {obj}/{obs} as is and {Obj}/{Obs} capitalised.
    """
    lang = base_language(language)
    template = PHRASES[lang].get(key) or PHRASES[DEFAULT_LANGUAGE][key]
    o = object_name(obj, lang) or "object"
    b = object_name(obs, lang)
    return template.format(obj=o, Obj=o.capitalize(), obs=b, Obs=b.capitalize())
