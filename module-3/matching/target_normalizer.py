# matching/target_normalizer.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 3: Target Matching
# Small, focused normalization layer that converts a raw target
# string (as it may arrive from Module 1 or from --target CLI)
# into a canonical COCO class name.
#
# Design decisions
# ────────────────
# • Module 1 already handles multilingual → English translation
#   and strips stop-words/articles via spaCy NLP.  When M3 is
#   wired to Module 1's live pipeline the `object` field it
#   returns is already clean (e.g. "bottle", not "my bottle").
#
# • When a user types `--target "my bottle"` directly, we still
#   need to handle the article/possessive stripping ourselves.
#   This normalizer handles that path too.
#
# • Keep alias mapping small and explicit for now.  Do NOT add
#   a large synonym corpus or call an LLM here.  The mapping
#   can be extended later in one place.
#
# • Objects that have no direct COCO-80 equivalent are NOT
#   silently mapped — they remain as-is so the matcher can
#   detect them as "unsupported_class".
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import re

# ── COCO-80 class set (YOLOv8n pretrained) ────────────────────
# Full list of the 80 COCO categories that YOLOv8n can detect.
# Keeping this here in ONE place lets us answer "is this target
# even supported?" without spinning up the YOLO model.
COCO_CLASSES: frozenset[str] = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
})

# ── Spoken-word alias → canonical COCO class name ─────────────
# Only add mappings where the spoken word differs from the COCO
# class name AND the mapping is unambiguous for this project's
# use-case (assisting visually impaired users find everyday objects).
#
# Deliberately NOT included:
#   "watch" -> "clock"   (a wristwatch is not a wall clock; misleading)
#   "glasses" -> any class  (no COCO glasses class)
#   "key" / "keys"          (no COCO key class)
#   "pen" / "pencil"        (no COCO pen class)
#   "slipper" / "shoe"      (no COCO slipper/shoe class)
#
_ALIASES: dict[str, str] = {
    # phone variants
    "phone":        "cell phone",
    "mobile":       "cell phone",
    "mobile phone": "cell phone",
    "smartphone":   "cell phone",
    "iphone":       "cell phone",
    "android":      "cell phone",

    # couch / sofa
    "sofa":         "couch",

    # television / monitor
    "television":   "tv",
    "monitor":      "tv",
    "screen":       "tv",
    "telly":        "tv",

    # fridge
    "fridge":       "refrigerator",

    # plant
    "plant":        "potted plant",
    "flower pot":   "potted plant",

    # dining table
    "table":        "dining table",

    # sports ball
    "ball":         "sports ball",
    "football":     "sports ball",
    "basketball":   "sports ball",
    "soccer ball":  "sports ball",

    # motorbike / bike
    "motorbike":    "motorcycle",
    "bike":         "bicycle",
}

# ── Stop-words/prefixes to strip before matching ─────────────
# Handles phrases like "my bottle", "the bottle", "a bottle",
# "find the bottle", "where is my phone".
_STRIP_PATTERN = re.compile(
    r"^\s*(my|your|the|a|an|find|get|locate|where is|where's|give me|i want|"
    r"show me|bring me|fetch|look for)\s+",
    re.IGNORECASE,
)


def normalize_target(raw: str) -> str:
    """
    Convert a raw target string into a canonical COCO class name.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Remove leading possessives / filler phrases  (my, the, ...).
    3. Lowercase.
    4. Apply alias mapping (phone -> cell phone, etc.).

    The result is compared against the YOLO detections by
    TargetMatcher.

    Parameters
    ----------
    raw : str
        The raw target string.  May come from Module 1's ``object``
        field or directly from the ``--target`` CLI argument.

    Returns
    -------
    str
        The normalized (lowercase, alias-resolved) target string.
        Still may not be a valid COCO class -- the caller decides
        what to do with unsupported targets.

    Examples
    --------
    normalize_target("my bottle")  -> 'bottle'
    normalize_target("PHONE")      -> 'cell phone'
    normalize_target("watch")      -> 'watch'
    """
    if not raw or not raw.strip():
        return ""

    # Step 1 — strip filler prefix
    cleaned = _STRIP_PATTERN.sub("", raw.strip())

    # Step 2 — lowercase
    cleaned = cleaned.lower().strip()

    # Step 3 — alias lookup
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]

    return cleaned


def is_supported_class(target: str) -> bool:
    """
    Return True if target (already normalized) is a COCO-80
    class that YOLOv8n can detect.

    Parameters
    ----------
    target : str
        A normalized target string (output of normalize_target).

    Returns
    -------
    bool
    """
    return target in COCO_CLASSES
