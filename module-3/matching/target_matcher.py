# matching/target_matcher.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 3: Target Matching
#
# TargetMatcher is the core M3 component.  It answers ONE question:
#
#   "Given the requested target object and the current YOLO
#    detections, is the target present in the scene?"
#
# It does NOT perform:
#   • Navigation (left/right/center instructions)
#   • Distance estimation
#   • Object tracking
#   • TTS
#   • Any camera control
#
# These are the responsibility of future milestones.
#
# Public API
# ──────────
#   matcher = TargetMatcher()
#   result  = matcher.match(raw_target, detections)
#
# Result schema (dict):
#   {
#       "target":        str,        # normalized target string
#       "found":         bool,       # True if target is in detections
#       "confidence":    float|None, # confidence of best matching detection
#       "bbox":          list|None,  # [x1, y1, x2, y2] of best match
#       "matched_class": str|None,   # COCO class name of best match
#       "reason":        str|None,   # set only for unsupported_class
#   }
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

from matching.target_normalizer import is_supported_class, normalize_target


class TargetMatcher:
    """
    Matches a requested target object against a list of YOLO detections.

    The matcher is stateless — each call to :meth:`match` is independent.
    This makes it safe to call from a multi-threaded camera loop and
    easy to unit-test in isolation.

    Usage
    -----
    ::

        from matching.target_matcher import TargetMatcher

        matcher    = TargetMatcher()
        detections = [
            {"class_name": "person",    "class_id": 0,  "confidence": 0.91, "bbox": [...]},
            {"class_name": "bottle",    "class_id": 39, "confidence": 0.72, "bbox": [...]},
            {"class_name": "bottle",    "class_id": 39, "confidence": 0.86, "bbox": [...]},
        ]
        result = matcher.match("my bottle", detections)
        # result["found"]      -> True
        # result["confidence"] -> 0.86
    """

    # ── Result-dict helpers ────────────────────────────────────

    @staticmethod
    def _result_found(
        target: str,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a result dict for a successfully matched target."""
        return {
            "target":        target,
            "found":         True,
            "confidence":    detection["confidence"],
            "bbox":          detection["bbox"],
            "matched_class": detection["class_name"],
            "reason":        None,
        }

    @staticmethod
    def _result_not_found(target: str, reason: str | None = None) -> dict[str, Any]:
        """Build a result dict for a target that was not found."""
        return {
            "target":        target,
            "found":         False,
            "confidence":    None,
            "bbox":          None,
            "matched_class": None,
            "reason":        reason,
        }

    # ── Public API ─────────────────────────────────────────────

    def match(
        self,
        raw_target: str,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Determine whether the requested target is present in the
        current set of YOLO detections.

        Parameters
        ----------
        raw_target : str
            The requested object name.  May be raw user input such as
            ``"my bottle"`` or the ``"object"`` field returned by
            Module 1's ``parse_query``.  Normalization is applied
            internally.
        detections : list[dict]
            Detection dicts produced by ``YOLODetector.detect()``.
            Each dict must have: ``class_name``, ``confidence``, ``bbox``.
            May be an empty list.

        Returns
        -------
        dict
            Consistent result schema::

                {
                    "target":        str,         # normalized target
                    "found":         bool,
                    "confidence":    float | None,
                    "bbox":          list  | None, # [x1, y1, x2, y2]
                    "matched_class": str   | None,
                    "reason":        str   | None, # "unsupported_class" or None
                }

        Notes
        -----
        * Matching is **case-insensitive** and compares the normalized
          target against the YOLO ``class_name`` (also lowercased).
        * When **multiple instances** of the same class are detected,
          the one with the **highest confidence** score is returned.
        * If the target maps to a class that is not in the COCO-80 set,
          ``reason`` is set to ``"unsupported_class"``.  The detection
          list is still searched (in case a future model supports it),
          but in practice an unsupported class will never appear.
        """
        # Step 1 — Normalize the raw target
        target = normalize_target(raw_target)

        # Step 2 — Handle empty or blank input
        if not target:
            return self._result_not_found(target or raw_target, reason=None)

        # Step 3 — Check COCO-80 support
        if not is_supported_class(target):
            return self._result_not_found(target, reason="unsupported_class")

        # Step 4 — Handle empty detection list gracefully
        if not detections:
            return self._result_not_found(target)

        # Step 5 — Find all detections whose class_name matches target
        # Comparison is lowercased on both sides for robustness.
        candidates = [
            det for det in detections
            if det["class_name"].lower() == target
        ]

        if not candidates:
            return self._result_not_found(target)

        # Step 6 — Pick the highest-confidence candidate
        best = max(candidates, key=lambda d: d["confidence"])

        return self._result_found(target, best)
