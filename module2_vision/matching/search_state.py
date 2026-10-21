# matching/search_state.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 3: Target Matching
# Simple search-state representation for the current milestone.
#
# Two states exist:
#
#   SEARCHING     — target is not visible in the current frame.
#                   The object may simply be out of frame.
#                   Do NOT assume the object does not exist.
#
#   TARGET_FOUND  — YOLO has detected the requested target class
#                   in the current frame with acceptable confidence.
#
# Future milestones will add:
#   NAVIGATING    — user is moving toward the target
#   ARRIVED       — user has reached the target
#   OBSTRUCTED    — obstacle between user and target
#
# These are NOT implemented here.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from enum import Enum


class SearchState(Enum):
    """
    Represents the current target-search state for a single video frame.

    Values
    ------
    SEARCHING     : The requested target is not present in the current
                    YOLO detections.  This does NOT mean the object is
                    absent from the scene — it may simply not be visible
                    from the current camera angle.

    TARGET_FOUND  : The requested target class has been detected by YOLO
                    with confidence above the threshold.

    Usage
    -----
    ::

        from matching.search_state import SearchState

        state = SearchState.from_match_result(match_result)

        if state == SearchState.TARGET_FOUND:
            print("Target is visible!")
        elif state == SearchState.SEARCHING:
            print("Still searching…")

    """

    SEARCHING    = "SEARCHING"
    TARGET_FOUND = "TARGET_FOUND"

    # ── Convenience factory ────────────────────────────────────

    @classmethod
    def from_match_result(cls, match_result: dict) -> "SearchState":
        """
        Derive the search state from a TargetMatcher result dict.

        Parameters
        ----------
        match_result : dict
            Result dict as returned by ``TargetMatcher.match()``.
            Must contain a boolean ``"found"`` key.

        Returns
        -------
        SearchState
            ``TARGET_FOUND`` if ``match_result["found"]`` is ``True``,
            ``SEARCHING`` otherwise (including unsupported-class results).
        """
        if match_result.get("found", False):
            return cls.TARGET_FOUND
        return cls.SEARCHING

    def __str__(self) -> str:
        return self.value
