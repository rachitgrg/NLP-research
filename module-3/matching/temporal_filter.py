# matching/temporal_filter.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Temporal smoothing of target matches
#
# A per-frame detector that succeeds ~70% of the time makes the
# target flicker FOUND / NOT FOUND, which turns into "Bottle lost"
# messages while the bottle is actually in view.  TargetTracker
# smooths TargetMatcher results over the last few frames:
#
#   • Confirm  : the target counts as present only after it was
#                matched in >= min_hits of the last `window` frames
#                (filters one-frame false positives).
#   • Hold     : once confirmed, a short gap (<= hold_s) keeps the
#                last bounding box instead of reporting "lost".
#
# Input and output use the TargetMatcher result schema, so the
# NavigationController does not need to know smoothing exists.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from collections import deque
from typing import Any, Optional

from config.config import (
    TARGET_CONFIRM_MIN_HITS,
    TARGET_CONFIRM_WINDOW,
    TARGET_HOLD_S,
)


class TargetTracker:
    """Smooths per-frame TargetMatcher results over time."""

    def __init__(
        self,
        window: int = TARGET_CONFIRM_WINDOW,
        min_hits: int = TARGET_CONFIRM_MIN_HITS,
        hold_s: float = TARGET_HOLD_S,
    ) -> None:
        self.window = max(1, window)
        self.min_hits = max(1, min(min_hits, self.window))
        self.hold_s = hold_s

        self._hits: deque[bool] = deque(maxlen=self.window)
        self._confirmed: bool = False
        self._last_found: Optional[dict[str, Any]] = None
        self._last_found_time: float = 0.0

    def update(self, match_result: dict[str, Any], now: float) -> dict[str, Any]:
        """
        Feed one frame's match result; return the smoothed result.

        Parameters
        ----------
        match_result : dict
            Output of ``TargetMatcher.match()`` for this frame.
        now : float
            Monotonic timestamp of this frame (seconds).
        """
        if match_result.get("reason") == "unsupported_class":
            return match_result

        found_now = bool(match_result.get("found")) and match_result.get("bbox") is not None
        self._hits.append(found_now)

        if found_now:
            self._last_found = match_result
            self._last_found_time = now
            if sum(self._hits) >= self.min_hits:
                self._confirmed = True
            return match_result if self._confirmed else self._not_found(match_result)

        # Missed this frame: hold the last confirmed box for a short gap.
        if (
            self._confirmed
            and self._last_found is not None
            and (now - self._last_found_time) <= self.hold_s
        ):
            return self._last_found

        # Gap longer than the hold: the target must be re-confirmed.
        self._confirmed = False
        return match_result

    def reset(self) -> None:
        self._hits.clear()
        self._confirmed = False
        self._last_found = None
        self._last_found_time = 0.0

    @staticmethod
    def _not_found(match_result: dict[str, Any]) -> dict[str, Any]:
        """Same schema as TargetMatcher, reporting 'not (yet) found'."""
        return {
            "target":        match_result.get("target"),
            "found":         False,
            "confidence":    None,
            "bbox":          None,
            "matched_class": None,
            "reason":        None,
        }
