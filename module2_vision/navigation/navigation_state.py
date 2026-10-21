# navigation/navigation_state.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 5: Navigation State Representation
#
# Finite State Machine representation for blind-assistive
# target finding and obstacle avoidance.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from guidance.obstacle_detector import ObstacleReport
from guidance.spatial_locator import SpatialLocation


class NavigationState(Enum):
    """
    Current phase of the assistive navigation lifecycle.
    """
    INITIALIZING = "INITIALIZING"  # Setting up pipeline
    LISTENING    = "LISTENING"     # Awaiting voice query
    SEARCHING    = "SEARCHING"     # Looking for target, not yet seen
    GUIDING      = "GUIDING"       # Target in view, guiding user towards it
    OBSTRUCTED   = "OBSTRUCTED"    # Physical obstacle blocking path to target
    ARRIVED      = "ARRIVED"       # Target within arm's reach
    LOST         = "LOST"          # Target was visible, but temporarily lost
    UNSUPPORTED  = "UNSUPPORTED"   # Target is not a supported class


@dataclass(frozen=True)
class NavigationStep:
    """
    Complete state evaluation for a single processing cycle/frame.
    """
    state: NavigationState
    target_name: str
    target_found: bool
    spoken_instruction: str
    spatial_location: Optional[SpatialLocation]
    obstacle_report: ObstacleReport
    is_arrived: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "target_name": self.target_name,
            "target_found": self.target_found,
            "spoken_instruction": self.spoken_instruction,
            "spatial_location": self.spatial_location.to_dict() if self.spatial_location else None,
            "obstacle_report": self.obstacle_report.to_dict(),
            "is_arrived": self.is_arrived,
        }
