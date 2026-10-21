# guidance/spatial_locator.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 4: Spatial Localization & Proximity
#
# Analyzes the bounding box of a detected target object within
# the camera frame and calculates:
#   1. Horizontal Zone: FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT
#   2. Proximity: FAR, APPROACHING, REACHED (based on area ratio)
#   3. Spoken directional instructions for visually impaired users
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from config.config import (
    ARRIVAL_AREA_THRESHOLD,
    PROXIMITY_APPROACHING_THRESHOLD,
    PROXIMITY_FAR_THRESHOLD,
    ZONE_FAR_LEFT_THRESHOLD,
    ZONE_FAR_RIGHT_THRESHOLD,
    ZONE_LEFT_THRESHOLD,
    ZONE_RIGHT_THRESHOLD,
)


class HorizontalZone(Enum):
    """
    Horizontal section of the camera's field of view where the target lies.
    """
    FAR_LEFT  = "far_left"
    LEFT      = "left"
    CENTER    = "center"
    RIGHT     = "right"
    FAR_RIGHT = "far_right"


class Proximity(Enum):
    """
    Estimated distance proximity based on target bounding-box area ratio.
    """
    FAR         = "far"
    APPROACHING = "approaching"
    REACHED     = "reached"


@dataclass(frozen=True)
class SpatialLocation:
    """
    Structured spatial localization results for a target bounding box.
    """
    zone: HorizontalZone
    zone_label: str
    direction_instruction: str
    normalized_x: float
    normalized_y: float
    area_ratio: float
    proximity: Proximity
    proximity_label: str
    is_reached: bool
    distance_instruction: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-compatible dictionary."""
        return {
            "zone": self.zone.value,
            "zone_label": self.zone_label,
            "direction_instruction": self.direction_instruction,
            "normalized_x": round(self.normalized_x, 4),
            "normalized_y": round(self.normalized_y, 4),
            "area_ratio": round(self.area_ratio, 4),
            "proximity": self.proximity.value,
            "proximity_label": self.proximity_label,
            "is_reached": self.is_reached,
            "distance_instruction": self.distance_instruction,
        }


class SpatialLocator:
    """
    Converts 2D bounding boxes into relative 3D spatial cues and verbal instructions.
    """

    def __init__(
        self,
        far_left_th: float = ZONE_FAR_LEFT_THRESHOLD,
        left_th: float = ZONE_LEFT_THRESHOLD,
        right_th: float = ZONE_RIGHT_THRESHOLD,
        far_right_th: float = ZONE_FAR_RIGHT_THRESHOLD,
        far_area_th: float = PROXIMITY_FAR_THRESHOLD,
        arrival_area_th: float = ARRIVAL_AREA_THRESHOLD,
    ) -> None:
        self.far_left_th = far_left_th
        self.left_th = left_th
        self.right_th = right_th
        self.far_right_th = far_right_th
        self.far_area_th = far_area_th
        self.arrival_area_th = arrival_area_th

    def locate(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> SpatialLocation:
        """
        Locate the target in the frame coordinate space.

        Parameters
        ----------
        bbox : list or tuple of 4 ints
            [x1, y1, x2, y2] pixel coordinates.
        frame_width : int
            Width of the video frame in pixels (> 0).
        frame_height : int
            Height of the video frame in pixels (> 0).

        Returns
        -------
        SpatialLocation
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame width and height must be strictly positive.")

        x1, y1, x2, y2 = bbox
        # Sanitize coordinate ordering
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        # Centroid in pixels
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        # Normalized coordinates clamped to [0.0, 1.0]
        norm_x = max(0.0, min(1.0, cx / frame_width))
        norm_y = max(0.0, min(1.0, cy / frame_height))

        # Calculate bounding box area ratio
        box_w = max(0, max_x - min_x)
        box_h = max(0, max_y - min_y)
        area_ratio = (box_w * box_h) / (frame_width * frame_height)

        # Determine horizontal zone
        if norm_x < self.far_left_th:
            zone = HorizontalZone.FAR_LEFT
            zone_label = "far left"
            dir_instruction = "Turn sharply left"
        elif norm_x < self.left_th:
            zone = HorizontalZone.LEFT
            zone_label = "left"
            dir_instruction = "Turn slightly left"
        elif norm_x <= self.right_th:
            zone = HorizontalZone.CENTER
            zone_label = "center"
            dir_instruction = "Straight ahead"
        elif norm_x <= self.far_right_th:
            zone = HorizontalZone.RIGHT
            zone_label = "right"
            dir_instruction = "Turn slightly right"
        else:
            zone = HorizontalZone.FAR_RIGHT
            zone_label = "far right"
            dir_instruction = "Turn sharply right"

        # Determine proximity and arrival
        if area_ratio >= self.arrival_area_th:
            proximity = Proximity.REACHED
            prox_label = "reached"
            is_reached = True
            dist_instruction = "Stop! Target is within reach."
        elif area_ratio >= self.far_area_th:
            proximity = Proximity.APPROACHING
            prox_label = "approaching"
            is_reached = False
            dist_instruction = "Walk forward, target approaching."
        else:
            proximity = Proximity.FAR
            prox_label = "far"
            is_reached = False
            dist_instruction = "Target is far, walk forward."

        return SpatialLocation(
            zone=zone,
            zone_label=zone_label,
            direction_instruction=dir_instruction,
            normalized_x=norm_x,
            normalized_y=norm_y,
            area_ratio=area_ratio,
            proximity=proximity,
            proximity_label=prox_label,
            is_reached=is_reached,
            distance_instruction=dist_instruction,
        )

    @staticmethod
    def format_guidance(location: SpatialLocation, target_name: str = "object") -> str:
        """
        Produce a short spoken guidance string ("Bottle, left.").

        Kept short on purpose: guidance repeats while the user moves, and
        each extra word delays the next update (~4.6 s for the old long
        form vs ~1-1.5 s for a short phrase, measured with Windows SAPI).
        """
        name = target_name.strip().capitalize() if target_name else "Object"

        if location.is_reached:
            return f"{name} is directly in front of you. Stop and reach out."

        if location.zone == HorizontalZone.CENTER:
            return f"{name}, straight ahead."
        elif location.zone == HorizontalZone.FAR_LEFT:
            return f"{name}, far left."
        elif location.zone == HorizontalZone.LEFT:
            return f"{name}, left."
        elif location.zone == HorizontalZone.RIGHT:
            return f"{name}, right."
        else:  # FAR_RIGHT
            return f"{name}, far right."
