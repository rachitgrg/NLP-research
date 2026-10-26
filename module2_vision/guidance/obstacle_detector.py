# guidance/obstacle_detector.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 5: Obstacle Detection & Avoidance
#
# Inspects YOLO detections to identify physical obstacles that
# block the user's immediate walking corridor.
# Generates real-time warnings and safe detour advice.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config.config import (
    OBSTACLE_CLASSES,
    OBSTACLE_CORRIDOR_LEFT,
    OBSTACLE_CORRIDOR_RIGHT,
    OBSTACLE_CORRIDOR_Y_MIN,
    OBSTACLE_DETECTION_ENABLED,
    OBSTACLE_HOLDER_CLASSES,
    OBSTACLE_MIN_AREA_RATIO,
)
from guidance.phrases import phrase


@dataclass(frozen=True)
class ObstacleHazard:
    """
    Describes an individual hazard identified in the walking path.
    """
    class_name: str
    confidence: float
    bbox: list[int]
    area_ratio: float
    normalized_cx: float
    detour_direction: str  # "left", "right", "stop"
    warning_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "area_ratio": round(self.area_ratio, 4),
            "normalized_cx": round(self.normalized_cx, 4),
            "detour_direction": self.detour_direction,
            "warning_text": self.warning_text,
        }


@dataclass(frozen=True)
class ObstacleReport:
    """
    Aggregated safety report for the current video frame.
    """
    has_hazard: bool
    primary_hazard: Optional[ObstacleHazard]
    hazards: list[ObstacleHazard]
    recommended_detour: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_hazard": self.has_hazard,
            "primary_hazard": self.primary_hazard.to_dict() if self.primary_hazard else None,
            "hazards": [h.to_dict() for h in self.hazards],
            "recommended_detour": self.recommended_detour,
        }


class ObstacleDetector:
    """
    Filters and evaluates YOLO detections to protect the user from walking into hazards.
    """

    def __init__(
        self,
        enabled: bool = OBSTACLE_DETECTION_ENABLED,
        corridor_left: float = OBSTACLE_CORRIDOR_LEFT,
        corridor_right: float = OBSTACLE_CORRIDOR_RIGHT,
        corridor_y_min: float = OBSTACLE_CORRIDOR_Y_MIN,
        min_area_ratio: float = OBSTACLE_MIN_AREA_RATIO,
        obstacle_classes: tuple[str, ...] = OBSTACLE_CLASSES,
        holder_classes: tuple[str, ...] = OBSTACLE_HOLDER_CLASSES,
        language: str = "en",
    ) -> None:
        self.language = language  # warnings via guidance/phrases.py
        self.holder_classes = set(c.lower() for c in holder_classes)
        self.enabled = enabled
        self.corridor_left = corridor_left
        self.corridor_right = corridor_right
        self.corridor_y_min = corridor_y_min
        self.min_area_ratio = min_area_ratio
        self.obstacle_classes = set(c.lower() for c in obstacle_classes)

    def evaluate(
        self,
        detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        target_class: Optional[str] = None,
        target_bbox: Optional[list[int]] = None,
    ) -> ObstacleReport:
        """
        Evaluate frame detections for walking hazards.

        Parameters
        ----------
        detections : list[dict]
            List of detection dictionaries from YOLODetector.
        frame_width : int
            Video frame width in pixels (> 0).
        frame_height : int
            Video frame height in pixels (> 0).
        target_class : str | None
            The class of the target object being searched for.
            Instances of the target class itself are excluded from being treated as obstacles.
        target_bbox : [x1, y1, x2, y2] | None
            Current (or recently seen) target box.  A holder-class object
            (default: person) whose box contains the target's centre is
            treated as holding the target, so it is not reported as
            blocking the way to it.  Without depth this cannot tell
            "holding" from "standing in front of"; see PROJECT_AUDIT.md.

        Returns
        -------
        ObstacleReport
        """
        if not self.enabled or not detections or frame_width <= 0 or frame_height <= 0:
            return ObstacleReport(
                has_hazard=False,
                primary_hazard=None,
                hazards=[],
                recommended_detour=None,
            )

        target_normalized = target_class.lower().strip() if target_class else None
        target_center: Optional[tuple[float, float]] = None
        if target_bbox and len(target_bbox) == 4:
            tx1, ty1, tx2, ty2 = target_bbox
            target_center = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
        hazards: list[ObstacleHazard] = []

        for det in detections:
            class_name = det.get("class_name", "").lower().strip()
            bbox = det.get("bbox")
            conf = float(det.get("confidence", 0.0))

            if not bbox or len(bbox) != 4:
                continue

            # Don't treat the user's requested target as an obstacle
            if target_normalized and class_name == target_normalized:
                continue

            # Check if this class is an obstacle category
            if class_name not in self.obstacle_classes:
                continue

            x1, y1, x2, y2 = bbox
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)

            # A person whose box contains the target is holding it: not a
            # blocker.  Furniture is NOT exempted: "bottle on the chair" and
            # "chair in front of the bottle" look identical in 2D, and the
            # chair may genuinely be in the way.
            if (
                target_center
                and class_name in self.holder_classes
                and min_x <= target_center[0] <= max_x
                and min_y <= target_center[1] <= max_y
            ):
                continue

            box_w = max(0, max_x - min_x)
            box_h = max(0, max_y - min_y)
            area_ratio = (box_w * box_h) / (frame_width * frame_height)

            # Check minimum size threshold
            if area_ratio < self.min_area_ratio:
                continue

            # Check vertical corridor / floor proximity (bottom of bbox must be near ground)
            norm_y2 = max_y / frame_height
            if norm_y2 < self.corridor_y_min:
                continue

            # Check horizontal corridor intersection
            norm_x1 = min_x / frame_width
            norm_x2 = max_x / frame_width

            corridor_intersect = not (
                norm_x2 < self.corridor_left or norm_x1 > self.corridor_right
            )

            if not corridor_intersect:
                continue

            # It's an active hazard in the corridor!
            cx = (min_x + max_x) / 2.0
            norm_cx = cx / frame_width

            # Determine detour direction based on obstacle center
            box_span = norm_x2 - norm_x1
            if box_span > 0.65 or area_ratio > 0.35:
                # Spans most of the walking path or very close
                detour = "stop"
                warning = phrase("obstacle_ahead", self.language, obs=class_name)
            elif norm_cx <= 0.50:
                # Obstacle is on the left side of path -> veer right
                detour = "right"
                warning = phrase("obstacle_left", self.language, obs=class_name)
            else:
                # Obstacle is on the right side of path -> veer left
                detour = "left"
                warning = phrase("obstacle_right", self.language, obs=class_name)

            hazards.append(
                ObstacleHazard(
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                    area_ratio=area_ratio,
                    normalized_cx=norm_cx,
                    detour_direction=detour,
                    warning_text=warning,
                )
            )

        if not hazards:
            return ObstacleReport(
                has_hazard=False,
                primary_hazard=None,
                hazards=[],
                recommended_detour=None,
            )

        # Sort hazards by area ratio descending (closest/largest hazard first)
        hazards.sort(key=lambda h: h.area_ratio, reverse=True)
        primary = hazards[0]

        return ObstacleReport(
            has_hazard=True,
            primary_hazard=primary,
            hazards=hazards,
            recommended_detour=primary.warning_text,
        )
