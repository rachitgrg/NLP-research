# tests/test_spatial_locator.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Unit tests for SpatialLocator
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from guidance.spatial_locator import (
    HorizontalZone,
    Proximity,
    SpatialLocation,
    SpatialLocator,
)


@pytest.fixture()
def locator() -> SpatialLocator:
    return SpatialLocator(
        far_left_th=0.20,
        left_th=0.40,
        right_th=0.60,
        far_right_th=0.80,
        far_area_th=0.05,
        arrival_area_th=0.22,
    )


class TestSpatialLocatorHorizontalZones:
    """Verify that horizontal zones are correctly identified."""

    def test_far_left_zone(self, locator):
        # Frame: 1000x1000. Bbox center x = 100 -> norm_x = 0.10 (< 0.20)
        loc = locator.locate([50, 400, 150, 600], 1000, 1000)
        assert loc.zone == HorizontalZone.FAR_LEFT
        assert loc.zone_label == "far left"
        assert "left" in loc.direction_instruction.lower()

    def test_left_zone(self, locator):
        # Bbox center x = 300 -> norm_x = 0.30 (0.20 <= norm_x < 0.40)
        loc = locator.locate([250, 400, 350, 600], 1000, 1000)
        assert loc.zone == HorizontalZone.LEFT
        assert loc.zone_label == "left"
        assert "slightly left" in loc.direction_instruction.lower()

    def test_center_zone(self, locator):
        # Bbox center x = 500 -> norm_x = 0.50 (0.40 <= norm_x <= 0.60)
        loc = locator.locate([450, 400, 550, 600], 1000, 1000)
        assert loc.zone == HorizontalZone.CENTER
        assert loc.zone_label == "center"
        assert "straight ahead" in loc.direction_instruction.lower()

    def test_right_zone(self, locator):
        # Bbox center x = 700 -> norm_x = 0.70 (0.60 < norm_x <= 0.80)
        loc = locator.locate([650, 400, 750, 600], 1000, 1000)
        assert loc.zone == HorizontalZone.RIGHT
        assert loc.zone_label == "right"
        assert "slightly right" in loc.direction_instruction.lower()

    def test_far_right_zone(self, locator):
        # Bbox center x = 900 -> norm_x = 0.90 (> 0.80)
        loc = locator.locate([850, 400, 950, 600], 1000, 1000)
        assert loc.zone == HorizontalZone.FAR_RIGHT
        assert loc.zone_label == "far right"
        assert "right" in loc.direction_instruction.lower()


class TestSpatialLocatorProximityAndArrival:
    """Verify distance estimation and arrival trigger."""

    def test_far_proximity(self, locator):
        # Small box: 50x50 in 1000x1000 = 0.0025 area ratio (< 0.05)
        loc = locator.locate([475, 475, 525, 525], 1000, 1000)
        assert loc.proximity == Proximity.FAR
        assert loc.is_reached is False
        assert "far" in loc.distance_instruction.lower()

    def test_approaching_proximity(self, locator):
        # Mid box: 300x300 in 1000x1000 = 0.09 area ratio (0.05 <= ratio < 0.22)
        loc = locator.locate([350, 350, 650, 650], 1000, 1000)
        assert loc.proximity == Proximity.APPROACHING
        assert loc.is_reached is False
        assert "approaching" in loc.distance_instruction.lower()

    def test_reached_arrival(self, locator):
        # Large box: 500x500 in 1000x1000 = 0.25 area ratio (>= 0.22)
        loc = locator.locate([250, 250, 750, 750], 1000, 1000)
        assert loc.proximity == Proximity.REACHED
        assert loc.is_reached is True
        assert "within reach" in loc.distance_instruction.lower()


class TestSpatialLocatorEdgeCases:
    """Handle unusual bounding box coordinates and invalid dimensions."""

    def test_inverted_coordinates_normalized(self, locator):
        # x1 > x2, y1 > y2
        loc = locator.locate([600, 600, 400, 400], 1000, 1000)
        assert loc.normalized_x == pytest.approx(0.50)
        assert loc.normalized_y == pytest.approx(0.50)
        assert loc.area_ratio == pytest.approx(0.04)

    def test_coordinates_exceeding_frame_clamped(self, locator):
        # Box extending past frame boundaries
        loc = locator.locate([-100, 400, 100, 600], 1000, 1000)
        assert 0.0 <= loc.normalized_x <= 1.0

    def test_zero_dimensions_raise_value_error(self, locator):
        with pytest.raises(ValueError):
            locator.locate([10, 10, 50, 50], 0, 1000)
        with pytest.raises(ValueError):
            locator.locate([10, 10, 50, 50], 1000, -5)

    def test_to_dict_format(self, locator):
        loc = locator.locate([450, 400, 550, 600], 1000, 1000)
        d = loc.to_dict()
        assert d["zone"] == "center"
        assert d["is_reached"] is False
        assert isinstance(d["area_ratio"], float)


class TestFormatGuidance:
    """Verify natural language spoken guidance outputs."""

    def test_format_guidance_reached(self, locator):
        loc = locator.locate([200, 200, 800, 800], 1000, 1000)
        msg = SpatialLocator.format_guidance(loc, "bottle")
        assert "Bottle is directly in front of you" in msg
        assert "reach out" in msg

    def test_format_guidance_center(self, locator):
        loc = locator.locate([450, 400, 550, 600], 1000, 1000)
        msg = SpatialLocator.format_guidance(loc, "bottle")
        assert "straight ahead" in msg

    def test_format_guidance_left_and_right(self, locator):
        loc_left = locator.locate([250, 400, 350, 600], 1000, 1000)
        msg_left = SpatialLocator.format_guidance(loc_left, "phone")
        assert "left" in msg_left

        loc_right = locator.locate([650, 400, 750, 600], 1000, 1000)
        msg_right = SpatialLocator.format_guidance(loc_right, "cup")
        assert "right" in msg_right
