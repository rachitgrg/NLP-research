# tests/test_assistive_pipeline.py
# ─────────────────────────────────────────────────────────────
# Module 3 – End-to-End Assistive Navigation Integration Tests
#
# Simulates the complete user scenario requested by the user:
#   1. Voice query: "where is my bottle" -> NLP extracts "bottle"
#   2. Frame 1: Bottle not in view -> "Bottle not seen. Turn slowly."
#   3. Frame 2: User pans, bottle appears in far-right -> "Bottle, far right."
#   4. Frame 3: User turns, bottle moves to center -> "Bottle, straight ahead."
#   5. Frame 4: Chair blocks walking corridor -> "Chair on left. Step slightly right."
#   6. Frame 5: User bypasses chair, approaches bottle -> "Bottle is directly in front of you. Stop and reach out."
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from audio_feedback.tts_speaker import TTSSpeaker
from guidance.obstacle_detector import ObstacleDetector
from guidance.spatial_locator import HorizontalZone, SpatialLocator
from matching.target_matcher import TargetMatcher
from navigation.navigation_controller import NavigationController
from navigation.navigation_state import NavigationState
from pipeline.m3_pipeline import extract_target_from_module1


class TestFullAssistiveWorkflow:
    """End-to-end simulation of the requested blind-assistive navigation flow."""

    def test_complete_user_flow(self):
        # ── Step 1: Voice input parsing (simulated Module 1 output) ──
        module1_voice_output = {
            "text": "where is my bottle",
            "language": "en",
            "intent": "find_object",
            "object": "bottle",
            "attributes": {},
            "keywords": ["bottle"],
        }
        target_info = extract_target_from_module1(module1_voice_output)
        assert target_info["target"] == "bottle"

        spoken_utterances = []

        def capture_speech(text, priority):
            spoken_utterances.append((text, priority))

        speaker = TTSSpeaker(enabled=True, use_mock=True, cooldown_s=0.0)
        speaker.on_speak = capture_speech

        controller = NavigationController(
            target_name=target_info["target"],
            matcher=TargetMatcher(),
            locator=SpatialLocator(arrival_area_th=0.20),
            obstacle_detector=ObstacleDetector(),
            speaker=speaker,
            searching_cooldown_s=0.0,
        )

        frame_w, frame_h = 1000, 1000

        # ── Step 2 / Frame 1: Camera opens, bottle is not found ───────
        frame1_detections = [
            {"class_name": "person", "confidence": 0.92, "bbox": [50, 50, 120, 200]}
        ]
        step1 = controller.process_frame(frame1_detections, frame_w, frame_h)
        assert step1.state == NavigationState.SEARCHING
        assert step1.target_found is False
        assert "bottle not seen" in step1.spoken_instruction.lower()

        # ── Step 3 / Frame 2: User pans camera, bottle spotted in far-right ──
        # x: 820..940 (norm cx = 880/1000 = 0.88 > 0.80 -> FAR_RIGHT)
        frame2_detections = [
            {"class_name": "bottle", "confidence": 0.86, "bbox": [820, 450, 940, 600]}
        ]
        step2 = controller.process_frame(frame2_detections, frame_w, frame_h)
        assert step2.state == NavigationState.GUIDING
        assert step2.target_found is True
        assert step2.spatial_location.zone == HorizontalZone.FAR_RIGHT
        assert "far right" in step2.spoken_instruction.lower()

        # ── Step 3 / Frame 3: User turns towards bottle, bottle is in center ──
        # x: 450..550 (norm cx = 500/1000 = 0.50 -> CENTER)
        frame3_detections = [
            {"class_name": "bottle", "confidence": 0.89, "bbox": [450, 450, 550, 600]}
        ]
        step3 = controller.process_frame(frame3_detections, frame_w, frame_h)
        assert step3.state == NavigationState.GUIDING
        assert step3.spatial_location.zone == HorizontalZone.CENTER
        assert "straight ahead" in step3.spoken_instruction.lower()

        # ── Step 5 / Frame 4: Obstacle (chair) blocks walking path ──────────
        # Chair in walking corridor (cx = 400, in left of corridor)
        frame4_detections = [
            {"class_name": "bottle", "confidence": 0.89, "bbox": [450, 450, 550, 600]},
            {"class_name": "chair",  "confidence": 0.94, "bbox": [280, 400, 520, 850]},
        ]
        step4 = controller.process_frame(frame4_detections, frame_w, frame_h)
        assert step4.state == NavigationState.OBSTRUCTED
        assert step4.obstacle_report.has_hazard is True
        assert "chair" in step4.spoken_instruction.lower()
        assert "step slightly right" in step4.spoken_instruction.lower()

        # ── Step 4 / Frame 5: User sidesteps chair, reaches bottle ───────────
        # Large bottle (> 20% area: 600x600 = 36% area)
        frame5_detections = [
            {"class_name": "bottle", "confidence": 0.95, "bbox": [200, 200, 800, 800]}
        ]
        for _ in range(controller.arrival_confirm_frames):
            step5 = controller.process_frame(frame5_detections, frame_w, frame_h)
        assert step5.state == NavigationState.ARRIVED
        assert step5.is_arrived is True
        assert "directly in front of you" in step5.spoken_instruction.lower()
        assert "reach out" in step5.spoken_instruction.lower()

        speaker.stop()
