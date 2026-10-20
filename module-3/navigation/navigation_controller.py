# navigation/navigation_controller.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 5: Assistive Navigation Controller
#
# Core orchestrator that combines:
#   1. TargetMatcher (YOLO object matching)
#   2. SpatialLocator (Directional & proximity calculation)
#   3. ObstacleDetector (Hazard collision avoidance)
#   4. TTSSpeaker (Spoken speech feedback for blind users)
#
# Emits coordinated state updates and voice guidance.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from audio_feedback.tts_speaker import SpeechPriority, TTSSpeaker
from config.config import ARRIVAL_CONFIRM_FRAMES, SEARCH_TIMEOUT_S
from guidance.obstacle_detector import ObstacleDetector, ObstacleReport
from guidance.spatial_locator import SpatialLocation, SpatialLocator
from matching.target_matcher import TargetMatcher
from navigation.navigation_state import NavigationState, NavigationStep

logger = logging.getLogger(__name__)


class NavigationController:
    """
    Coordinates target navigation and obstacle avoidance for blind users.
    """

    def __init__(
        self,
        target_name: str,
        matcher: Optional[TargetMatcher] = None,
        locator: Optional[SpatialLocator] = None,
        obstacle_detector: Optional[ObstacleDetector] = None,
        speaker: Optional[TTSSpeaker] = None,
        searching_cooldown_s: float = 3.5,
        arrival_confirm_frames: int = ARRIVAL_CONFIRM_FRAMES,
        search_timeout_s: float = SEARCH_TIMEOUT_S,
    ) -> None:
        self.raw_target = target_name
        self.target_name = target_name.strip().lower()

        self.matcher = matcher or TargetMatcher()
        self.locator = locator or SpatialLocator()
        self.obstacle_detector = obstacle_detector or ObstacleDetector()
        self.speaker = speaker or TTSSpeaker(use_mock=True)

        self.searching_cooldown_s = searching_cooldown_s
        self.arrival_confirm_frames = max(1, arrival_confirm_frames)
        self.search_timeout_s = search_timeout_s

        # Tracking state
        self.state: NavigationState = NavigationState.SEARCHING
        self.last_seen_target_time: float = 0.0
        self.last_search_prompt_time: float = 0.0
        self.ever_found: bool = False
        self.has_arrived: bool = False

        # Session completion: None while searching, else "arrived" / "timeout".
        # The camera loop ends the session once this is set.
        self.finish_reason: Optional[str] = None
        self._arrival_streak: int = 0
        self._session_start_time: Optional[float] = None

    def process_frame(
        self,
        detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> NavigationStep:
        """
        Process a single visual frame or snapshot and generate navigational instructions.

        Parameters
        ----------
        detections : list[dict]
            Detections from YOLODetector.
        frame_width : int
            Width of the camera frame in pixels.
        frame_height : int
            Height of the camera frame in pixels.

        Returns
        -------
        NavigationStep
        """
        now = time.monotonic()
        if self._session_start_time is None:
            self._session_start_time = now

        # Step 1: Match target object
        match_result = self.matcher.match(self.target_name, detections)
        is_found = match_result.get("found", False)
        reason = match_result.get("reason")

        # Step 2: Handle unsupported classes
        if reason == "unsupported_class":
            self.state = NavigationState.UNSUPPORTED
            # Nothing to search for — end the session after this message.
            self.finish_reason = "unsupported"
            instruction = f"{self.target_name.capitalize()} is not detectable by the vision model."
            self.speaker.speak(instruction, priority=SpeechPriority.EMERGENCY)
            empty_obstacle = ObstacleReport(
                has_hazard=False, primary_hazard=None, hazards=[], recommended_detour=None
            )
            return NavigationStep(
                state=self.state,
                target_name=self.target_name,
                target_found=False,
                spoken_instruction=instruction,
                spatial_location=None,
                obstacle_report=empty_obstacle,
                is_arrived=False,
            )

        # Step 2b: Search timeout, measured from the last sighting (or session
        # start).  Checked before obstacles so a persistent obstacle (e.g. a
        # person in view) cannot keep the search alive forever.
        if not is_found and self.search_timeout_s > 0:
            last_seen = self.last_seen_target_time if self.ever_found else self._session_start_time
            if (now - last_seen) >= self.search_timeout_s:
                self.finish_reason = "timeout"
                instruction = f"Could not find the {self.target_name}. Stopping the search."
                self.speaker.speak(instruction, priority=SpeechPriority.ARRIVAL, force=True)
                return NavigationStep(
                    state=self.state,
                    target_name=self.target_name,
                    target_found=False,
                    spoken_instruction=instruction,
                    spatial_location=None,
                    obstacle_report=ObstacleReport(
                        has_hazard=False, primary_hazard=None, hazards=[], recommended_detour=None
                    ),
                    is_arrived=False,
                )

        # Step 3: Evaluate obstacles in path
        matched_class = match_result.get("matched_class")
        obstacle_report = self.obstacle_detector.evaluate(
            detections=detections,
            frame_width=frame_width,
            frame_height=frame_height,
            target_class=matched_class,
        )

        # Priority Check: Immediate Obstacle Hazard
        if obstacle_report.has_hazard and obstacle_report.primary_hazard:
            hazard = obstacle_report.primary_hazard
            self.state = NavigationState.OBSTRUCTED
            instruction = hazard.warning_text
            # Not forced: called every frame while the hazard persists; the
            # speaker repeats identical text only after its cooldown.
            self.speaker.speak(instruction, priority=SpeechPriority.EMERGENCY)

            # If target is also visible, compute its location for reporting
            target_location: Optional[SpatialLocation] = None
            if is_found and match_result.get("bbox"):
                target_location = self.locator.locate(
                    match_result["bbox"], frame_width, frame_height
                )

            return NavigationStep(
                state=self.state,
                target_name=self.target_name,
                target_found=is_found,
                spoken_instruction=instruction,
                spatial_location=target_location,
                obstacle_report=obstacle_report,
                is_arrived=False,
            )

        # Step 4: Handle Target Found
        if is_found and match_result.get("bbox"):
            self.ever_found = True
            self.last_seen_target_time = now
            bbox = match_result["bbox"]

            spatial_loc = self.locator.locate(bbox, frame_width, frame_height)

            if spatial_loc.is_reached:
                self.state = NavigationState.ARRIVED
                self.has_arrived = True
                self._arrival_streak += 1
                instruction = f"{self.target_name.capitalize()} is directly in front of you. Stop and reach out."
                self.speaker.speak(instruction, priority=SpeechPriority.ARRIVAL)
                if self._arrival_streak >= self.arrival_confirm_frames:
                    self.finish_reason = "arrived"
            else:
                self._arrival_streak = 0
                self.state = NavigationState.GUIDING
                instruction = self.locator.format_guidance(spatial_loc, self.target_name)
                self.speaker.speak(instruction, priority=SpeechPriority.GUIDANCE)

            return NavigationStep(
                state=self.state,
                target_name=self.target_name,
                target_found=True,
                spoken_instruction=instruction,
                spatial_location=spatial_loc,
                obstacle_report=obstacle_report,
                is_arrived=self.has_arrived,
            )

        # Step 5: Handle Target Not Found
        self._arrival_streak = 0
        if self.ever_found:
            self.state = NavigationState.LOST
            instruction = f"{self.target_name.capitalize()} lost. Turn back slowly."
        else:
            self.state = NavigationState.SEARCHING
            instruction = f"{self.target_name.capitalize()} not seen. Turn slowly."

        # Speak searching prompt with pacing cooldown
        if (now - self.last_search_prompt_time) >= self.searching_cooldown_s:
            if self.speaker.speak(instruction, priority=SpeechPriority.STATUS):
                self.last_search_prompt_time = now

        return NavigationStep(
            state=self.state,
            target_name=self.target_name,
            target_found=False,
            spoken_instruction=instruction,
            spatial_location=None,
            obstacle_report=obstacle_report,
            is_arrived=False,
        )

    def reset(self, new_target: Optional[str] = None) -> None:
        """Reset state tracking for a new search session."""
        if new_target:
            self.raw_target = new_target
            self.target_name = new_target.strip().lower()
        self.state = NavigationState.SEARCHING
        self.last_seen_target_time = 0.0
        self.last_search_prompt_time = 0.0
        self.ever_found = False
        self.has_arrived = False
        self.finish_reason = None
        self._arrival_streak = 0
        self._session_start_time = None
        self.speaker.clear()
