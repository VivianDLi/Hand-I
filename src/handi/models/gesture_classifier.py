from pathlib import Path

from handi.config import GESTURE_CONFIG_PATH
from handi.types import (
    Gesture,
    GestureClassifierInterface,
    GestureConfig,
    GestureResult,
    LandmarkPrediction,
    TrackingResult,
)


class GestureClassifier(GestureClassifierInterface):
    def __init__(self, config: GestureConfig | Path | str | None = None):
        if config is None:
            self.config = GestureConfig.load_from_file(GESTURE_CONFIG_PATH)
        elif isinstance(config, GestureConfig):
            self.config = config
        elif isinstance(config, (str | Path)):
            self.config = GestureConfig.load_from_file(config)
        else:
            raise ValueError(
                "Invalid config type. Must be GestureConfig, str, Path, or None."
            )
        self.prev_time_ms = 0.0

    def _classify_gesture(
        self, hand_landmarks: LandmarkPrediction, duration: float
    ) -> Gesture | None:
        curr_gesture = None
        valid_gestures = [
            g
            for g in self.config.gestures
            if g.check_gesture(hand_landmarks.angles)
        ]
        self.config.durations = [
            dur + duration if g in valid_gestures else 0.0
            for g, dur in zip(
                self.config.gestures, self.config.durations, strict=True
            )
        ]
        for gesture, dur in zip(
            self.config.gestures, self.config.durations, strict=True
        ):
            if dur >= gesture.time_delay:
                curr_gesture = gesture
                break  # Return the first valid gesture that meets the time delay
        return curr_gesture

    def classify_gesture(self, landmarks: TrackingResult) -> None:
        """Classify the gesture based on the provided landmarks."""
        # Get duration
        duration = landmarks.timestamp - self.prev_time_ms
        self.prev_time_ms = landmarks.timestamp
        # Separate gestures by hand
        lh_gesture = (
            self._classify_gesture(landmarks.left_hand, duration)
            if landmarks.left_hand
            else None
        )
        rh_gesture = (
            self._classify_gesture(landmarks.right_hand, duration)
            if landmarks.right_hand
            else None
        )
        # Process gestures
        self.process_results(lh_gesture, rh_gesture, landmarks)

    def _process_results(
        self,
        lh_gesture: Gesture | None,
        rh_gesture: Gesture | None,
        landmarks: TrackingResult,
    ) -> GestureResult:
        """Process the landmarks to classify the gesture."""
        return GestureResult(
            landmark_result=landmarks,
            left_hand=lh_gesture,
            right_hand=rh_gesture,
        )

    def start(self) -> None:
        """Start the gesture classifier."""
        self.config.reset_durations()
        self.is_running = True

    def stop(self) -> None:
        """Stop the gesture classifier and release resources."""
        self.config.reset_durations()
        self.is_running = False
