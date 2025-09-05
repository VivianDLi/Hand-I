from src.handi.types import GestureResult, TrackingResult


class GestureClassifier:
    def __init__(self):
        pass

    def classify_gesture(self, landmarks):
        """Classify the gesture based on the provided landmarks."""
        raise NotImplementedError

    def _process_results(self, result: TrackingResult) -> GestureResult:
        """Process the landmarks to classify the gesture."""
        raise NotImplementedError

    def start(self) -> None:
        """Start the gesture classifier."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the gesture classifier and release resources."""
        raise NotImplementedError
