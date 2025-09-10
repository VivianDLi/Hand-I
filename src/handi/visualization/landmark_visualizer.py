import numpy as np

from handi.types import (
    GestureResult,
    PostInterface,
    StreamResult,
    TrackingResult,
)


class LandmarkVisualizer(PostInterface):
    def _show_frame(
        self,
        frame: np.ndarray,
        landmarks: TrackingResult | None = None,
        gestures: GestureResult | None = None,
    ) -> None:
        """Draw landmarks on the frame."""
        import cv2

        out_frame = frame.copy()
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        if landmarks is not None:
            for hand in [landmarks.left_hand, landmarks.right_hand]:
                if hand is not None:
                    x = 0
                    y = 0
                    color = (255, 0, 0) if hand.handedness else (0, 0, 255)
                    for landmark in hand.landmarks.values():
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        out_frame = cv2.circle(
                            out_frame,
                            center=(x, y),
                            radius=5,
                            color=color,
                            thickness=-1,
                        )
        if gestures is not None:
            for i, gesture in enumerate(
                [gestures.left_hand, gestures.right_hand]
            ):
                if gesture is not None:
                    color = (255, 0, 0) if i == 0 else (0, 0, 255)
                    out_frame = cv2.putText(
                        out_frame,
                        f"{gesture.name}",
                        (10, 30 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        cv2.imshow("Camera", out_frame)
        cv2.waitKey(1)

    def visualize_landmarks(
        self, result: StreamResult | TrackingResult | GestureResult
    ):
        match result:
            case StreamResult():
                data = result.data
                self._show_frame(data)
            case TrackingResult():
                data = result.original_image
                self._show_frame(data, landmarks=result)
            case GestureResult():
                data = result.landmark_result.original_image
                self._show_frame(
                    data, landmarks=result.landmark_result, gestures=result
                )
            case _:
                raise ValueError("Unsupported result type for visualization.")

    def process_results(self, data: GestureResult) -> None:
        self.visualize_landmarks(data)

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True

    def stop(self) -> None:
        import cv2

        if not self.is_running:
            return
        cv2.destroyAllWindows()
        self.is_running = False
