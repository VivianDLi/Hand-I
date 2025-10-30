import numpy as np

from handi.types import (
    Angle,
    GestureResult,
    Landmark,
    PostInterface,
    StreamResult,
    TrackingResult,
)

COORD_OFFSETS = {
    Landmark.WRIST: (0, 0),
    Landmark.INDEX_FINGER_MCP: (-5, -20),
    Landmark.MIDDLE_FINGER_MCP: (0, -20),
    Landmark.RING_FINGER_MCP: (5, -20),
    Landmark.PINKY_FINGER_MCP: (10, -20),
}


class LandmarkVisualizer(PostInterface):
    def __init__(self, *args, draw_angles: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.draw_angles = draw_angles

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
            for i, hand in enumerate([landmarks.left_hand, landmarks.right_hand]):
                if hand is not None:
                    x = 0
                    y = 0
                    color = (255, 0, 0) if hand.handedness else (0, 0, 255)
                    # Draw landmarks on hand
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
                    # Draw hand angles
                    hand_center = (100, 50 + i * 50)
                    start_points = {
                        landmark: (
                            hand_center[0] + offset[0],
                            hand_center[1] + offset[1],
                        )
                        for landmark, offset in COORD_OFFSETS.items()
                    }
                    for angle in Angle:
                        # Draw connecting lines
                        start_landmark, end_landmark = Landmark(
                            angle.value[0]
                        ), Landmark(angle.value[1])
                        if (
                            start_landmark in hand.landmarks
                            and end_landmark in hand.landmarks
                        ):
                            start_coords = hand.landmarks[start_landmark]
                            end_coords = hand.landmarks[end_landmark]
                            start_point = (
                                int(start_coords.x * frame.shape[1]),
                                int(start_coords.y * frame.shape[0]),
                            )
                            end_point = (
                                int(end_coords.x * frame.shape[1]),
                                int(end_coords.y * frame.shape[0]),
                            )
                            out_frame = cv2.line(
                                out_frame,
                                pt1=start_point,
                                pt2=end_point,
                                color=color,
                                thickness=2,
                            )
                            if self.draw_angles:
                                out_frame = cv2.putText(
                                    out_frame,
                                    f"{angle.value[0]:.2f}, {angle.value[1]:.2f}",
                                    (start_point[0] + 5, start_point[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    color,
                                    2,
                                    cv2.LINE_AA,
                                )
                        # Draw normalized hand position
                        if angle in hand.angles:
                            coords = hand.angles[angle]
                            # Assuming palm axis is facing out of the screen
                            r = int(10 * np.sin(coords.phi))
                            x, y = -int(r * np.sin(coords.theta)), int(
                                r * np.cos(coords.theta)
                            )
                            start_point = start_points[Landmark(angle.value[0])]
                            end_point = (
                                start_point[0] + x,
                                start_point[1] - y,
                            )
                            out_frame = cv2.line(
                                out_frame,
                                pt1=start_point,
                                pt2=end_point,
                                color=color,
                                thickness=2,
                            )
                            start_points[Landmark(angle.value[1])] = end_point
        if gestures is not None:
            for i, gesture in enumerate([gestures.left_hand, gestures.right_hand]):
                if gesture is not None:
                    color = (255, 0, 0) if i == 0 else (0, 0, 255)
                    out_frame = cv2.putText(
                        out_frame,
                        f"{gesture.name}",
                        (50, 50 + i * 50),
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
