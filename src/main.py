import cv2
import numpy as np

from handi.io.camera import CameraStream
from handi.models.mediapipe_interface import MediapipeInterface
from handi.types import EventManager, TrackingResult


def show_frame(frame: np.ndarray, landmarks: TrackingResult) -> None:
    """Draw landmarks on the frame."""
    out_frame = frame.copy()
    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
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
    cv2.imshow("Camera", out_frame)
    cv2.waitKey(1)


if __name__ == "__main__":
    event_manager = EventManager()
    stream = CameraStream(0)
    interface = MediapipeInterface()

    event_manager.connect_stream(stream)
    event_manager.connect_predictor(interface)
    interface.landmark_predicted.connect(show_frame)

    try:
        interface.start()
        stream.start()
    except KeyboardInterrupt:
        pass

    # Release the capture and writer objects
    stream.stop()
    interface.stop()
    cv2.destroyAllWindows()
