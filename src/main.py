import cv2
import numpy as np

from handi.io.camera import CameraStream
from handi.models.mediapipe_interface import MediapipeInterface
from handi.types import EventManager, TrackingResult


def show_frame(frame: np.ndarray, landmarks: TrackingResult) -> None:
    """Draw landmarks on the frame."""
    for hand in [landmarks.left_hand, landmarks.right_hand]:
        if hand is not None:
            x = 0
            y = 0
            for landmark in hand.landmarks.values():
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow('Camera', frame)

if __name__ == "__main__":
    # Open the default camera
    cam = cv2.VideoCapture(0)
    event_manager = EventManager()
    stream = CameraStream(0)
    interface = MediapipeInterface()

    event_manager.connect_stream(stream)
    event_manager.connect_predictor(interface)
    interface.landmark_predicted.connect(show_frame)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    interface.start()
    stream.start()

    # Release the capture and writer objects
    stream.stop()
    interface.stop()
    event_manager.close()
    cv2.destroyAllWindows()
