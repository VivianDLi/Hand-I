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
    cv2.imshow("Camera", frame)


if __name__ == "__main__":
    # Try different camera indices
    for i in range(5):  # Test indices 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index: {i}")
            # You can add code here to read frames and display them
            # ret, frame = cap.read()
            # if ret:
            #    cv2.imshow('Camera Test', frame)
            #    cv2.waitKey(1)
            cap.release()
            break  # Exit loop once a working camera is found
        else:
            print(f"No camera found at index: {i}")

        if (
            not cap.isOpened() and i == 4
        ):  # If loop completes without finding a camera
            print(
                "Error: Could not open any camera. Check connections and permissions."
            )
    exit()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        exit()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
    exit()

    event_manager = EventManager()
    stream = CameraStream(0)
    interface = MediapipeInterface()

    event_manager.connect_stream(stream)
    event_manager.connect_predictor(interface)
    interface.landmark_predicted.connect(show_frame)

    interface.start()
    stream.start()

    # Release the capture and writer objects
    stream.stop()
    interface.stop()
    event_manager.close()
    cv2.destroyAllWindows()
