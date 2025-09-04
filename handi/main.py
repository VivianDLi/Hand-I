from typing import List
import cv2
import numpy as np
from psygnal import Signal

from handi.models.mediapipe_interface import MediapipeInterface
from handi.types import LandmarkResult

def show_frame(frame: np.ndarray, landmarks: List[LandmarkResult]) -> None:
    """Draw landmarks on the frame."""
    for hand in landmarks:
        for landmark in hand.landmarks.values():
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow('Camera', frame)

if __name__ == "__main__":
    # Signals
    frame_received = Signal(np.ndarray, int)
    landmark_predicted = Signal(np.ndarray, List[LandmarkResult])
    
    # Open the default camera
    cam = cv2.VideoCapture(0)
    interface = MediapipeInterface(landmark_predicted)
    
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Connect signals
    frame_received.connect(interface.predict_landmarks)
    landmark_predicted.connect(show_frame)

    while True:
        ret, frame = cam.read()
        
        if ret:
            frame_received.emit(frame, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()