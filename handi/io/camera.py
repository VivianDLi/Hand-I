import cv2
import numpy as np

from handi.types import StreamInterface

class CameraStream(StreamInterface):
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id

    def read_frame(self) -> np.ndarray:
        if not self.is_streaming:
            return np.empty((0, 0, 3), dtype=np.uint8)
        ret, frame = self.cap.read()
        if not ret:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame

    def start_stream(self):
        if self.is_streaming:
            return
        self.cap = cv2.VideoCapture(self.camera_id)
        self.is_streaming = True
        
        # Stream loop
        while self.is_streaming:
            self.read_frame()

    def stop_stream(self):
        if not self.is_streaming:
            return
        self.cap.release()
        self.is_streaming = False
        self.camera_id = self.camera_id