import cv2
import numpy as np

from handi.types import EventDataType, StreamInterface


class CameraStream(StreamInterface):
    def __init__(self, camera_id: int = 0):
        self.data_type = EventDataType.IMAGE
        self.camera_id = camera_id

    def _read_frame(self) -> tuple[np.ndarray, int]:
        if not self.is_streaming:
            return np.empty((0, 0, 3), dtype=np.uint8), 0
        ret, frame = self.cap.read()
        if not ret:
            return np.empty((0, 0, 3), dtype=np.uint8), 0
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        return frame, timestamp_ms

    def start(self):
        if self.is_streaming:
            return
        self.cap = cv2.VideoCapture(self.camera_id)
        self.is_streaming = True

        # Stream loop
        while self.is_streaming:
            self.read_frame()
            if cv2.waitKey(1) == ord("q"):
                break

    def stop(self):
        if not self.is_streaming:
            return
        self.cap.release()
        self.is_streaming = False
        self.camera_id = self.camera_id
