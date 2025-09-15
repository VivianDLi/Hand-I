import logging

import numpy as np

from handi.types import EventDataType, StreamInterface, StreamResult


class CameraStream(StreamInterface):
    def __init__(self, camera_id: int = 0):
        self.data_type = EventDataType.IMAGE
        self.camera_id = camera_id

    def _read_frame(self) -> StreamResult:
        import cv2

        if not self.is_streaming:
            return StreamResult(np.empty((0, 0, 3), dtype=np.uint8), -1)
        ret, frame = self.cap.read()
        if not ret:
            return StreamResult(np.empty((0, 0, 3), dtype=np.uint8), -1)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return StreamResult(rgb_frame, timestamp_ms)

    def start(self):
        import cv2

        if self.is_streaming:
            return
        self.cap = cv2.VideoCapture(self.camera_id)
        # Check if the webcam is opened correctly
        if self.cap.isOpened() and not self.cap.read()[0]:
            self.cap.release()
            logging.warning(
                "Unable to access the camera. Trying with CAP_DSHOW backend."
            )
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_streaming = True

        while self.cap.isOpened() and self.is_streaming:
            ret = self.read_frame()
            if not ret:
                logging.error("Failed to read frame from camera.")
                break

    def stop(self):
        if not self.is_streaming:
            return
        self.cap.release()
        self.is_streaming = False
