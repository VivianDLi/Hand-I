from pathlib import Path
from typing import List, override

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult
from psygnal import Signal

from handi.types import Landmark, LandmarkCoords, LandmarkResult, LandmarkPredictorInterface

MODEL_PATH = Path(__file__) / "pretrained" / "hand_landmarker.task"

class MediapipeInterface(LandmarkPredictorInterface):
    def __init__(self, landmark_predicted: Signal):
        self.base_options = BaseOptions(model_asset_path=MODEL_PATH)
        self.running_mode = RunningMode.LIVE_STREAM
        self.options = HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=self.running_mode,
            num_hands=2,
            results_callback=self.process_results
        )
        self.is_running = False

        self.landmark_signal = landmark_predicted

    @override
    def predict_landmarks(self, image: np.ndarray, frame_timestamp_ms: int) -> None:
        """Implementation using Mediapipe for landmark prediction, connect to data received signal."""
        if not self.is_running:
            raise RuntimeError("MediapipeInterface is not running. Call start() before predicting landmarks.")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        self.landmark_predictor.detect_async(mp_image, frame_timestamp_ms)
    
    @override
    def process_results(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> List[LandmarkResult]:
        """Implementation to send results after prediction, connect to data result signal."""
        landmarks = []
        for hand in result.hands:
            hand_landmarks = []
            for landmark in hand.landmarks:
                hand_landmarks.append(Landmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility
                ))
            landmarks.append(LandmarkResult(landmarks=hand_landmarks))
        self.landmark_signal.emit(output_image.numpy_view(), landmarks)
        return landmarks

    @override
    def start(self) -> None:
        self.landmark_predictor = HandLandmarker.create_from_options(self.options)
        self.is_running = True
    
    @override
    def close(self) -> None:
        self.landmark_predictor.close()
        self.is_running = False