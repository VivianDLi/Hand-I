from pathlib import Path
from typing import override

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as RunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)

from handi.types import (
    EventDataType,
    Landmark,
    LandmarkCoords,
    LandmarkPredictorInterface,
    LandmarkResult,
    TrackingResult,
)

MODEL_PATH = Path(__file__).parent / "pretrained" / "hand_landmarker.task"


class MediapipeInterface(LandmarkPredictorInterface):
    def __init__(self):
        self.data_type = EventDataType.IMAGE
        self.base_options = BaseOptions(model_asset_path=str(MODEL_PATH))
        self.running_mode = RunningMode.VIDEO
        self.options = HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=self.running_mode,
            num_hands=2,
        )

    @override
    def predict_landmarks(
        self, image: np.ndarray, frame_timestamp_ms: int
    ) -> None:
        """Implementation using Mediapipe for landmark prediction, connect to data received signal."""
        if not self.is_running:
            raise RuntimeError(
                "MediapipeInterface is not running. Call start() before predicting landmarks."
            )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.landmark_predictor.detect_for_video(
            mp_image, frame_timestamp_ms
        )
        self.process_results(result, mp_image, frame_timestamp_ms)

    @override
    def _process_results(
        self,
        result: HandLandmarkerResult,
        original_image: mp.Image,
        timestamp_ms: int,
    ) -> tuple[np.ndarray, TrackingResult]:
        """Implementation to send results after prediction, connect to data result signal."""
        lh = None
        rh = None
        num_hands = len(result.hand_landmarks)
        for i in range(num_hands):
            hand_landmarks = result.hand_landmarks[i]
            hand_world_landmarks = result.hand_world_landmarks[i]
            handedness = result.handedness[i][0].category_name
            landmarks = {
                Landmark(idx): LandmarkCoords(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                )
                for idx, landmark in enumerate(hand_landmarks)
            }
            world_landmarks = {
                Landmark(idx): LandmarkCoords(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                )
                for idx, landmark in enumerate(hand_world_landmarks)
            }
            if handedness == "Left" and lh is None:
                lh = LandmarkResult(
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    handedness=False,
                )
            elif handedness == "Right" and rh is None:
                rh = LandmarkResult(
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    handedness=True,
                )
            else:
                # If there's already a left/right hand detected, we skip additional hands
                continue
        landmarks = TrackingResult(
            left_hand=lh, right_hand=rh, timestamp=timestamp_ms
        )
        return original_image.numpy_view(), landmarks

    @override
    def start(self) -> None:
        self.landmark_predictor = HandLandmarker.create_from_options(
            self.options
        )
        self.is_running = True

    @override
    def stop(self) -> None:
        self.landmark_predictor.close()
        self.is_running = False
