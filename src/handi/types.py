from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from psygnal import Signal

#### General Dataclasses for representing data structures ####


class EventType(Enum):
    STREAM_RECEIVED = "stream_received"
    LANDMARK_PREDICTED = "landmark_predicted"
    GESTURE_CLASSIFIED = "gesture_classified"


class EventDataType(Enum):
    IMAGE = "image"
    DVS = "dvs"
    EMG = "emg"


class Landmark(Enum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_FINGER_MCP = 17
    PINKY_FINGER_PIP = 18
    PINKY_FINGER_DIP = 19
    PINKY_FINGER_TIP = 20


class Angle(Enum):
    WRIST_THUMB = (0, 1)
    THUMB_CMC_PIP = (1, 2)
    INDEX_CMC_PIP = (5, 6)
    MIDDLE_CMC_PIP = (9, 10)
    RING_CMC_PIP = (13, 14)
    PINKY_CMC_PIP = (17, 18)
    THUMB_PIP_DIP = (2, 3)
    INDEX_PIP_DIP = (6, 7)
    MIDDLE_PIP_DIP = (10, 11)
    RING_PIP_DIP = (14, 15)
    PINKY_PIP_DIP = (18, 19)
    THUMB_DIP_TIP = (3, 4)
    INDEX_DIP_TIP = (7, 8)
    MIDDLE_DIP_TIP = (11, 12)
    RING_DIP_TIP = (15, 16)
    PINKY_DIP_TIP = (19, 20)


@dataclass
class LandmarkCoords:
    """Represents a single hand landmark coordinate.

    Attributes:
        x (float): The x-coordinate of the landmark.
        y (float): The y-coordinate of the landmark.
        z (float): The z-coordinate of the landmark.
    """

    x: float
    y: float
    z: float

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass(order=True)
class AngleCoords:
    """Represents a single angle coordinate.

    Attributes:
        theta (float): The theta (xy-plane) angle in degrees.
        phi (float): The phi (z-axis) angle in degrees.
    """

    theta: float
    phi: float


@dataclass
class Gesture:
    """Configuration defining a gesture for classification.

    Attributes:
        name (str): The name of the gesture.
        min_value (Dict[Landmark, LandmarkCoords]): The minimum value in world coordinates for each landmark of the hand. Assumes a right hand.
        max_value (Dict[Landmark, LandmarkCoords]): The maximum value in world coordinates for each landmark of the hand. Assumes a right hand.
        time_delay (float): The time delay in seconds for the gesture to be held before classification.
    """

    name: str
    time_delay: float
    thresholds: dict[Angle, tuple[AngleCoords, AngleCoords]]

    def check_gesture(
        self, angles: dict[Angle, AngleCoords], duration: float
    ) -> bool:
        """Check if the given angles match the gesture thresholds."""
        matching = [
            angle in angles and min_val <= angles[angle] <= max_val
            for angle, (min_val, max_val) in self.thresholds.items()
        ]
        return all(matching) and duration >= self.time_delay


@dataclass
class GestureConfig:
    gestures: list[Gesture]

    def load_from_file(self, file_path: str) -> None:
        """Load gesture configurations from a .yaml config file."""
        raise NotImplementedError

    def save_to_file(self, file_path: str) -> None:
        """Save gesture configurations to a .yaml config file."""
        raise NotImplementedError


#### Interface Dataclasses for representing data to pass between components ####


@dataclass
class LandmarkResult:
    """Result of landmark prediction.

    Attributes:
        landmarks (Dict[Landmark, LandmarkCoords]): A dictionary containing the predicted landmarks with their coordinates.
        world_landmarks (Dict[Landmark, LandmarkCoords]): A dictionary containing the predicted landmarks in world coordinates.
        angles (Dict[str, float]): A dictionary containing the calculated angles based on the landmarks.
        handedness (bool): Indicates if the hand is left (False) or right (True).
    """

    landmarks: dict[Landmark, LandmarkCoords]
    world_landmarks: dict[Landmark, LandmarkCoords]
    handedness: bool

    def __post_init__(self):
        self.angles: dict[Angle, float] = self._calculate_hand_angles()

    def _calculate_hand_angles(self) -> dict[Angle, float]:
        """Calculates the angles between the landmarks of the hand."""
        angles = {}
        palm_normal = np.cross(
            (
                self.world_landmarks[Landmark.WRIST].coords
                - self.world_landmarks[Landmark.INDEX_FINGER_MCP].coords
            ),
            (
                self.world_landmarks[Landmark.WRIST].coords
                - self.world_landmarks[Landmark.PINKY_FINGER_MCP].coords
            ),
        )  # calculate wrist axis as reference
        palm_normal = palm_normal / np.linalg.norm(palm_normal)  # normalize
        for angle in Angle:
            # Calculate angle between two points
            l1, l2 = Landmark(angle.value[0]), Landmark(angle.value[1])
            assert (
                l1 in self.landmarks and l2 in self.landmarks
            ), f"Landmarks {l1} and {l2} must be present to calculate angle {angle}"
            vec = (
                self.world_landmarks[l2].coords
                - self.world_landmarks[l1].coords
            )
            vec = vec / np.linalg.norm(vec)  # normalize
            z_proj = (
                np.array([0, 0, 1])
                - np.dot(np.array([0, 0, 1]), palm_normal) * palm_normal
            )
            vec_proj = vec - np.dot(vec, palm_normal) * palm_normal
            theta = (
                np.arccos(np.dot(vec_proj, z_proj)) * 180 / np.pi
            )  # angle in xy-plane
            phi = (
                np.arccos(np.dot(vec, palm_normal)) * 180 / np.pi
            )  # angle from z-axis
            angles[angle] = AngleCoords(theta=theta, phi=phi)
        return angles


@dataclass
class TrackingResult:
    """Result of tracking hands.

    Attributes:
        left_hand (Optional[LandmarkResult]): The result of landmark prediction for the left hand.
        right_hand (Optional[LandmarkResult]): The result of landmark prediction for the right hand.
        timestamp (float): The timestamp when the tracking was performed.
    """

    left_hand: LandmarkResult | None = None
    right_hand: LandmarkResult | None = None
    timestamp: float = 0.0


@dataclass
class GestureResult:
    """Result of gesture classification.

    Attributes:
        gesture (str): The name of the classified gesture.
        duration (float): The duration in seconds for which the gesture was held.
        handedness (bool): Indicates if the hand is left (False) or right (True).
    """

    gesture: str
    duration: float
    handedness: bool


#### Abstract Base Classes for Interface Components ####
class StreamInterface(ABC):
    """Abstract base class for stream interfaces."""

    data_type: EventDataType
    frame_read = Signal(
        np.ndarray, int
    )  # Signal emitting a frame and its timestamp in ms
    is_streaming: bool = False

    @abstractmethod
    def _read_frame(self) -> tuple[np.ndarray, int]:
        """Read a frame from the data stream."""
        raise NotImplementedError

    def read_frame(self, *args) -> bool:
        """Read a frame and emit the frame read signal."""
        frame, timestamp_ms = self._read_frame()
        if timestamp_ms == -1:
            return False
        self.frame_read.emit(frame, timestamp_ms)
        return True

    @abstractmethod
    def start(self):
        """Start the data stream."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """Stop the data stream."""
        raise NotImplementedError


class LandmarkPredictorInterface(ABC):
    """Abstract base class for landmark predictor interfaces."""

    data_type: EventDataType
    landmark_predicted = Signal(np.ndarray, TrackingResult)
    is_running: bool = False

    @abstractmethod
    def predict_landmarks(
        self, image: np.ndarray, frame_timestamp_ms: int
    ) -> None:
        """Predict landmarks from the given image. To be connected to frame received signal."""
        raise NotImplementedError

    @abstractmethod
    def _process_results(
        self, result, original_image, timestamp_ms: int
    ) -> tuple[np.ndarray, TrackingResult]:
        """Processes the results into standard form after landmark prediction."""
        raise NotImplementedError

    def process_results(
        self, result, original_image, timestamp_ms: int
    ) -> None:
        """Processes the results into standard form after landmark prediction. To emit the landmark predicted signal."""
        data, landmarks = self._process_results(
            result, original_image, timestamp_ms
        )
        self.landmark_predicted.emit(data, landmarks)

    @abstractmethod
    def start(self) -> None:
        """Start the landmark predictor."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Close the landmark predictor and release resources."""
        raise NotImplementedError


class GestureClassifierInterface(ABC):
    """Abstract base class for gesture classifier interfaces."""

    gesture_classified = Signal(GestureResult)
    is_running: bool = False

    @abstractmethod
    def classify_gesture(self, landmarks: TrackingResult) -> None:
        """Classify the gesture based on the given landmarks."""
        raise NotImplementedError

    @abstractmethod
    def _process_results(self, result) -> GestureResult:
        """Processes the results into standard form after gesture classification."""
        raise NotImplementedError

    def process_results(self, result) -> None:
        """Processes the results into standard form after gesture classification. To emit the gesture classified signal."""
        gesture = self._process_results(result)
        self.gesture_classified.emit(gesture)

    @abstractmethod
    def start(self) -> None:
        """Start the gesture classifier."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Close the gesture classifier and release resources."""
        raise NotImplementedError


#### Event Manager for managing connections between components ####
class EventManager:
    """Class for managing events."""

    def __init__(self):
        self.streams: dict[EventDataType, list[StreamInterface]] = {
            dtype: [] for dtype in EventDataType
        }
        self.predictors: dict[
            EventDataType, list[LandmarkPredictorInterface]
        ] = {dtype: [] for dtype in EventDataType}
        self.classifiers: list[GestureClassifierInterface] = []

    def connect_stream(self, stream: StreamInterface):
        """Connect a stream to the event manager."""
        if stream not in self.streams[stream.data_type]:
            self.streams[stream.data_type].append(stream)
            # Connect to all valid predictors
            for predictor in self.predictors[stream.data_type]:
                stream.frame_read.connect(predictor.predict_landmarks)

    def disconnect_stream(self, stream: StreamInterface):
        """Disconnect a stream from the event manager."""
        if stream in self.streams[stream.data_type]:
            # Disconnect from all connected predictors
            stream.frame_read.disconnect()
        self.streams[stream.data_type].remove(stream)

    def connect_predictor(self, predictor: LandmarkPredictorInterface):
        """Connect a landmark predictor to the event manager."""
        if predictor not in self.predictors[predictor.data_type]:
            self.predictors[predictor.data_type].append(predictor)
            # Connect all valid streams
            for stream in self.streams[predictor.data_type]:
                stream.frame_read.connect(predictor.predict_landmarks)
            # Connect to all classifiers
            for classifier in self.classifiers:
                predictor.landmark_predicted.connect(
                    classifier.classify_gesture
                )
        self.predictors[predictor.data_type].append(predictor)

    def disconnect_predictor(self, predictor: LandmarkPredictorInterface):
        """Disconnect a landmark predictor from the event manager."""
        if predictor in self.predictors[predictor.data_type]:
            # Disconnect from all connected streams
            for stream in self.streams[predictor.data_type]:
                stream.frame_read.disconnect(predictor.predict_landmarks)
            # Disconnect from all classifiers
            predictor.landmark_predicted.disconnect()
        self.predictors[predictor.data_type].remove(predictor)

    def connect_classifier(self, classifier: GestureClassifierInterface):
        """Connect a gesture classifier to the event manager."""
        if classifier not in self.classifiers:
            self.classifiers.append(classifier)
            # Connect all valid predictors
            for dtype in self.predictors:
                for predictor in self.predictors[dtype]:
                    predictor.landmark_predicted.connect(
                        classifier.classify_gesture
                    )

    def disconnect_classifier(self, classifier: GestureClassifierInterface):
        """Disconnect a gesture classifier from the event manager."""
        if classifier in self.classifiers:
            # Disconnect all connected predictors
            for dtype in self.predictors:
                for predictor in self.predictors[dtype]:
                    predictor.landmark_predicted.disconnect(
                        classifier.classify_gesture
                    )
        self.classifiers.remove(classifier)

    def close(self):
        """Close all connected components."""
        for dtype_streams in self.streams.values():
            for stream in dtype_streams:
                stream.stop()
        for dtype_predictors in self.predictors.values():
            for predictor in dtype_predictors:
                predictor.stop()
        for classifier in self.classifiers:
            classifier.stop()
