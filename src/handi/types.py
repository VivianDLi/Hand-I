import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
    THUMB_CMC_MCP = (1, 2)
    THUMB_MCP_IP = (2, 3)
    INDEX_MCP_PIP = (5, 6)
    MIDDLE_MCP_PIP = (9, 10)
    RING_MCP_PIP = (13, 14)
    PINKY_MCP_PIP = (17, 18)
    THUMB_IP_DIP = (3, 4)
    INDEX_PIP_DIP = (6, 7)
    MIDDLE_PIP_DIP = (10, 11)
    RING_PIP_DIP = (14, 15)
    PINKY_PIP_DIP = (18, 19)
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
        theta (float): The theta (xy-plane) angle in radians.
        phi (float): The phi (z-axis) angle in radians.
    """

    @staticmethod
    def from_degrees(theta: float, phi: float) -> "AngleCoords":
        """Create AngleCoords from degrees."""
        return AngleCoords(
            theta=np.deg2rad(theta),
            phi=np.deg2rad(phi),
        )

    @staticmethod
    def to_degrees(theta: float, phi: float) -> dict[str, float]:
        """Convert AngleCoords to degrees."""
        return {
            "theta": np.rad2deg(theta),
            "phi": np.rad2deg(phi),
        }

    theta: float
    phi: float


@dataclass
class LandmarkPrediction:
    """Result of landmark prediction.

    Attributes:
        landmarks (Dict[Landmark, LandmarkCoords]): A dictionary containing the predicted landmarks with their coordinates.
        world_landmarks (Dict[Landmark, LandmarkCoords]): A dictionary containing the predicted landmarks in world coordinates.
        angles (Dict[Angle, AngleCoords]): A dictionary containing the calculated angles based on the landmarks.
        handedness (bool): Indicates if the hand is left (False) or right (True).
    """

    landmarks: dict[Landmark, LandmarkCoords]
    world_landmarks: dict[Landmark, LandmarkCoords]
    handedness: bool

    def __post_init__(self):
        self.angles: dict[Angle, AngleCoords] = self._calculate_hand_angles()

    def _calculate_hand_angles(self) -> dict[Angle, AngleCoords]:
        """Calculates the angles between the landmarks of the hand."""
        angles = {}
        palm_normal = np.cross(
            (
                self.world_landmarks[Landmark.INDEX_FINGER_MCP].coords
                - self.world_landmarks[Landmark.WRIST].coords
            ),
            (
                self.world_landmarks[Landmark.PINKY_FINGER_MCP].coords
                - self.world_landmarks[Landmark.WRIST].coords
            ),
        )  # calculate wrist axis as reference
        palm_normal = palm_normal / np.linalg.norm(palm_normal)  # normalize
        for angle in Angle:
            # Calculate angle between two points
            l1, l2 = Landmark(angle.value[0]), Landmark(angle.value[1])
            assert (
                l1 in self.world_landmarks and l2 in self.world_landmarks
            ), f"Landmarks {l1} and {l2} must be present to calculate angle {angle}"
            vec = self.world_landmarks[l2].coords - self.world_landmarks[l1].coords
            vec = vec / np.linalg.norm(vec)  # normalize
            vec_perp = (
                vec - np.dot(vec, palm_normal) * palm_normal
            )  # perpendicular component
            vec_ref = (
                self.world_landmarks[Landmark.MIDDLE_FINGER_MCP].coords
                - self.world_landmarks[Landmark.WRIST].coords
            )
            vec_ref = vec_ref / np.linalg.norm(
                vec_ref
            )  # normalized vertical vector in palm plane
            theta = np.arccos(np.dot(vec_perp, vec_ref))  # angle in xy-plane
            # check direction
            if np.dot(np.cross(vec_ref, vec_perp), palm_normal) >= 0:
                theta = -theta
            phi = np.arccos(np.dot(vec, palm_normal))  # angle from z-axis
            angles[angle] = AngleCoords(theta=theta, phi=phi)
        return angles


@dataclass
class Gesture:
    """Configuration defining a gesture for classification.

    Attributes:
        name (str): The name of the gesture.
        time_delay (float): The time delay in seconds for the gesture to be held before classification.
        thresholds (Dict[Angle, Tuple[AngleCoords, AngleCoords]]): The minimum and maximum values for each angle of the hand.
    """

    name: str
    time_delay: float
    thresholds: dict[Angle, tuple[AngleCoords, AngleCoords]]

    def check_gesture(self, angles: dict[Angle, AngleCoords]) -> bool:
        """Check if the given angles match the gesture thresholds."""
        matching = [
            angle in angles and min_val <= angles[angle] <= max_val
            for angle, (min_val, max_val) in self.thresholds.items()
        ]
        return all(matching)


@dataclass
class GestureConfig:
    gestures: list[Gesture]
    durations: list[float]

    @staticmethod
    def load_from_file(file_path: Path | str) -> "GestureConfig":
        """Load gesture configurations from a .yaml config file."""
        with open(file_path) as f:
            gesture_dict = json.load(f)
        gestures = []
        for name, data in gesture_dict.items():
            gesture = Gesture(
                name=name,
                time_delay=data["time_delay"],
                thresholds={
                    angle: (
                        AngleCoords.from_degrees(**min_val),
                        AngleCoords.from_degrees(**max_val),
                    )
                    for angle, (min_val, max_val) in data["thresholds"].items()
                },
            )
            gestures.append(gesture)
        return GestureConfig(gestures=gestures, durations=[0.0 for _ in gestures])

    def reset_durations(self) -> None:
        """Reset the durations for all gestures."""
        self.durations = [0.0 for _ in self.gestures]

    def save_to_file(self, file_path: str) -> None:
        """Save gesture configurations to a .yaml config file."""
        gesture_dict = {}
        for gesture in self.gestures:
            gesture_dict[gesture.name] = {
                "time_delay": gesture.time_delay,
                "thresholds": {
                    angle: (
                        AngleCoords.to_degrees(theta=min_val.theta, phi=min_val.phi),
                        AngleCoords.to_degrees(theta=max_val.theta, phi=max_val.phi),
                    )
                    for angle, (min_val, max_val) in gesture.thresholds.items()
                },
            }
        with open(file_path, "w") as f:
            json.dump(gesture_dict, f, indent=4)


#### Interface Dataclasses for representing data to pass between components ####


@dataclass
class StreamResult:
    """Result of a data stream.

    Attributes:
        data (np.ndarray): The data from the stream.
        timestamp (int): The timestamp when the data was received.
    """

    data: np.ndarray
    timestamp: int


@dataclass
class TrackingResult:
    """Result of tracking hands.

    Attributes:
        left_hand (Optional[LandmarkPrediction]): The result of landmark prediction for the left hand.
        right_hand (Optional[LandmarkPrediction]): The result of landmark prediction for the right hand.
        timestamp (float): The timestamp when the tracking was performed.
    """

    original_image: np.ndarray
    left_hand: LandmarkPrediction | None = None
    right_hand: LandmarkPrediction | None = None
    timestamp: float = 0.0


@dataclass
class GestureResult:
    """Result of gesture classification.

    Attributes:
        left_hand (Optional[Gesture]): The result of gesture classification for the left hand.
        right_hand (Optional[Gesture]): The result of gesture classification for the right hand.
    """

    landmark_result: TrackingResult
    left_hand: Gesture | None = None
    right_hand: Gesture | None = None


#### Abstract Base Classes for Interface Components ####
class StreamInterface(ABC):
    """Abstract base class for stream interfaces."""

    data_type: EventDataType
    frame_read = Signal(StreamResult)  # Signal emitting a frame and its timestamp in ms
    is_streaming: bool = False

    @abstractmethod
    def _read_frame(self, *args, **kwargs) -> StreamResult:
        """Read a frame from the data stream."""
        raise NotImplementedError

    def read_frame(self, *args, **kwargs) -> bool:
        """Read a frame and emit the frame read signal."""
        result = self._read_frame(*args, **kwargs)
        if result.timestamp == -1:
            return False
        self.frame_read.emit(result)
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
    landmark_predicted = Signal(TrackingResult)
    is_running: bool = False
    
    def __init__(self, rolling_average_window: int = 0):
        super().__init__()
        self.rolling_average_window = rolling_average_window
        self._landmark_history = {
            "left": {landmark: [] for landmark in Landmark},
            "right": {landmark: [] for landmark in Landmark},
        }

    @abstractmethod
    def predict_landmarks(self, data: StreamResult) -> None:
        """Predict landmarks from the given image. To be connected to frame received signal."""
        raise NotImplementedError

    @abstractmethod
    def _process_results(self, *args, **kwargs) -> TrackingResult:
        """Processes the results into standard form after landmark prediction."""
        raise NotImplementedError

    def process_results(self, *args, **kwargs) -> None:
        """Processes the results into standard form after landmark prediction. To emit the landmark predicted signal."""
        landmarks = self._process_results(*args, **kwargs)
        if self.rolling_average_window > 1:
            # Apply rolling average smoothing to landmarks
            self._landmark_history.append(landmarks)
            if len(self._landmark_history) > self.rolling_average_window:
                self._landmark_history.pop(0)
            avg_screen_coords = {
                landmark: np.mean([hand.landmarks.values()[landmark].coords for hand in [land.left_hand for land in self._landmark_history if land.left_hand is not None]], axis=0)
                for landmark in Landmark
            }
            avg_screen_coords = np.mean([land.left_hand.landmarks for land in self._landmark_history], axis=0)
            avg_world_coords = np.mean([land.left_hand.world_landmarks for land in self._landmark_history], axis=0)
            avg_handedness = np.mean([land.left_hand.handedness for land in self._landmark_history], axis=0)
            avg_left_hand = LandmarkPrediction(
                landmarks=avg_screen_coords,
                world_landmarks=avg_world_coords,
                handedness=False,
            )
            for hand in [landmarks.left_hand, landmarks.right_hand]:
                if hand is not None:
                    for landmark in hand.landmarks.values():
                        if not hasattr(landmark, "history"):
                            landmark.history = []
                        landmark.history.append(
                            np.array([landmark.x, landmark.y, landmark.z])
                        )
                        if len(landmark.history) > self.rolling_average_window:
                            landmark.history.pop(0)
                        avg_coords = np.mean(
                            landmark.history, axis=0
                        )
                        landmark.x, landmark.y, landmark.z = (
                            avg_coords[0],
                            avg_coords[1],
                            avg_coords[2],
                        )
        self.landmark_predicted.emit(landmarks)

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

    config: GestureConfig
    gesture_classified = Signal(GestureResult)
    is_running: bool = False

    @abstractmethod
    def classify_gesture(self, landmarks: TrackingResult) -> None:
        """Classify the gesture based on the given landmarks."""
        raise NotImplementedError

    @abstractmethod
    def _process_results(self, *args, **kwargs) -> GestureResult:
        """Processes the results into standard form after gesture classification."""
        raise NotImplementedError

    def process_results(self, *args, **kwargs) -> None:
        """Processes the results into standard form after gesture classification. To emit the gesture classified signal."""
        gesture = self._process_results(*args, **kwargs)
        self.gesture_classified.emit(gesture)

    @abstractmethod
    def start(self) -> None:
        """Start the gesture classifier."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Close the gesture classifier and release resources."""
        raise NotImplementedError


class PostInterface(ABC):
    """Abstract base class for post-processing interfaces."""

    is_running: bool = False

    @abstractmethod
    def process_results(self, data: GestureResult) -> None:
        """Process the results from landmark prediction or gesture classification."""
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """Start the post-processor."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Close the post-processor and release resources."""
        raise NotImplementedError


#### Event Manager for managing connections between components ####
class EventManager:
    """Class for managing events."""

    def __init__(self):
        self.streams: dict[EventDataType, list[StreamInterface]] = {
            dtype: [] for dtype in EventDataType
        }
        self.predictors: dict[EventDataType, list[LandmarkPredictorInterface]] = {
            dtype: [] for dtype in EventDataType
        }
        self.classifiers: list[GestureClassifierInterface] = []
        self.post_processors: list[PostInterface] = []

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
                predictor.landmark_predicted.connect(classifier.classify_gesture)
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
                    predictor.landmark_predicted.connect(classifier.classify_gesture)

    def disconnect_classifier(self, classifier: GestureClassifierInterface):
        """Disconnect a gesture classifier from the event manager."""
        if classifier in self.classifiers:
            # Disconnect all connected predictors
            for dtype in self.predictors:
                for predictor in self.predictors[dtype]:
                    predictor.landmark_predicted.disconnect(classifier.classify_gesture)
            # Disconnect from all post-processors
            classifier.gesture_classified.disconnect()
        self.classifiers.remove(classifier)

    def connect_post_processor(self, processor: PostInterface):
        """Connect a post-processor to the event manager."""
        if processor not in self.post_processors:
            self.post_processors.append(processor)
            # Connect all valid classifiers
            for classifier in self.classifiers:
                classifier.gesture_classified.connect(processor.process_results)

    def disconnect_post_processor(self, processor: PostInterface):
        """Disconnect a post-processor from the event manager."""
        if processor in self.post_processors:
            # Disconnect all connected classifiers
            for classifier in self.classifiers:
                classifier.gesture_classified.disconnect(processor.process_results)
        self.post_processors.remove(processor)

    def open(self):
        """Open all connected components."""
        for dtype_predictors in self.predictors.values():
            for predictor in dtype_predictors:
                predictor.start()
        for classifier in self.classifiers:
            classifier.start()
        for processor in self.post_processors:
            processor.start()
        for dtype_streams in self.streams.values():
            for stream in dtype_streams:
                stream.start()

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
        for processor in self.post_processors:
            processor.stop()
