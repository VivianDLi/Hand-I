from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

import numpy as np
from psygnal import Signal

#### General Dataclasses for representing data structures ####
class Landmark(Enum):
    0 = "WRIST"
    1 = "THUMB_CMC"
    2 = "THUMB_MCP"
    3 = "THUMB_IP"
    4 = "THUMB_TIP"
    5 = "INDEX_FINGER_MCP"
    6 = "INDEX_FINGER_PIP"
    7 = "INDEX_FINGER_DIP"
    8 = "INDEX_FINGER_TIP"
    9 = "MIDDLE_FINGER_MCP"
    10 = "MIDDLE_FINGER_PIP"
    11 = "MIDDLE_FINGER_DIP"
    12 = "MIDDLE_FINGER_TIP"
    13 = "RING_FINGER_MCP"
    14 = "RING_FINGER_PIP"
    15 = "RING_FINGER_DIP"
    16 = "RING_FINGER_TIP"
    17 = "PINKY_FINGER_MCP"
    18 = "PINKY_FINGER_PIP"
    19 = "PINKY_FINGER_DIP"
    20 = "PINKY_FINGER_TIP"

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

#### Interface Dataclasses for representing data to pass between components ####

@dataclass
class GestureConfig:
    """Configuration defining a gesture for classification.
    
    Attributes:
        name (str): The name of the gesture.
        min_value (Dict[Landmark, LandmarkCoords]): The minimum value in world coordinates for each landmark of the hand. Assumes a right hand.
        max_value (Dict[Landmark, LandmarkCoords]): The maximum value in world coordinates for each landmark of the hand. Assumes a right hand.
        time_delay (float): The time delay in seconds for the gesture to be held before classification.    
    """
    name: str
    min_value: Dict[Landmark, LandmarkCoords]
    max_value: Dict[Landmark, LandmarkCoords]
    time_delay: float
  
@dataclass
class LandmarkResult:
    """Result of landmark prediction.
    
    Attributes:
        landmarks (Dict[Landmark, LandmarkCoords]): A dictionary containing the predicted landmarks with their coordinates.
        world_landmarks (Dict[Landmark, LandmarkCoords]): A dictionary containing the predicted landmarks in world coordinates.
        angles (Dict[str, float]): A dictionary containing the calculated angles based on the landmarks.
        handedness (bool): Indicates if the hand is left (False) or right (True).
    """
    landmarks: Dict[Landmark, LandmarkCoords]
    world_landmarks: Dict[Landmark, LandmarkCoords]
    angles: Dict[str, float]
    handedness: bool

@dataclass
class TrackingResult:
    """Result of tracking hands.
    
    Attributes:
        left_hand (Optional[LandmarkResult]): The result of landmark prediction for the left hand.
        right_hand (Optional[LandmarkResult]): The result of landmark prediction for the right hand.
        timestamp (float): The timestamp when the tracking was performed.
    """
    left_hand: Optional[LandmarkResult] = None
    right_hand: Optional[LandmarkResult] = None
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
class EventManager(ABC):
    """Abstract base class for event management."""
    def emit_event(self, event_name: str, data: Optional[Dict] = None):
        """Emit an event with the given name and optional data."""
        raise NotImplementedError

    def connect_event(self, event_name: str, callback):
        """Connect a callback function to an event."""
        raise NotImplementedError

    def disconnect_event(self, event_name: str, callback):
        """Disconnect a callback function from an event."""
        raise NotImplementedError

class StreamInterface(ABC):
    """Abstract base class for stream interfaces."""
    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Read a frame from the data stream."""
        raise NotImplementedError

    @abstractmethod
    def start_stream(self):
        """Start the data stream."""
        raise NotImplementedError

    @abstractmethod
    def stop_stream(self):
        """Stop the data stream."""
        raise NotImplementedError
    
class LandmarkPredictorInterface(ABC):
    """Abstract base class for landmark predictor interfaces."""
    @abstractmethod
    def predict_landmarks(self, image: np.ndarray, frame_timestamp_ms: int) -> None:
        """Predict landmarks from the given image."""
        raise NotImplementedError
    
    @abstractmethod
    def process_results(self, result, original_image, timestamp_ms) -> List[LandmarkResult]:
        """Processes the results into standard form after landmark prediction."""
        raise NotImplementedError
    
    @abstractmethod
    def start(self) -> None:
        """Start the landmark predictor."""
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> None:
        """Close the landmark predictor and release resources."""
        raise NotImplementedError
    
class GestureClassifierInterface(ABC):
    """Abstract base class for gesture classifier interfaces."""
    @abstractmethod
    def classify_gesture(self, landmarks: Dict[Landmark, LandmarkCoords]) -> str:
        """Classify the gesture based on the given landmarks."""
        raise NotImplementedError