#### What is needed?
# Main Interface class
    # Created by other applications to communicate between Hand-I and application controls
    # Exposes signals for each possible gesture and landmark prediction
    # Exposes slots to send data to Hand-I
    # Exposes events for data received, data sent, after landmark prediction, and after gesture classification
    # Exposes methods to start and stop gesture recognition
    # Exposes methods to change gesture classification settings
    # Exposes methods to change landmark prediction settings
# Model class
    # Receives data from Hand-I and processes it into landmarks
    # Additionally calculates interim angles
# Gesture classifier
    # Based on a config file, classifies gestures based on landmark positions
    # Simple tree classifier
# Training module
    # Allows training of models from training data in /data
# Visualization module
    # Enables testing by visualizing landmarks (optional) and classified gestures (optional) with data in real-time
    
from handi.types import ManagerInterface

class HandIInterface(ManagerInterface):
    def __init__(self):
        """Initialize the Hand-I interface."""
        super().__init__()
        
    