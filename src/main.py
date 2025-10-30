from handi.io.camera import CameraStream
from handi.models.gesture_classifier import GestureClassifier
from handi.models.mediapipe_interface import MediapipeInterface
from handi.types import EventManager
from handi.visualization.landmark_visualizer import LandmarkVisualizer

if __name__ == "__main__":
    event_manager = EventManager()
    stream = CameraStream(0)
    interface = MediapipeInterface(rolling_average_window=5)
    classifier = GestureClassifier()
    visualizer = LandmarkVisualizer(draw_angles=True)

    event_manager.connect_stream(stream)
    event_manager.connect_predictor(interface)
    event_manager.connect_classifier(classifier)
    event_manager.connect_post_processor(visualizer)

    try:
        event_manager.open()
    except KeyboardInterrupt:
        print("Connection closed")
    finally:
        event_manager.close()
