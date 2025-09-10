from pathlib import Path

PRETRAINED_MODEL_PATH = Path(__file__).parent / "models" / "pretrained"
HAND_MODEL_PATH = PRETRAINED_MODEL_PATH / "hand_landmarker.task"
GESTURE_CONFIG_PATH = PRETRAINED_MODEL_PATH / "gestures.json"
