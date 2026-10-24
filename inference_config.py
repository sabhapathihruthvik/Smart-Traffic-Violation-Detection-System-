from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
SAMPLE_DIR = ROOT_DIR / "data" / "sample_data"
ANALYSIS_DIR = ROOT_DIR / "analysis_results"

VEHICLE_MODEL_NAME = "yolo26m.pt"
HELMET_MODEL_NAME = "helmet_detector2.pt"
PLATE_MODEL_NAME = "yolo26m_face_lp.pt"

VEHICLE_CONFIDENCE = 0.15
VEHICLE_IOU = 0.50
VEHICLE_IMAGE_SIZE = 640
VEHICLE_TRACKER = "bytetrack.yaml"

HELMET_CONFIDENCE = 0.25
HELMET_IMAGE_SIZE = 640
HELMET_RIDER_REGION_TOP_SCALE = 1.0
HELMET_RIDER_REGION_SIDE_SCALE = 0.20

PLATE_CONFIDENCE = 0.263
PLATE_IMAGE_SIZE = 640
OCR_MIN_CONFIDENCE = 0.0

OPERATIONS = ("vehicle", "helmet", "plate", "ocr", "wrong_direction", "red_light")
OPERATION_DEPENDENCIES = {
    "vehicle": (),
    "helmet": ("vehicle",),
    "plate": ("vehicle",),
    "ocr": ("vehicle", "plate"),
    "wrong_direction": ("vehicle",),
    "red_light": ("vehicle",),
}
LANE_DIRECTIONS = (
    "TOP_TO_BOTTOM",
    "BOTTOM_TO_TOP",
    "LEFT_TO_RIGHT",
    "RIGHT_TO_LEFT",
)
DEFAULT_LANE_DIRECTION = "BOTTOM_TO_TOP"
DEFAULT_LANE_POLYGON = (
    (0.43, 0.08),
    (0.57, 0.08),
    (0.90, 0.98),
    (0.10, 0.98),
)
DIRECTION_WINDOW_SECONDS = 0.5
DIRECTION_HISTORY_LENGTH = 60
LANE_ENTRY_CONFIRMATION_FRAMES = 2
DIRECTION_CONFIRMATION_FRAMES = 4
MIN_DIRECTION_DISPLACEMENT_RATIO = 0.008
DIRECTION_DOMINANCE_RATIO = 1.25
DIRECTION_MIN_COSINE = 0.65
TRACK_STATE_TIMEOUT_SECONDS = 5
DEBUG_WRONG_DIRECTION = True
DEFAULT_RED_LIGHT_ROI = (0.82, 0.02, 0.98, 0.25)
DEFAULT_RED_LIGHT_STOP_LINE = ((0.10, 0.75), (0.90, 0.75))
DEFAULT_RED_LIGHT_CROSSING_DIRECTION = "NEGATIVE_TO_POSITIVE"
RED_LIGHT_STATE_HISTORY = 5
RED_LIGHT_STATE_CONFIRMATION_FRAMES = 3
RED_LIGHT_MIN_ACTIVE_PIXELS = 4
RED_LIGHT_MIN_ACTIVE_ROI_RATIO = 0.001
RED_LIGHT_COLOR_DOMINANCE_RATIO = 1.35
RED_LIGHT_CROSSING_DEADBAND_RATIO = 0.012
RED_LIGHT_CROSSING_CONFIRMATION_FRAMES = 2
RED_LIGHT_MAX_CROSSING_GAP_SECONDS = 3.0
RED_LIGHT_COLORS = {
    "RED": (160, 160, 160),
    "YELLOW": (0, 220, 255),
    "GREEN": (0, 210, 0),
    "UNKNOWN": (160, 160, 160),
}
RED_LIGHT_VIOLATION_COLOR = (0, 0, 255)
INFERENCE_BATCH_SIZE = 8

# These model labels are mapped to the vehicle categories used by the UI.
VEHICLE_CLASS_ALIASES = {
    "bike": "bike",
    "bicycle": "bike",
    "motorcycle": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}
TRACKED_VEHICLES = ("bike", "motorcycle", "car", "bus", "truck")
HELMET_VEHICLES = frozenset({"bike", "motorcycle"})

VEHICLE_COLORS = {
    "bike": (255, 0, 0),
    "motorcycle": (255, 0, 0),
    "car": (0, 165, 255),
    "bus": (255, 0, 255),
    "truck": (255, 255, 0),
}
HELMET_COLOR = (0, 220, 130)
PLATE_COLOR = (0, 220, 255)
WRONG_DIRECTION_COLORS = {
    "UNKNOWN": (160, 160, 160),
    "NOT_RELEVANT": (140, 140, 140),
    "IN_LANE": (0, 200, 255),
    "CORRECT_DIRECTION": (0, 210, 0),
    "WRONG_DIRECTION": (0, 0, 255),
}

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
