# backend/config.py

# --- Path and URL Configuration ---
YOLO_COCO_SEGMENTATION_MODEL_PATH = 'backend/resource/models/yolo11n-seg.pt' # Main model for general detection
YOLO_ELECTRONICS_MODEL_PATH = 'backend/resource/models/electro.pt' # Electronics-specific model
YOLO_FACE_MODEL_PATH = 'backend/resource/models/face.pt' # Face detection model
YOLO_EMOTION_MODEL_PATH = 'backend/resource/models/emotion.pt' # Emotion analysis model
# YOLO_PCB_MODEL_PATH = 'backend/resource/models/circuit-board.pt'
YOLO_PCB_MODEL_PATH = 'backend/resource/models/pcb.pt'
YOLO_STUDYDESKITEM_MODEL_PATH = 'backend/resource/models/study-desk-item.pt' # Study desk item detection model
YOLO_WINDOWSELEMENT_MODEL_PATH = 'backend/resource/models/windows-element.pt' # Windows element detection model

OLLAMA_API_URL = 'http://localhost:11434/api/generate'
DEEPSEEK_MODEL_NAME = 'deepseek-r1:8b'
METADATA_PATH = 'backend/data/capture_metadata.json'
IMAGE_PATH = 'backend/data/study-desk.jpg'
RESPONSE_LOG_PATH = 'backend/data/response_log.json'

# Multi-model fusion detection configuration
MULTI_MODEL_DETECTION_CONFIG = [
    {
        # First model: YOLOv8 official segmentation model
        'model_path': YOLO_COCO_SEGMENTATION_MODEL_PATH, 
        'classes_to_keep': ['person', 'bus', 'stop sign', 'handbag', 'cup', 'chair', 'couch', 'dinning table', 'laptop', 'mouse', 'keyboard', 'cell phone']
    },
    {
        # Second model: Study desk item detection model
        'model_path': YOLO_STUDYDESKITEM_MODEL_PATH,
        'classes_to_keep': ['Gag', 'Charging-cable', 'Earphones', 'Keys', 'Markers', 'Mobile phone', 'Mouse', 'Screen', 'Pen', 'StudentID_card', 'Wallet', 'Watch', 'Water bottle', 'iPad-Air', 'iPad-Pro'] # <-- 根据您的需求修改
    }
]

# --- Hierarchical Detection Configuration ---
# Define which primary categories should trigger which YOLO model for secondary detection
RECURSIVE_DETECTION_CONFIG = {
    "person": {
        'model_path': YOLO_FACE_MODEL_PATH,
        'post_rules': [
            {
                'class': 'face',          # Rule applies to the 'face' subcategory
                'max_detections': 1,      # Keep at most 1 detection
                'strategy': 'highest_confidence' # Keep the one with highest confidence
            }
        ],
        'sub_config': {
            "face": {
                'model_path': YOLO_EMOTION_MODEL_PATH,
                'post_rules': [
                    {
                        'class': '*',
                        'max_detections': 1,
                        'strategy': 'highest_confidence'
                    }
                ],
                'sub_config': {}
                }
            }
    },
    "laptop": {
        'model_path': YOLO_WINDOWSELEMENT_MODEL_PATH,
        'post_rules': [
            {
                'class': 'activewindow',         
                'max_detections': 1,         
                'strategy': 'highest_confidence' 
            },
        ],
        'sub_config': {}
    },
    "Screen": {
        'model_path': YOLO_WINDOWSELEMENT_MODEL_PATH,
        'post_rules': [
            {
                'class': 'activewindow',         
                'max_detections': 1,         
                'strategy': 'highest_confidence' 
            },
        ],
        'sub_config': {}
    }
}

# --- Model and Logic Configuration ---

# Color name mapping (H value range 0-360)
COLOR_NAMES = {
    (0, 15): 'red', (15, 45): 'orange', (45, 75): 'yellow',
    (75, 165): 'green', (165, 195): 'cyan', (195, 255): 'blue',
    (255, 345): 'purple', (345, 360): 'red'
}

# --- OCR Configuration ---
# Classes in this list will trigger OCR text recognition
OCR_ENABLED_CLASSES = [
    'laptop','keyboard','Screen'
]