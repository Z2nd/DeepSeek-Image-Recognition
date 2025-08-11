# backend/config.py

# --- 路径与URL配置 ---
YOLO_COCO_SEGMENTATION_MODEL_PATH = 'backend/resource/models/yolov8n-seg.pt' # 主模型，用于通用检测
YOLO_ELECTRONICS_MODEL_PATH = 'backend/resource/models/electro.pt' # 电子元器件专用模型
YOLO_FACE_MODEL_PATH = 'backend/resource/models/face.pt' # 人脸检测模型
YOLO_EMOTION_MODEL_PATH = 'backend/resource/models/emotion.pt' # 情感分析模型
# YOLO_PCB_MODEL_PATH = 'backend/resource/models/circuit-board.pt'
YOLO_PCB_MODEL_PATH = 'backend/resource/models/pcb.pt'
YOLO_STUDYDESKITEM_MODEL_PATH = 'backend/resource/models/study-desk-item.pt' # 学习桌物品检测模型
YOLO_WINDOWSELEMENT_MODEL_PATH = 'backend/resource/models/windows-element.pt' # 窗口元素检测模型

OLLAMA_API_URL = 'http://localhost:11434/api/generate'
DEEPSEEK_MODEL_NAME = 'deepseek-r1:8b'
METADATA_PATH = 'backend/data/capture_metadata.json'
IMAGE_PATH = 'backend/data/study-desk.jpg'
RESPONSE_LOG_PATH = 'backend/data/response_log.json'

# 新增：多模型融合检测配置
MULTI_MODEL_DETECTION_CONFIG = [
    {
        # 第一个模型：YOLOv11官方分割模型
        'model_path': YOLO_COCO_SEGMENTATION_MODEL_PATH, # <-- 替换成您的官方模型路径
        # 从这个模型中，我们只保留这些通用类别
        'classes_to_keep': ['person', 'bus', 'stop sign', 'handbag', 'cup', 'chair', 'couch', 'dinning table', 'laptop', 'mouse', 'keyboard', 'cell phone'] # <-- 根据您的需求修改
    },
    {
        # 第二个模型：书桌物品检测模型
        'model_path': YOLO_STUDYDESKITEM_MODEL_PATH, # <-- 替换成您的书桌模型路径
        # 从这个模型中，我们只保留这些它更擅长的特定类别
        'classes_to_keep': ['Apple-Pencil', 'Gag', 'Calculator', 'Charging-cable', 'Earphones', 'Keyboard', 'Keys', 'Laptoop', 'Markers', 'Mobile phone', 'Mouse', 'PC', 'Screen', 'Pen', 'StudentID_card', 'Wallet', 'Watch', 'Water bottle', 'iPad-Air', 'iPad-Pro'] # <-- 根据您的需求修改
    }
]

# --- 分层检测配置 ---
# 定义哪个主类别应该触发哪个YOLO模型进行二次检测
RECURSIVE_DETECTION_CONFIG = {
    # 当主模型检测到 'person' 时，使用 'electro.pt' 在其区域内进行二次检测
    # 注意：这里的 'person' 是一个示例，您应该换成主模型能识别的容器类物体，
    # 比如 'circuit board' (如果您的主模型能识别它的话)。
    # 假设您的best.pt能识别'pcb'类别，可以写成 'pcb': YOLO_ELECTRONICS_MODEL_PATH
    "person": {
        'model_path': YOLO_FACE_MODEL_PATH,
        'sub_config': {
            "face": {
                'model_path': YOLO_EMOTION_MODEL_PATH,
                'sub_config': {}
                }
            }
    },
    "laptop": {
        'model_path': YOLO_WINDOWSELEMENT_MODEL_PATH,
        'sub_config': {}
    },"Laptop": {
        'model_path': YOLO_WINDOWSELEMENT_MODEL_PATH,
        'sub_config': {}
    },"Screen": {
        'model_path': YOLO_WINDOWSELEMENT_MODEL_PATH,
        'sub_config': {}
    }
}

# --- 模型与逻辑配置 ---

# 颜色名称映射 (H值范围 0-360)
COLOR_NAMES = {
    (0, 15): 'red', (15, 45): 'orange', (45, 75): 'yellow',
    (75, 165): 'green', (165, 195): 'cyan', (195, 255): 'blue',
    (255, 345): 'purple', (345, 360): 'red'
}

# --- OCR 配置 ---
# 在这个列表中的类别，将会触发OCR文字识别
OCR_ENABLED_CLASSES = [
    'laptop','mouse','keyboard','Screen'
    # 您未来可以添加其他类别，例如 'book', 'sign', 'license_plate' 等
]