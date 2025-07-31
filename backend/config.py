# backend/config.py

# --- 路径与URL配置 ---
YOLO_MODEL_PATH = 'backend/resource/models/yolov8n-seg.pt'
# YOLO_MODEL_PATH = 'backend/resource/models/yolo11n.pt's
SECONDARY_MODEL_TYPE = 'mobilenet_v3_small'
OLLAMA_API_URL = 'http://localhost:11434/api/generate'
DEEPSEEK_MODEL_NAME = 'deepseek-r1:8b'
METADATA_PATH = 'backend/data/capture_metadata.json'
# IMAGE_PATH = 'backend/data/captured_image.jpg'
IMAGE_PATH = 'capture1.jpg'
RESPONSE_LOG_PATH = 'backend/resource/response_log.json'

# --- 模型与逻辑配置 ---
# COCO80 类别分组定义
CATEGORY_GROUPS = {
    'person': [0],
    'animal': [14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 41, 44],
    'vehicle': [2, 3, 4, 5, 6, 7, 8, 9],
    'other': [i for i in range(80) if i not in [0, 2, 3, 4, 5, 6, 7, 8, 9, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 41, 44]]
}

# 动物子分类映射 (示例)
ANIMAL_SUBCLASSES = {
    'dog': ['golden retriever', 'labrador', 'husky', 'bulldog'],
    'cat': ['persian', 'siamese', 'maine coon']
}

# 交通工具子分类映射 (示例)
VEHICLE_SUBCLASSES = {
    'car': ['sedan', 'suv', 'pickup'],
    'bus': ['city bus', 'school bus']
}

# 颜色名称映射 (H值范围 0-360)
COLOR_NAMES = {
    (0, 15): 'red', (15, 45): 'orange', (45, 75): 'yellow',
    (75, 165): 'green', (165, 195): 'cyan', (195, 255): 'blue',
    (255, 345): 'purple', (345, 360): 'red'
}