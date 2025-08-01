# backend/vision/drawing.py
import cv2
import numpy as np
import random

# 创建一个颜色列表用于不同的实例分割掩码
# 为了可复现性，我们固定随机种子
random.seed(42)
COLOR_PALETTE = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(100)]

def draw_segmentation_masks(image: np.ndarray, detections: list, alpha: float = 0.4):
    """
    在图像上绘制半透明的实例分割掩码。

    Args:
        image (np.ndarray): 原始BGR图像。
        detections (list): Detection对象的列表。
        alpha (float): 叠加掩码的透明度。

    Returns:
        np.ndarray: 叠加了掩码的图像。
    """
    overlay = image.copy()
    for i, detection in enumerate(detections):
        mask = detection.mask
        if mask is None:
            continue

        # 为每个实例选择一个颜色
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        
        # 将掩码应用到图像上
        # mask > 0 的部分会被着色
        overlay[mask > 0] = color

    # 将带有颜色的遮罩与原始图像按权重混合
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def draw_enhanced_annotations(image: np.ndarray, detections_list: list):
    """
    在图像上绘制边界框和标签，现在支持递归绘制子检测。
    """
    annotated_image = image.copy()
    img_height, img_width, _ = annotated_image.shape

    # 使用一个辅助函数来处理绘制，以便递归
    def _draw_single_detection(img, detection, is_sub_detection=False):
        bbox = detection.get("bbox")
        if not bbox: return
        
        x1, y1, x2, y2 = map(int, bbox)
        class_name = detection.get("class", "N/A")
        conf = detection.get("confidence", 0)
        
        # 为主检测和子检测使用不同颜色
        box_color = (255, 165, 0) if is_sub_detection else (0, 255, 0) # 子检测用橙色，主检测用绿色
        label_text_color = (255, 255, 255) if is_sub_detection else (0, 0, 0) # 子检测用白色字

        box_thickness = max(1, int(img_width / 600))
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)
        
        label = f"{class_name} {conf:.2f}"
        font_scale = max(0.4, (x2 - x1) / 300)
        font_thickness = max(1, int(font_scale * 1.5))
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), baseline = cv2.getTextSize(label, font_face, font_scale, font_thickness)
        
        label_y = y1 - 5
        if label_y - text_height < 0:
            label_y = y1 + text_height + 5
            
        cv2.rectangle(img, (x1, label_y - text_height - baseline), (x1 + text_width, label_y + baseline), box_color, -1)
        cv2.putText(img, label, (x1, label_y), font_face, font_scale, label_text_color, font_thickness, cv2.LINE_AA)

    # 遍历主检测列表
    for detection in detections_list:
        _draw_single_detection(annotated_image, detection, is_sub_detection=False)
        
        # 如果有子检测，递归绘制它们
        if 'features' in detection and 'sub_detections' in detection['features']:
            sub_detections_dicts = [vars(d) for d in detection['features']['sub_detections']]
            for sub_detection in sub_detections_dicts:
                 _draw_single_detection(annotated_image, sub_detection, is_sub_detection=True)

    return annotated_image