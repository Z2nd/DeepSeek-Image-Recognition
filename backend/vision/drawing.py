# backend/vision/drawing.py
import cv2
import numpy as np

def draw_enhanced_annotations(image, detections_list):
    """
    在图像上绘制更清晰、自适应的标注（边界框、掩码和标签）。

    Args:
        image (np.array): 需要被标注的原始图像 (BGR格式)。
        detections_list (list): 包含检测结果字典的列表。

    Returns:
        np.array: 已经添加了标注的图像。
    """
    annotated_image = image.copy()
    img_height, img_width, _ = annotated_image.shape

    for detection in detections_list:
        # --- 1. 解包检测信息 ---
        bbox = detection.get("bbox")
        if not bbox:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        class_name = detection.get("class", "N/A")
        conf = detection.get("confidence", 0)
        color_name = detection.get("color_name", "")
        sub_class = detection.get("sub_class", "unknown")
        
        # --- 2. 绘制边界框 ---
        box_color = (0, 255, 0) # 绿色边界框
        # 根据图像大小自适应调整线条粗细
        box_thickness = max(1, int(img_width / 500))
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # --- 3. 准备和绘制标签 ---
        # 构造标签文本
        if sub_class != 'unknown':
            label = f"{class_name} ({sub_class}, {color_name}) {conf:.2f}"
        else:
            label = f"{class_name} ({color_name}) {conf:.2f}"

        # 根据边界框大小自适应调整字体大小
        font_scale = max(0.5, (x2 - x1) / 250)
        font_thickness = max(1, int(font_scale * 2))
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        # 计算文本框大小
        (text_width, text_height), baseline = cv2.getTextSize(label, font_face, font_scale, font_thickness)
        
        # 优化标签位置：如果上方空间不足，则将标签放在框内
        label_y = y1 - 10
        if label_y - text_height < 0:
            label_y = y1 + text_height + 10
            
        # 绘制文本背景以增加对比度
        cv2.rectangle(annotated_image, (x1, label_y - text_height - baseline), (x1 + text_width, label_y + baseline), box_color, -1)
        
        # 绘制文本
        cv2.putText(annotated_image, label, (x1, label_y), font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return annotated_image