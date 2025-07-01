import cv2
import numpy as np
import json
import requests
from ultralytics import YOLO
from sklearn.cluster import KMeans

def detect_objects_yolo(image_bgr, yolo_model):
    """
    使用YOLO分割模型进行实例分割，提取类别、置信度、边界框、掩码面积、颜色直方图和K-Means主导颜色RGB。
    Args:
        image_bgr: BGR格式的NumPy图像数组
        yolo_model: 加载的YOLO分割模型
    Returns:
        detections_list: 检测结果列表，包含类别、置信度、边界框、掩码面积、直方图和主导颜色RGB
        annotated_image: 标注后的图像（BGR格式，含边界框、掩码轮廓和主导颜色）
    """
    try:
        if yolo_model is None:
            print("Error: YOLO model is not loaded.")
            return [], image_bgr

        # 进行目标检测和分割
        results = yolo_model(image_bgr)
        detections_list = []
        annotated_image = image_bgr.copy()

        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

            for idx, box in enumerate(boxes):
                # 提取边界框坐标、类别和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = yolo_model.names[cls]

                # 获取掩码（如果存在）
                mask_area = 0.0
                mask = None
                if masks is not None and idx < len(masks):
                    mask = masks.data[idx].cpu().numpy()  # 掩码为布尔数组
                    mask_area = np.sum(mask) / (image_bgr.shape[0] * image_bgr.shape[1])  # 面积占比
                    # 绘制掩码轮廓
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_image, contours, -1, (255, 0, 0), 2)

                # 提取掩码或边界框区域
                roi = image_bgr * mask[..., None] if mask is not None else image_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # 计算颜色直方图（RGB，16 bins/通道）
                hist_r = cv2.calcHist([roi], [2], mask.astype(np.uint8) if mask is not None else None, [8], [0, 256])
                hist_g = cv2.calcHist([roi], [1], mask.astype(np.uint8) if mask is not None else None, [8], [0, 256])
                hist_b = cv2.calcHist([roi], [0], mask.astype(np.uint8) if mask is not None else None, [8], [0, 256])
                # 归一化直方图
                hist_r = hist_r.flatten().tolist()
                hist_g = hist_g.flatten().tolist()
                hist_b = hist_b.flatten().tolist()
                hist_sum = sum(hist_r) + sum(hist_g) + sum(hist_b)
                if hist_sum > 0:
                    hist_r = [x / hist_sum for x in hist_r]
                    hist_g = [x / hist_sum for x in hist_g]
                    hist_b = [x / hist_sum for x in hist_b]

                # 使用K-Means提取主导颜色（RGB）
                pixels = roi[mask > 0] if mask is not None else roi.reshape(-1, 3)
                if len(pixels) > 0:
                    kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
                    kmeans.fit(pixels)
                    dominant_color_rgb = kmeans.cluster_centers_[0].astype(int).tolist()  # [R, G, B]
                else:
                    dominant_color_rgb = [0, 0, 0]  # 默认值

                # 添加到检测结果
                detections_list.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "dominant_color_rgb": dominant_color_rgb,
                    "mask_area": float(mask_area),
                    "color_histogram": {
                        "r": hist_r,
                        "g": hist_g,
                        "b": hist_b
                    }
                })

                # 绘制边界框和标签（包含RGB值）
                label = f"{class_name} (RGB: {dominant_color_rgb}) {conf:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections_list, annotated_image

    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        return [], image_bgr

def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """
    将检测结果格式化为JSON字符串，包含主导颜色RGB、掩码面积、直方图和拍摄时间。
    Args:
        detections_list: 检测结果列表
        image_shape: 图像尺寸 (height, width, channels)
        capture_time: 拍摄时间字符串（可选）
    Returns:
        JSON字符串
    """
    if not detections_list:
        data = {
            "image_height": image_shape[0],
            "image_width": image_shape[1],
            "detections": [],
            "message": "No objects detected in the image."
        }
    else:
        formatted_detections = []
        for detection in detections_list:
            formatted_detections.append({
                "class": detection["class"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "dominant_color_rgb": detection["dominant_color_rgb"],
                "mask_area": detection["mask_area"],
                "color_histogram": detection["color_histogram"]
            })
        data = {
            "image_height": image_shape[0],
            "image_width": image_shape[1],
            "detections": formatted_detections
        }
    
    if capture_time:
        data["capture_time"] = capture_time
    
    return json.dumps(data)

def answer_question_with_deepseek(json_detections, question, ollama_api_url, model_name):
    """
    根据检测结果、掩码面积、直方图、主导颜色RGB、拍摄时间和用户问题，使用DeepSeek生成回答。
    Args:
        json_detections: 检测结果的JSON字符串
        question: 用户输入的问题
        ollama_api_url: Ollama API的URL
        model_name: DeepSeek模型名称
    Returns:
        回答文本或错误信息
    """
    try:
        detections = json.loads(json_detections)
        image_height = detections["image_height"]
        image_width = detections["image_width"]
        capture_time = detections.get("capture_time", "unknown")

        if not detections["detections"]:
            prompt = (
                f"The image is {image_height} pixels high and {image_width} pixels wide. "
                f"No objects were detected. The image was captured at {capture_time}. "
                f"Please answer the following question based on this information:\n"
                f"Question: {question}"
            )
        else:
            prompt = (
                "You are an AI assistant that answers questions about an image based on structured detection data from a YOLO segmentation model. "
                "The input data is provided in JSON format. The arguements of bounding box are [x_center, y_center, width, height]"
                f"The image is {image_height} pixels high and {image_width} pixels wide. "
                f"The image was captured at {capture_time}. "
                f"Detected objects data:\n{json.dumps(detections, indent=2)}\n"
                "Based on this data, answer the following question in concise, natural language. "
                f"Question: {question}"
            )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(ollama_api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "No answer generated.")

    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to Ollama API."
    except requests.exceptions.Timeout:
        return "Error: Ollama API request timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP error occurred: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response format from Ollama API."
    except Exception as e:
        return f"Error in answering question: {str(e)}"

def process_image_and_describe(image_bgr, yolo_model, model_name, ollama_api_url, capture_time=None):
    """
    处理图像并返回检测结果和标注图像，供后续提问和直方图显示。
    Args:
        image_bgr: BGR格式的NumPy图像数组
        yolo_model: 加载的YOLO分割模型
        model_name: DeepSeek模型名称
        ollama_api_url: Ollama API的URL
        capture_time: 拍摄时间字符串（可选）
    Returns:
        json_detections: 检测结果的JSON字符串（包含时间、掩码面积、直方图和RGB）
        annotated_image: 标注后的图像（BGR格式）
    """
    try:
        # 进行目标检测和分割
        detections_list, annotated_image = detect_objects_yolo(image_bgr, yolo_model)

        # 格式化检测结果
        json_detections = format_detections_as_json_for_llm(detections_list, image_bgr.shape, capture_time)

        return json_detections, annotated_image

    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return json.dumps({"message": f"Processing failed: {str(e)}"}), image_bgr