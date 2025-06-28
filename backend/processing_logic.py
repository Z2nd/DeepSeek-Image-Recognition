import cv2
import numpy as np
import json
import requests
from ultralytics import YOLO

def detect_objects_yolo(image_bgr, yolo_model):
    """
    使用YOLO模型进行目标检测，并提取每个检测对象的颜色。
    Args:
        image_bgr: BGR格式的NumPy图像数组
        yolo_model: 加载的YOLO模型
    Returns:
        detections_list: 检测结果列表，包含类别、置信度、边界框和主要颜色
        annotated_image: 标注后的图像（BGR格式）
    """
    try:
        if yolo_model is None:
            print("Error: YOLO model is not loaded.")
            return [], image_bgr

        # 进行目标检测
        results = yolo_model(image_bgr)
        detections_list = []
        annotated_image = image_bgr.copy()

        # 定义颜色映射（HSV范围）
        color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
            'blue': [(100, 100, 100), (130, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)],
            'yellow': [(20, 100, 100), (30, 255, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'white': [(0, 0, 200), (180, 20, 255)]
        }

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 提取边界框坐标、类别和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = yolo_model.names[cls]

                # 提取边界框区域
                roi = image_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # 转换为HSV颜色空间
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # 计算主要颜色
                dominant_color = None
                max_count = 0
                for color_name, ranges in color_ranges.items():
                    if len(ranges) == 4:  # 处理红色（两段范围）
                        mask1 = cv2.inRange(roi_hsv, ranges[0], ranges[1])
                        mask2 = cv2.inRange(roi_hsv, ranges[2], ranges[3])
                        mask = cv2.bitwise_or(mask1, mask2)
                    else:
                        mask = cv2.inRange(roi_hsv, ranges[0], ranges[1])
                    count = cv2.countNonZero(mask)
                    if count > max_count:
                        max_count = count
                        dominant_color = color_name

                # 默认颜色（若未识别）
                dominant_color = dominant_color or "unknown"

                # 添加到检测结果
                detections_list.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "color": dominant_color
                })

                # 在图像上绘制边界框和标签
                label = f"{class_name} ({dominant_color}) {conf:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections_list, annotated_image

    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        return [], image_bgr

def format_detections_as_json_for_llm(detections_list, image_shape):
    """
    将检测结果格式化为JSON字符串，包含颜色信息，供LLM使用。
    Args:
        detections_list: 检测结果列表
        image_shape: 图像尺寸 (height, width, channels)
    Returns:
        JSON字符串
    """
    if not detections_list:
        return json.dumps({
            "image_height": image_shape[0],
            "image_width": image_shape[1],
            "detections": [],
            "message": "No objects detected in the image."
        })

    formatted_detections = []
    for detection in detections_list:
        formatted_detections.append({
            "class": detection["class"],
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "color": detection["color"]
        })

    return json.dumps({
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "detections": formatted_detections
    })

def generate_text_with_deepseek(json_detections, ollama_api_url, model_name):
    """
    使用DeepSeek生成自然语言描述。
    Args:
        json_detections: 检测结果的JSON字符串
        ollama_api_url: Ollama API的URL
        model_name: DeepSeek模型名称
    Returns:
        生成的描述或错误信息
    """
    try:
        detections = json.loads(json_detections)
        if not detections["detections"]:
            prompt = (f"The image is {detections['image_height']} pixels high and "
                     f"{detections['image_width']} pixels wide. No objects were detected.")
        else:
            prompt = f"""
You are an AI assistant that describes images based on a list of detected objects provided in JSON format.
The following is a JSON list of objects detected in the image. Each object has:
- 'class': The identified type of the object.
- 'confidence': The model's confidence in this detection (a value between 0.0 and 1.0).
- 'bbox': A list of four integers [x_center , y_center, width, height], where [x,y] is the top-left corner of the bounding box.
- 'color': The dominant color detected in the object.
Detected objects data:
{json_detections}
Based on this structured data, please generate a concise and natural language description of the scene. Try to infer relationships between objects and their general locations (e.g., "left side", "center", "behind another object") rather than just listing the raw bounding box coordinates. Focus on creating a human-like, coherent description of what the image likely contains.
"""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(ollama_api_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "No description generated.")

    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to Ollama API."
    except requests.exceptions.Timeout:
        return "Error: Ollama API request timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP error occurred: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response format from Ollama API."
    except Exception as e:
        return f"Error in text generation: {str(e)}"

def answer_question_about_image(json_detections, question, ollama_api_url, model_name):
    """
    根据图像检测结果回答用户的问题。
    Args:
        json_detections: 检测结果的JSON字符串
        question: 用户的问题
        ollama_api_url: Ollama API的URL
        model_name: DeepSeek模型名称
    Returns:
        答案或错误信息
    """
    try:
        if not question.strip():
            return "Error: Empty question provided."

        detections = json.loads(json_detections)
        
        # 定义输入数据结构
        input_structure = {
            "image_info": {
                "height": detections["image_height"],
                "width": detections["image_width"]
            },
            "objects": [
                {
                    "class": det["class"],
                    "color": det["color"],
                    "position": {
                        "top_left": {"x": det["bbox"][0], "y": det["bbox"][1]},
                        "bottom_right": {"x": det["bbox"][2], "y": det["bbox"][3]}
                    },
                    "confidence": det["confidence"]
                } for det in detections["detections"]
            ]
        }

        # 构造提示
        prompt = (
            "You are an AI that answers questions about images based on structured detection data. "
            f"For this image, the data is:\n{json.dumps(input_structure, indent=2)}\n"
            f"User question: {question}\n"
            "Answer the question in natural language based on the provided data. "
            "If the question cannot be answered with the available data, respond with 'The available data does not provide enough information to answer this question.'"
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(ollama_api_url, json=payload, timeout=30)
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

def process_image_and_describe(image_bgr, yolo_model, model_name, ollama_api_url):
    """
    处理图像并生成描述，包含颜色信息，并返回检测结果的JSON以支持提问。
    Args:
        image_bgr: BGR格式的NumPy图像数组
        yolo_model: 加载的YOLO模型
        model_name: DeepSeek模型名称
        ollama_api_url: Ollama API的URL
    Returns:
        natural_language_description: DeepSeek生成的描述
        annotated_image: 标注后的图像（BGR格式）
        json_detections: 检测结果的JSON字符串
    """
    try:
        # 进行目标检测（包括颜色检测）
        detections_list, annotated_image = detect_objects_yolo(image_bgr, yolo_model)

        # 格式化检测结果
        json_detections = format_detections_as_json_for_llm(detections_list, image_bgr.shape)

        # 生成自然语言描述
        natural_language_description = generate_text_with_deepseek(json_detections, ollama_api_url, model_name)

        return natural_language_description, annotated_image, json_detections

    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return f"Processing failed: {str(e)}", image_bgr, None