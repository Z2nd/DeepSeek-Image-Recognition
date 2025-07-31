# app.py
import cv2
import os
import json
import sys
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2  # 在PC上测试时注释掉

# 导入新模块和配置
from backend import config
from backend.vision.detection import detect_objects_yolo
from backend.vision.models import load_secondary_model
from backend.llm.formatting import format_detections_as_json_for_llm
from backend.llm.interaction import answer_question_with_deepseek

class ImageProcessor:
    def __init__(self):
        """初始化配置并加载模型。"""
        print("Loading YOLO model...")
        try:
            self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
            print(f"YOLO model '{config.YOLO_MODEL_PATH}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
            
        print("Loading secondary classification model...")
        self.secondary_model = load_secondary_model(config.SECONDARY_MODEL_TYPE, num_classes=4) # 示例类别数

    def capture_image(self):
        """
        Capture an image from the Raspberry Pi camera using picamera2 and save metadata.
        Returns:
            image_bgr: Captured image in BGR format
            capture_time: Timestamp of capture
        """
        try:
            # Initialize Picamera2
            picam2 = Picamera2()
            # Configure for still capture (use RGB format, convert to BGR)
            config = picam2.create_still_configuration(main={"size": (640, 480), "format": "RGB888"})
            picam2.configure(config)
            picam2.start()
            
            try:
                # Capture image in BGR format
                image_bgr = picam2.capture_array()
                
                # Get timestamp
                capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Image captured at {capture_time}.")
                
                # Save image in BGR format
                cv2.imwrite(self.IMAGE_SAVE_PATH, image_bgr)
                print(f"Image saved to {self.IMAGE_SAVE_PATH}.")
                
                # Save metadata
                metadata = {"capture_time": capture_time}
                with open(self.METADATA_PATH, 'w') as f:
                    json.dump(metadata, f)
                print(f"Metadata saved to {self.METADATA_PATH}.")
                
                return image_bgr, capture_time
            
            finally:
                picam2.stop()
                picam2.close()
        
        except Exception as e:
            print(f"Error capturing image with picamera2: {str(e)}")
            return None, None

    def process_and_annotate(self, image_bgr, capture_time):
        """处理图像并返回结果。"""
        if self.yolo_model is None or image_bgr is None:
            return json.dumps({"message": "Processing failed."}), image_bgr
        
        print("Starting image analysis...")
        detections_list, annotated_image = detect_objects_yolo(
            image_bgr, self.yolo_model, self.secondary_model
        )
        json_detections = format_detections_as_json_for_llm(
            detections_list, image_bgr.shape, capture_time
        )
        print("Analysis complete.")
        return json_detections, annotated_image

    def display_annotated_image(self, annotated_image):
        """显示带标注的图像。"""
        if annotated_image is not None:
            cv2.imshow("Annotated Image", annotated_image)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def question_loop(self, json_detections):
        """处理用户问答循环。"""
        response_log = []
        
        print("\nEnter your questions about the image (type 'quit' to exit).")
        while True:
            question = input("Question: ").strip()
            if question.lower() == 'quit':
                with open(config.RESPONSE_LOG_PATH, 'w') as f:
                    json.dump(response_log, f, indent=2)
                print(f"Performance metrics saved to {config.RESPONSE_LOG_PATH}.")
                break
            if not question: continue
            
            print("Generating answer...")
            final_answer, _, metrics = answer_question_with_deepseek(
                json_detections, question, config.OLLAMA_API_URL, config.DEEPSEEK_MODEL_NAME
            )
            print(f"Answer: {final_answer}")
            print(f"Performance: Time={metrics.get('inference_time', 0):.2f}s, "
                  f"Memory={metrics.get('memory_mb', 0):.2f}MB, "
                  f"Status={metrics.get('status', 'unknown')}\n")
            response_log.append({"question": question, "answer": final_answer, "metrics": metrics})

    def run(self):
        """执行主流程。"""
        image_bgr, capture_time = self.capture_image()
        if image_bgr is None: return
        
        json_detections, annotated_image = self.process_and_annotate(image_bgr, capture_time)
        self.display_annotated_image(annotated_image)
        self.question_loop(json_detections)

if __name__ == "__main__":
    processor = ImageProcessor()
    processor.run()