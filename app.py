# app.py
# 这是一个为硬件（如树莓派）设计的完整应用程序脚本。
# 它使用 picamera2 库从摄像头捕获图像，
# 然后运行一个支持递归检测的视觉分析流水线，
# 最后通过命令行界面与大型语言模型进行交互式问答。

import cv2
import os
import json
import sys
from datetime import datetime
from ultralytics import YOLO

# 尝试导入硬件特定的库，如果失败则给出提示
try:
    from picamera2 import Picamera2
    IS_HARDWARE = True
except ImportError:
    print("警告: picamera2 库未找到。程序将以模拟模式运行，加载本地文件。")
    IS_HARDWARE = False

# 导入重构后的项目模块和配置
from backend import config
from backend.vision.pipelines import VisionPipeline
from backend.vision.analyzers import GroupingAnalyzer, ColorAnalyzer, RecursiveYOLOAnalyzer
from backend.llm.formatting import format_detections_as_json_for_llm
from backend.llm.interaction import answer_question_with_deepseek

class ImageProcessor:
    """
    封装了图像捕获、视觉分析和问答交互的完整流程。
    """
    def __init__(self):
        """初始化配置、加载模型并构建视觉处理流水线。"""
        print("正在加载主YOLO模型...")
        try:
            self.yolo_model = YOLO(config.YOLO_GENERAL_MODEL_PATH)
            print(f"主模型 '{config.YOLO_GENERAL_MODEL_PATH}' 加载成功。")
        except Exception as e:
            print(f"错误: 无法加载主YOLO模型: {e}")
            sys.exit(1)
        
        # 定义将在每个检测层级应用的基础分析器
        base_analyzers = [
            GroupingAnalyzer(),
            ColorAnalyzer(),
        ]
        
        # 创建递归分析器的实例，它将协调所有的分层检测
        recursive_analyzer = RecursiveYOLOAnalyzer(
            initial_config=config.RECURSIVE_DETECTION_CONFIG,
            other_analyzers=base_analyzers
        )
        
        # 创建最终的视觉流水线实例
        self.pipeline = VisionPipeline(self.yolo_model, recursive_analyzer)
        print("视觉流水线构建完成。")

    def capture_image(self):
        """
        从硬件摄像头捕获图像。如果不在硬件上运行，则加载本地文件作为替代。
        """
        capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if IS_HARDWARE:
            print("正在从Picamera2摄像头捕获图像...")
            try:
                picam2 = Picamera2()
                # 配置预览和捕获分辨率
                config_still = picam2.create_still_configuration(main={"size": (1280, 720)}, lores={"size": (640, 480)}, display="lores")
                picam2.configure(config_still)
                picam2.start()
                
                # 捕获为NumPy数组 (RGB格式)
                image_rgb = picam2.capture_array()
                picam2.stop()
                picam2.close()

                # 将RGB转换为OpenCV使用的BGR格式
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # 保存捕获的图像
                cv2.imwrite(config.IMAGE_SAVE_PATH, image_bgr)
                print(f"图像成功捕获并保存至: {config.IMAGE_SAVE_PATH}")
                
                return image_bgr, capture_time
            except Exception as e:
                print(f"错误: 摄像头捕获失败: {e}")
                return None, None
        else:
            # 模拟模式：加载本地文件
            print(f"模拟捕获模式: 正在从 '{config.IMAGE_SAVE_PATH}' 加载图像...")
            image_bgr = cv2.imread(config.IMAGE_SAVE_PATH)
            if image_bgr is None:
                print(f"错误: 无法从 {config.IMAGE_SAVE_PATH} 加载图像。请确保文件存在。")
                return None, None
            print("图像加载成功。")
            return image_bgr, capture_time

    def process_and_get_results(self, image_bgr, capture_time):
        """运行完整的视觉流水线并格式化结果。"""
        if image_bgr is None:
            return None, None

        print("正在运行视觉分析流水线...")
        detections, annotated_image = self.pipeline.run(image_bgr)
        
        json_detections = format_detections_as_json_for_llm(
            detections, image_bgr.shape, capture_time
        )
        print("流水线处理完成。")
        return json_detections, annotated_image

    def save_annotated_image(self, annotated_image):
        """将带标注的图像保存到文件，而不是尝试在窗口中显示。"""
        if annotated_image is not None:
            save_path = 'backend/data/annotated_image.jpg'
            try:
                cv2.imwrite(save_path, annotated_image)
                print(f"已标注的图像已保存至: {save_path}")
            except Exception as e:
                print(f"错误: 保存已标注图像失败: {e}")

    def question_loop(self, json_detections):
        """处理用户在命令行中的问答循环。"""
        if json_detections is None:
            print("没有检测结果可供提问。")
            return

        response_log = []
        
        print("\n--- 问答环节 ---")
        print("请输入您关于图像的问题 (输入 'quit' 退出):")
        while True:
            question = input("问题: ").strip()
            if question.lower() == 'quit':
                try:
                    with open(config.RESPONSE_LOG_PATH, 'w', encoding='utf-8') as f:
                        json.dump(response_log, f, indent=2, ensure_ascii=False)
                    print(f"问答日志已保存至: {config.RESPONSE_LOG_PATH}")
                except Exception as e:
                    print(f"错误: 保存日志文件失败: {e}")
                break
            if not question:
                continue
            
            print("正在生成回答...")
            final_answer, _, metrics = answer_question_with_deepseek(
                json_detections, question, config.OLLAMA_API_URL, config.DEEPSEEK_MODEL_NAME
            )
            print(f"回答: {final_answer}")
            print(f"性能: 时间={metrics.get('inference_time', 0):.2f}s, "
                  f"内存={metrics.get('memory_mb', 0):.2f}MB, "
                  f"状态={metrics.get('status', 'unknown')}\n")
            
            response_log.append({
                "question": question, 
                "answer": final_answer, 
                "metrics": metrics,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    def run(self):
        """执行从捕获到问答的完整主流程。"""
        image_bgr, capture_time = self.capture_image()
        if image_bgr is None:
            print("程序因图像获取失败而退出。")
            return
        
        json_detections, annotated_image = self.process_and_get_results(image_bgr, capture_time)
        
        self.save_annotated_image(annotated_image)
        
        self.question_loop(json_detections)

if __name__ == "__main__":
    processor = ImageProcessor()
    processor.run()
