# app.py (Fixed Version)
#
# This version corrects the ImportError by using the new refactored functions
# from the interaction.py module.

import cv2
import os
import json
import sys
from datetime import datetime

# Attempt to import hardware-specific library; set a flag accordingly.
try:
    from picamera2 import Picamera2
    import time
    IS_HARDWARE = True
    print("picamera2 library found. Running in HARDWARE mode.")
except ImportError:
    # This is a normal warning when not on a Raspberry Pi.
    # print("WARNING: picamera2 library not found. Running in SIMULATION mode (loading local image).")
    IS_HARDWARE = False

# ------------ TESTING MODE ------------
# IS_HARDWARE = False
# --------------------------------------

# Import all necessary modules and configurations from the backend.
from backend import config
from backend.vision.pipelines import VisionPipeline
from backend.vision.analyzers import (
    ColorAnalyzer,
    RecursiveYOLOAnalyzer,
    GridPositionAnalyzer,
    OCRAnalyzer
)
from backend.llm.formatting import format_detections_as_json_for_llm
# Corrected import: Use the new refactored functions
from backend.llm.interaction import predict_response_time, stream_answer


class Application:
    """
    Encapsulates the entire application flow from image capture to interactive Q&A.
    """
    def __init__(self):
        """
        Initializes configurations, loads models, and constructs the vision pipeline.
        """
        print("\nInitializing application...")

        try:
            temp_img = cv2.imread(config.IMAGE_PATH)
            if temp_img is None:
                raise FileNotFoundError(f"Fallback image not found at {config.IMAGE_PATH}")
            self.image_height, self.image_width, _ = temp_img.shape
            print(f"Image dimensions set to: {self.image_width}x{self.image_height}")
        except Exception as e:
            print(f"ERROR: Could not load fallback image to determine dimensions. Exiting. Error: {e}")
            sys.exit(1)

        print("Initializing analyzers...")
        color_analyzer = ColorAnalyzer()
        ocr_analyzer = OCRAnalyzer()
        grid_analyzer = GridPositionAnalyzer(self.image_width, self.image_height)

        recursive_analyzer = RecursiveYOLOAnalyzer(
            initial_config=config.RECURSIVE_DETECTION_CONFIG,
            other_analyzers=[color_analyzer, ocr_analyzer]
        )

        self.pipeline = VisionPipeline(
            recursive_analyzer=recursive_analyzer,
            post_analyzers=[grid_analyzer]
        )
        print("Vision pipeline constructed successfully.")

    def capture_image(self):
        """
        Captures an image from the camera if on hardware, otherwise loads from a file.
        """
        if IS_HARDWARE:
            print("\nCapturing image from PiCamera...")
            try:
                picam2 = Picamera2()
                capture_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
                picam2.configure(capture_config)
                picam2.start()
                time.sleep(2)
                image_array = picam2.capture_array()
                picam2.stop()
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                print("Image captured successfully.")
                return image_bgr, datetime.now()
            except Exception as e:
                print(f"ERROR: Failed to capture image from camera: {e}")
                return None, None
        else:
            print(f"\nLoading image from file: {config.IMAGE_PATH}")
            image_bgr = cv2.imread(config.IMAGE_PATH)
            if image_bgr is None:
                print(f"ERROR: Failed to load image from {config.IMAGE_PATH}")
                return None, None
            print("Image loaded successfully.")
            return image_bgr, datetime.now()

    def process_image(self, image_bgr, capture_time):
        """
        Runs the full vision pipeline on the provided image.
        """
        print("\nRunning vision pipeline... (This may take a moment)")
        start_time = time.time() if 'time' in globals() else 0

        all_detections, annotated_image = self.pipeline.run(image_bgr)

        end_time = time.time() if 'time' in globals() else 0
        if start_time > 0:
            print(f"Pipeline finished in {end_time - start_time:.2f} seconds.")

        output_path = os.path.join(os.path.dirname(config.IMAGE_PATH), "annotated_output.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")

        json_detections = format_detections_as_json_for_llm(
            all_detections, image_bgr.shape, capture_time.strftime("%Y-%m-%d %H:%M:%S")
        )

        return json_detections

    def interactive_qa(self, json_detections):
        """
        Starts an interactive command-line Q&A session with the LLM.
        """
        print("\n" + "="*50)
        print("AI Vision Assistant Ready. Ask me about the image.")
        print("   Type your question and press Enter. Type 'exit' or 'quit' to finish.")
        print("="*50)

        response_log = []

        while True:
            try:
                question = input("> ")
                if question.lower() in ['exit', 'quit']:
                    break
                if not question:
                    continue

                # 1. Call the prediction function
                predicted_time = predict_response_time(json_detections, question)
                print(f"(Estimated time: {predicted_time:.1f} seconds)")

                # 2. Call the streaming function
                print("\nAnswer: ", end='', flush=True)
                full_response, metrics = stream_answer(
                    json_detections, question, config.OLLAMA_API_URL, config.DEEPSEEK_MODEL_NAME
                )

                # 3. Display performance metrics
                if metrics:
                    total_s = metrics.get('total_duration', 0) / 1_000_000_000
                    load_s = metrics.get('load_duration', 0) / 1_000_000_000
                    eval_count = metrics.get('eval_count', 0)
                    eval_s = metrics.get('eval_duration', 0) / 1_000_000_000
                    speed_tps = (eval_count / eval_s) if eval_s > 0 else float('inf')

                    print("\n" + "---")
                    print(f"Time -> Total: {total_s:.2f}s | Load: {load_s:.2f}s")
                    print(f"Speed -> {speed_tps:.1f} tokens/sec ({eval_count} tokens)")
                    print("---")
                else:
                    print("\n")

                response_log.append({
                    "question": question,
                    "answer": full_response,
                    "metrics": metrics,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except (KeyboardInterrupt, EOFError):
                break

        try:
            with open(config.RESPONSE_LOG_PATH, 'w', encoding='utf-8') as f:
                json.dump(response_log, f, indent=2, ensure_ascii=False)
            print(f"\nQ&A log saved to: {config.RESPONSE_LOG_PATH}")
        except Exception as e:
            print(f"ERROR: Could not save log file: {e}")

    def run(self):
        """
        Executes the main application flow.
        """
        image_bgr, capture_time = self.capture_image()
        if image_bgr is None:
            print("\nApplication exiting due to image acquisition failure.")
            return

        json_detections = self.process_image(image_bgr, capture_time)

        json_output_path = os.path.join(os.path.dirname(config.IMAGE_PATH), "detection_results.json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            f.write(json_detections)
        print(f"Detection JSON saved to: {json_output_path}")

        self.interactive_qa(json_detections)
        print("\nGoodbye!")


if __name__ == "__main__":
    app = Application()
    app.run()
