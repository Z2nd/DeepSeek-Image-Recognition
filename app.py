# app.py (Hardware Deployment Version)
#
# This is the final, consolidated script designed for hardware deployment (e.g., Raspberry Pi).
# It incorporates all recent features:
# 1. Multi-Model Fusion: Combines results from different YOLO models for initial detection.
# 2. OCR Analysis: Extracts text from specified object classes.
# 3. Recursive Analysis: Performs detailed sub-detection on specified objects.
# 4. Post-Processing Rules: Applies logic like "one keyboard per laptop".
# 5. Hardware/Simulation Mode: Uses picamera2 if available, otherwise falls back to loading a local image file.

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
    print("WARNING: picamera2 library not found. Running in SIMULATION mode (loading local image).")
    IS_HARDWARE = False

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
from backend.llm.interaction import answer_question_with_deepseek


class Application:
    """
    Encapsulates the entire application flow from image capture to interactive Q&A.
    """
    def __init__(self):
        """
        Initializes configurations, loads models, and constructs the vision pipeline.
        """
        print("\nInitializing application...")
        
        # We need the image dimensions to initialize the GridPositionAnalyzer.
        # We'll get them by pre-loading the fallback image.
        try:
            temp_img = cv2.imread(config.IMAGE_PATH)
            if temp_img is None:
                raise FileNotFoundError(f"Fallback image not found at {config.IMAGE_PATH}")
            self.image_height, self.image_width, _ = temp_img.shape
            print(f"Image dimensions set to: {self.image_width}x{self.image_height}")
        except Exception as e:
            print(f"ERROR: Could not load fallback image to determine dimensions. Exiting. Error: {e}")
            sys.exit(1)

        # 1. Initialize all individual analyzers.
        print("Initializing analyzers...")
        color_analyzer = ColorAnalyzer()
        ocr_analyzer = OCRAnalyzer()
        grid_analyzer = GridPositionAnalyzer(self.image_width, self.image_height)
        
        # 2. The RecursiveYOLOAnalyzer now orchestrates other analyzers that run on each detection.
        recursive_analyzer = RecursiveYOLOAnalyzer(
            initial_config=config.RECURSIVE_DETECTION_CONFIG,
            other_analyzers=[color_analyzer, ocr_analyzer] # Color and OCR run on every relevant object
        )

        # 3. Construct the main vision pipeline.
        # The pipeline now internally handles model loading based on the config.
        self.pipeline = VisionPipeline(
            recursive_analyzer=recursive_analyzer,
            post_analyzers=[grid_analyzer] # Grid position is a post-processing step on the final list
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
                # Configure the camera for high-resolution still capture
                capture_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
                picam2.configure(capture_config)
                picam2.start()
                time.sleep(2) # Allow time for sensor to adjust
                image_array = picam2.capture_array()
                picam2.stop()
                # Convert RGB (from PiCamera) to BGR (for OpenCV)
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
        start_time = time.time()
        
        all_detections, annotated_image = self.pipeline.run(image_bgr)
        
        end_time = time.time()
        print(f"Pipeline finished in {end_time - start_time:.2f} seconds.")
        
        # Save the annotated image for review
        output_path = os.path.join(os.path.dirname(config.IMAGE_PATH), "annotated_output.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")
        
        # Format the results into JSON for the LLM
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
                
                print("Thinking...")
                final_answer, full_response, metrics = answer_question_with_deepseek(
                    json_detections, question, config.OLLAMA_API_URL, config.DEEPSEEK_MODEL_NAME
                )
                print(f"\nAnswer: {final_answer}\n")
                
                response_log.append({
                    "question": question, 
                    "answer": final_answer, 
                    "full_response": full_response,
                    "metrics": metrics,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except (KeyboardInterrupt, EOFError):
                break # Allow exiting with Ctrl+C or Ctrl+D
        
        # Save the conversation log
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
        
        # Save the generated JSON for debugging purposes
        json_output_path = os.path.join(os.path.dirname(config.IMAGE_PATH), "detection_results.json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            f.write(json_detections)
        print(f"Detection JSON saved to: {json_output_path}")
        
        self.interactive_qa(json_detections)
        print("\nGoodbye!")


if __name__ == "__main__":
    app = Application()
    app.run()
