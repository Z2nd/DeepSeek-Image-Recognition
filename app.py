import cv2
import os
import json
import sys
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2
import backend.processing_logic as processing_logic

class ImageProcessor:
    """
    A class to handle image capture, YOLO segmentation, and question answering on Raspberry Pi.
    """
    def __init__(self):
        """Initialize configurations and load YOLO model."""
        self.YOLO_MODEL_PATH = 'backend/resource/yolov8n-seg.pt'
        self.OLLAMA_API_URL = 'http://localhost:11434/api/generate'
        self.DEEPSEEK_MODEL_NAME = 'deepseek-r1:8b'
        self.METADATA_PATH = 'backend/resource/capture_metadata.json'
        self.IMAGE_SAVE_PATH = 'backend/resource/captured_image.jpg'
        
        # Create resource directory
        os.makedirs('backend/resource', exist_ok=True)
        
        # Load YOLO model
        print("Loading YOLO segmentation model...")
        try:
            self.yolo_model = YOLO(self.YOLO_MODEL_PATH)
            print(f"YOLO model '{self.YOLO_MODEL_PATH}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

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
        """
        Process the image with YOLO segmentation and return results.
        Args:
            image_bgr: Image in BGR format
            capture_time: Timestamp of capture
        Returns:
            json_detections: JSON string of detection results
            annotated_image: Annotated image with bounding boxes and masks
        """
        if self.yolo_model is None or image_bgr is None:
            print("Error: Model or image not loaded.")
            return json.dumps({"message": "Processing failed."}), image_bgr
        
        print("Starting image segmentation...")
        json_detections, annotated_image = processing_logic.process_image_and_describe(
            image_bgr,
            self.yolo_model,
            self.DEEPSEEK_MODEL_NAME,
            self.OLLAMA_API_URL,
            capture_time
        )
        print("Segmentation complete.")
        return json_detections, annotated_image

    def display_annotated_image(self, annotated_image):
        """
        Display the annotated image in a window.
        Args:
            annotated_image: Image with bounding boxes and mask contours
        """
        if annotated_image is not None:
            cv2.imshow("Annotated Image", annotated_image)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No annotated image available.")

    def question_loop(self, json_detections):
      """
      Handle user questions about the image, display and log performance metrics.
      Args:
          json_detections: JSON string of detection results
      """
      performance_log = []
      log_file = 'backend/resource/performance_log.json'
      
      print(f"Image captured at: {json.loads(json_detections).get('capture_time', 'unknown')}")
      print("Enter your questions about the image (type 'quit' to exit).")
      
      while True:
          question = input("Question: ").strip()
          sys.stdout.flush()
          
          if question.lower() == 'quit':
              # Save performance log
              with open(log_file, 'w') as f:
                  json.dump(performance_log, f, indent=2)
              print(f"Performance metrics saved to {log_file}.")
              break
          if not question:
              print("Please enter a valid question.")
              sys.stdout.flush()
              continue
          
          print("Generating answer, please wait...")
          sys.stdout.flush()
          
          answer, metrics = processing_logic.answer_question_with_deepseek(
              json_detections,
              question,
              self.OLLAMA_API_URL,
              self.DEEPSEEK_MODEL_NAME
          )
          print(f"Answer: {answer}")
          print(f"Performance: Inference Time={metrics['inference_time']:.2f}s, "
                f"Memory={metrics['memory_mb']:.2f}MB, "
                f"CPU={metrics['cpu_percent']:.1f}%, "
                f"Retries={metrics['retry_attempts']}, "
                f"Status={metrics['status']}")
          sys.stdout.flush()
          
          # Append metrics to log
          performance_log.append(metrics)
          print()  # Add spacing for readability

    def run(self):
        """Main method to execute the image processing and questioning pipeline."""
        # Capture image
        image_bgr, capture_time = self.capture_image()
        if image_bgr is None:
            print("Exiting due to capture failure.")
            return
        
        # Process and annotate image
        json_detections, annotated_image = self.process_and_annotate(image_bgr, capture_time)
        
        # Display annotated image
        self.display_annotated_image(annotated_image)
        
        # Start question loop
        self.question_loop(json_detections)

if __name__ == "__main__":
    processor = ImageProcessor()
    processor.run()