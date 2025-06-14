import cv2
from ultralytics import YOLO
from PIL import Image # For displaying images in Notebook
import os
import time # For potential delays if needed
import traceback

import backend.processing_logic as processing_logic

# --- Configuration ---
# CHOOSE OPERATION MODE: "file" or "camera"
OPERATION_MODE = "file"  # <<<< CHANGE THIS TO "camera" TO USE CAMERA MODE
# OPERATION_MODE = "camera"

# YOLO Configuration
YOLO_MODEL_NAME = 'backend/resource/yolov8n.pt'

# Ollama / DeepSeek Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEEPSEEK_MODEL_NAME = "deepseek-r1:8b" # 默认使用 "deepseek-llm"，请根据 `ollama list` 的结果修改

# Input Image Path (used only if OPERATION_MODE is "file")
INPUT_IMAGE_PATH = "backend/resource/bus.jpg" # 例如: "test_images/bus.jpg"

# Camera Configuration (used only if OPERATION_MODE is "camera")
CAMERA_INDEX = 0 # 0 for default camera, change if you have multiple cameras

# --- Sanity Checks ---
print("Libraries imported.")
print(f"Selected Operation Mode: {OPERATION_MODE}")
print(f"Using YOLO model: {YOLO_MODEL_NAME}")
print(f"Ollama API URL: {OLLAMA_API_URL}")
print(f"Attempting to use DeepSeek model: {DEEPSEEK_MODEL_NAME}")

if DEEPSEEK_MODEL_NAME == "your-deepseek-model-name": # Default placeholder check
    print("\n⚠️ WARNING: 'DEEPSEEK_MODEL_NAME' is set to a placeholder. Please update it with your actual model name from Ollama (run `ollama list`).")

if OPERATION_MODE == "file":
    if not os.path.exists(INPUT_IMAGE_PATH) and INPUT_IMAGE_PATH != "path/to/your/image.jpg":
        print(f"\n⚠️ WARNING: Input image not found at '{INPUT_IMAGE_PATH}'. Please check the path.")
    elif INPUT_IMAGE_PATH == "path/to/your/image.jpg":
         print(f"\n⚠️ WARNING: Please update 'INPUT_IMAGE_PATH' to your desired image for file mode.")
elif OPERATION_MODE not in ["file", "camera"]:
    print(f"\n⚠️ ERROR: Invalid 'OPERATION_MODE' ('{OPERATION_MODE}'). Choose 'file' or 'camera'.")


# Load YOLO Model (common for both modes)
try:
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print(f"YOLO model '{YOLO_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None




# --- Preliminary Checks ---
if yolo_model is None:
    print("Skipping workflow: YOLO model failed to load. Please check Cell 1.")
elif DEEPSEEK_MODEL_NAME == "your-deepseek-model-name" or not DEEPSEEK_MODEL_NAME:
    print("Skipping workflow: DeepSeek model name not configured correctly. Please check Cell 1.")
elif OPERATION_MODE not in ["file", "camera"]:
    print(f"Skipping workflow: Invalid 'OPERATION_MODE' ('{OPERATION_MODE}') in Cell 1. Choose 'file' or 'camera'.")
else:
    # --- FILE MODE ---
    if OPERATION_MODE == "file":
        print(f"--- Running in FILE mode for image: {INPUT_IMAGE_PATH} ---")
        if not os.path.exists(INPUT_IMAGE_PATH) or INPUT_IMAGE_PATH == "path/to/your/image.jpg":
            print(f"Skipping file mode: Input image path '{INPUT_IMAGE_PATH}' is invalid or not set. Please check Cell 1.")
        else:
            try:
                input_image_bgr = cv2.imread(INPUT_IMAGE_PATH)
                if input_image_bgr is None:
                    raise FileNotFoundError(f"Could not read image at {INPUT_IMAGE_PATH}")

                print("\n--- Original Image ---")
                input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
                # display(Image.fromarray(input_image_rgb))
                # Image.fromarray(input_image_rgb).show()  # Display the original image

                # Call the shared processing function
                print("\n--- Processing Image ---")
                description, annotated_image_bgr = processing_logic.process_image_and_describe(
                    input_image_bgr, yolo_model, DEEPSEEK_MODEL_NAME, OLLAMA_API_URL
                )

                # --- Output Results ---
                print("\n--- Annotated Image (YOLO Detections) ---")
                if annotated_image_bgr is not None:
                    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                    # display(Image.fromarray(annotated_image_rgb))
                    Image.fromarray(annotated_image_rgb).show()
                else:
                    print("Annotated image is not available.")

                print("\n--- Natural Language Description ---")
                print(description)

            except FileNotFoundError as e_file:
                print(f"Error in file mode: {e_file}")
            except Exception as e_file_main:
                print(f"An unexpected error occurred in file mode: {e_file_main}")
                traceback.print_exc()

    # --- CAMERA MODE ---
    elif OPERATION_MODE == "camera":
        print(f"--- Running in CAMERA mode (Camera Index: {CAMERA_INDEX}) ---")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"Error: Could not open camera with index {CAMERA_INDEX}. Please check if the camera is connected and not in use.")
        else:
            print("Camera opened successfully.")
            print("Instructions: Focus the OpenCV window. Press 'c' to capture and process the current frame. Press 'q' to quit.")

            window_name_annotated = "Annotated Frame (Press 'c' for next, 'q' to quit)"
            cv2.namedWindow(window_name_annotated, cv2.WINDOW_AUTOSIZE)

            try:
                while True:
                    print("\nWaiting for your command... (Focus OpenCV window: Press 'c' to capture, 'q' to quit)")

                    key = cv2.waitKey(0) & 0xFF # Wait indefinitely for a key press

                    if key == ord('q'):
                        print("Quit command received. Exiting camera mode.")
                        break
                    elif key == ord('c'):
                        print("Capture command received.")
                        ret, frame_bgr = cap.read()
                        if not ret:
                            print("Error: Could not read frame from camera. Exiting.")
                            break

                        print(f"\n--- Processing new frame from camera ---")

                        # Call the shared processing function
                        description, annotated_frame_bgr = processing_logic.process_image_and_describe(
                            frame_bgr, yolo_model, DEEPSEEK_MODEL_NAME, OLLAMA_API_URL
                        )

                        # --- Output Results ---
                        print("\n--- Natural Language Description ---")
                        print(description)

                        print("\n--- Annotated Frame (YOLO Detections) ---")
                        if annotated_frame_bgr is not None:
                            cv2.imshow(window_name_annotated, annotated_frame_bgr)
                        else:
                             # If annotation failed, show the original captured frame
                            cv2.imshow(window_name_annotated, frame_bgr)
                            print("Annotated frame is not available, showing original captured frame.")
                        cv2.waitKey(1) # Allow window to update

                        print("-" * 70) # Separator for next iteration

                    else:
                        # Optional: Handle other key presses if needed
                        print(f"Key '{chr(key)}' pressed. Press 'c' to capture or 'q' to quit.")
                        pass

            except KeyboardInterrupt:
                print("Camera mode interrupted by user (Ctrl+C).")
            except Exception as e_cam_main:
                print(f"An unexpected error occurred in camera mode: {e_cam_main}")
                traceback.print_exc()
            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                    print("Camera released.")
                cv2.destroyAllWindows()
                print("OpenCV windows closed.")

