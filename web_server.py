# web_server.py
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import backend.processing_logic as processing_logic

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='frontend/template', static_folder='frontend/static')

# --- Configuration ---
YOLO_MODEL_NAME = 'backend/resource/yolov8n.pt'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEEPSEEK_MODEL_NAME = "deepseek-r1:8b" 

# --- Load YOLO Model ---
print("Loading YOLO model...")
try:
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print(f"YOLO model '{YOLO_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# --- Define Routes ---

# Route for the main page
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

# Route for processing the uploaded image
@app.route('/process_image', methods=['POST'])
def process_image_route():
    """Receives an image, processes it, and returns the results."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and yolo_model:
        try:
            # 1. Read image from upload
            image_bytes = file.read()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 2. Call your existing processing logic
            description, annotated_image_bgr = processing_logic.process_image_and_describe(
                image_bgr, yolo_model, DEEPSEEK_MODEL_NAME, OLLAMA_API_URL
            )
            
            # 3. Encode the annotated image to send back to the browser
            _, buffer = cv2.imencode('.jpg', annotated_image_bgr)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{annotated_image_base64}"

            # 4. Return results as JSON
            return jsonify({
                'description': description,
                'annotated_image': image_data_url
            })

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({'error': f'An error occurred during processing: {e}'}), 500
            
    return jsonify({'error': 'Processing failed'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Setting host='0.0.0.0' makes it accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True)