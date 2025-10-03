# backend/llm/interaction.py
import json
import requests
import time
import os
from transformers import AutoTokenizer

# --- Configuration & Global Variables ---
METRICS_FILE_PATH = os.path.join(os.path.dirname(__file__), 'llm_metrics.json')
ESTIMATED_COMPLETION_TOKENS = 500 # Estimated average answer token count
BASE_LATENCY_SECONDS = 1.5 # Base latency for model loading/network delays

# --- Tokenizer Initialization ---
try:
    print("Initializing tokenizer for prompt analysis...")
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
    print("Tokenizer initialized.")
except Exception as e:
    print(f"Warning: Could not initialize tokenizer. Prompt token counts will be estimated. Error: {e}")
    tokenizer = None

# --- Performance Tracker ---
class MetricsTracker:
    """
    Tracks historical token generation speed to estimate response times.
    """
    def __init__(self, filepath):
        """
        Initialize the metrics tracker.

        Args:
            filepath (str): Path to save/load JSON metrics.
        """
        self.filepath = filepath
        self.metrics = {
            "total_runs": 0,
            "avg_tokens_per_second": 2.0  # Initial conservative estimate
        }
        self.load()

    def load(self):
        """Load metrics from JSON file if it exists."""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                self.metrics = json.load(f)
    
    def save(self):
        """Save current metrics to JSON file."""
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def update(self, new_eval_count, new_eval_duration_ns):
        """
        Update metrics based on a new run.

        Args:
            new_eval_count (int): Number of tokens generated.
            new_eval_duration_ns (int): Duration in nanoseconds for token generation.
        """
        if new_eval_duration_ns == 0:
            return

        # Calculate the speed for this run
        current_tokens_per_sec = new_eval_count / (new_eval_duration_ns / 1_000_000_000)
        
        # Update the average speed using a moving average to avoid outliers having too much impact
        total = self.metrics["total_runs"]
        avg = self.metrics["avg_tokens_per_second"]
        new_avg = (total * avg + current_tokens_per_sec) / (total + 1)
        
        self.metrics["avg_tokens_per_second"] = new_avg
        self.metrics["total_runs"] += 1
        self.save()

    def get_avg_speed(self):
        return self.metrics["avg_tokens_per_second"]

# Initialize tracker instance
tracker = MetricsTracker(METRICS_FILE_PATH)

# --- Core Functional Functions ---
def _build_prompt(json_detections, question):
    """
    Build a prompt string for the LLM based on JSON detections and a user question.

    Args:
        json_detections (str): JSON string of detection results.
        question (str): User question about the image.

    Returns:
        str: Formatted prompt string for LLM input.
    """
    detections_data = json.loads(json_detections)
    # The original is_sequence check is omitted here; add back if needed
    return (
        "You are an AI assistant that answers questions about an image based on structured detection data. "
        f"The image is {detections_data.get('image_height')}x{detections_data.get('image_width')} pixels, "
        f"captured at {detections_data.get('capture_time')}.\n"
        f"""
You are an expert visual analyst. Your task is to describe an image based on the provided JSON data.

Detected objects data:
{json_detections}

Rules:
1.Summarize First: For general questions like "What do you see?" or "Describe the image," provide a overall summary in a single sentence in final response. Only mention the main objects and the total count of items.
2.Do Not List All Details: Do not list the properties (like bbox, confidence, color, text, etc) of every single object unless the user asks for details. Your goal is to be concise initially.
3.Answer Specifics Directly: If the user asks about a specific object (e.g., "What color is the ESP32?" or "Tell me more about the components on the right"), use the detailed information from the JSON to provide a direct and detailed answer.
4.The (0,0) coordinate is the top-left corner of the image.

Based on these rules, answer the following user question:
Question: {question}
"""
    )

def predict_response_time(json_detections, question):
    """Predict LLM response time based on historical data."""
    prompt = _build_prompt(json_detections, question)
    
    # 1. Get average generation speed
    avg_speed = tracker.get_avg_speed()  # tokens/sec

    # 2. Estimate generation time
    prediction_time = ESTIMATED_COMPLETION_TOKENS / avg_speed

    # 3. (Optional) More accurately estimate prompt processing time
    if tokenizer:
        prompt_token_count = len(tokenizer.encode(prompt))
        # Assume prompt processing speed is 5x generation speed (usually faster)
        prompt_eval_time = prompt_token_count / (avg_speed * 1.5)
        prediction_time += prompt_eval_time

    return prediction_time + BASE_LATENCY_SECONDS

def stream_answer(json_detections, question, ollama_api_url, model_name):
    """Perform streaming API call and return final performance metrics."""
    prompt = _build_prompt(json_detections, question)
    payload = {"model": model_name, "prompt": prompt, "stream": True}
    
    final_answer_chunks = []
    run_metrics = {}

    try:
        response = requests.post(ollama_api_url, json=payload, timeout=1800, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                content = chunk.get("response", "")
                print(content, end='', flush=True)
                final_answer_chunks.append(content)
                
                if chunk.get("done"):
                    # get metrics from the last chunk
                    run_metrics['eval_count'] = chunk.get('eval_count', 0)
                    run_metrics['eval_duration'] = chunk.get('eval_duration', 0)
                    run_metrics['total_duration'] = chunk.get('total_duration', 0)
                    run_metrics['load_duration'] = chunk.get('load_duration', 0)
        
        full_response = "".join(final_answer_chunks)
        # update tracker with the final metrics
        if run_metrics.get('eval_count'):
            tracker.update(run_metrics['eval_count'], run_metrics['eval_duration'])

        return full_response, run_metrics

    except requests.exceptions.RequestException as e:
        print(f"\nError during API call: {e}")
        return f"Error: {str(e)}", {}