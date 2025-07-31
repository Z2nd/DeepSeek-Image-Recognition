# backend/llm/interaction.py
import json
import requests
import time
import psutil
import re

def answer_question_with_deepseek(json_detections, question, ollama_api_url, model_name, max_retries=3, retry_delay=2):
    """使用DeepSeek生成问题的答案。"""
    metrics = {"question": question, "start_time": time.time(), "status": "pending"}
    process = psutil.Process()
    
    try:
        detections_data = json.loads(json_detections)
        is_sequence = "frames" in detections_data

        if is_sequence:
            prompt = (
                "You are an AI assistant analyzing a sequence of image frames. "
                f"Analyze the objects and their changes over time based on the following data:\n"
                f"Data: {json_detections}\n\n"
                f"Please answer the following user question in concise, natural descriptive language, one sentence:\n"
                f"Question: {question}"
            )
        else:
            prompt = (
                "You are an AI assistant that answers questions about an image based on structured detection data. "
                f"The image is {detections_data.get('image_height')}x{detections_data.get('image_width')} pixels, "
                f"captured at {detections_data.get('capture_time')}.\n"
                "Detected objects data:\n"
                f"{json_detections}\n"
                "Answer the following question in concise, natural language:\n"
                f"Question: {question}"
            )

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                payload = {"model": model_name, "prompt": prompt, "stream": False}
                response = requests.post(ollama_api_url, json=payload, timeout=1200)
                response.raise_for_status()
                
                response_json = response.json()
                final_answer = response_json.get("response", "No answer generated.").strip()
                
                metrics.update({
                    "inference_time": time.time() - start_time,
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "retry_attempts": attempt, "status": "success"
                })
                return final_answer, response_json, metrics
            except (requests.exceptions.RequestException) as e:
                print(f"API attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        raise ConnectionError(f"Failed to get response from Ollama API after {max_retries} attempts.")

    except Exception as e:
        metrics.update({"status": "failed", "error": str(e)})
        return f"Error: {str(e)}", {}, metrics