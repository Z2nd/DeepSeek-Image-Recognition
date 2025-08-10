# backend/llm/interaction.py
import json
import requests
import time
import psutil
import re

def answer_question_with_deepseek(json_detections, question, ollama_api_url, model_name, max_retries=3, retry_delay=2):
    """Generate answers to questions using DeepSeek."""
    metrics = {"question": question, "start_time": time.time(), "status": "pending"}
    process = psutil.Process()
    
    try:
        prompt = (
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

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                payload = {"model": model_name, "prompt": prompt, "stream": False}
                response = requests.post(ollama_api_url, json=payload, timeout=1200)
                response.raise_for_status()
                
                response_json = response.json()
                full_response = response_json.get("response", "No answer generated.").strip()
                final_answer = re.sub(r'<think>.*?</think>', ' ', full_response, flags=re.DOTALL)
                
                metrics.update({
                    "inference_time": time.time() - start_time,
                    "total_duration": response_json.get("total_duration", 0),
                    "load_duration": response_json.get("load_duration", 0),
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "retry_attempts": attempt, "status": "success"
                })
                return final_answer, full_response, metrics
            except (requests.exceptions.RequestException) as e:
                print(f"API attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        raise ConnectionError(f"Failed to get response from Ollama API after {max_retries} attempts.")

    except Exception as e:
        metrics.update({"status": "failed", "error": str(e)})
        return f"Error: {str(e)}", {}, metrics