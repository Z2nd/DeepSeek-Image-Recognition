# backend/vision/ocr.py
import easyocr
import numpy as np

# 初始化OCR Reader。我们在这里将其创建为全局变量，
# 这样模型只需要加载一次，可以避免在每次调用时都重新加载，从而提高效率。
# ['en'] 表示我们只识别英文。如果您需要识别中文，可以改成 ['ch_sim', 'en']
print("Initializing EasyOCR Reader...")
reader = easyocr.Reader(['en']) 
print("EasyOCR Reader initialized.")

def get_text_from_image(roi: np.ndarray) -> str:
    """
    使用EasyOCR从图像区域(ROI)中提取文本。
    
    Args:
        roi (np.ndarray): 包含待识别文本的图像区域 (BGR格式)。
        
    Returns:
        str: 识别出的所有文本拼接成的单个字符串，如果没有识别到则返回空字符串。
    """
    if roi is None or roi.size == 0:
        return ""
        
    try:
        # EasyOCR的 readtext 方法返回一个结果列表，每个结果包含 [bbox, text, confidence]
        results = reader.readtext(roi)
        
        if not results:
            return ""
            
        # 我们将所有识别到的文本片段用空格连接成一个字符串
        recognized_texts = [res[1] for res in results]
        return " ".join(recognized_texts)
        
    except Exception as e:
        print(f"An error occurred during OCR processing: {e}")
        return ""