# backend/vision/ocr.py
import easyocr
import numpy as np

# Initialize a global EasyOCR Reader to avoid reloading the model on each call.
# Only English ('en') is recognized here. Add other languages like ['ch_sim', 'en'] if needed.
print("Initializing EasyOCR Reader...")
reader = easyocr.Reader(['en']) 
print("EasyOCR Reader initialized.")

def get_text_from_image(roi: np.ndarray) -> str:
    """
    Extract text from a region of interest (ROI) using EasyOCR.

    Args:
        roi (np.ndarray): Image region containing text in BGR format.

    Returns:
        str: Concatenated text recognized from the ROI. Returns an empty string if no text is detected.
    """
    if roi is None or roi.size == 0:
        return ""
        
    try:
        # Use EasyOCR's readtext method to extract text.
        # Each result is a list: [bounding_box, text, confidence]
        results = reader.readtext(roi)
        
        if not results:
            return ""
            
        # Concatenate all detected text segments into a single string
        recognized_texts = [res[1] for res in results]
        return " ".join(recognized_texts)
        
    except Exception as e:
        print(f"An error occurred during OCR processing: {e}")
        return ""