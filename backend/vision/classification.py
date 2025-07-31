# backend/vision/classification.py
import cv2
import torch
from torchvision import transforms

def preprocess_roi_for_classification(roi, target_size=(224, 224)):
    """为二次分类预处理ROI。"""
    try:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(roi_rgb).unsqueeze(0)
    except Exception as e:
        print(f"Error preprocessing ROI: {str(e)}")
        return None

def secondary_classification(roi, model, class_names):
    """对ROI进行二次分类以预测子类别。"""
    try:
        if model is None or roi is None or not class_names:
            return 'unknown', 0.0
        
        roi_tensor = preprocess_roi_for_classification(roi)
        if roi_tensor is None:
            return 'unknown', 0.0
        
        with torch.no_grad():
            outputs = model(roi_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            sub_class = class_names[predicted_idx.item()]
        
        return sub_class, float(confidence)
    except Exception as e:
        print(f"Error in secondary classification: {str(e)}")
        return 'unknown', 0.0