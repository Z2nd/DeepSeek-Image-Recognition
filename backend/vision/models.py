# backend/vision/models.py
import torch
from torchvision import models

def load_secondary_model(model_type='mobilenet_v3_small', num_classes=10):
    """加载用于二次分类的预训练模型。"""
    try:
        if model_type == 'mobilenet_v3_small':
            # 'pretrained' is deprecated, using 'weights' instead
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading secondary model: {str(e)}")
        return None