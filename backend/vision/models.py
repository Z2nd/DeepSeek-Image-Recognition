# backend/vision/models.py
import torch
from torchvision import models

def load_secondary_model(model_type='mobilenet_v3_small', num_classes=10):
    """
    Load a pre-trained secondary classification model for further analysis.

    Args:
        model_type (str, optional): Type of model to load. Currently supports 'mobilenet_v3_small'. Defaults to 'mobilenet_v3_small'.
        num_classes (int, optional): Number of output classes for the classifier. Defaults to 10.

    Returns:
        torch.nn.Module | None: The loaded PyTorch model set to evaluation mode, or None if loading fails.
    """
    try:
        if model_type == 'mobilenet_v3_small':
            # Load pretrained MobileNetV3 small model from torchvision
            # Note: 'pretrained' is deprecated, using 'weights' instead
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # Replace the classifier's final layer to match num_classes
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        # Set the model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading secondary model: {str(e)}")
        return None