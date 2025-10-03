# backend/vision/features.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from backend import config

def get_dominant_color(image, mask=None, color_space='HSL', n_clusters=1):
    """
    Extract the dominant color from a region of an image using K-Means clustering.

    Args:
        image (np.ndarray): Input image in BGR format.
        mask (np.ndarray, optional): Binary mask specifying the region of interest. Defaults to None.
        color_space (str, optional): Color space to use ('HSV' or 'HSL'). Defaults to 'HSL'.
        n_clusters (int, optional): Number of clusters for K-Means. Defaults to 1.

    Returns:
        tuple[list[float], str]: The dominant color as a list of floats [C1, C2, C3], and its approximate color name.
    """
    # Extract pixels from the region of interest
    if mask is not None:
        pixels = image[mask > 0]
    else:
        pixels = image.reshape(-1, 3)

    if len(pixels) == 0:
        return [0, 0, 0], 'unknown'

    # Convert color space if needed
    if color_space == 'HSV':
        pixels_converted = cv2.cvtColor(pixels[np.newaxis, :, :], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    elif color_space == 'HSL':
        pixels_converted = cv2.cvtColor(pixels[np.newaxis, :, :], cv2.COLOR_BGR2HLS).reshape(-1, 3)
    else:
        raise ValueError("Unsupported color space. Use 'HSV' or 'HSL'.")

    # Apply K-Means clustering to find dominant color
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(pixels_converted)
    dominant_color = kmeans.cluster_centers_[0].astype(float).tolist()
    
    # Convert hue to 0-360 scale and determine approximate color name
    hue = dominant_color[0] * 2  # OpenCV Hue (0-180) to 0-360
    color_name = 'unknown'
    for (h_min, h_max), name in config.COLOR_NAMES.items():
        if h_min <= hue <= h_max:
            color_name = name
            break
    
    # Adjust for grayscale/near-neutral colors
    saturation = dominant_color[1]
    value_or_lightness = dominant_color[2]
    if (color_space == 'HSV' and (saturation < 25 or value_or_lightness < 25)) or \
       (color_space == 'HSL' and saturation < 25):
        if value_or_lightness < 50:
            color_name = 'black'
        elif value_or_lightness < 200:
            color_name = 'gray'
        else:
            color_name = 'white'
            
    return dominant_color, color_name