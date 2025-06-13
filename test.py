import cv2
import matplotlib.pyplot as plt

image_path = 'bus.jpg'  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not read image from {image_path}")
else:
    print(f"Image loaded successfully from {image_path}")
    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

