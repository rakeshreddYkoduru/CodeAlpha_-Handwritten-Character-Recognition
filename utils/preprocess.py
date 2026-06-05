import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path_or_buf):
    """
    Preprocess a raw image to match model input format (28x28 grayscale, normalized).
    """
    if isinstance(image_path_or_buf, str):
        img = cv2.imread(image_path_or_buf, cv2.IMREAD_GRAYSCALE)
    else:
        # Assume it's a buffer/PIL image
        img = np.array(Image.open(image_path_or_buf).convert('L'))

    # Thresholding to remove noise and make it binary
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours to crop to the character
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Add padding
        pad = 4
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1], w + 2*pad)
        h = min(img.shape[0], h + 2*pad)
        
        img = img[y:y+h, x:x+w]

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to 0-1
    img = img.astype('float32') / 255.0
    
    # Add channel dimension
    img = np.expand_dims(img, axis=0) # (1, 28, 28)
    
    return img

def get_emnist_mapping():
    # EMNIST Balanced mapping (47 classes)
    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    return list(mapping)
