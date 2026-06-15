import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path_or_buf):
    """
    Preprocess a raw image to match model input format (28x28 grayscale, normalized).
    Must match training preprocessing: EMNIST images are transposed, so we
    transpose the input to match what the model learned.
    """
    if isinstance(image_path_or_buf, str):
        img = cv2.imread(image_path_or_buf, cv2.IMREAD_GRAYSCALE)
    else:
        # Assume it's a buffer/PIL image
        img = np.array(Image.open(image_path_or_buf).convert('L'))

    # Invert: canvas sends white-on-black, but ensure we have white char on black bg
    # EMNIST uses white-on-black (character pixels = 255, background = 0)
    # Check if image is mostly white (>128 mean) indicating black-on-white
    if np.mean(img) > 128:
        img = 255 - img

    # Thresholding to remove noise
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Find contours to crop to the character
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Crop to bounding box
        img = img[y:y+h, x:x+w]

    # Make the image square by padding with black
    h, w = img.shape
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        img = np.pad(img, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    elif w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        img = np.pad(img, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

    # Add border padding (like EMNIST has ~2px border)
    img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # CRITICAL: EMNIST images are transposed. The model was trained on transposed
    # images (torchvision applies this automatically). We must transpose user input
    # to match the training data orientation.
    img = np.transpose(img)

    # Normalize to 0-1
    img = img.astype('float32') / 255.0
    
    # Apply the SAME normalization used during training
    # Training used: transforms.Normalize((0.1751,), (0.3332,))
    img = (img - 0.1751) / 0.3332
    
    # Add channel dimension
    img = np.expand_dims(img, axis=0)  # (1, 28, 28)
    
    return img

def get_emnist_mapping():
    # EMNIST Balanced mapping (47 classes)
    # This maps class index to character
    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    return list(mapping)
