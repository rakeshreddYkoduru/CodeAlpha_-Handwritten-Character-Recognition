from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import base64
import io
from PIL import Image
import sys
import os

# Add src and utils to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import CharacterCNN
from utils.preprocess import preprocess_image, get_emnist_mapping

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 47 # EMNIST Balanced
model = CharacterCNN(num_classes=num_classes).to(device)

model_path = os.path.join(os.path.dirname(__file__), '../models/character_cnn.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
else:
    print("Warning: Model weights not found. Predictions will be random.")

mapping = get_emnist_mapping()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    
    # Decode base64 image
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = io.BytesIO(image_bytes)
    
    # Preprocess
    processed_img = preprocess_image(image) # (1, 28, 28)
    
    # Convert to tensor
    tensor_img = torch.from_numpy(processed_img).unsqueeze(0).to(device) # (1, 1, 28, 28)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    char = mapping[predicted.item()]
    conf = confidence.item()
    
    return jsonify({
        'character': char,
        'confidence': float(conf),
        'all_predictions': {mapping[i]: float(probabilities[0][i]) for i in range(len(mapping))}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
