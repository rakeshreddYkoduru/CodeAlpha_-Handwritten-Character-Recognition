# Handwritten Character Recognition System

A state-of-the-art Handwritten Character Recognition (HCR) system built with PyTorch and Flask.

## 🚀 Features
- **Deep Learning Model**: Convolutional Neural Network (CNN) trained on MNIST/EMNIST.
- **Premium Web Interface**: Modern, responsive drawing canvas with real-time confidence feedback.
- **Image Processing**: Robust preprocessing including automated thresholding, cropping, and normalization.
- **Future Ready**: Includes CRNN architecture for extending to full word and sentence recognition.

## 📂 Project Structure
- `src/model.py`: CNN architecture definitions.
- `src/train.py`: Training pipeline using EMNIST Balanced (47 classes).
- `src/crnn_extension.py`: Sequence modeling architecture for words.
- `utils/preprocess.py`: Image handling and cleaning logic.
- `app/server.py`: Flask API serving the model and UI.
- `app/static/`: Frontend assets (HTML, CSS, JS).

## 🛠️ Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
The EMNIST dataset files are too large for GitHub (>25MB), so they are downloaded separately:
```bash
python download_data.py
```
*This downloads the EMNIST Balanced dataset (~85MB compressed) into the `data/` folder.*

### 3. Train the Model
```bash
python src/train.py
```
*This will train the CNN and save weights to `models/character_cnn.pth`.*

### 4. Launch the Application
```bash
python app/server.py
```
Open `http://localhost:5000` in your browser.


## 🧠 Sequence Modeling (CRNN)
The project is designed to be extendable. The `src/crnn_extension.py` file contains the architecture for word recognition. To implement words:
1. Use a dataset like IAM or MJ Synthetic.
2. Use the provided CRNN architecture.
3. Train using `torch.nn.CTCLoss`.
