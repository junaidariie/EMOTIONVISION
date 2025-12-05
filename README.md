# Face Expression Recognition with ResNet18

A production-ready facial emotion recognition web application built with PyTorch, Streamlit, and transfer learning using ResNet18. Detects 5 emotions (Angry, Fear, Happy, Sad, Surprise) from uploaded face images with confidence scores and emoji visualization.

APP LINK : https://emotionvision-c4fw8qjwfkesqpabqiaavt.streamlit.app/

## Features

- **Transfer Learning**: Fine-tuned ResNet18 backbone with frozen early layers and trainable final layers
- **Real-time Web UI**: Streamlit app for instant image uploads and predictions
- **High Accuracy**: Achieves ~84% validation accuracy on custom emotion dataset
- **Production Ready**: Model artifacts saved, GPU/CPU compatible, clean inference pipeline
- **Interactive Results**: Confidence metrics, progress bars, emoji feedback, side-by-side image display

## Project Structure

```
FaceExpressionResnet/
├── artifacts/
│   └── FaceExpressionResnet_18.pth     # Trained model weights
├── predict_helper_cv.py                 # Model loading & inference
├── app.py                               # Streamlit web application
├── FaceExpressionResnet.ipynb           # Training notebook (data prep + model training)
├── requirements.txt                      # Dependencies
└── README.md
```

## Quick Start

1. **Clone & Install**

```bash
git clone <your-repo>
cd FaceExpressionResnet
pip install -r requirements.txt
```

2. **Download Model** (or train using notebook)

Ensure `artifacts/FaceExpressionResnet_18.pth` exists

3. **Run Web App**

```bash
streamlit run app.py
```

Open `http://localhost:8501` and upload face images!

## Model Architecture

**FaceExpressionResNet** - Custom ResNet18 head:

```
ResNet18(pretrained)
├── Freeze layers 1-3 (feature extraction)
├── Unfreeze layer4 + BatchNorm (fine-tuning)
└── Custom FC: Dropout(0.4) → Linear(512→5)
```

- Input: 128x128 RGB faces
- Output: 5-class softmax (Angry/Fear/Happy/Sad/Surprise)
- ImageNet normalization: `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`

## Training Results

| Model | Train Acc | Val Acc | Epochs | Dataset |
|-------|-----------|---------|--------|---------|
| Custom CNN | 68% | 65% | 30 | 30k train / 10k val |
| **ResNet18** | **95%** | **84%** | **15** | **30k train / 10k val** |

**Confusion Matrix Highlights**: Strong Happy/Surprise detection, moderate Angry/Fear/Sad.

## Training (Optional)

1. Run `FaceExpressionResnet.ipynb` in Google Colab
2. Downloads Kaggle "human-face-emotions" dataset
3. Auto-splits: 6000 train + 2000 val per class (5 classes)
4. Trains → Saves `FaceExpressionResnet_18.pth`

## Usage Example

```python
from predict_helper_cv import predict_image

result = predict_image("test_image.jpg")
# {'emotion': 'Happy', 'confidence': 0.92}
```

**Streamlit Demo**:

1. Upload JPG/PNG (face image)
2. View original + prediction side-by-side
3. See emotion + confidence % + emoji
4. Progress bar shows model certainty

## Improvements Made

- **Data**: Organized train/val folders (RAF-DB style)
- **Augmentation**: Horizontal flip, rotation, color jitter
- **Optimizer**: AdamW + weight decay + label smoothing
- **Transfer Learning**: Only train final layers (efficient)
- **Deployment**: Clean separation (model/inference/UI)

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Supported Emotions

| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Fear | 😨 |
| Happy | 😊 |
| Sad | 😢 |
| Surprise | 😲 |

