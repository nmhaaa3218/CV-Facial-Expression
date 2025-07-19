# Facial Expression Recognition System

## Project Overview

This project implements a state-of-the-art facial expression recognition system using deep learning techniques. The system can classify facial expressions into 7 different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

**Final Results:**
- **Accuracy:** 68.74%
- **GFLOPs:** 0.046421376
- **Efficiency:** 14.81 (Accuracy/GFLOPs ratio)

## Dataset

The system uses the **FER-2013** dataset, which contains:
- **Training set:** 28,709 grayscale images (48x48 pixels)
- **Public test set:** 3,589 images (used for validation)
- **Private test set:** 3,589 images (used for final evaluation)

All images are automatically registered and centered, with faces occupying similar amounts of space in each image.

## Architecture & Implementation

### Base Model: EfficientNet-B0

The core architecture is based on **EfficientNet-B0** with the following modifications:

1. **Input Layer Adaptation:** Modified the first convolutional layer to accept single-channel (grayscale) input instead of RGB
2. **Classifier Head:** Replaced the final classification layer to output 7 classes (emotions)
3. **Pretrained Weights:** Utilized ImageNet pretrained weights for better feature extraction

```python
class EfficientNetB0(expression_model):
    def __init__(self, number_of_class: int, in_channel: int = 1):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Adapt input layer for grayscale
        old_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=in_channel,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Replace classifier for 7 emotion classes
        in_feat = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_feat, number_of_class)
```

### Training Strategy

#### 1. Data Augmentation Techniques

Multiple data augmentation strategies were experimented with:

**Version 1 (Basic):** Standard normalization only
- Accuracy: 62.25%

**Version 2 (CRP):** Crop + Rotate + Flip
- Random horizontal flip (p=0.5)
- Random crop with 4-pixel padding
- Random rotation (±10 degrees)
- Accuracy: 66.66%

**Version 3 (CRP + Brightness/Contrast):** Enhanced color augmentation
- All CRP techniques
- ColorJitter with brightness=0.2, contrast=0.2
- Accuracy: 64.51%

**Version 4 (CRP + CutOut):** Occlusion-based augmentation
- All CRP techniques
- RandomErasing with p=0.5, scale=(0.01, 0.11)
- Accuracy: 65.22%

**Version 5 (MixUp + CutMix):** Advanced mixing techniques
- MixUp and CutMix data augmentation
- Accuracy: 64.69%

#### 2. Training Parameters

**Optimizer:** AdamW with parameter-wise weight decay
- Learning rate: 1e-3 (base), 1e-2 (max)
- Weight decay: 0.1
- Beta values: (0.9, 0.999)

**Scheduler:** OneCycleLR
- Warmup: 10% of training
- Cosine annealing
- Div factor: 10
- Final div factor: 1000

**Training Configuration:**
- Batch size: 64
- Epochs: 300 (with early stopping)
- Patience: 20 epochs
- Gradient clipping: 2.0
- Label smoothing: 0.1

#### 3. Ensemble Strategy

The final system uses an **ensemble of two EfficientNet-B0 models**:

1. **Model 1:** `model_efficientnetb0_v2_tuned_2.pth` (CRP augmentation)
   - Individual accuracy: 66.66%
   - Weight: 0.45

2. **Model 2:** `model_efficientnetb0_v5_tuned.pth` (MixUp + CutMix)
   - Individual accuracy: 64.69%
   - Weight: 0.55

**Ensemble Method:** Weighted Average of Softmax Probabilities
```python
def weighted_average_probs(models, img_tensor, weights):
    probs_list = []
    for model in models:
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    
    probs_array = np.stack(probs_list, axis=0)
    weights = np.array(weights).reshape(-1, 1, 1)
    weighted_probs = (probs_array * weights).sum(axis=0)
    return weighted_probs[0], probs_list
```

## Performance Analysis

### Individual Model Results

| Model | Augmentation | Accuracy | GFLOPs | Efficiency |
|-------|-------------|----------|--------|------------|
| EfficientNet-B0 | None | 62.25% | 0.046 | 13.53 |
| EfficientNet-B0 | CRP | 66.66% | 0.046 | 14.49 |
| EfficientNet-B0 | CRP + Brightness/Contrast | 64.51% | 0.046 | 14.02 |
| EfficientNet-B0 | CRP + CutOut | 65.22% | 0.046 | 14.18 |
| EfficientNet-B0 | MixUp + CutMix | 64.69% | 0.046 | 14.06 |

### Ensemble Results

**Final Ensemble Performance:**
- **Accuracy:** 68.74%
- **GFLOPs:** 0.046421376 (approximately 2x single model)
- **Efficiency:** 14.81 (Accuracy/GFLOPs)

### Ablation Study

| Method | CRP | Brightness/Contrast | CutOut | MixUp/CutMix | Accuracy |
|--------|-----|-------------------|--------|--------------|----------|
| Baseline | N | N | N | N | 62.25% |
| Version 2 | Y | N | N | N | 66.66% |
| Version 3 | Y | Y | N | N | 64.51% |
| Version 4 | Y | N | Y | N | 65.22% |
| Version 5 | Y | N | N | Y | 64.69% |
| **Ensemble** | **Y** | **N** | **N** | **Y** | **68.74%** |

## Computational Efficiency

The system achieves excellent computational efficiency:

- **Single Model GFLOPs:** 0.046
- **Ensemble GFLOPs:** 0.046421376
- **Efficiency Ratio:** 14.81

This efficiency is achieved through:
1. **EfficientNet-B0 architecture:** Designed for mobile/edge deployment
2. **Grayscale input:** Reduces computational overhead
3. **Optimized ensemble:** Only 2 models instead of larger ensembles
4. **Model compression:** Efficient parameter usage

## Deployment: Streamlit Web Application

The project includes a **Streamlit web application** (`streamlit_app.py`) that provides:

### Features:
1. **Image Upload:** Users can upload face images for emotion analysis
2. **Webcam Integration:** Real-time emotion detection from webcam feed
3. **Face Detection:** YOLO-based face detection with confidence scores
4. **Multi-face Support:** Processes multiple faces in a single image
5. **Ensemble Predictions:** Shows both ensemble and individual model results
6. **Visualization:** Displays emotion probabilities as bar charts

### Technical Implementation:
- **Face Detection:** YOLOv12n-face model for robust face localization
- **Image Preprocessing:** Automatic cropping, resizing, and normalization
- **Real-time Processing:** Optimized for live webcam feeds
- **User-friendly Interface:** Intuitive design with clear results display

## File Structure

```
CV-Facial-Expression/
├── Facial_Expression_Recognition_2025.ipynb  # Main training notebook
├── model.py                                  # Model architecture definitions
├── streamlit_app.py                         # Web application
├── requirements.txt                          # Python dependencies
├── model_efficientnetb0_v2_tuned_2.pth      # Ensemble model 1
├── model_efficientnetb0_v5_tuned.pth        # Ensemble model 2
├── yolov12n-face.pt                         # Face detection model
├── notebook.pdf                             # Project documentation
└── README.md                                # This file
```

## Installation & Usage Instructions

### 1. Install Dependencies

Install all required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- **PyTorch 2.5.1** - Deep learning framework
- **TorchVision 0.20.1** - Computer vision utilities
- **Streamlit 1.40.1** - Web application framework
- **Ultralytics 8.3.167** - YOLO face detection
- **OpenCV 4.11.0** - Computer vision library
- **NumPy 1.26.4** - Numerical computing
- **Pandas 2.2.3** - Data manipulation
- **Pillow 11.0.0** - Image processing
- **fvcore 0.1.5** - FLOPs calculation

### 2. Running the Web Application

```bash
streamlit run streamlit_app.py
```

**Features:**
- Upload face images for emotion analysis
- Real-time webcam emotion detection
- Multi-face detection and classification
- Ensemble and individual model predictions
- Visual probability charts

### 3. Training/Evaluation

1. **Open the Jupyter notebook:**
```bash
jupyter notebook Facial_Expression_Recognition_2025.ipynb
```

2. **Follow the notebook cells for:**
   - Data loading and preprocessing
   - Model training with different augmentations
   - Ensemble evaluation
   - Performance analysis

## Key Innovations

1. **Adaptive Input Layer:** Modified EfficientNet for grayscale facial images
2. **Progressive Augmentation:** Systematic exploration of data augmentation techniques
3. **Efficient Ensemble:** Lightweight 2-model ensemble with optimal weighting
4. **Real-time Deployment:** Web application with face detection and emotion classification
5. **Computational Efficiency:** Optimized for edge deployment with low GFLOPs

## Limitations & Future Work

### Current Limitations:
1. **Dataset Bias:** FER-2013 may not represent all demographic groups equally
2. **Emotion Granularity:** Limited to 7 discrete emotion categories
3. **Context Dependency:** May not capture complex emotional states
4. **Lighting Sensitivity:** Performance may vary under different lighting conditions

### Future Improvements:
1. **Multi-modal Fusion:** Incorporate audio and text for better emotion understanding
2. **Temporal Modeling:** Use video sequences for more accurate emotion detection
3. **Domain Adaptation:** Improve performance across different ethnicities and ages
4. **Real-time Optimization:** Further reduce inference time for mobile deployment
5. **Continuous Emotions:** Support for continuous emotion space instead of discrete categories

## Technical Specifications

- **Framework:** PyTorch 2.0+
- **Architecture:** EfficientNet-B0 (modified)
- **Input:** 48x48 grayscale images
- **Output:** 7-class emotion classification
- **Training:** AdamW optimizer, OneCycleLR scheduler
- **Deployment:** Streamlit web application with YOLO face detection

## Authors

- **Manh Ha Nguyen** (a1840406)
- **Le Thuy An Phan** (a1874923)

**Subject:** Computer Vision  
**Year:** 2025  
**Competition:** Facial Expression Recognition/Classification

---

*This project demonstrates advanced deep learning techniques for facial expression recognition, achieving competitive accuracy while maintaining computational efficiency suitable for real-world deployment.* 