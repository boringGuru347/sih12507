"# Kolam Design Classifier 🎨

A deep learning project that classifies different types of Kolam designs using EfficientNet-B0 with transfer learning.

## 🎯 Project Overview

This project implements a complete pipeline for training a Kolam design classifier using:
- **EfficientNet-B0** as the base model with ImageNet pretrained weights
- **Transfer Learning** to adapt the model for Kolam classification
- **HuggingFace Datasets** to load the `ayshthkr/kolam_dataset`
- **PyTorch** for deep learning implementation

## 📁 Project Structure

```
Kolam Classifier/
├── kolam_classifier.py    # Main training script
├── inference.py          # Inference script for predictions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── image.png           # Sample image for testing
├── test.png           # Another test image
└── images/            # Directory with kolam images
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9 or later
- CUDA-compatible GPU (optional but recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Training the Model

Run the main training script:
```bash
python kolam_classifier.py
```

This will:
- Load the HuggingFace dataset `ayshthkr/kolam_dataset`
- Split data into 70% train, 15% validation, 15% test
- Apply data augmentation and preprocessing
- Train EfficientNet-B0 for 10 epochs
- Evaluate on test set
- Save the trained model as `efficientnet_kolam.pth`
- Generate training plots and confusion matrix

### 2. Making Predictions

After training, use the inference script:
```bash
python inference.py
```

This will load the trained model and make predictions on sample images.

### 3. Custom Image Prediction

You can modify the `inference.py` script to predict on your own images:

```python
predictor = KolamPredictor()
result = predictor.predict_image('path/to/your/image.jpg')
```

## 📊 Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 224×224 pixels
- **Output Classes**: 8 Kolam categories
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss

## 🔄 Data Processing Pipeline

### Training Data Augmentation:
- Random horizontal flip (50%)
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Random crop after resize

### Preprocessing:
- Resize to 224×224 pixels
- Normalize with ImageNet mean/std values

## 📈 Training Process

The training pipeline includes:
1. **Dataset Loading**: Load from HuggingFace and create custom splits
2. **Data Augmentation**: Apply transformations for robustness
3. **Model Creation**: EfficientNet-B0 with custom classifier head
4. **Training Loop**: 10 epochs with validation monitoring
5. **Early Stopping**: Save best model based on validation accuracy
6. **Evaluation**: Comprehensive metrics on test set

## 📊 Evaluation Metrics

The model is evaluated using:
- **Overall Accuracy**
- **Per-class Precision, Recall, F1-score**
- **Confusion Matrix**
- **Training/Validation Loss and Accuracy Plots**

## 💾 Model Saving

The trained model is saved with:
- Model state dictionary
- Class names
- Number of classes

This allows easy loading for inference without retraining.

## 🔍 Features

### Training Script (`kolam_classifier.py`)
- ✅ Automatic dataset loading and splitting
- ✅ Data augmentation and preprocessing
- ✅ Transfer learning with EfficientNet-B0
- ✅ Training progress monitoring
- ✅ Validation during training
- ✅ Comprehensive evaluation
- ✅ Visualization of results
- ✅ Model saving

### Inference Script (`inference.py`)
- ✅ Load trained model
- ✅ Predict single images
- ✅ Top-3 predictions with confidence scores
- ✅ Visualization of results
- ✅ Error handling

## 🎨 Kolam Categories

The model classifies 8 different types of Kolams:
(Categories will be displayed after loading the dataset)

## 🔧 Customization

### Hyperparameters
You can modify training parameters in `kolam_classifier.py`:
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Batch size for training
- Image augmentation parameters

### Model Architecture
- Change base model (e.g., ResNet, DenseNet)
- Modify classifier head
- Adjust dropout rates

## 📋 System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (optional)
- **Storage**: 2GB free space for dataset and models

## 🐛 Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce batch size
2. **Dataset loading error**: Check internet connection
3. **Import errors**: Ensure all dependencies are installed

### Performance Tips:
- Use GPU for faster training
- Adjust batch size based on available memory
- Experiment with learning rate scheduling

## 📜 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please create an issue in the repository.

---

**Happy Kolam Classification! 🎨✨**" 
