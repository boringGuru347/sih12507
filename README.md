"# Kolam Design Classifier 🎨

A deep learning project that classifies different types of Kolam designs using a few-shot learning approach with EfficientNet-B0 backbone.

## 🎯 Project Overview

This project implements a few-shot learning classifier for Kolam designs using:
- **Prototypical Networks** for few-shot learning
- **EfficientNet-B0** as the feature extractor backbone
- **HuggingFace Datasets** to load the `ayshthkr/kolam_dataset`
- **PyTorch** for deep learning implementation

## 📁 Project Structure

```
Kolam Classifier/
├── few_shot_kolam_classifier.py    # Few-shot classifier implementation
├── few_shot_kolam_classifier.pth   # Trained model weights
├── kolam_classifier_wrapper.py     # Wrapper class for easy usage
├── inference.py                    # Inference script for predictions
├── requirements_fewshot.txt        # Python dependencies
└── README.md                      # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9 or later
- CUDA-compatible GPU (optional but recommended)

### Install Dependencies
```bash
pip install -r requirements_fewshot.txt
```

## 🚀 Usage

### 1. Using the Pre-trained Model

The project includes a pre-trained few-shot Kolam classifier. Use the wrapper class for easy predictions:

```python
from kolam_classifier_wrapper import KolamClassifier
from PIL import Image

# Initialize classifier
classifier = KolamClassifier()

# Load and predict on an image
image = Image.open('your_kolam_image.jpg')
result = classifier.predict_kolam_type(image)

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 2. Using the Inference Script

Run the inference script directly:
```bash
python inference.py
```

This will load the trained model and make predictions on sample images.

### 3. Training a New Model

If you want to train your own model, use the few-shot classifier:
```bash
python few_shot_kolam_classifier.py
```

## 📊 Model Architecture

- **Architecture**: Prototypical Networks (Few-shot Learning)
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Feature Dimension**: 1280 → 256 (with projection layer)
- **Input Size**: 224×224 pixels
- **Output Classes**: 8 Kolam categories
- **Learning Approach**: Few-shot learning with prototypical networks

## 🔄 Data Processing Pipeline

### Few-shot Learning Setup:
- **Support Set**: Small number of examples per class for prototypes
- **Query Set**: Test examples for classification
- **Episode-based Training**: Simulates few-shot scenarios during training

### Preprocessing:
- Resize to 224×224 pixels
- Normalize with ImageNet mean/std values
- Data augmentation during training (horizontal flip, rotation, color jitter)

## 📈 Training Process

The few-shot learning pipeline includes:
1. **Dataset Loading**: Load from HuggingFace `ayshthkr/kolam_dataset`
2. **Episode Creation**: Generate training episodes with support/query sets
3. **Prototype Learning**: Learn class prototypes from support examples
4. **Distance-based Classification**: Classify queries based on prototype distances
5. **Model Saving**: Save trained prototypical network weights

## 📊 Evaluation Metrics

The few-shot model is evaluated using:
- **Episode-based Accuracy**: Performance on few-shot episodes
- **Confidence Scores**: Probability distributions for predictions
- **Class-wise Performance**: Individual class prediction accuracy

## 💾 Model Loading

The pre-trained model (`few_shot_kolam_classifier.pth`) includes:
- Prototypical network state dictionary
- Class names mapping
- Feature extraction backbone weights

Load easily using the wrapper class for immediate inference.

## 🔍 Features

### Few-Shot Classifier (`few_shot_kolam_classifier.py`)
- ✅ Prototypical Networks implementation
- ✅ EfficientNet-B0 feature extraction
- ✅ Episode-based training
- ✅ Few-shot learning capabilities
- ✅ Model saving and loading

### Classifier Wrapper (`kolam_classifier_wrapper.py`)
- ✅ Easy-to-use interface for predictions
- ✅ Confidence score calculation
- ✅ Class name mapping
- ✅ Error handling and validation

### Inference Script (`inference.py`)
- ✅ Load pre-trained model
- ✅ Predict on sample images
- ✅ Confidence scores and probabilities
- ✅ Visual results display

## 🎨 Kolam Categories

The model classifies 8 different types of Kolams:
- **geometric**: Geometric patterns
- **butterfly**: Butterfly-shaped designs
- **pulli**: Dot-based kolams
- **flower**: Floral patterns
- **naga**: Snake-inspired designs
- **sikku**: Line-based interlaced patterns
- **kamal**: Lotus-inspired designs
- **traditional**: Classical traditional patterns

## 🔧 Customization

### Model Parameters
You can modify parameters in `few_shot_kolam_classifier.py`:
- `feature_dim`: Feature dimension for embeddings
- `backbone`: Feature extraction backbone model
- Episode configuration for training

### Wrapper Configuration
Modify `kolam_classifier_wrapper.py` for:
- Custom class mappings
- Confidence thresholds
- Input preprocessing options

## 📋 System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (optional)
- **Storage**: 2GB free space for dataset and models

## 🐛 Troubleshooting

### Common Issues:
1. **Model loading error**: Ensure `few_shot_kolam_classifier.pth` exists
2. **Import errors**: Install dependencies with `pip install -r requirements_fewshot.txt`
3. **CUDA issues**: Model automatically uses CPU if CUDA unavailable
4. **Image format errors**: Ensure images are in supported formats (JPG, PNG)

### Performance Tips:
- Use GPU for faster inference
- PIL Image objects work best for predictions
- Ensure images are properly formatted before prediction

## 📜 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please create an issue in the repository.

---

**Happy Kolam Classification! 🎨✨**" 
