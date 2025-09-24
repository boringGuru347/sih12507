# Kolam Classifier - Binary Classification (Sikku vs Non-Sikku)"# Kolam Design Classifier 🎨



A deep learning project for classifying traditional South Indian Kolam patterns into Sikku and Non-Sikku categories using PyTorch and computer vision techniques.A deep learning project that classifies different types of Kolam designs using a few-shot learning approach with EfficientNet-B0 backbone.



## 🎯 Project Overview## 🎯 Project Overview



This project implements a binary classifier to distinguish between two types of Kolam designs:This project implements a few-shot learning classifier for Kolam designs using:

- **Sikku Kolam**: Continuous line patterns- **Prototypical Networks** for few-shot learning

- **Non-Sikku Kolam**: Discontinuous line patterns- **EfficientNet-B0** as the feature extractor backbone

- **HuggingFace Datasets** to load the `ayshthkr/kolam_dataset`

## 🚀 Key Features- **PyTorch** for deep learning implementation



- **High Accuracy**: 99.8% accuracy on test dataset## 📁 Project Structure

- **Robust Preprocessing**: 9-step OpenCV preprocessing pipeline

- **Data Augmentation**: Comprehensive augmentation for better generalization```

- **Production Ready**: Complete training and testing pipelineKolam Classifier/

- **Balanced Performance**: Excellent performance on both classes despite class imbalance├── few_shot_kolam_classifier.py    # Few-shot classifier implementation

├── few_shot_kolam_classifier.pth   # Trained model weights

## 📊 Performance Metrics├── kolam_classifier_wrapper.py     # Wrapper class for easy usage

├── inference.py                    # Inference script for predictions

- **Overall Accuracy**: 99.8% (629/630 correct predictions)├── requirements_fewshot.txt        # Python dependencies

- **Non-Sikku Class**: 93.8% recall, 100% precision└── README.md                      # This file

- **Sikku Class**: 100% recall, 99.8% precision```

- **F1-Score**: 98.3% overall

## 🔧 Installation & Setup

## 🛠️ Technical Stack

### Prerequisites

- **Framework**: PyTorch- Python 3.9 or later

- **Architecture**: EfficientNet-B0 backbone with custom classifier- CUDA-compatible GPU (optional but recommended)

- **Preprocessing**: OpenCV-based image enhancement

- **Data Augmentation**: Rotations, flips, perspective transforms, color jitter### Install Dependencies

- **Training**: Early stopping, train/validation split (80/20)```bash

pip install -r requirements_fewshot.txt

## 📁 Project Structure```



```## 🚀 Usage

Kolam Classifier/

├── optimized_kolam_trainer.py    # Main training script### 1. Using the Pre-trained Model

├── test_binary_classifier.py     # Testing and evaluation script

├── preprocessing.py               # Image preprocessing pipelineThe project includes a pre-trained few-shot Kolam classifier. Use the wrapper class for easy predictions:

├── binary_training_curves.png    # Training visualization

├── kolam_dataset/                 # Dataset folder```python

│   ├── Sikku/                     # Sikku kolam images (664 images)from kolam_classifier_wrapper import KolamClassifier

│   └── Non - Sikku/               # Non-Sikku kolam images (81 images)from PIL import Image

└── .gitignore                     # Git ignore file

```# Initialize classifier

classifier = KolamClassifier()

## 🏃‍♂️ Quick Start

# Load and predict on an image

### Prerequisitesimage = Image.open('your_kolam_image.jpg')

```bashresult = classifier.predict_kolam_type(image)

pip install torch torchvision opencv-python pillow matplotlib seaborn scikit-learn numpy

```print(f"Predicted class: {result['predicted_class']}")

print(f"Confidence: {result['confidence']:.2%}")

### Training the Model```

```bash

python optimized_kolam_trainer.py### 2. Using the Inference Script

```

Run the inference script directly:

### Testing the Model```bash

```bashpython inference.py

python test_binary_classifier.py```

```

This will load the trained model and make predictions on sample images.

## 📈 Training Process

### 3. Training a New Model

1. **Data Loading**: Loads images from Sikku and Non-Sikku folders

2. **Preprocessing**: Applies 9-step OpenCV enhancement pipelineIf you want to train your own model, use the few-shot classifier:

3. **Augmentation**: Random rotations, flips, perspective transforms```bash

4. **Training**: EfficientNet-B0 with custom classifier headpython few_shot_kolam_classifier.py

5. **Validation**: 80/20 train/validation split```

6. **Early Stopping**: Stops training when validation accuracy plateaus

## 📊 Model Architecture

## 🔬 Preprocessing Pipeline

- **Architecture**: Prototypical Networks (Few-shot Learning)

The preprocessing module applies the following steps:- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)

1. Noise reduction (Gaussian blur)- **Feature Dimension**: 1280 → 256 (with projection layer)

2. Histogram equalization (CLAHE)- **Input Size**: 224×224 pixels

3. Gaussian blur- **Output Classes**: 8 Kolam categories

4. Edge enhancement- **Learning Approach**: Few-shot learning with prototypical networks

5. Morphological operations

6. Contrast enhancement## 🔄 Data Processing Pipeline

7. Additional noise reduction

8. Final contrast adjustment### Few-shot Learning Setup:

9. Brightness normalization- **Support Set**: Small number of examples per class for prototypes

- **Query Set**: Test examples for classification

## 📊 Dataset Information- **Episode-based Training**: Simulates few-shot scenarios during training



- **Total Images**: 745### Preprocessing:

- **Sikku Images**: 664 (89%)- Resize to 224×224 pixels

- **Non-Sikku Images**: 81 (11%)- Normalize with ImageNet mean/std values

- **Image Formats**: JPG, JPEG, PNG- Data augmentation during training (horizontal flip, rotation, color jitter)

- **Resolution**: Variable (preprocessed to 224x224)

## 📈 Training Process

## 🎯 Model Architecture

The few-shot learning pipeline includes:

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)1. **Dataset Loading**: Load from HuggingFace `ayshthkr/kolam_dataset`

- **Feature Extractor**: 1280 → 512 → 256 → 128 features2. **Episode Creation**: Generate training episodes with support/query sets

- **Classifier**: 128 → 2 classes (Sikku/Non-Sikku)3. **Prototype Learning**: Learn class prototypes from support examples

- **Activation**: ReLU with Dropout (0.3)4. **Distance-based Classification**: Classify queries based on prototype distances

- **Loss Function**: CrossEntropyLoss5. **Model Saving**: Save trained prototypical network weights

- **Optimizer**: Adam with learning rate scheduling

## 📊 Evaluation Metrics

## 📋 Results Summary

The few-shot model is evaluated using:

### Confusion Matrix- **Episode-based Accuracy**: Performance on few-shot episodes

```- **Confidence Scores**: Probability distributions for predictions

             Predicted- **Class-wise Performance**: Individual class prediction accuracy

Actual    Non-Sikku  Sikku

Non-Sikku        15      1## 💾 Model Loading

Sikku             0    614

```The pre-trained model (`few_shot_kolam_classifier.pth`) includes:

- Prototypical network state dictionary

### Class-wise Performance- Class names mapping

- **Non-Sikku**: 93.8% accuracy (15/16 correct)- Feature extraction backbone weights

- **Sikku**: 100% accuracy (614/614 correct)

Load easily using the wrapper class for immediate inference.

## 🔧 Configuration

## 🔍 Features

Key training parameters:

- **Batch Size**: 16### Few-Shot Classifier (`few_shot_kolam_classifier.py`)

- **Learning Rate**: 0.001- ✅ Prototypical Networks implementation

- **Max Epochs**: 100- ✅ EfficientNet-B0 feature extraction

- **Early Stopping Patience**: 5- ✅ Episode-based training

- **Validation Split**: 20%- ✅ Few-shot learning capabilities

- **Data Augmentation**: Enabled for training- ✅ Model saving and loading



## 📝 Usage Examples### Classifier Wrapper (`kolam_classifier_wrapper.py`)

- ✅ Easy-to-use interface for predictions

### Training a New Model- ✅ Confidence score calculation

```python- ✅ Class name mapping

from optimized_kolam_trainer import train_optimized_model- ✅ Error handling and validation

train_optimized_model()

```### Inference Script (`inference.py`)

- ✅ Load pre-trained model

### Testing Trained Model- ✅ Predict on sample images

```python- ✅ Confidence scores and probabilities

from test_binary_classifier import test_binary_classifier- ✅ Visual results display

test_binary_classifier()

```## 🎨 Kolam Categories



### Using PreprocessingThe model classifies 8 different types of Kolams:

```python- **geometric**: Geometric patterns

from preprocessing import KolamPreprocessor- **butterfly**: Butterfly-shaped designs

preprocessor = KolamPreprocessor()- **pulli**: Dot-based kolams

processed_image = preprocessor.preprocess_image(image_array)- **flower**: Floral patterns

```- **naga**: Snake-inspired designs

- **sikku**: Line-based interlaced patterns

## 🤝 Contributing- **kamal**: Lotus-inspired designs

- **traditional**: Classical traditional patterns

1. Fork the repository

2. Create a feature branch## 🔧 Customization

3. Commit your changes

4. Push to the branch### Model Parameters

5. Create a Pull RequestYou can modify parameters in `few_shot_kolam_classifier.py`:

- `feature_dim`: Feature dimension for embeddings

## 📄 License- `backbone`: Feature extraction backbone model

- Episode configuration for training

This project is open source and available under the MIT License.

### Wrapper Configuration

## 🙏 AcknowledgmentsModify `kolam_classifier_wrapper.py` for:

- Custom class mappings

- Traditional Kolam art form and cultural heritage- Confidence thresholds

- OpenCV and PyTorch communities- Input preprocessing options

- EfficientNet architecture by Google Research

## 📋 System Requirements

## 📞 Contact

- **RAM**: 8GB minimum, 16GB recommended

For questions or collaborations, please open an issue in this repository.- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (optional)

- **Storage**: 2GB free space for dataset and models

---

## 🐛 Troubleshooting

**Note**: This project is designed for educational and research purposes in computer vision and cultural pattern recognition.
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
