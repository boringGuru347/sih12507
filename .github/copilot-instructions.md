# Kolam Binary Classifier - AI Agent Instructions

## Project Overview
This is a PyTorch-based binary classification system for traditional South Indian Kolam patterns, distinguishing between **Sikku** (continuous line) and **Non-Sikku** (discontinuous line) designs. The project uses EfficientNet-B0 backbone with custom preprocessing and achieves 99.8% accuracy.

## Architecture & Key Components

### Core Model Pipeline
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet) → feature extraction
- **Feature Pipeline**: 1280 → 512 → 256 → 128 features with ReLU + Dropout(0.3)
- **Classification Head**: 128 → 2 classes (Binary classification)
- **Model Class**: `PrototypicalNetwork` in both training and testing files

### Critical Preprocessing Pipeline (`preprocessing.py`)
The `KolamPreprocessor` class implements a **9-step OpenCV pipeline**:
1. Gaussian blur (noise reduction)
2. Adaptive/Otsu/Binary thresholding
3. Morphological operations (connect broken strokes)
4. Contour detection and cropping (optional)
5. Resize to 224×224
6. Convert to 3-channel RGB
7. Normalize to [0,1]

**Usage Pattern**: Always use `preprocess_image()` function or `KolamPreprocessor` class before model inference.

## Project Structure & Workflows

### Dataset Organization
```
kolam_dataset/
├── Sikku/           # 664 continuous line pattern images
└── Non - Sikku/     # 81 discontinuous pattern images  
```

### Key Training Workflow (`optimized_kolam_trainer.py`)
```bash
python optimized_kolam_trainer.py
```
- **Train/Val Split**: 80/20 with random shuffling
- **Augmentation**: Rotations (360°), flips, perspective transforms, color jitter
- **Early Stopping**: Patience=5 epochs
- **Outputs**: `binary_kolam_classifier.pth`, `binary_training_curves.png`

### Testing Workflow (`test_binary_classifier.py`)
```bash
python test_binary_classifier.py
```
- Loads `binary_kolam_classifier.pth`
- Tests entire dataset with detailed metrics
- Generates confusion matrix and class-wise accuracy

## Development Patterns & Conventions

### Model Loading Pattern
```python
# Standard model loading across the project
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Dataset Class Pattern
- All dataset classes inherit from `torch.utils.data.Dataset`
- Use `OptimizedKolamDataset` for training, `TestKolamDataset` for testing
- Always include preprocessing option: `use_preprocessing=True`
- Standard transforms: ImageNet normalization `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`

### Error Handling Conventions
- Always include try-catch blocks around image loading
- Fallback to black images `np.zeros((224, 224, 3))` for failed preprocessing
- Use `warnings.filterwarnings('ignore')` to suppress model warnings

### File Naming & Outputs
- Model files: `*.pth` format with comprehensive checkpoint data
- Visualizations: `*_curves.png`, `test_results.png`
- Logs: `test_results.txt` for detailed metrics

## Configuration & Hyperparameters

### Standard Training Config
- **Batch Size**: 16 (memory-optimized for GPU)
- **Learning Rate**: 0.001 with StepLR scheduler (step_size=20, gamma=0.5)
- **Max Epochs**: 100 with early stopping
- **Image Size**: 224×224 (EfficientNet standard)
- **Device Detection**: Auto CUDA/CPU selection

### Preprocessing Config (Global)
```python
PREPROCESS_CONFIG = {
    'PREPROCESS': True,
    'TARGET_SIZE': (224, 224),
    'BLUR_KERNEL_SIZE': 5,
    'THRESHOLD_METHOD': 'adaptive'
}
```

## Development Commands

### Essential Commands
```bash
# Train original model (imbalanced)
python optimized_kolam_trainer.py

# Train balanced model (recommended)
python balanced_kolam_trainer.py --method undersample --target 80

# Test original model
python test_binary_classifier.py

# Test balanced model
python test_balanced_classifier.py

# Test preprocessing pipeline
python preprocessing.py
```

### Balanced Training Options
```bash
# Undersample majority class (recommended for small datasets)
python balanced_kolam_trainer.py --method undersample --target 80

# Use weighted loss (for full dataset)
python balanced_kolam_trainer.py --method weighted

# Oversample minority class (memory intensive)
python balanced_kolam_trainer.py --method oversample
```

### Dependencies Installation
```bash
pip install torch torchvision opencv-python pillow matplotlib seaborn scikit-learn numpy
```

## Critical Implementation Details

### Memory Management
- Use `num_workers=0` in DataLoaders (Windows compatibility)
- Model automatically switches to CPU if CUDA unavailable
- Images loaded as PIL → NumPy → PIL conversion cycle for preprocessing

### Class Mapping Convention
- **Class 0**: Non-Sikku (minority class - 83 images across jpg/jpeg/png)
- **Class 1**: Sikku (majority class - 664 images)
- **Class Names**: `['Non-Sikku', 'Sikku']` (fixed order)
- **Class Imbalance**: 11.1% Non-Sikku vs 88.9% Sikku

### Balanced Training Approach
- **balanced_kolam_trainer.py**: Handles class imbalance with undersampling/weighted loss
- **Undersampling**: Reduces majority class to match minority (80 samples each)
- **Weighted Loss**: Uses inverse class frequency weighting
- **Performance**: Balanced accuracy ~95.3% vs standard accuracy ~93.4%

### Evaluation Metrics Focus
- **Primary**: Overall accuracy, per-class precision/recall
- **Visualization**: Confusion matrix with seaborn heatmaps
- **Class Imbalance**: Handle with detailed per-class metrics reporting

## Common Issues & Solutions

### Model Loading Issues
- Ensure `binary_kolam_classifier.pth` exists before testing
- Use `weights_only=False` for complete checkpoint loading
- Check device compatibility (CUDA vs CPU)

### Preprocessing Failures
- Preprocessing returns black image as fallback for corrupted inputs
- Use `visualize_preprocessing_steps()` for debugging pipeline
- Toggle preprocessing with `PREPROCESS_CONFIG['PREPROCESS'] = False`

### Dataset Path Issues
- Use `glob.glob()` for flexible file discovery
- Handle mixed file extensions (`.jpg`, `.jpeg`, `.png`)
- Absolute paths preferred for cross-platform compatibility

When working with this codebase, always consider the cultural significance of Kolam art and maintain high accuracy standards for this traditional pattern recognition task.