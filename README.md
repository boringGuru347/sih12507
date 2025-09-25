# Kolam Pattern Classifier 🎨

A web-based deep learning application for classifying traditional South Indian Kolam patterns into Sikku and Non-Sikku categories using PyTorch, computer vision, and Streamlit.

## 🎯 Project Overview

This project provides a user-friendly web interface for classifying Kolam designs into two categories:
- **Sikku Kolam**: Continuous line patterns without lifting the hand
- **Non-Sikku Kolam**: Patterns with dots or discrete elements

The application features:
- **Web Interface**: Interactive Streamlit app for easy image upload and classification
- **Advanced Preprocessing**: 9-step OpenCV pipeline for robust image enhancement
- **High Performance**: Trained EfficientNet-B0 model with excellent accuracy
- **Real-time Prediction**: Instant classification with confidence scores

## 🚀 Key Features

- **🌐 Web Application**: User-friendly Streamlit interface
- **📱 Easy Upload**: Drag-and-drop image upload functionality
- **🔍 Advanced Preprocessing**: Robust OpenCV-based image enhancement
- **⚡ Fast Inference**: Real-time predictions with confidence scores
- **📊 Detailed Results**: Probability breakdown and interpretation
- **🎨 Visual Interface**: Clean, intuitive design with result visualization

## 📁 Project Structure

```
sih12507/
├── streamlit_app.py                           # Main web application
├── preprocessing.py                           # Image preprocessing pipeline
├── balanced_kolam_classifier_undersample.pth # Trained model weights
├── requirements.txt                           # Python dependencies
├── balanced_kolam_trainer.py                 # Training script (optional)
├── optimized_kolam_trainer.py                # Original training script (optional)
├── kolam_dataset/                             # Training dataset
│   ├── Sikku/                                 # Sikku kolam images
│   └── Non - Sikku/                           # Non-Sikku kolam images
├── README.md                                  # Project documentation
└── .gitignore                                 # Git ignore file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9 or later
- CUDA-compatible GPU (optional but recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install streamlit torch torchvision opencv-python pillow numpy scikit-learn matplotlib seaborn tqdm
```

## 🚀 Usage

### Running the Web Application
```bash
streamlit run streamlit_app.py
```

The web application will open in your browser at `http://localhost:8501`

### Using the Web Interface
1. **Upload Image**: Click "Choose an image file" and select a Kolam pattern image
2. **View Image**: The uploaded image will be displayed
3. **Classify**: Click "🔮 Classify Pattern" to get predictions
4. **View Results**: See the predicted class, confidence score, and probability breakdown

## 🛠️ Technical Stack

- **Framework**: PyTorch
- **Web Interface**: Streamlit
- **Architecture**: EfficientNet-B0 backbone with custom classifier
- **Preprocessing**: OpenCV-based image enhancement
- **Image Processing**: PIL/Pillow for image handling
- **Visualization**: Matplotlib, Seaborn

## 🎯 Model Architecture

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Feature Extractor**: 1280 → 512 → 256 → 128 features
- **Classifier**: 128 → 2 classes (Sikku/Non-Sikku)
- **Activation**: ReLU with Dropout (0.3)
- **Input Size**: 224×224 pixels
- **Output**: Binary classification with confidence scores

## 🔬 Preprocessing Pipeline

The preprocessing module applies the following 9-step enhancement:

1. **Noise Reduction**: Gaussian blur for noise removal
2. **Histogram Equalization**: CLAHE for contrast enhancement
3. **Gaussian Blur**: Additional smoothing
4. **Edge Enhancement**: Sharpen important features
5. **Morphological Operations**: Shape refinement
6. **Contrast Enhancement**: Adaptive contrast adjustment
7. **Noise Reduction**: Final noise cleanup
8. **Contrast Adjustment**: Final contrast optimization
9. **Brightness Normalization**: Consistent brightness levels

## 📊 Dataset Information

- **Total Images**: 745
- **Sikku Images**: 664 (89%)
- **Non-Sikku Images**: 81 (11%)
- **Image Formats**: JPG, JPEG, PNG
- **Processing**: Resized to 224×224 pixels

## 🎨 Features

### Web Application (`streamlit_app.py`)
- ✅ Interactive file upload interface
- ✅ Real-time image preview
- ✅ One-click classification
- ✅ Confidence score display
- ✅ Probability breakdown visualization
- ✅ Result interpretation (High/Medium/Low confidence)
- ✅ Responsive design

### Preprocessing (`preprocessing.py`)
- ✅ 9-step OpenCV enhancement pipeline
- ✅ Configurable preprocessing parameters
- ✅ Multiple input format support
- ✅ Robust error handling
- ✅ Visualization of preprocessing steps

### Model Training (Optional Scripts)
- ✅ `balanced_kolam_trainer.py`: Advanced training with class balancing
- ✅ `optimized_kolam_trainer.py`: Original training implementation
- ✅ Multiple balancing techniques (undersample, weighted loss, oversample)
- ✅ Training visualization and metrics

## 🔍 About Kolam Patterns

### Sikku Kolam
- Continuous line patterns without lifting the hand
- Traditional geometric designs
- Connected motifs forming closed loops
- Intricate interlaced patterns

### Non-Sikku Kolam
- Patterns with dots or discrete elements
- May include separate, unconnected components
- Can have lifting points in the drawing process
- Often feature dot-based foundations

## 📋 System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: CUDA-compatible GPU with 2GB+ VRAM (optional)
- **Storage**: 1GB free space for dataset and models
- **Python**: 3.9 or later

## 🐛 Troubleshooting

### Common Issues:
1. **Streamlit not recognized**: Install with `pip install -r requirements.txt`
2. **Model loading error**: Ensure `balanced_kolam_classifier_undersample.pth` exists
3. **Import errors**: Install all dependencies with `pip install -r requirements.txt`
4. **CUDA issues**: Model automatically uses CPU if CUDA unavailable
5. **Image format errors**: Ensure images are in supported formats (JPG, PNG, JPEG)

### Performance Tips:
- Use GPU for faster inference
- Supported image formats: JPG, JPEG, PNG
- Optimal image size: 224×224 pixels (auto-resized)
- For best results, use clear, well-lit Kolam images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Traditional Kolam art form and cultural heritage
- OpenCV and PyTorch communities
- EfficientNet architecture by Google Research
- Streamlit for making web apps accessible

## 📞 Support

If you encounter any issues or have questions, please create an issue in the repository.

---

**Happy Kolam Classification! 🎨✨**