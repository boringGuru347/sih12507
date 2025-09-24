import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import io
import os
import sys

# Add current directory to path for importing preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import preprocessing functions
try:
    from preprocessing import preprocess_image
except ImportError as e:
    st.error(f"Could not import preprocessing module: {e}")
    st.stop()

class KolamClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(KolamClassifier, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
        
        # Remove the original classifier (replace with identity)
        self.backbone.classifier = nn.Identity()
        
        # Add feature extractor layers (matching saved model architecture)
        self.feature_extractor = nn.Sequential(
            nn.Linear(1280, 512),    # feature_extractor.0
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),     # feature_extractor.3
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(256, 128),     # feature_extractor.6
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Final classifier layer
        self.classifier = nn.Linear(128, num_classes)  # classifier
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)  # Output: [batch, 1280]
        
        # Pass through feature extractor
        extracted = self.feature_extractor(features)  # Output: [batch, 128]
        
        # Final classification
        output = self.classifier(extracted)  # Output: [batch, 2]
        return output

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Initialize model
        model = KolamClassifier(num_classes=2)
        
        # Load trained weights
        model_path = 'balanced_kolam_classifier_undersample.pth'
        if os.path.exists(model_path):
            # Use weights_only=False for compatibility with older saved models
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            st.success("✅ Model loaded successfully!")
            return model, checkpoint.get('epoch', 'Unknown'), checkpoint.get('val_accuracy', 'Unknown')
        else:
            st.error(f"❌ Model file not found: {model_path}")
            return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def preprocess_for_inference(image):
    """Preprocess uploaded image for model inference"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Ensure the image is in the right format (uint8)
        if img_array.dtype != np.uint8:
            # If it's float, assume it's normalized to [0,1] and convert to [0,255]
            if img_array.dtype in [np.float32, np.float64]:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        # Handle different image formats
        if len(img_array.shape) == 2:
            # Grayscale image, convert to 3-channel
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                # RGBA image, convert to RGB first, then to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif img_array.shape[2] == 3:
                # RGB image, convert to BGR for OpenCV
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply preprocessing pipeline
        processed_img = preprocess_image(img_array)
        
        # Convert processed image to proper format for display
        # preprocess_image returns float32 in [0,1] range or RGB format
        if processed_img.dtype in [np.float32, np.float64]:
            # If float, convert to uint8 [0,255] range
            if processed_img.max() <= 1.0:
                processed_display = (processed_img * 255).astype(np.uint8)
            else:
                processed_display = processed_img.astype(np.uint8)
        else:
            processed_display = processed_img
        
        # Convert back to PIL for display
        if len(processed_display.shape) == 2:
            # Grayscale result
            processed_pil = Image.fromarray(processed_display, mode='L')
        else:
            # Color result - check if it's already RGB or needs BGR->RGB conversion
            if processed_display.shape[2] == 3:
                # Assume it's RGB from preprocessing (not BGR)
                processed_pil = Image.fromarray(processed_display)
            else:
                processed_pil = Image.fromarray(cv2.cvtColor(processed_display, cv2.COLOR_BGR2RGB))
        
        # Convert to tensor for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Handle grayscale images - convert to RGB for the model
        if processed_pil.mode == 'L':
            processed_pil = processed_pil.convert('RGB')
        elif processed_pil.mode == 'RGBA':
            processed_pil = processed_pil.convert('RGB')
        
        tensor = transform(processed_pil).unsqueeze(0)
        
        return tensor, processed_pil
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None, None

def predict_image(model, image_tensor):
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Class mapping
            class_names = {0: 'Non-Sikku', 1: 'Sikku'}
            
            return class_names[predicted_class], confidence, probabilities[0].tolist()
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None, None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Kolam Pattern Classifier",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🎨 Kolam Pattern Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Upload an image to classify whether it contains Sikku or Non-Sikku Kolam patterns</p>", unsafe_allow_html=True)
    
    # Load model
    model, epoch, val_accuracy = load_model()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        
        if model is not None:
            # Format validation accuracy properly
            val_acc_display = f"{val_accuracy:.1f}%" if isinstance(val_accuracy, (int, float)) else str(val_accuracy)
            
            st.markdown(f"""
            <div class='metric-container'>
            <h4>Model Details</h4>
            <ul>
            <li><strong>Architecture:</strong> EfficientNet-B0 with Feature Extractor</li>
            <li><strong>Training Epoch:</strong> {epoch}</li>
            <li><strong>Validation Accuracy:</strong> {val_acc_display}</li>
            <li><strong>Classes:</strong> Sikku, Non-Sikku</li>
            <li><strong>Input Size:</strong> 224×224 pixels</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Model Performance")
            st.markdown("""
            - **Overall Accuracy:** 93.4%
            - **Balanced Accuracy:** 95.3%
            - **Non-Sikku Recall:** 97.6%
            - **Sikku Recall:** 92.9%
            """)
            
        st.markdown("### ℹ️ About Kolam")
        st.markdown("""
        **Sikku Kolam:** Traditional patterns drawn without lifting the finger, creating continuous loops.
        
        **Non-Sikku Kolam:** Patterns that may have breaks or are drawn by lifting the finger.
        """)
    
    # Main content
    if model is None:
        st.error("⚠️ Please ensure the model file 'balanced_kolam_classifier_undersample.pth' is in the current directory.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a Kolam pattern for classification"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Original Image")
            # Display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Uploaded Image", use_container_width=True)
            
            # Display image info
            st.markdown(f"""
            <div class='metric-container'>
            <strong>Image Info:</strong><br>
            Size: {original_image.size[0]} × {original_image.size[1]} pixels<br>
            Mode: {original_image.mode}<br>
            Format: {original_image.format}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔄 Processed Image")
            
            # Preprocess image
            with st.spinner("Processing image..."):
                image_tensor, processed_image = preprocess_for_inference(original_image)
            
            if image_tensor is not None and processed_image is not None:
                # Display processed image
                st.image(processed_image, caption="Processed Image (Ready for Classification)", use_container_width=True)
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    prediction, confidence, probabilities = predict_image(model, image_tensor)
                
                if prediction is not None:
                    # Display prediction results
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Main prediction
                    confidence_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.6 else "#dc3545"
                    st.markdown(f"""
                    <div class='prediction-box'>
                    <h3 style='color: {confidence_color}; margin: 0;'>Prediction: {prediction}</h3>
                    <p style='margin: 0.5rem 0 0 0;'>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability bars
                    st.markdown("### 📊 Class Probabilities")
                    col_prob1, col_prob2 = st.columns(2)
                    
                    with col_prob1:
                        st.metric("Non-Sikku", f"{probabilities[0]:.1%}")
                        st.progress(probabilities[0])
                    
                    with col_prob2:
                        st.metric("Sikku", f"{probabilities[1]:.1%}")
                        st.progress(probabilities[1])
                    
                    # Confidence interpretation
                    if confidence > 0.9:
                        st.success("🎯 Very High Confidence - The model is very certain about this prediction.")
                    elif confidence > 0.8:
                        st.success("✅ High Confidence - The model is confident about this prediction.")
                    elif confidence > 0.6:
                        st.warning("⚠️ Moderate Confidence - The model has some uncertainty.")
                    else:
                        st.error("❓ Low Confidence - The prediction may be unreliable.")
    
    # Additional information
    with st.expander("🔍 How it works"):
        st.markdown("""
        ### Preprocessing Pipeline:
        1. **Gaussian Blur** - Reduces noise
        2. **Adaptive Thresholding** - Converts to binary image
        3. **Morphological Operations** - Cleans up the image
        4. **Contour Detection** - Identifies pattern boundaries
        5. **Resize** - Standardizes to 224×224 pixels
        6. **Normalization** - Prepares for neural network
        
        ### Model Architecture:
        - **Backbone:** EfficientNet-B0 (pre-trained on ImageNet)
        - **Classification Head:** Fully connected layer with dropout
        - **Training:** Balanced dataset with undersampling technique
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Built with Streamlit and PyTorch | Kolam Pattern Classification</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()