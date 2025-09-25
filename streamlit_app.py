import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import io
from preprocessing import preprocess_image

class PrototypicalNetwork(nn.Module):
    """Modified network for kolam classification - matches the trained model"""
    
    def __init__(self, num_classes=2):
        super(PrototypicalNetwork, self).__init__()
        
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280
        self.num_classes = num_classes
        
        # Feature extractor (matching the saved model)
        self.feature_extractor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Classifier head
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Extract features
        backbone_features = self.backbone(x)
        features = self.feature_extractor(backbone_features)
        
        # Classify
        logits = self.classifier(features)
        return logits

@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        # Load the checkpoint
        checkpoint = torch.load('balanced_kolam_classifier_undersample.pth', 
                               map_location='cpu', weights_only=False)
        
        # Check if it's a complete checkpoint or just state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # It's a complete checkpoint
            model = PrototypicalNetwork(num_classes=2)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # It's just the state_dict
            model = PrototypicalNetwork(num_classes=2)
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image):
    """Make prediction on uploaded image."""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Convert from (H, W, C) to (C, H, W) and add batch dimension
        if len(processed_image.shape) == 3:
            processed_image = processed_image.transpose(2, 0, 1)  # (H, W, C) to (C, H, W)
        
        image_tensor = torch.FloatTensor(processed_image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class to label
        class_labels = {0: "Non-Sikku", 1: "Sikku"}
        predicted_label = class_labels[predicted_class]
        
        return predicted_label, confidence, probabilities[0].tolist()
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    st.set_page_config(
        page_title="Kolam Pattern Classifier",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Kolam Pattern Classifier")
    st.markdown("Upload an image to classify whether it's a **Sikku** or **Non-Sikku** Kolam pattern")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a Kolam pattern image for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add prediction button
            if st.button("🔮 Classify Pattern", type="primary"):
                with st.spinner("Analyzing the Kolam pattern..."):
                    predicted_label, confidence, probabilities = predict_image(model, image)
                    
                    if predicted_label is not None:
                        # Store results in session state
                        st.session_state.prediction_result = {
                            'label': predicted_label,
                            'confidence': confidence,
                            'probabilities': probabilities
                        }
    
    with col2:
        st.header("Prediction Results")
        
        if hasattr(st.session_state, 'prediction_result') and st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            # Display main prediction
            if result['label'] == "Sikku":
                st.success(f"🎯 **Prediction: {result['label']}**")
            else:
                st.info(f"🎯 **Prediction: {result['label']}**")
            
            # Display confidence
            st.metric("Confidence", f"{result['confidence']:.2%}")
            
            # Display probability breakdown
            st.subheader("Probability Breakdown")
            
            labels = ["Non-Sikku", "Sikku"]
            probs = result['probabilities']
            
            for i, (label, prob) in enumerate(zip(labels, probs)):
                st.write(f"**{label}**: {prob:.2%}")
                st.progress(prob)
            
            # Add interpretation
            st.subheader("Interpretation")
            if result['confidence'] > 0.8:
                st.success("🎯 High confidence prediction")
            elif result['confidence'] > 0.6:
                st.warning("⚠️ Moderate confidence prediction")
            else:
                st.error("❌ Low confidence prediction - results may be unreliable")
        
        else:
            st.info("👆 Upload an image and click 'Classify Pattern' to see results")
    
    # Add information section
    st.markdown("---")
    st.subheader("About Kolam Patterns")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Sikku Kolam:**
        - Continuous line patterns without lifting the hand
        - Traditional geometric designs
        - Connected motifs forming closed loops
        """)
    
    with info_col2:
        st.markdown("""
        **Non-Sikku Kolam:**
        - Patterns with dots or discrete elements
        - May include separate, unconnected components
        - Can have lifting points in the drawing process
        """)

if __name__ == "__main__":
    main()