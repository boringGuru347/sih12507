"""
Kolam Classifier Inference Script
This script loads a trained model and predicts kolam types from new images.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class KolamPredictor:
    """Class for making predictions with trained kolam classifier"""
    
    def __init__(self, model_path='efficientnet_kolam.pth', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.transform = None
        
        self.load_model(model_path)
        self.setup_transforms()
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model information
            self.class_names = checkpoint['class_names']
            num_classes = checkpoint['num_classes']
            
            # Create model architecture
            self.model = models.efficientnet_b0(pretrained=False)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_features, num_classes)
            )
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully!")
            print(f"Classes: {self.class_names}")
            print(f"Device: {self.device}")
            
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found!")
            print("Please train the model first by running kolam_classifier.py")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def predict_image(self, image_path):
        """Predict kolam type for a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            # Get top prediction
            predicted_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[predicted_idx].item()
            predicted_class = self.class_names[predicted_idx]
            
            # Get top 3 predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            top3_predictions = [(self.class_names[idx], prob.item()) 
                              for idx, prob in zip(top3_idx, top3_prob)]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top3_predictions': top3_predictions,
                'original_image': original_image
            }
            
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found!")
            return None
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def visualize_prediction(self, result, image_path):
        """Visualize prediction results"""
        if result is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display original image
        ax1.imshow(result['original_image'])
        ax1.set_title(f'Input Image\n{image_path}')
        ax1.axis('off')
        
        # Display prediction results
        classes = [pred[0] for pred in result['top3_predictions']]
        confidences = [pred[1] for pred in result['top3_predictions']]
        
        colors = ['green' if i == 0 else 'orange' for i in range(len(classes))]
        bars = ax2.barh(classes, confidences, color=colors)
        
        ax2.set_xlabel('Confidence')
        ax2.set_title(f'Top 3 Predictions\nPredicted: {result["predicted_class"]} ({result["confidence"]:.2%})')
        ax2.set_xlim(0, 1)
        
        # Add confidence values on bars
        for bar, conf in zip(bars, confidences):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.2%}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'prediction_{image_path.split("/")[-1].split("\\")[-1]}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function for inference"""
    print("🔍 Kolam Classifier Inference")
    print("=" * 30)
    
    # Initialize predictor
    try:
        predictor = KolamPredictor()
    except:
        return
    
    # Example usage - you can change this to any image path
    test_images = [
        'image.png',  # The image in your workspace
        'test.png'    # The test image in your workspace
    ]
    
    for image_path in test_images:
        print(f"\nProcessing: {image_path}")
        print("-" * 30)
        
        result = predictor.predict_image(image_path)
        
        if result:
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nTop 3 Predictions:")
            for i, (class_name, conf) in enumerate(result['top3_predictions'], 1):
                print(f"  {i}. {class_name}: {conf:.2%}")
            
            # Visualize results
            predictor.visualize_prediction(result, image_path)
        else:
            print("Failed to process image")

if __name__ == "__main__":
    main()