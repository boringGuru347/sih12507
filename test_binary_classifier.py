"""
Test Binary Kolam Classifier on Sikku vs Non-Sikku Dataset
This script loads the trained binary classifier and evaluates it on the test dataset.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from preprocessing import KolamPreprocessor
import warnings
warnings.filterwarnings('ignore')

class PrototypicalNetwork(nn.Module):
    """Binary kolam classifier architecture"""
    
    def __init__(self, num_classes=2):
        super(PrototypicalNetwork, self).__init__()
        
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280
        self.num_classes = num_classes
        
        # Feature extractor
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

class TestKolamDataset(Dataset):
    """Dataset for testing binary classification"""
    
    def __init__(self, image_paths, labels, transform=None, use_preprocessing=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_preprocessing = use_preprocessing
        
        # Initialize preprocessor if needed
        if use_preprocessing:
            self.preprocessor = KolamPreprocessor()
            print("🔥 Test dataset with preprocessing enabled")
        else:
            self.preprocessor = None
            print("🔥 Test dataset with preprocessing disabled")
        
        self.class_names = ['Non-Sikku', 'Sikku']
        self.num_classes = len(self.class_names)
        
        print(f"📊 Test Dataset Summary:")
        print(f"   Total samples: {len(self.image_paths)}")
        print(f"   Classes: {self.num_classes}")
        for i, class_name in enumerate(self.class_names):
            count = labels.count(i)
            print(f"   {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image from local path
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        label = self.labels[idx]
        
        # Apply preprocessing if enabled
        if self.use_preprocessing and self.preprocessor:
            img_array = np.array(image)
            processed_array = self.preprocessor.preprocess_image(img_array)
            # Convert back to PIL for transforms
            if processed_array is not None:
                image = Image.fromarray((processed_array * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_test_transforms():
    """Get transforms for testing (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate_model_detailed(model, dataloader, device, class_names):
    """Detailed evaluation of the model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    correct_predictions = []
    incorrect_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track correct and incorrect predictions
            for i in range(len(labels)):
                actual = labels[i].item()
                pred = predicted[i].item()
                confidence = probs[i][pred].item()
                
                sample_info = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'actual': actual,
                    'predicted': pred,
                    'confidence': confidence,
                    'actual_class': class_names[actual],
                    'predicted_class': class_names[pred]
                }
                
                if actual == pred:
                    correct_predictions.append(sample_info)
                else:
                    incorrect_predictions.append(sample_info)
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Calculate precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Overall metrics
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision_avg,
        'recall': recall_avg,
        'f1': f1_avg,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions
    }

def test_binary_classifier():
    """Test the trained binary classifier"""
    
    print("🚀 Testing Binary Kolam Classifier (Sikku vs Non-Sikku)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        # Load test dataset
        print("\n📥 Loading test dataset...")
        dataset_folder = "kolam_dataset"
        
        # Load Non-Sikku images
        non_sikku_paths = glob.glob(f"{dataset_folder}/Non - Sikku/*.jpg")
        non_sikku_labels = [0] * len(non_sikku_paths)  # 0 for Non-Sikku
        
        # Load Sikku images
        sikku_paths = glob.glob(f"{dataset_folder}/Sikku/*.jpg")
        sikku_labels = [1] * len(sikku_paths)  # 1 for Sikku
        
        # Combine both classes
        test_paths = non_sikku_paths + sikku_paths
        test_labels = non_sikku_labels + sikku_labels
        
        print(f"Found {len(non_sikku_paths)} Non-Sikku images")
        print(f"Found {len(sikku_paths)} Sikku images")
        print(f"Total test images: {len(test_paths)}")
        
        # Create test dataset
        test_transform = get_test_transforms()
        test_dataset = TestKolamDataset(
            test_paths, test_labels,
            transform=test_transform,
            use_preprocessing=True
        )
        
        # Create test data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        
        # Load trained model
        print("\n🤖 Loading trained model...")
        model_path = "binary_kolam_classifier.pth"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file {model_path} not found!")
            return
        
        # Initialize model architecture
        model = PrototypicalNetwork(num_classes=2).to(device)
        
        # Load saved weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"   Classes: {checkpoint.get('class_names', ['Non-Sikku', 'Sikku'])}")
        
        # Test the model
        print("\n🧪 Testing model on dataset...")
        class_names = ['Non-Sikku', 'Sikku']
        test_results = evaluate_model_detailed(model, test_loader, device, class_names)
        
        # Print results
        print(f"\n📊 Test Results:")
        print(f"   Overall Accuracy: {test_results['accuracy']:.3f}")
        print(f"   Overall Precision: {test_results['precision']:.3f}")
        print(f"   Overall Recall: {test_results['recall']:.3f}")
        print(f"   Overall F1-Score: {test_results['f1']:.3f}")
        
        print(f"\n📈 Per-Class Metrics:")
        for i, class_name in enumerate(class_names):
            if i < len(test_results['per_class_precision']):
                print(f"   {class_name}:")
                print(f"     Precision: {test_results['per_class_precision'][i]:.3f}")
                print(f"     Recall: {test_results['per_class_recall'][i]:.3f}")
                print(f"     F1-Score: {test_results['per_class_f1'][i]:.3f}")
                print(f"     Support: {test_results['support'][i]} samples")
        
        # Print confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = test_results['confusion_matrix']
        print("             Predicted")
        print("Actual    Non-Sikku  Sikku")
        print(f"Non-Sikku    {cm[0][0]:6d}  {cm[0][1]:5d}")
        print(f"Sikku        {cm[1][0]:6d}  {cm[1][1]:5d}")
        
        # Calculate accuracy per class
        non_sikku_accuracy = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        sikku_accuracy = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
        
        print(f"\n🎯 Class-wise Accuracy:")
        print(f"   Non-Sikku: {non_sikku_accuracy:.3f} ({cm[0][0]}/{cm[0][0] + cm[0][1]})")
        print(f"   Sikku: {sikku_accuracy:.3f} ({cm[1][1]}/{cm[1][0] + cm[1][1]})")
        
        # Print prediction summary
        correct_count = len(test_results['correct_predictions'])
        incorrect_count = len(test_results['incorrect_predictions'])
        total_count = correct_count + incorrect_count
        
        print(f"\n📋 Prediction Summary:")
        print(f"   Correct predictions: {correct_count}/{total_count} ({correct_count/total_count:.3f})")
        print(f"   Incorrect predictions: {incorrect_count}/{total_count} ({incorrect_count/total_count:.3f})")
        
        # Show incorrect predictions if any
        if incorrect_count > 0:
            print(f"\n❌ Incorrect Predictions:")
            for i, pred_info in enumerate(test_results['incorrect_predictions'][:10]):  # Show first 10
                print(f"   {i+1}. Actual: {pred_info['actual_class']}, "
                      f"Predicted: {pred_info['predicted_class']}, "
                      f"Confidence: {pred_info['confidence']:.3f}")
        else:
            print(f"\n🎉 Perfect classification! No incorrect predictions!")
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Class-wise accuracy
        plt.subplot(1, 2, 2)
        accuracies = [non_sikku_accuracy, sikku_accuracy]
        colors = ['lightcoral', 'lightblue']
        bars = plt.bar(class_names, accuracies, color=colors)
        plt.title('Class-wise Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n💾 Test results visualization saved to test_results.png")
        
        # Save detailed results
        with open('test_results.txt', 'w') as f:
            f.write("Binary Kolam Classifier Test Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Dataset: {len(test_paths)} images\n")
            f.write(f"Non-Sikku: {len(non_sikku_paths)} images\n")
            f.write(f"Sikku: {len(sikku_paths)} images\n\n")
            f.write(f"Overall Accuracy: {test_results['accuracy']:.3f}\n")
            f.write(f"Overall Precision: {test_results['precision']:.3f}\n")
            f.write(f"Overall Recall: {test_results['recall']:.3f}\n")
            f.write(f"Overall F1-Score: {test_results['f1']:.3f}\n\n")
            f.write("Per-Class Metrics:\n")
            for i, class_name in enumerate(class_names):
                if i < len(test_results['per_class_precision']):
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {test_results['per_class_precision'][i]:.3f}\n")
                    f.write(f"  Recall: {test_results['per_class_recall'][i]:.3f}\n")
                    f.write(f"  F1-Score: {test_results['per_class_f1'][i]:.3f}\n")
                    f.write(f"  Support: {test_results['support'][i]} samples\n\n")
        
        print(f"📄 Detailed results saved to test_results.txt")
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_binary_classifier()