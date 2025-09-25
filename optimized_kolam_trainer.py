"""
Optimized Kolam Classifier Training with Smart Preprocessing
This version uses caching and batch optimization for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from collections import Counter, defaultdict
import random
import time
import json
from copy import deepcopy
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing module
from preprocessing import KolamPreprocessor

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_transforms(is_training=True, augment=True):
    """Get transforms for training and validation"""
    
    # Base transforms for all datasets
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if is_training and augment:
        # Training transforms with augmentation
        train_transforms = [
            transforms.Resize((256, 256)),  # Slightly larger for cropping
            transforms.RandomRotation(degrees=360),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(train_transforms)
    else:
        # Validation/test transforms (deterministic)
        return transforms.Compose(base_transforms)

def save_sample_augmentations(dataset, num_samples=5):
    """Save sample augmented images for visualization"""
    os.makedirs("samples/augmented", exist_ok=True)
    
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        
        # Convert tensor back to PIL for saving
        if isinstance(img, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Convert to PIL
            img = transforms.ToPILImage()(img)
        
        img.save(f"samples/augmented/sample_{i}_label_{label}.png")
    
    print(f"💾 Saved {min(num_samples, len(dataset))} sample augmented images to samples/augmented/")

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / len(dataloader)
    
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
        'loss': avg_loss,
        'precision': precision_avg,
        'recall': recall_avg,
        'f1': f1_avg,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

class OptimizedKolamDataset(Dataset):
    """Local dataset for Images folder with train/val split support"""
    
    def __init__(self, image_paths, labels, transform=None, use_preprocessing=True, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_preprocessing = use_preprocessing
        self.is_training = is_training
        
        # Initialize preprocessor if needed
        if use_preprocessing:
            self.preprocessor = KolamPreprocessor()
            mode = "training" if is_training else "validation"
            print(f"🔥 {mode.capitalize()} dataset with preprocessing enabled")
        else:
            self.preprocessor = None
            mode = "training" if is_training else "validation"
            print(f"🔥 {mode.capitalize()} dataset with preprocessing disabled")
        
        self.class_names = list(set(labels))
        self.num_classes = len(self.class_names)
        
        print(f"📊 {mode.capitalize()} Dataset Summary:")
        print(f"   Total samples: {len(self.image_paths)}")
        print(f"   Classes: {self.num_classes}")
        for class_name in self.class_names:
            count = labels.count(class_name)
            print(f"   {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image from local path
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.class_names.index(self.labels[idx])
        
        # Apply preprocessing if enabled
        if self.use_preprocessing and self.preprocessor:
            img_array = np.array(image)
            processed_array = self.preprocessor.preprocess_image_array(img_array)
            # Convert back to PIL for transforms
            image = Image.fromarray((processed_array * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    


class PrototypicalNetwork(nn.Module):
    """Modified network for single-class kolam classification"""
    
    def __init__(self, num_classes=1):
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
        
        print(f"Kolam Classifier initialized with EfficientNet-B0")
        print(f"Feature dimension: 1280 -> 128 -> {num_classes}")
    
    def forward(self, x):
        # Extract features
        backbone_features = self.backbone(x)
        features = self.feature_extractor(backbone_features)
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """Extract features without classification"""
        backbone_features = self.backbone(x)
        features = self.feature_extractor(backbone_features)
        return features
    


def train_optimized_model():
    """Train the optimized kolam classifier with train/val split and augmentation"""
    
    # Configuration
    val_split = 0.2
    augment = True
    batch_size = 16
    max_epochs = 100
    learning_rate = 0.001
    patience = 5  # Early stopping patience
    
    print("🚀 Starting Binary Kolam Classifier Training (Sikku vs Non-Sikku)")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Validation split: {val_split}")
    print(f"   Augmentation: {augment}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    try:
        # Load kolam_dataset with Sikku and Non-Sikku classes
        print("\n📥 Loading kolam_dataset with Sikku and Non-Sikku classes...")
        dataset_folder = "kolam_dataset"
        
        # Load images from both folders
        import glob
        
        # Load Non-Sikku images
        non_sikku_paths = glob.glob(f"{dataset_folder}/Non - Sikku/*.jpg")
        non_sikku_labels = ['Non-Sikku'] * len(non_sikku_paths)
        
        # Load Sikku images
        sikku_paths = glob.glob(f"{dataset_folder}/Sikku/*.jpg")
        sikku_labels = ['Sikku'] * len(sikku_paths)
        
        # Combine both classes
        image_paths = non_sikku_paths + sikku_paths
        labels = non_sikku_labels + sikku_labels
        
        print(f"Found {len(non_sikku_paths)} Non-Sikku images")
        print(f"Found {len(sikku_paths)} Sikku images")
        print(f"Total: {len(image_paths)} images")
        
        # Shuffle and split dataset
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)
        
        # Calculate split sizes
        total_size = len(image_paths)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # Split the data
        train_paths = list(image_paths[:train_size])
        train_labels = list(labels[:train_size])
        val_paths = list(image_paths[train_size:])
        val_labels = list(labels[train_size:])
        
        print(f"\n📊 Dataset Split:")
        print(f"   Total images: {total_size}")
        print(f"   Training set: {train_size} images ({(1-val_split)*100:.0f}%)")
        print(f"   Validation set: {val_size} images ({val_split*100:.0f}%)")
        
        # Get transforms
        train_transform = get_transforms(is_training=True, augment=augment)
        val_transform = get_transforms(is_training=False, augment=False)
        
        # Create datasets
        print("\n🔬 Creating datasets...")
        train_dataset = OptimizedKolamDataset(
            train_paths, train_labels,
            transform=train_transform, 
            use_preprocessing=True,
            is_training=True
        )
        
        val_dataset = OptimizedKolamDataset(
            val_paths, val_labels,
            transform=val_transform, 
            use_preprocessing=True,
            is_training=False
        )
        
        # Save sample augmented images
        print("\n�️ Saving sample augmented images...")
        save_sample_augmentations(train_dataset, num_samples=5)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        # Initialize model
        print("\n🤖 Initializing model...")
        num_classes = len(train_dataset.class_names)
        print(f"Number of classes: {num_classes} ({train_dataset.class_names})")
        model = PrototypicalNetwork(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Training tracking
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"\n🏋️ Starting training with validation...")
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            
            # Validation phase
            val_metrics = evaluate_model(model, val_loader, device, train_dataset.class_names)
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping check
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{max_epochs}: "
                      f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.3f}, "
                      f"Val Loss = {val_metrics['loss']:.4f}, Val Acc = {val_metrics['accuracy']:.3f}, "
                      f"Time = {epoch_time:.2f}s, Elapsed = {elapsed/60:.1f}min")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
                print(f"   Best validation accuracy: {best_val_acc:.3f}")
                break
        
        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation on validation set
        print(f"\n✅ Training completed!")
        total_time = time.time() - start_time
        
        print(f"\n📊 Final Evaluation on Validation Set:")
        final_metrics = evaluate_model(model, val_loader, device, train_dataset.class_names)
        
        print(f"   Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"   Precision: {final_metrics['precision']:.3f}")
        print(f"   Recall: {final_metrics['recall']:.3f}")
        print(f"   F1-Score: {final_metrics['f1']:.3f}")
        print(f"   Validation Loss: {final_metrics['loss']:.4f}")
        
        print(f"\n📈 Training Summary:")
        print(f"   Total training time: {total_time/60:.1f} minutes")
        print(f"   Training images: {len(train_dataset)}")
        print(f"   Validation images: {len(val_dataset)}")
        print(f"   Best validation accuracy: {best_val_acc:.3f}")
        
        # Print confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = final_metrics['confusion_matrix']
        print(cm)
        
        # Save model
        model_path = "binary_kolam_classifier.pth"
        torch.save({
            'model_state_dict': best_model_state if best_model_state else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'final_metrics': final_metrics,
            'class_names': train_dataset.class_names,
            'config': {
                'val_split': val_split,
                'augment': augment,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'max_epochs': max_epochs,
                'patience': patience
            }
        }, model_path)
        
        print(f"💾 Model saved to {model_path}")
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        # Training and validation loss
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Training and validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Confusion Matrix
        plt.subplot(1, 3, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=train_dataset.class_names,
                   yticklabels=train_dataset.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('binary_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves and confusion matrix saved to binary_training_curves.png")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_optimized_model()