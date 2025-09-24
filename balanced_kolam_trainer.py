"""
Balanced Kolam Classifier Training with Equal Class Distribution
This script balances the dataset by undersampling the majority class (Sikku)
or using weighted loss to handle class imbalance.
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
import glob
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
        # Enhanced training transforms with more aggressive augmentation for minority class
        train_transforms = [
            transforms.Resize((256, 256)),  # Slightly larger for cropping
            transforms.RandomRotation(degrees=360),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.7, 1.3),
                shear=15
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(train_transforms)
    else:
        # Validation/test transforms (deterministic)
        return transforms.Compose(base_transforms)

def load_balanced_dataset(dataset_folder, balance_method='undersample', target_per_class=None):
    """
    Load and balance the dataset
    
    Args:
        dataset_folder: Path to kolam_dataset folder
        balance_method: 'undersample', 'weighted', or 'oversample'
        target_per_class: Target number of samples per class (for undersampling)
    
    Returns:
        Tuple of (image_paths, labels, class_weights)
    """
    print(f"📊 Loading dataset with balance method: {balance_method}")
    
    # Load Non-Sikku images (all formats)
    non_sikku_patterns = [
        f"{dataset_folder}/Non - Sikku/*.jpg",
        f"{dataset_folder}/Non - Sikku/*.jpeg", 
        f"{dataset_folder}/Non - Sikku/*.png"
    ]
    non_sikku_paths = []
    for pattern in non_sikku_patterns:
        non_sikku_paths.extend(glob.glob(pattern))
    
    # Load Sikku images (all formats)
    sikku_patterns = [
        f"{dataset_folder}/Sikku/*.jpg",
        f"{dataset_folder}/Sikku/*.jpeg",
        f"{dataset_folder}/Sikku/*.png"
    ]
    sikku_paths = []
    for pattern in sikku_patterns:
        sikku_paths.extend(glob.glob(pattern))
    
    print(f"Found {len(non_sikku_paths)} Non-Sikku images")
    print(f"Found {len(sikku_paths)} Sikku images")
    
    # Balance the dataset
    if balance_method == 'undersample':
        # Undersample majority class to match minority class
        min_class_size = min(len(non_sikku_paths), len(sikku_paths))
        if target_per_class:
            min_class_size = min(min_class_size, target_per_class)
        
        print(f"Undersampling to {min_class_size} samples per class")
        
        # Randomly sample from each class
        random.shuffle(non_sikku_paths)
        random.shuffle(sikku_paths)
        
        non_sikku_paths = non_sikku_paths[:min_class_size]
        sikku_paths = sikku_paths[:min_class_size]
        
        # Create balanced dataset
        image_paths = non_sikku_paths + sikku_paths
        labels = ['Non-Sikku'] * len(non_sikku_paths) + ['Sikku'] * len(sikku_paths)
        
        # Equal class weights
        class_weights = torch.tensor([1.0, 1.0])
        
    elif balance_method == 'weighted':
        # Use all data but with weighted loss
        image_paths = non_sikku_paths + sikku_paths
        labels = ['Non-Sikku'] * len(non_sikku_paths) + ['Sikku'] * len(sikku_paths)
        
        # Calculate inverse class weights
        total_samples = len(image_paths)
        non_sikku_weight = total_samples / (2 * len(non_sikku_paths))
        sikku_weight = total_samples / (2 * len(sikku_paths))
        
        class_weights = torch.tensor([non_sikku_weight, sikku_weight])
        print(f"Class weights: Non-Sikku={non_sikku_weight:.3f}, Sikku={sikku_weight:.3f}")
        
    elif balance_method == 'oversample':
        # Oversample minority class to match majority class
        max_class_size = max(len(non_sikku_paths), len(sikku_paths))
        print(f"Oversampling to {max_class_size} samples per class")
        
        # Oversample the minority class (Non-Sikku)
        while len(non_sikku_paths) < max_class_size:
            non_sikku_paths.extend(non_sikku_paths[:min(len(non_sikku_paths), max_class_size - len(non_sikku_paths))])
        
        non_sikku_paths = non_sikku_paths[:max_class_size]
        sikku_paths = sikku_paths[:max_class_size]
        
        image_paths = non_sikku_paths + sikku_paths
        labels = ['Non-Sikku'] * len(non_sikku_paths) + ['Sikku'] * len(sikku_paths)
        
        # Equal class weights
        class_weights = torch.tensor([1.0, 1.0])
    
    # Shuffle the combined dataset
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    
    print(f"Final dataset: {len(image_paths)} total images")
    print(f"  Non-Sikku: {labels.count('Non-Sikku')} images")
    print(f"  Sikku: {labels.count('Sikku')} images")
    
    return list(image_paths), list(labels), class_weights

class BalancedKolamDataset(Dataset):
    """Balanced dataset for kolam classification"""
    
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
        
        self.class_names = ['Non-Sikku', 'Sikku']
        self.num_classes = len(self.class_names)
        
        print(f"📊 {mode.capitalize()} Dataset Summary:")
        print(f"   Total samples: {len(self.image_paths)}")
        print(f"   Classes: {self.num_classes}")
        for class_name in self.class_names:
            count = labels.count(class_name)
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
            
        label = self.class_names.index(self.labels[idx])
        
        # Apply preprocessing if enabled
        if self.use_preprocessing and self.preprocessor:
            img_array = np.array(image)
            processed_array = self.preprocessor.preprocess_image_array(img_array)
            # Convert back to PIL for transforms
            if processed_array is not None:
                image = Image.fromarray((processed_array * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class PrototypicalNetwork(nn.Module):
    """Binary kolam classifier architecture"""
    
    def __init__(self, num_classes=2):
        super(PrototypicalNetwork, self).__init__()
        
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280
        self.num_classes = num_classes
        
        # Feature extractor with enhanced capacity for balanced learning
        self.feature_extractor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Slightly higher dropout for better generalization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier head
        self.classifier = nn.Linear(128, num_classes)
        
        print(f"Balanced Kolam Classifier initialized with EfficientNet-B0")
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

def train_balanced_model(balance_method='undersample', target_per_class=80):
    """Train the balanced kolam classifier"""
    
    # Configuration
    val_split = 0.2
    augment = True
    batch_size = 16
    max_epochs = 120  # More epochs for balanced learning
    learning_rate = 0.0005  # Slightly lower learning rate
    patience = 8  # Increased patience for balanced training
    
    print("🚀 Starting Balanced Binary Kolam Classifier Training")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Balance method: {balance_method}")
    print(f"   Target per class: {target_per_class}")
    print(f"   Validation split: {val_split}")
    print(f"   Augmentation: {augment}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    try:
        # Load balanced dataset
        print("\n📥 Loading and balancing dataset...")
        dataset_folder = "kolam_dataset"
        
        image_paths, labels, class_weights = load_balanced_dataset(
            dataset_folder, 
            balance_method=balance_method,
            target_per_class=target_per_class
        )
        
        # Split dataset
        total_size = len(image_paths)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # Stratified split to maintain class balance
        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        
        # Split each class separately to maintain balance
        non_sikku_indices = [i for i, label in enumerate(labels) if label == 'Non-Sikku']
        sikku_indices = [i for i, label in enumerate(labels) if label == 'Sikku']
        
        # Split Non-Sikku
        non_sikku_val_size = int(len(non_sikku_indices) * val_split)
        random.shuffle(non_sikku_indices)
        
        for i in non_sikku_indices[:non_sikku_val_size]:
            val_paths.append(image_paths[i])
            val_labels.append(labels[i])
        
        for i in non_sikku_indices[non_sikku_val_size:]:
            train_paths.append(image_paths[i])
            train_labels.append(labels[i])
        
        # Split Sikku
        sikku_val_size = int(len(sikku_indices) * val_split)
        random.shuffle(sikku_indices)
        
        for i in sikku_indices[:sikku_val_size]:
            val_paths.append(image_paths[i])
            val_labels.append(labels[i])
        
        for i in sikku_indices[sikku_val_size:]:
            train_paths.append(image_paths[i])
            train_labels.append(labels[i])
        
        print(f"\n📊 Balanced Dataset Split:")
        print(f"   Training set: {len(train_paths)} images")
        print(f"     Non-Sikku: {train_labels.count('Non-Sikku')} images")
        print(f"     Sikku: {train_labels.count('Sikku')} images")
        print(f"   Validation set: {len(val_paths)} images")
        print(f"     Non-Sikku: {val_labels.count('Non-Sikku')} images")
        print(f"     Sikku: {val_labels.count('Sikku')} images")
        
        # Get transforms
        train_transform = get_transforms(is_training=True, augment=augment)
        val_transform = get_transforms(is_training=False, augment=False)
        
        # Create datasets
        print("\n🔬 Creating balanced datasets...")
        train_dataset = BalancedKolamDataset(
            train_paths, train_labels,
            transform=train_transform, 
            use_preprocessing=True,
            is_training=True
        )
        
        val_dataset = BalancedKolamDataset(
            val_paths, val_labels,
            transform=val_transform, 
            use_preprocessing=True,
            is_training=False
        )
        
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
        print("\n🤖 Initializing balanced model...")
        model = PrototypicalNetwork(num_classes=2).to(device)
        
        # Use weighted loss if specified
        if balance_method == 'weighted':
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        
        # Training tracking
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"\n🏋️ Starting balanced training...")
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
            val_metrics = evaluate_model(model, val_loader, device, ['Non-Sikku', 'Sikku'])
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
        final_metrics = evaluate_model(model, val_loader, device, ['Non-Sikku', 'Sikku'])
        
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
        print(f"   Balance method: {balance_method}")
        
        # Print confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = final_metrics['confusion_matrix']
        print("             Predicted")
        print("Actual    Non-Sikku  Sikku")
        print(f"Non-Sikku    {cm[0][0]:6d}  {cm[0][1]:5d}")
        print(f"Sikku        {cm[1][0]:6d}  {cm[1][1]:5d}")
        
        # Save model
        model_path = f"balanced_kolam_classifier_{balance_method}.pth"
        torch.save({
            'model_state_dict': best_model_state if best_model_state else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'final_metrics': final_metrics,
            'class_names': ['Non-Sikku', 'Sikku'],
            'balance_method': balance_method,
            'class_weights': class_weights,
            'config': {
                'balance_method': balance_method,
                'target_per_class': target_per_class,
                'val_split': val_split,
                'augment': augment,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'max_epochs': max_epochs,
                'patience': patience
            }
        }, model_path)
        
        print(f"💾 Balanced model saved to {model_path}")
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        # Training and validation loss
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title(f'Training Curves - {balance_method.title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Training and validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Confusion Matrix
        plt.subplot(1, 3, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Sikku', 'Sikku'],
                   yticklabels=['Non-Sikku', 'Sikku'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        curves_path = f'balanced_training_curves_{balance_method}.png'
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved to {curves_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Balanced Kolam Classifier')
    parser.add_argument('--method', type=str, default='undersample', 
                       choices=['undersample', 'weighted', 'oversample'],
                       help='Balancing method: undersample, weighted, or oversample')
    parser.add_argument('--target', type=int, default=80,
                       help='Target number of samples per class (for undersampling)')
    
    args = parser.parse_args()
    
    print(f"🎯 Training with balance method: {args.method}")
    if args.method == 'undersample':
        print(f"🎯 Target samples per class: {args.target}")
    
    train_balanced_model(balance_method=args.method, target_per_class=args.target)