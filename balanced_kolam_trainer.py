import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from preprocessing import KolamDataset, augment_dataset

class KolamClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(KolamClassifier, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Remove the final classifier
        self.backbone.classifier = nn.Identity()
        
        # Add custom feature extractor layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        # Pass through feature extractor
        features = self.feature_extractor(features)
        # Final classification
        output = self.classifier(features)
        return output

def train_balanced_classifier(data_dir, balance_method='undersample', num_epochs=20, batch_size=16, learning_rate=1e-4):
    """
    Train a balanced Kolam classifier using different balancing techniques.
    
    Args:
        data_dir: Path to the dataset directory
        balance_method: 'undersample', 'weighted_loss', or 'oversample'
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
    """
    
    print(f"Training with balance method: {balance_method}")
    
    # Create dataset
    dataset = KolamDataset(data_dir)
    print(f"Original dataset size: {len(dataset)}")
    
    # Get class distribution
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    print(f"Class distribution: {dict(class_counts)}")
    
    # Apply balancing technique
    if balance_method == 'undersample':
        # Undersample majority class
        min_count = min(class_counts.values())
        balanced_indices = []
        class_indices = {0: [], 1: []}
        
        for i, label in enumerate(labels):
            class_indices[label].append(i)
        
        for class_label, indices in class_indices.items():
            np.random.shuffle(indices)
            balanced_indices.extend(indices[:min_count])
        
        np.random.shuffle(balanced_indices)
        dataset = torch.utils.data.Subset(dataset, balanced_indices)
        print(f"Undersampled dataset size: {len(dataset)}")
        
    elif balance_method == 'oversample':
        # Oversample minority class using augmentation
        dataset = augment_dataset(dataset, target_size=max(class_counts.values()) * 2)
        print(f"Oversampled dataset size: {len(dataset)}")
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = KolamClassifier(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Define loss function
    if balance_method == 'weighted_loss':
        # Compute class weights
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted loss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate detailed metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Sikku', 'Sikku'], 
                yticklabels=['Non-Sikku', 'Sikku'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('binary_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    model_filename = f'balanced_kolam_classifier_{balance_method}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"\nModel saved as: {model_filename}")
    
    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Dataset directory
    data_dir = "kolam_dataset"
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found!")
        exit(1)
    
    # Train models with different balancing techniques
    methods = ['undersample', 'weighted_loss', 'oversample']
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {method.upper()} METHOD")
        print(f"{'='*60}")
        
        model, metrics = train_balanced_classifier(
            data_dir=data_dir,
            balance_method=method,
            num_epochs=20,
            batch_size=16,
            learning_rate=1e-4
        )
        
        results[method] = metrics
        
        print(f"\n{method.upper()} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON OF METHODS")
    print(f"{'='*60}")
    
    for method, metrics in results.items():
        print(f"{method.upper():<15} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    
    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['f1'])
    print(f"\nBest method: {best_method.upper()} with F1-Score: {results[best_method]['f1']:.4f}")