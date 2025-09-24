"""
Few-Shot Kolam Design Classifier using Prototypical Networks
This approach is specifically designed for scenarios with limited samples per class.

Key Features:
- Prototypical Networks for few-shot learning
- Episodic training with support/query sets
- Optimized for classes with 2-100 samples
- Distance-based classification instead of traditional softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datasets import load_dataset
from collections import Counter, defaultdict
import random
import time
from huggingface_hub.errors import HfHubHTTPError
import json
from copy import deepcopy
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class KolamFewShotDataset(Dataset):
    """Dataset for few-shot learning with episodic training"""
    
    def __init__(self, hf_dataset, transform=None, mode='train'):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.mode = mode
        
        # Organize data by class
        self.class_to_images = defaultdict(list)
        self.class_names = []
        
        for idx, item in enumerate(hf_dataset):
            label = item['label']
            if label not in self.class_names:
                self.class_names.append(label)
            self.class_to_images[label].append(idx)
        
        self.num_classes = len(self.class_names)
        print(f"Dataset initialized with {len(hf_dataset)} samples across {self.num_classes} classes")
        
        # Print class distribution
        for class_name in self.class_names:
            count = len(self.class_to_images[class_name])
            print(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        label = self.class_names.index(item['label'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def create_episode(self, n_way=5, k_shot=5, q_query=10):
        """Create an episode for few-shot learning"""
        # Select n_way classes
        selected_classes = random.sample(range(self.num_classes), min(n_way, self.num_classes))
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for class_idx, selected_class in enumerate(selected_classes):
            class_images = self.class_to_images[self.class_names[selected_class]]
            
            # Handle classes with fewer samples than k_shot + q_query
            available_samples = len(class_images)
            actual_k_shot = min(k_shot, available_samples // 2)
            actual_q_query = min(q_query, available_samples - actual_k_shot)
            
            if actual_k_shot == 0 or actual_q_query == 0:
                continue
            
            # Sample support and query sets
            sampled_indices = random.sample(class_images, actual_k_shot + actual_q_query)
            support_indices = sampled_indices[:actual_k_shot]
            query_indices = sampled_indices[actual_k_shot:actual_k_shot + actual_q_query]
            
            # Add support samples
            for idx in support_indices:
                image, _ = self.__getitem__(idx)
                support_images.append(image)
                support_labels.append(class_idx)
            
            # Add query samples
            for idx in query_indices:
                image, _ = self.__getitem__(idx)
                query_images.append(image)
                query_labels.append(class_idx)
        
        return {
            'support_images': torch.stack(support_images) if support_images else torch.empty(0),
            'support_labels': torch.tensor(support_labels),
            'query_images': torch.stack(query_images) if query_images else torch.empty(0),
            'query_labels': torch.tensor(query_labels),
            'n_way': len(set(support_labels))
        }

class PrototypicalNetwork(nn.Module):
    """Prototypical Network for Few-Shot Learning"""
    
    def __init__(self, backbone='efficientnet_b0', feature_dim=1280):
        super(PrototypicalNetwork, self).__init__()
        
        # Feature extractor backbone
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # Remove the classifier layer
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 1280
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.backbone.fc = nn.Identity()
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Optional feature projection layer
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )
        
        print(f"Prototypical Network initialized with {backbone} backbone")
        print(f"Feature dimension: {self.feature_dim} -> 256")
    
    def forward(self, x):
        """Extract features from input images"""
        features = self.backbone(x)
        features = self.projector(features)
        return features
    
    def compute_prototypes(self, support_features, support_labels, n_way):
        """Compute class prototypes from support set"""
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            if class_mask.sum() > 0:
                class_features = support_features[class_mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
        
        return torch.stack(prototypes) if prototypes else torch.empty(0, support_features.size(-1))
    
    def classify(self, query_features, prototypes):
        """Classify query samples based on distance to prototypes"""
        # Compute euclidean distances
        distances = torch.cdist(query_features, prototypes)
        # Convert to logits (negative distances)
        logits = -distances
        return logits

class FewShotKolamTrainer:
    """Trainer for Few-Shot Kolam Classification"""
    
    def __init__(self, device=None, cache_dir=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir or "kolam_cache"
        
        # Training parameters
        self.n_way = 5  # Number of classes per episode
        self.k_shot = 3  # Number of support samples per class
        self.q_query = 5  # Number of query samples per class
        self.episodes_per_epoch = 100
        self.num_epochs = 50
        
        print(f"Few-Shot Trainer initialized on {self.device}")
        print(f"Episode configuration: {self.n_way}-way, {self.k_shot}-shot, {self.q_query} queries")
    
    def load_dataset_with_retry(self, max_retries=3):
        """Load Kolam dataset with retry mechanism"""
        for attempt in range(max_retries):
            try:
                print(f"Loading dataset from HuggingFace... (Attempt {attempt + 1}/{max_retries})")
                
                dataset = load_dataset(
                    "ayshthkr/kolam_dataset", 
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                
                print("Dataset loaded successfully!")
                return dataset['train']
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def get_transforms(self):
        """Get data transforms for few-shot learning"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_splits(self, dataset, train_ratio=0.7, val_ratio=0.15):
        """Create train/val/test splits while maintaining class distribution"""
        from sklearn.model_selection import train_test_split
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, item in enumerate(dataset):
            class_indices[item['label']].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_name, indices in class_indices.items():
            if len(indices) < 3:  # Too few samples for splitting
                # Put all in training set for few-shot learning
                train_indices.extend(indices)
                continue
            
            # Split while maintaining class distribution
            train_idx, temp_idx = train_test_split(
                indices, 
                train_size=train_ratio, 
                random_state=42
            )
            
            if len(temp_idx) >= 2:
                val_idx, test_idx = train_test_split(
                    temp_idx, 
                    train_size=val_ratio / (val_ratio + (1 - train_ratio - val_ratio)), 
                    random_state=42
                )
            else:
                val_idx = temp_idx
                test_idx = []
            
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)
        
        print(f"Dataset splits created:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples") 
        print(f"  Test: {len(test_indices)} samples")
        
        return train_indices, val_indices, test_indices
    
    def train_episode(self, model, episode, optimizer, criterion=None):
        """Train on a single episode"""
        model.train()
        
        support_images = episode['support_images'].to(self.device)
        support_labels = episode['support_labels'].to(self.device)
        query_images = episode['query_images'].to(self.device)
        query_labels = episode['query_labels'].to(self.device)
        n_way = episode['n_way']
        
        if support_images.size(0) == 0 or query_images.size(0) == 0:
            return 0.0, 0.0
        
        # Extract features
        support_features = model(support_images)
        query_features = model(query_images)
        
        # Compute prototypes
        prototypes = model.compute_prototypes(support_features, support_labels, n_way)
        
        if prototypes.size(0) == 0:
            return 0.0, 0.0
        
        # Classify queries
        logits = model.classify(query_features, prototypes)
        
        # Compute loss
        loss = F.cross_entropy(logits, query_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        return loss.item(), accuracy.item()
    
    def validate_episode(self, model, episode):
        """Validate on a single episode"""
        model.eval()
        
        with torch.no_grad():
            support_images = episode['support_images'].to(self.device)
            support_labels = episode['support_labels'].to(self.device)
            query_images = episode['query_images'].to(self.device)
            query_labels = episode['query_labels'].to(self.device)
            n_way = episode['n_way']
            
            if support_images.size(0) == 0 or query_images.size(0) == 0:
                return 0.0, 0.0
            
            # Extract features
            support_features = model(support_images)
            query_features = model(query_images)
            
            # Compute prototypes
            prototypes = model.compute_prototypes(support_features, support_labels, n_way)
            
            if prototypes.size(0) == 0:
                return 0.0, 0.0
            
            # Classify queries
            logits = model.classify(query_features, prototypes)
            
            # Compute loss and accuracy
            loss = F.cross_entropy(logits, query_labels)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == query_labels).float().mean()
            
            return loss.item(), accuracy.item()
    
    def train(self, dataset):
        """Main training loop for few-shot learning"""
        print("\n🚀 Starting Few-Shot Learning Training")
        print("=" * 60)
        
        # Create transforms
        train_transform, val_transform = self.get_transforms()
        
        # Create splits
        train_indices, val_indices, test_indices = self.create_splits(dataset)
        
        # Create subset datasets
        train_subset = dataset.select(train_indices)
        val_subset = dataset.select(val_indices) if val_indices else train_subset
        
        # Create few-shot datasets
        train_dataset = KolamFewShotDataset(train_subset, train_transform, 'train')
        val_dataset = KolamFewShotDataset(val_subset, val_transform, 'val')
        
        # Initialize model
        model = PrototypicalNetwork(backbone='efficientnet_b0').to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training metrics
        best_val_acc = 0.0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        print(f"Training for {self.num_epochs} epochs...")
        print(f"Episodes per epoch: {self.episodes_per_epoch}")
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            epoch_train_losses = []
            epoch_train_accs = []
            
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            for episode_idx in range(self.episodes_per_epoch):
                # Create training episode
                episode = train_dataset.create_episode(
                    n_way=min(self.n_way, train_dataset.num_classes),
                    k_shot=self.k_shot,
                    q_query=self.q_query
                )
                
                loss, acc = self.train_episode(model, episode, optimizer)
                
                if loss > 0:  # Valid episode
                    epoch_train_losses.append(loss)
                    epoch_train_accs.append(acc)
                
                if episode_idx % 20 == 0:
                    current_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
                    current_acc = np.mean(epoch_train_accs) if epoch_train_accs else 0.0
                    print(f"  Episode {episode_idx:3d}/{self.episodes_per_epoch}, "
                          f"Loss: {current_loss:.4f}, Acc: {current_acc:.2%}")
            
            # Validation phase
            model.eval()
            epoch_val_losses = []
            epoch_val_accs = []
            
            val_episodes = min(20, self.episodes_per_epoch // 5)  # Fewer validation episodes
            
            for _ in range(val_episodes):
                episode = val_dataset.create_episode(
                    n_way=min(self.n_way, val_dataset.num_classes),
                    k_shot=self.k_shot,
                    q_query=self.q_query
                )
                
                loss, acc = self.validate_episode(model, episode)
                
                if loss > 0:  # Valid episode
                    epoch_val_losses.append(loss)
                    epoch_val_accs.append(acc)
            
            # Calculate epoch metrics
            avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
            avg_train_acc = np.mean(epoch_train_accs) if epoch_train_accs else 0.0
            avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
            avg_val_acc = np.mean(epoch_val_accs) if epoch_val_accs else 0.0
            
            # Store metrics
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2%}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2%}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                print(f"✅ New best validation accuracy: {best_val_acc:.2%}")
                
                # Save model
                model_save_path = "few_shot_kolam_classifier.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_names': train_dataset.class_names,
                    'model_config': {
                        'backbone': 'efficientnet_b0',
                        'feature_dim': 256,
                        'n_way': self.n_way,
                        'k_shot': self.k_shot
                    }
                }, model_save_path)
        
        print(f"\n✅ Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        
        return model, {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def evaluate_model(self, model, dataset, test_indices):
        """Comprehensive evaluation of the few-shot model"""
        if not test_indices:
            print("No test samples available for evaluation")
            return
        
        print("\n📊 Evaluating Few-Shot Model on Test Set")
        print("=" * 50)
        
        # Create test dataset
        test_subset = dataset.select(test_indices)
        _, val_transform = self.get_transforms()
        test_dataset = KolamFewShotDataset(test_subset, val_transform, 'test')
        
        model.eval()
        all_predictions = []
        all_labels = []
        test_episodes = 50
        
        print(f"Running {test_episodes} test episodes...")
        
        for episode_idx in range(test_episodes):
            episode = test_dataset.create_episode(
                n_way=min(self.n_way, test_dataset.num_classes),
                k_shot=self.k_shot,
                q_query=self.q_query
            )
            
            with torch.no_grad():
                support_images = episode['support_images'].to(self.device)
                support_labels = episode['support_labels'].to(self.device)
                query_images = episode['query_images'].to(self.device)
                query_labels = episode['query_labels'].to(self.device)
                n_way = episode['n_way']
                
                if support_images.size(0) == 0 or query_images.size(0) == 0:
                    continue
                
                # Extract features and classify
                support_features = model(support_images)
                query_features = model(query_images)
                prototypes = model.compute_prototypes(support_features, support_labels, n_way)
                
                if prototypes.size(0) > 0:
                    logits = model.classify(query_features, prototypes)
                    predictions = logits.argmax(dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(query_labels.cpu().numpy())
        
        if all_predictions:
            accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
            print(f"Overall Test Accuracy: {accuracy:.2%}")
            
            # Per-class analysis
            unique_labels = sorted(set(all_labels))
            for label in unique_labels:
                label_mask = np.array(all_labels) == label
                if label_mask.sum() > 0:
                    label_acc = np.mean(np.array(all_predictions)[label_mask] == label)
                    class_name = test_dataset.class_names[label] if label < len(test_dataset.class_names) else f"Class_{label}"
                    print(f"  {class_name}: {label_acc:.2%}")
        
        return model

def main():
    """Main function to run few-shot learning training"""
    print("🔥 Few-Shot Kolam Design Classifier")
    print("=" * 60)
    
    # Initialize trainer
    trainer = FewShotKolamTrainer(cache_dir="kolam_cache")
    
    try:
        # Load dataset
        dataset = trainer.load_dataset_with_retry()
        
        # Train few-shot model
        model, metrics = trainer.train(dataset)
        
        # Evaluate model
        _, _, test_indices = trainer.create_splits(dataset)
        trainer.evaluate_model(model, dataset, test_indices)
        
        print("\n🎉 Few-Shot Learning Pipeline Completed Successfully!")
        print("Model saved as 'few_shot_kolam_classifier.pth'")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()