#!/usr/bin/env python3
"""
Breast Cancer Classification Trainer
Enterprise-ready classification model for breast imaging
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

class BreastClassificationDataset(Dataset):
    """Dataset for breast cancer classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load DICOM image
        ds = pydicom.dcmread(self.image_paths[idx])
        image = ds.pixel_array.astype(np.float32)
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to 3-channel if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

class BreastClassificationModel(nn.Module):
    """ResNet-based classification model for breast cancer"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Use ResNet50 as backbone
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify first layer for single channel input
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class BreastClassificationTrainer:
    """Enterprise-ready breast cancer classification trainer"""
    
    def __init__(self, data_dir, config=None):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default configuration
        self.config = {
            'num_classes': 3,  # Normal, Benign, Malignant
            'batch_size': 16,
            'learning_rate': 1e-4,
            'epochs': 50,
            'patience': 10,
            'image_size': 224,
            'pretrained': True,
            'weight_decay': 1e-4,
            'class_names': ['Normal', 'Benign', 'Malignant']
        }
        
        if config:
            self.config.update(config)
            
        # Setup logging
        self.setup_logging()
        
        # Create output directory
        self.output_dir = Path('trained_classification_models')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🏥 Breast Cancer Classification Trainer")
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"💻 Device: {self.device}")
        print(f"📦 Output Directory: {self.output_dir}")
        print(f"⚙️  Configuration: {json.dumps(self.config, indent=2)}")
        
    def setup_logging(self):
        """Setup logging for training"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"classification_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """Load and prepare classification data"""
        print("📊 Loading classification data...")
        
        # Get all DICOM files
        dicom_files = list(self.data_dir.rglob("*.dcm"))
        print(f"Found {len(dicom_files)} DICOM files")
        
        # Create labels based on directory structure
        # This is a simplified approach - in practice, you'd have proper labels
        labels = []
        valid_files = []
        
        for file_path in dicom_files:
            # Simple heuristic: if path contains 'cancer', label as malignant
            # if path contains 'normal', label as normal
            # otherwise, label as benign
            path_str = str(file_path).lower()
            
            if 'cancer' in path_str or 'malignant' in path_str:
                label = 2  # Malignant
            elif 'normal' in path_str or 'benign' in path_str:
                label = 0  # Normal
            else:
                label = 1  # Benign (default)
                
            labels.append(label)
            valid_files.append(file_path)
        
        print(f"📊 Label distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = self.config['class_names'][label]
            print(f"   {class_name}: {count} files")
        
        return valid_files, labels
    
    def create_transforms(self):
        """Create data transforms for training and validation"""
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def train_model(self):
        """Train the classification model"""
        print("🚀 Starting classification training...")
        
        # Load data
        image_paths, labels = self.load_data()
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"📊 Training samples: {len(train_paths)}")
        print(f"📊 Validation samples: {len(val_paths)}")
        
        # Create transforms
        train_transform, val_transform = self.create_transforms()
        
        # Create datasets
        train_dataset = BreastClassificationDataset(train_paths, train_labels, train_transform)
        val_dataset = BreastClassificationDataset(val_paths, val_labels, val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=4
        )
        
        # Create model
        model = BreastClassificationModel(
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{self.config["epochs"]}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
            
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Accuracy: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'config': self.config
                }, self.output_dir / 'best_classification_model.pth')
                print(f'  ✅ New best model saved (Val Acc: {val_acc:.4f})')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f'⏹️  Early stopping triggered after {self.config["patience"]} epochs without improvement')
                break
            
            scheduler.step(avg_train_loss)
        
        # Final evaluation
        self.evaluate_model(model, val_loader)
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'config': self.config
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"🎉 Training Complete!")
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
        
        return model, history
    
    def evaluate_model(self, model, val_loader):
        """Evaluate the trained model"""
        print("📊 Evaluating model...")
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Classification report
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.config['class_names'],
            output_dict=True
        )
        
        print("📊 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.config['class_names']))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.config['class_names'],
                   yticklabels=self.config['class_names'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()
        
        # Save evaluation results
        evaluation_results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'config': self.config
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return report

def main():
    """Main training function"""
    # Configuration
    config = {
        'num_classes': 3,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 50,
        'patience': 10,
        'image_size': 224,
        'pretrained': True
    }
    
    # Create trainer
    trainer = BreastClassificationTrainer('consolidated_training_data', config)
    
    # Train model
    model, history = trainer.train_model()
    
    print("🎉 Classification training complete!")
    print("📁 Model saved to: trained_classification_models/")
    print("📊 Check evaluation_results.json for detailed metrics")

if __name__ == "__main__":
    main()


