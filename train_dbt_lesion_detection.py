#!/usr/bin/env python3
"""
DBT Lesion Detection Training Script
Trains a deep learning model to detect lesions in Digital Breast Tomosynthesis (DBT) images
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from pathlib import Path
import pydicom
from PIL import Image
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import training log parser
from training_log_parser import TrainingLogParser

# Configuration
class Config:
    # Data paths
    DATA_ROOT = Path("/home/ubuntu/mri_app/dbt_complete_training_data")
    MODEL_SAVE_PATH = Path("/home/ubuntu/mri_app/models")
    LOGS_PATH = Path("/home/ubuntu/mri_app/logs")
    
    # Model parameters
    IMAGE_SIZE = (512, 512)  # Resize DBT slices to this size
    SLICE_SAMPLING = 'middle'  # 'middle', 'random', 'all'
    NUM_CLASSES = 2  # 0: no lesion, 1: lesion
    
    # Training parameters
    BATCH_SIZE = 8  # Small batch due to large images
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    
    # Class balancing
    USE_WEIGHTED_SAMPLING = True
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    Config.LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    log_filename = Config.LOGS_PATH / f"dbt_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DBTDataset(Dataset):
    """Dataset class for DBT lesion detection"""
    
    def __init__(self, data_dir, split='train', transform=None, slice_sampling='middle'):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.slice_sampling = slice_sampling
        
        # Load patient data
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Calculate class distribution
        labels = [sample['label'] for sample in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
    
    def _load_samples(self):
        """Load all patient samples"""
        for case_type in ['positive', 'negative']:
            case_dir = self.data_dir / case_type
            if not case_dir.exists():
                continue
                
            label = 1 if case_type == 'positive' else 0
            
            for patient_dir in case_dir.iterdir():
                if not patient_dir.is_dir():
                    continue
                
                # Find DICOM files
                dicom_files = list(patient_dir.glob('*.dcm'))
                if not dicom_files:
                    continue
                
                # Load annotation file
                annotation_file = patient_dir / f"{patient_dir.name}_annotations.json"
                annotations = {}
                if annotation_file.exists():
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                
                # Add sample
                self.samples.append({
                    'patient_id': patient_dir.name,
                    'dicom_files': dicom_files,
                    'label': label,
                    'annotations': annotations,
                    'case_type': case_type
                })
    
    def _load_dicom_slice(self, dicom_path, slice_idx=None):
        """Load a specific slice from DICOM file"""
        try:
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array
            
            # Handle 3D volumes (DBT)
            if len(pixel_array.shape) == 3:
                if slice_idx is None:
                    # Use middle slice by default
                    slice_idx = pixel_array.shape[0] // 2
                elif slice_idx >= pixel_array.shape[0]:
                    slice_idx = pixel_array.shape[0] - 1
                
                slice_data = pixel_array[slice_idx, :, :]
            else:
                # 2D image
                slice_data = pixel_array
            
            # Normalize to 0-255
            if slice_data.max() > 255:
                slice_data = ((slice_data / slice_data.max()) * 255).astype(np.uint8)
            else:
                slice_data = slice_data.astype(np.uint8)
            
            return slice_data
            
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            return None
    
    def _get_slice_with_lesion(self, dicom_files, annotations):
        """Get a slice that contains a lesion"""
        if not annotations.get('lesions'):
            return None, None
        
        # Try to find a slice with lesions
        for lesion in annotations['lesions']:
            slice_idx = lesion.get('slice', 0)
            
            # Try each DICOM file
            for dicom_file in dicom_files:
                slice_data = self._load_dicom_slice(dicom_file, slice_idx)
                if slice_data is not None:
                    return slice_data, slice_idx
        
        return None, None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Select slice strategy
        slice_data = None
        slice_idx = None
        
        if sample['label'] == 1:  # Positive case
            if self.slice_sampling == 'lesion':
                # Try to get slice with lesion
                slice_data, slice_idx = self._get_slice_with_lesion(
                    sample['dicom_files'], 
                    sample['annotations']
                )
        
        # Fallback to first DICOM file, middle slice
        if slice_data is None:
            dicom_file = sample['dicom_files'][0]
            slice_data = self._load_dicom_slice(dicom_file)
        
        if slice_data is None:
            # Create dummy data if loading fails
            slice_data = np.zeros((512, 512), dtype=np.uint8)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(slice_data).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['patient_id']

class DBTLesionNet(nn.Module):
    """CNN for DBT lesion detection"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(DBTLesionNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_transforms():
    """Get data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_weighted_sampler(dataset):
    """Create weighted sampler for handling class imbalance"""
    labels = [sample['label'] for sample in dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, labels, patient_ids) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, logger):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, labels, patient_ids in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
        logger.info(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%, AUC: {auc:.4f}")
    except:
        auc = 0.0
        logger.info(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc, auc, all_predictions, all_labels

def save_model(model, optimizer, epoch, loss, acc, auc, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'auc': auc,
        'config': {
            'image_size': Config.IMAGE_SIZE,
            'num_classes': Config.NUM_CLASSES,
            'model_architecture': 'DBTLesionNet'
        }
    }, filepath)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Main training function"""
    # Setup
    logger = setup_logging()
    Config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize training log parser
    log_parser = TrainingLogParser()
    
    logger.info("Starting DBT Lesion Detection Training")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Data root: {Config.DATA_ROOT}")
    
    # Check if data exists
    if not Config.DATA_ROOT.exists():
        logger.error(f"Data directory not found: {Config.DATA_ROOT}")
        return
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = DBTDataset(Config.DATA_ROOT, 'train', train_transform)
    val_dataset = DBTDataset(Config.DATA_ROOT, 'val', val_transform)
    
    # Create data loaders
    if Config.USE_WEIGHTED_SAMPLING:
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = DBTLesionNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Loss function and optimizer
    criterion = FocalLoss(
        alpha=Config.FOCAL_LOSS_ALPHA, 
        gamma=Config.FOCAL_LOSS_GAMMA
    )
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    logger.info("Starting training loop...")
    
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, logger
        )
        
        # Validate
        val_loss, val_acc, val_auc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, Config.DEVICE, logger
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log metrics to structured log file
        log_parser.log_metrics({
            'epoch': epoch + 1,
            'total_epochs': Config.NUM_EPOCHS,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'auc': val_auc
        })
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            
            model_path = Config.MODEL_SAVE_PATH / "best_dbt_lesion_model.pth"
            save_model(model, optimizer, epoch, val_loss, val_acc, val_auc, model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    final_model_path = Config.MODEL_SAVE_PATH / "final_dbt_lesion_model.pth"
    save_model(model, optimizer, epoch, val_loss, val_acc, val_auc, final_model_path)
    
    # Plot training history
    plot_path = Config.MODEL_SAVE_PATH / "training_history.png"
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    logger.info("Training completed!")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Models saved to: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
