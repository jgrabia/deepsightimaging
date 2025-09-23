import pydicom
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BalancedBreastDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Create class mapping
        unique_classes = sorted(dataframe['class_label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        file_path = row['file_path']
        class_label = row['class_label']
        
        try:
            # Read DICOM file
            ds = pydicom.dcmread(file_path)
            image = ds.pixel_array.astype(np.float32)
            
            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Resize image to standard size (224x224)
            from PIL import Image
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_image = pil_image.resize((224, 224), Image.LANCZOS)
            image = np.array(pil_image).astype(np.float32) / 255.0
            
            # Convert to 3-channel (repeat grayscale)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=0)
            else:
                image = np.transpose(image, (2, 0, 1))
            
            # Convert to tensor
            image = torch.FloatTensor(image)
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            # Get class index
            class_idx = self.class_to_idx[class_label]
            
            return image, class_idx
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy image
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, 0

class BalancedBreastClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(BalancedBreastClassifier, self).__init__()
        
        # Use a pre-trained ResNet backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BalancedBreastTrainer:
    def __init__(self, data_dir='consolidated_training_data', config=None):
        self.data_dir = Path(data_dir)
        self.config = config or {
            'epochs': 50,
            'learning_rate': 1e-4,
            'batch_size': 16,
            'patience': 15,
            'train_split': 0.8
        }
        
        # Load balanced dataset
        self.df = pd.read_csv('balanced_dataset.csv')
        print(f"Loaded balanced dataset: {len(self.df)} files")
        print(f"Classes: {self.df['class_label'].value_counts().to_dict()}")
        
        # Create class mapping
        unique_classes = sorted(self.df['class_label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(unique_classes)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Class mapping: {self.class_to_idx}")
        
    def create_data_loaders(self):
        # Split data
        train_df, val_df = train_test_split(
            self.df, 
            test_size=1-self.config['train_split'], 
            stratify=self.df['class_label'],
            random_state=42
        )
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        # Create datasets
        train_dataset = BalancedBreastDataset(train_df)
        val_dataset = BalancedBreastDataset(val_df)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        return train_loader, val_loader
    
    def train_model(self):
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()
        
        # Create model
        model = BalancedBreastClassifier(self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Training on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_balanced_breast_model.pth')
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        model.load_state_dict(torch.load('best_balanced_breast_model.pth'))
        
        # Final evaluation
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Classification report
        target_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        report = classification_report(all_targets, all_predictions, target_names=target_names)
        print(f'\nClassification Report:\n{report}')
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(all_targets, all_predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        
        plt.tight_layout()
        plt.savefig('balanced_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return model, report

    def run_training(self):
        print("Starting Balanced Breast MRI Classification Training...")
        print("=" * 60)
        
        model, report = self.train_model()
        
        print("\nTraining completed!")
        print(f"Best model saved as: best_balanced_breast_model.pth")
        print(f"Results plot saved as: balanced_training_results.png")
        
        return model, report

if __name__ == "__main__":
    trainer = BalancedBreastTrainer()
    trainer.run_training()
