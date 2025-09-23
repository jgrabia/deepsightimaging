#!/usr/bin/env python3
"""
Training Log Parser for DBT Lesion Detection
Extracts metrics from training output and saves to structured log file
"""

import re
import json
import os
from datetime import datetime
from pathlib import Path

class TrainingLogParser:
    def __init__(self, log_file_path="/home/ubuntu/mri_app/training_progress.log"):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
    def parse_training_line(self, line):
        """Parse a single training output line for metrics"""
        metrics = {}
        
        try:
            # Parse epoch information
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                metrics['epoch'] = int(epoch_match.group(1))
                metrics['total_epochs'] = int(epoch_match.group(2))
            
            # Parse training metrics
            train_loss_match = re.search(r'Train Loss: ([\d.]+), Train Acc: ([\d.]+)%', line)
            if train_loss_match:
                metrics['train_loss'] = float(train_loss_match.group(1))
                metrics['train_acc'] = float(train_loss_match.group(2))
            
            # Parse validation metrics
            val_loss_match = re.search(r'Val Loss: ([\d.]+), Val Acc: ([\d.]+)%, AUC: ([\d.]+)', line)
            if val_loss_match:
                metrics['val_loss'] = float(val_loss_match.group(1))
                metrics['val_acc'] = float(val_loss_match.group(2))
                metrics['auc'] = float(val_loss_match.group(3))
            
            # Parse timestamp
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                metrics['timestamp'] = timestamp_match.group(1)
            else:
                metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Parse batch progress
            batch_match = re.search(r'(\d+)%\|.*\| (\d+)/(\d+)', line)
            if batch_match:
                metrics['batch_progress'] = int(batch_match.group(1))
                metrics['current_batch'] = int(batch_match.group(2))
                metrics['total_batches'] = int(batch_match.group(3))
            
            # Parse batch metrics
            batch_loss_match = re.search(r'Loss=([\d.]+), Acc=([\d.]+)%', line)
            if batch_loss_match:
                metrics['batch_loss'] = float(batch_loss_match.group(1))
                metrics['batch_acc'] = float(batch_loss_match.group(2))
            
            return metrics if metrics else None
            
        except Exception as e:
            print(f"Error parsing line: {e}")
            return None
    
    def log_metrics(self, metrics):
        """Log metrics to structured log file"""
        if not metrics:
            return
        
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write to log file
        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def parse_and_log(self, line):
        """Parse line and log metrics if found"""
        metrics = self.parse_training_line(line)
        if metrics:
            self.log_metrics(metrics)
            return metrics
        return None

def test_parser():
    """Test the parser with sample training output"""
    parser = TrainingLogParser()
    
    # Sample training output lines
    test_lines = [
        "Epoch 1/50",
        "Training:  43%|▊ | 163/382 [4:25:43<3:16:22, 53.80s/it, Loss=0.0505, Acc=50.54%]",
        "2025-09-20 04:08:08,504 - INFO - Train Loss: 0.0576, Train Acc: 49.90%",
        "Validation: 100%|████| 82/82 [2:14:18<00:00, 98.28s/it, Loss=0.0322, Acc=87.77%]",
        "2025-09-20 06:22:27,186 - INFO - Val Loss: 0.0376, Val Acc: 87.77%, AUC: 0.5144",
        "2025-09-20 06:22:29,883 - INFO - New best model saved with AUC: 0.5144"
    ]
    
    print("Testing Training Log Parser:")
    print("=" * 50)
    
    for line in test_lines:
        print(f"Input: {line}")
        metrics = parser.parse_and_log(line)
        if metrics:
            print(f"Parsed: {json.dumps(metrics, indent=2)}")
        else:
            print("No metrics found")
        print("-" * 30)

if __name__ == "__main__":
    test_parser()


