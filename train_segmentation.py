import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path

from model.segmentation import create_segmentation_model
from data.segmentation_dataset import create_segmentation_data_loaders
from utils.metrics import compute_segmentation_metrics

class SegmentationTrainer:
    """Trainer class for brain tumor segmentation model."""
    
    def __init__(self, model, device, config):
        """
        Initialize the trainer.
        
        Args:
            model: The segmentation model
            device: Device to train on (cuda/cpu)
            config: Configuration dictionary
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Setup directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = Path(config['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {
            'dice': 0,
            'iou': 0,
            'precision': 0,
            'recall': 0
        }
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                metrics = compute_segmentation_metrics(pred_masks, masks)
            
            # Update progress
            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': metrics['dice']
            })
        
        # Calculate epoch averages
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return epoch_loss, epoch_metrics
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        val_metrics = {
            'dice': 0,
            'iou': 0,
            'precision': 0,
            'recall': 0
        }
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                metrics = compute_segmentation_metrics(pred_masks, masks)
                
                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] += v
        
        # Calculate validation averages
        num_batches = len(val_loader)
        val_loss /= num_batches
        for k in val_metrics:
            val_metrics[k] /= num_batches
        
        return val_loss, val_metrics
    
    def save_checkpoint(self, epoch, val_loss, val_metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }
        
        # Save best model based on validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
        
        # Save latest model
        torch.save(checkpoint, self.checkpoint_dir / 'latest_model.pth')
    
    def train(self, train_loader, val_loader, num_epochs):
        """Train the model."""
        self.best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate(val_loader)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            for metric, value in train_metrics.items():
                self.writer.add_scalar(f'{metric}/train', value, epoch)
            for metric, value in val_metrics.items():
                self.writer.add_scalar(f'{metric}/val', value, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print("Train Metrics:", {k: f"{v:.4f}" for k, v in train_metrics.items()})
            print("Val Metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})

def main():
    parser = argparse.ArgumentParser(description='Train brain tumor segmentation model')
    parser.add_argument('--train_image_dir', type=str, required=True,
                      help='Directory containing training images')
    parser.add_argument('--train_mask_dir', type=str, required=True,
                      help='Directory containing training masks')
    parser.add_argument('--val_image_dir', type=str, required=True,
                      help='Directory containing validation images')
    parser.add_argument('--val_mask_dir', type=str, required=True,
                      help='Directory containing validation masks')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256,
                      help='Image size for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/segmentation',
                      help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/segmentation',
                      help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'img_size': args.img_size,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_segmentation_data_loaders(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        val_image_dir=args.val_image_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Create model
    model = create_segmentation_model()
    model = model.to(device)
    
    # Create trainer
    trainer = SegmentationTrainer(model, device, config)
    
    # Train model
    trainer.train(train_loader, val_loader, args.num_epochs)

if __name__ == '__main__':
    main() 