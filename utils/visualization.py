"""
Visualization utilities for brain tumor detection.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
import os
import sys
from model.grad_cam import GradCAM
from typing import List, Dict, Optional
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc
from PIL import Image

# ImageNet normalization constants
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Class names
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES

def plot_batch_samples(images, labels, predictions=None, class_names=CLASS_NAMES, num_samples=8):
    """
    Plot a batch of samples with their labels and predictions.
    
    Args:
        images (torch.Tensor): Batch of images
        labels (torch.Tensor): True labels
        predictions (torch.Tensor, optional): Predicted labels
        class_names (list): List of class names
        num_samples (int): Number of samples to plot
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if predictions is not None and isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Limit number of samples
    num_samples = min(num_samples, len(images))
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Get image, label and prediction
        img = images[i].transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        label = labels[i]
        pred = predictions[i] if predictions is not None else None
        
        # Plot image
        axes[i].imshow(img)
        title = f"True: {class_names[label]}"
        if pred is not None:
            title += f"\nPred: {class_names[pred]}"
            color = "green" if pred == label else "red"
            axes[i].set_title(title, color=color)
        else:
            axes[i].set_title(title)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training history.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_accs (list): Training accuracies
        val_accs (list): Validation accuracies
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

def plot_gradcam_results(images, labels, predictions, gradcam_images, class_names=CLASS_NAMES, num_samples=8):
    """
    Plot Grad-CAM results.
    
    Args:
        images (list): Original images
        labels (list): True labels
        predictions (list): Predicted labels
        gradcam_images (list): Grad-CAM visualization images
        class_names (list): List of class names
        num_samples (int): Number of samples to plot
    """
    # Limit number of samples
    num_samples = min(num_samples, len(images))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Get original image
        axes[i, 0].imshow(images[i])
        title = f"Original\nTrue: {class_names[labels[i]]}"
        axes[i, 0].set_title(title)
        axes[i, 0].axis('off')
        
        # Get Grad-CAM image
        axes[i, 1].imshow(gradcam_images[i])
        title = f"Grad-CAM\nPred: {class_names[predictions[i]]}"
        color = "green" if predictions[i] == labels[i] else "red"
        axes[i, 1].set_title(title, color=color)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_model_predictions(model, dataloader, device, class_names=CLASS_NAMES, num_samples=8):
    """
    Visualize model predictions on a batch of samples.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader for visualization
        device (torch.device): Device to use
        class_names (list): List of class names
        num_samples (int): Number of samples to visualize
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of samples
    images, labels, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Plot batch samples
    images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    fig = plot_batch_samples(images_np, labels_np, predictions_np, class_names, num_samples)
    return fig

def visualize_gradcam(model, dataloader, device, class_names, num_samples=8):
    """
    Generate Grad-CAM visualizations for a batch of images
    """
    # Initialize GradCAM
    gradcam = GradCAM(model)
    
    # Get a batch of images
    images, labels, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Generate Grad-CAM maps
    gradcam_maps = []
    for i in range(num_samples):
        gradcam_map = gradcam(images[i:i+1], target_class=preds[i].item())
        if gradcam_map is not None:
            gradcam_maps.append(gradcam_map)
        else:
            print(f"Failed to generate Grad-CAM for sample {i}")
            gradcam_maps.append(np.zeros((224, 224)))  # Default size
    
    # Create figure
    fig = plt.figure(figsize=(20, 4 * ((num_samples + 3) // 4)))
    
    for i in range(num_samples):
        # Original image
        ax = plt.subplot(((num_samples + 3) // 4), 4, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * std + mean, 0, 1)
        plt.imshow(img)
        
        # Overlay Grad-CAM
        heatmap = gradcam_maps[i]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.float32(heatmap) / 255 * 0.5 + np.float32(img)
        overlay = overlay / np.max(overlay)
        plt.imshow(overlay, alpha=0.5)
        
        plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    return fig

class ModelVisualizer:
    """Visualization tools for model architecture and training."""
    
    def __init__(self, log_dir: str):
        """
        Initialize visualizer.
        
        Args:
            log_dir (str): Directory to save visualizations
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
    def plot_model_architecture(self, model: nn.Module, input_shape: tuple, save_path: Optional[str] = None):
        """
        Plot model architecture diagram.
        
        Args:
            model (nn.Module): PyTorch model
            input_shape (tuple): Input tensor shape
            save_path (str, optional): Path to save the plot
        """
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Plot model graph
        self.writer.add_graph(model, dummy_input)
        
        if save_path:
            # Save architecture diagram
            plt.figure(figsize=(15, 10))
            plt.title('Model Architecture')
            plt.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_attention_maps(self, 
                          attention_maps: Dict[str, torch.Tensor],
                          save_path: Optional[str] = None):
        """
        Plot attention maps.
        
        Args:
            attention_maps (Dict[str, torch.Tensor]): Dictionary of attention maps
            save_path (str, optional): Path to save the plot
        """
        n_maps = len(attention_maps)
        fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))
        
        for i, (name, attn_map) in enumerate(attention_maps.items()):
            if n_maps > 1:
                ax = axes[i]
            else:
                ax = axes
                
            # Convert to numpy and normalize
            attn_map = attn_map.squeeze().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            
            # Plot attention map
            ax.imshow(attn_map, cmap='viridis')
            ax.set_title(name)
            ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_training_metrics(self,
                            metrics: Dict[str, List[float]],
                            save_path: Optional[str] = None):
        """
        Plot training metrics.
        
        Args:
            metrics (Dict[str, List[float]]): Dictionary of metric values
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(15, 5))
        
        for name, values in metrics.items():
            plt.plot(values, label=name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str],
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): List of class names
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_scores: np.ndarray,
                       class_names: List[str],
                       save_path: Optional[str] = None):
        """
        Plot ROC curves.
        
        Args:
            y_true (np.ndarray): True labels
            y_scores (np.ndarray): Predicted scores
            class_names (List[str]): List of class names
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_segmentation_results(self,
                                image: np.ndarray,
                                ground_truth: np.ndarray,
                                prediction: np.ndarray,
                                save_path: Optional[str] = None):
        """
        Plot segmentation results.
        
        Args:
            image (np.ndarray): Original image
            ground_truth (np.ndarray): Ground truth mask
            prediction (np.ndarray): Predicted mask
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(ground_truth, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def close(self):
        """Close the tensorboard writer."""
        self.writer.close()