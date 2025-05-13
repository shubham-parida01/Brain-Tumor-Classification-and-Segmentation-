"""
Test script for verifying new implementations.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import DWTPreprocessor
from model.feature_selection import CSAFeatureSelector
from model.attention import MultiScaleAttention
from utils.visualization import ModelVisualizer
from config import Config

def test_dwt_preprocessing():
    """Test DWT preprocessing implementation."""
    print("\nTesting DWT Preprocessing...")
    
    # Initialize preprocessor
    dwt = DWTPreprocessor(
        wavelet='db4',
        levels=3,
        mode='soft',
        threshold_method='BayesShrink'
    )
    
    # Create a sample image
    sample_image = np.random.rand(224, 224, 3) * 255
    sample_image = sample_image.astype(np.uint8)
    
    # Apply preprocessing
    preprocessed = dwt(sample_image)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed)
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    plt.savefig('test_drive/dwt_test.png')
    plt.close()
    
    print("✓ DWT preprocessing test completed")

def test_csa_feature_selection():
    """Test CSA feature selection implementation."""
    print("\nTesting CSA Feature Selection...")
    
    # Initialize selector
    csa = CSAFeatureSelector(
        n_crows=50,
        max_iter=100,
        flight_length=2.0,
        awareness_prob=0.1
    )
    
    # Create sample features and labels
    n_samples = 100
    n_features = 50
    features = torch.randn(n_samples, n_features)
    labels = torch.randint(0, 4, (n_samples,))
    
    # Run feature selection
    selected_mask, fitness_history = csa.select_features(features, labels)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(selected_mask.reshape(1, -1), cmap='gray')
    plt.title('Selected Features')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.plot(fitness_history)
    plt.title('Fitness History')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    
    plt.savefig('test_drive/csa_test.png')
    plt.close()
    
    print("✓ CSA feature selection test completed")

def test_attention_mechanisms():
    """Test attention mechanisms implementation."""
    print("\nTesting Attention Mechanisms...")
    
    # Initialize attention module
    attention = MultiScaleAttention(in_channels=64)
    
    # Create sample input
    batch_size = 4
    channels = 64
    height = 224
    width = 224
    x = torch.randn(batch_size, channels, height, width)
    
    # Apply attention
    output = attention(x)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Input Feature Map')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Output Feature Map')
    plt.axis('off')
    
    plt.savefig('test_drive/attention_test.png')
    plt.close()
    
    print("✓ Attention mechanisms test completed")

def test_visualization():
    """Test visualization tools."""
    print("\nTesting Visualization Tools...")
    
    # Initialize visualizer
    visualizer = ModelVisualizer(log_dir='test_drive/logs')
    
    # Test model architecture visualization
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3)
            self.attention = MultiScaleAttention(64)
            self.fc = torch.nn.Linear(64, 4)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.attention(x)
            x = x.mean(dim=[2, 3])
            x = self.fc(x)
            return x
    
    model = DummyModel()
    visualizer.plot_model_architecture(model, (3, 224, 224), 'test_drive/model_architecture.png')
    
    # Test attention map visualization
    attention_maps = {
        'channel': torch.randn(1, 64, 224, 224),
        'spatial': torch.randn(1, 64, 224, 224)
    }
    visualizer.plot_attention_maps(attention_maps, 'test_drive/attention_maps.png')
    
    # Test training metrics visualization
    metrics = {
        'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'accuracy': [0.6, 0.7, 0.8, 0.9, 0.95]
    }
    visualizer.plot_training_metrics(metrics, 'test_drive/training_metrics.png')
    
    # Test confusion matrix visualization
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    visualizer.plot_confusion_matrix(y_true, y_pred, class_names, 'test_drive/confusion_matrix.png')
    
    # Test ROC curves visualization
    y_scores = np.random.rand(8, 4)
    visualizer.plot_roc_curves(y_true, y_scores, class_names, 'test_drive/roc_curves.png')
    
    # Test segmentation results visualization
    image = np.random.rand(224, 224, 3)
    ground_truth = np.random.rand(224, 224) > 0.5
    prediction = np.random.rand(224, 224) > 0.5
    visualizer.plot_segmentation_results(image, ground_truth, prediction, 'test_drive/segmentation_results.png')
    
    visualizer.close()
    print("✓ Visualization tools test completed")

def main():
    """Run all tests."""
    print("Starting test runs for new implementations...")
    
    # Create test directory if it doesn't exist
    os.makedirs('test_drive', exist_ok=True)
    
    # Run tests
    test_dwt_preprocessing()
    test_csa_feature_selection()
    test_attention_mechanisms()
    test_visualization()
    
    print("\nAll tests completed successfully!")
    print("Check the 'test_drive' directory for visualization outputs.")

if __name__ == "__main__":
    main() 