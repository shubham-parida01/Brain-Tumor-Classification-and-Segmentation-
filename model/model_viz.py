import torch
from network import BrainTumorClassifier
from segmentation import BrainTumorSegmentation
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_classifier():
    """Create visualization for the classifier model"""
    model = BrainTumorClassifier()
    
    # Create summary string
    summary = []
    summary.append("Brain Tumor Classification Model Architecture")
    summary.append("=" * 50)
    summary.append("\nBackbone: ResNet50")
    summary.append("Input size: (3, 224, 224)")
    summary.append(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Add detailed architecture information
    summary.append("\nDetailed Architecture:")
    summary.append("1. Feature Extraction (ResNet50)")
    summary.append("   - Initial conv: 7x7, 64 channels, stride 2")
    summary.append("   - MaxPool: 3x3, stride 2")
    summary.append("   - ResNet blocks: [3, 4, 6, 3]")
    summary.append("   - Output channels: [64, 128, 256, 512]")
    
    summary.append("\n2. Feature Pyramid Network (FPN)")
    summary.append("   - Lateral connections: 256 channels")
    summary.append("   - Top-down pathway with 3x3 convolutions")
    summary.append("   - Feature levels: P3, P4, P5, P6, P7")
    summary.append("   - Output channels: 256 at all levels")
    
    summary.append("\n3. Attention Mechanisms")
    summary.append("   - Channel Attention:")
    summary.append("     * Global average pooling")
    summary.append("     * MLP: 256 -> 128 -> 256")
    summary.append("     * Sigmoid activation")
    summary.append("   - Spatial Attention:")
    summary.append("     * 7x7 convolution")
    summary.append("     * Sigmoid activation")
    
    summary.append("\n4. Classification Head")
    summary.append("   - Global Average Pooling")
    summary.append("   - Dropout (p=0.5)")
    summary.append("   - FC layers: 256 -> 512 -> 256 -> 4")
    summary.append("   - BatchNorm after each FC layer")
    summary.append("   - ReLU activation")
    
    summary.append("\n5. Training Details")
    summary.append("   - Optimizer: Adam (lr=0.001)")
    summary.append("   - Loss: Cross Entropy")
    summary.append("   - Data augmentation: Random rotation, flip, color jitter")
    summary.append("   - Batch size: 32")
    
    # Save summary to file
    os.makedirs("research_visuals/model_architecture", exist_ok=True)
    with open("research_visuals/model_architecture/classifier_architecture.txt", "w") as f:
        f.write("\n".join(summary))

def visualize_segmentation():
    """Create visualization for the segmentation model"""
    model = BrainTumorSegmentation()
    
    # Create summary string
    summary = []
    summary.append("Brain Tumor Segmentation Model Architecture")
    summary.append("=" * 50)
    summary.append("\nArchitecture: Enhanced DeepLabV3+")
    summary.append("Input size: (3, 256, 256)")
    summary.append(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Add detailed architecture information
    summary.append("\nDetailed Architecture:")
    summary.append("1. Encoder Path")
    summary.append("   - Initial conv block:")
    summary.append("     * 7x7 conv, 64 channels, stride 2")
    summary.append("     * BatchNorm + ReLU")
    summary.append("   - Encoder blocks:")
    summary.append("     * Block 1: 64 channels, 2 layers")
    summary.append("     * Block 2: 128 channels, 2 layers")
    summary.append("     * Block 3: 256 channels, 2 layers")
    summary.append("     * Block 4: 512 channels, 2 layers")
    summary.append("   - Each block: Conv + BatchNorm + ReLU")
    summary.append("   - Max pooling: 2x2, stride 2")
    
    summary.append("\n2. ASPP Module")
    summary.append("   - Dilated convolutions:")
    summary.append("     * 1x1 conv: 256 channels")
    summary.append("     * 3x3 conv: rates [6, 12, 18], 256 channels")
    summary.append("   - Global average pooling branch")
    summary.append("   - Concatenation of all branches")
    summary.append("   - 1x1 conv to reduce channels")
    
    summary.append("\n3. Decoder Path")
    summary.append("   - Attention Gates:")
    summary.append("     * Query: Low-level features")
    summary.append("     * Key: High-level features")
    summary.append("     * Value: Low-level features")
    summary.append("   - Skip connections from encoder")
    summary.append("   - Upsampling blocks:")
    summary.append("     * 3x3 transposed conv")
    summary.append("     * BatchNorm + ReLU")
    
    summary.append("\n4. Final Output")
    summary.append("   - 1x1 convolution")
    summary.append("   - Single channel output")
    summary.append("   - Sigmoid activation")
    
    summary.append("\n5. Training Details")
    summary.append("   - Optimizer: Adam (lr=0.0001)")
    summary.append("   - Loss: Dice + BCE")
    summary.append("   - Data augmentation: Random rotation, flip, elastic transform")
    summary.append("   - Batch size: 16")
    
    # Save summary to file
    os.makedirs("research_visuals/model_architecture", exist_ok=True)
    with open("research_visuals/model_architecture/segmentation_architecture.txt", "w") as f:
        f.write("\n".join(summary))

def plot_performance_metrics():
    """Plot and save performance metrics"""
    # Create directory for performance metrics
    os.makedirs("research_visuals/performance_metrics", exist_ok=True)
    
    # Example metrics (replace with actual metrics from your training)
    metrics = {
        'Classifier': {
            'Accuracy': 0.95,
            'Precision': 0.94,
            'Recall': 0.93,
            'F1-Score': 0.935
        },
        'Segmentation': {
            'Dice Score': 0.89,
            'IoU': 0.85,
            'Precision': 0.88,
            'Recall': 0.87
        }
    }
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics['Classifier']))
    width = 0.35
    
    plt.bar(x - width/2, list(metrics['Classifier'].values()), width, label='Classifier')
    plt.bar(x + width/2, list(metrics['Segmentation'].values()), width, label='Segmentation')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.xticks(x, list(metrics['Classifier'].keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('research_visuals/performance_metrics/model_performance.png')
    plt.close()

if __name__ == "__main__":
    print("Generating model architecture visualizations...")
    visualize_classifier()
    visualize_segmentation()
    plot_performance_metrics()
    print("Done! Visualizations saved in research_visuals/") 