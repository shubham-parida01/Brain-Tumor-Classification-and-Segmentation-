import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from network import BrainTumorClassifier
from segmentation import BrainTumorSegmentation
import os

def visualize_attention_maps(model, input_tensor, save_path):
    """Visualize attention maps from the model"""
    model.eval()
    with torch.no_grad():
        # Get intermediate features
        features = model.get_intermediate_features(input_tensor)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Attention Mechanism Visualization', fontsize=16)
        
        # Plot channel attention
        channel_attn = model.channel_attention(features)
        axes[0, 0].imshow(channel_attn[0].cpu().numpy(), cmap='viridis')
        axes[0, 0].set_title('Channel Attention')
        
        # Plot spatial attention
        spatial_attn = model.spatial_attention(features)
        axes[0, 1].imshow(spatial_attn[0].cpu().numpy(), cmap='viridis')
        axes[0, 1].set_title('Spatial Attention')
        
        # Plot feature maps
        feature_maps = features[0].cpu().numpy()
        axes[1, 0].imshow(np.mean(feature_maps, axis=0), cmap='viridis')
        axes[1, 0].set_title('Average Feature Maps')
        
        # Plot attention-weighted features
        weighted_features = features * channel_attn.unsqueeze(-1).unsqueeze(-1)
        axes[1, 1].imshow(np.mean(weighted_features[0].cpu().numpy(), axis=0), cmap='viridis')
        axes[1, 1].set_title('Attention-Weighted Features')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def visualize_feature_pyramid(model, input_tensor, save_path):
    """Visualize feature pyramid network outputs"""
    model.eval()
    with torch.no_grad():
        # Get FPN features
        fpn_features = model.get_fpn_features(input_tensor)
        
        # Create figure
        n_levels = len(fpn_features)
        fig, axes = plt.subplots(1, n_levels, figsize=(20, 4))
        fig.suptitle('Feature Pyramid Network Visualization', fontsize=16)
        
        for i, feat in enumerate(fpn_features):
            feat_np = feat[0].cpu().numpy()
            axes[i].imshow(np.mean(feat_np, axis=0), cmap='viridis')
            axes[i].set_title(f'P{i+3}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def visualize_segmentation_attention(model, input_tensor, save_path):
    """Visualize segmentation model attention gates"""
    model.eval()
    with torch.no_grad():
        # Get attention gate outputs
        attn_gates = model.get_attention_gates(input_tensor)
        
        # Create figure
        n_gates = len(attn_gates)
        fig, axes = plt.subplots(2, n_gates, figsize=(20, 8))
        fig.suptitle('Segmentation Attention Gates', fontsize=16)
        
        for i, (gate, skip) in enumerate(attn_gates):
            # Plot attention gate
            gate_np = gate[0].cpu().numpy()
            axes[0, i].imshow(np.mean(gate_np, axis=0), cmap='viridis')
            axes[0, i].set_title(f'Gate {i+1}')
            axes[0, i].axis('off')
            
            # Plot skip connection
            skip_np = skip[0].cpu().numpy()
            axes[1, i].imshow(np.mean(skip_np, axis=0), cmap='viridis')
            axes[1, i].set_title(f'Skip {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    # Create directories
    os.makedirs("research_visuals/attention_mechanisms", exist_ok=True)
    os.makedirs("research_visuals/feature_maps", exist_ok=True)
    
    # Initialize models
    classifier = BrainTumorClassifier()
    segmentation = BrainTumorSegmentation()
    
    # Create dummy input
    classifier_input = torch.randn(1, 3, 224, 224)
    segmentation_input = torch.randn(1, 3, 256, 256)
    
    # Generate visualizations
    print("Generating attention mechanism visualizations...")
    visualize_attention_maps(classifier, classifier_input, 
                           "research_visuals/attention_mechanisms/classifier_attention.png")
    visualize_feature_pyramid(classifier, classifier_input,
                            "research_visuals/feature_maps/classifier_fpn.png")
    visualize_segmentation_attention(segmentation, segmentation_input,
                                   "research_visuals/attention_mechanisms/segmentation_attention.png")
    
    print("Done! Visualizations saved in research_visuals/")

if __name__ == "__main__":
    main() 