import os
import torch
import argparse
from PIL import Image
from model.network import BrainTumorClassifier
from model.segmentation import BrainTumorSegmentation
from model.grad_cam import GradCAM
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from config import test_transforms, CLASS_NAMES
import cv2
import torch.nn as nn

class BrainTumorVisualizer:
    """Class for visualizing brain tumor detection results."""
    
    def __init__(self, classifier_model, device):
        self.classifier = classifier_model
        self.device = device
        self.grad_cam = GradCAM(self.classifier)
        
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor_image = test_transforms(image).unsqueeze(0).to(self.device)
        return image, tensor_image
        
    def visualize_prediction(self, image_path, save_path):
        # Load and preprocess image
        original_image, tensor_image = self.preprocess_image(image_path)
        
        # Get model prediction and confidence
        with torch.no_grad():
            output = self.classifier(tensor_image)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
        
        # Get Grad-CAM attention
        attention_map = self.grad_cam(tensor_image, pred.item())
        attention_map = cv2.resize(attention_map, original_image.size[::-1])
        
        # Create visualization with better spacing
        plt.figure(figsize=(20, 5))
        plt.subplots_adjust(wspace=0.3)
        
        # Original Image
        plt.subplot(131)
        plt.imshow(original_image)
        plt.title(f'Original Image\nPrediction: {CLASS_NAMES[pred]}\nConfidence: {confidence.item():.2%}',
                 pad=10, fontsize=10)
        plt.axis('off')
        
        # Grad-CAM Attention
        plt.subplot(132)
        heatmap = plt.imshow(attention_map, cmap='hot')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04)
        plt.title('Attention Heatmap', pad=10, fontsize=10)
        plt.axis('off')
        
        # Grad-CAM Overlay
        plt.subplot(133)
        plt.imshow(original_image)
        overlay = plt.imshow(attention_map, alpha=0.6, cmap='hot')
        plt.colorbar(overlay, fraction=0.046, pad=0.04)
        plt.contour(attention_map, levels=5, colors='yellow', alpha=0.5, linewidths=1)
        plt.title('Attention Overlay', pad=10, fontsize=10)
        plt.axis('off')
        
        # Save visualization
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            save_path = save_path + '.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to: {save_path}")

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load classification model
    classification_model = BrainTumorClassifier()
    checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
    classification_model.load_state_dict(checkpoint['model_state_dict'])
    classification_model = classification_model.to(device)
    classification_model.eval()
    
    # Verify input image exists
    image_path = args.image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create save path
    save_path = os.path.join(args.output_dir, f"viz_{os.path.basename(image_path)}")
    
    # Create visualizer and generate visualization
    visualizer = BrainTumorVisualizer(classification_model, device)
    visualizer.visualize_prediction(image_path, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize brain tumor detection results')
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                      help='Path to classification model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='results/visualization',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    main(args) 