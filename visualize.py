import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import cv2
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

from config import *
from data.dataset import BrainMRIDataset
from model.network import create_model
from utils.visualization import plot_confusion_matrix, visualize_gradcam
from model.grad_cam import GradCAM

class BrainTumorVisualizer:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.target_layer = self.model.layer4[-1]  # Last layer of ResNet
        self.class_names = ['no tumor', 'glioma', 'meningioma', 'pituitary']
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """Load the model from checkpoint"""
        # Create model
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different state dict structures
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remove 'model.' prefix from keys if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        """Preprocess a single image for model input"""
        # Load and convert image
        image = Image.open(image_path)
        
        # Convert to RGB if grayscale
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        original_size = image.size
        
        # Apply preprocessing
        tensor_image = self.preprocess(image)
        
        return image, tensor_image, original_size

    def visualize_prediction(self, image_path, save_path=None):
        """Visualize model prediction for a single image with Grad-CAM"""
        # Load and preprocess image
        original_image, input_tensor, original_size = self.preprocess_image(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            # Print probabilities for all classes
            print("\nPrediction probabilities:")
            for i, class_name in enumerate(self.class_names):
                print(f"{class_name}: {probabilities[0][i].item():.2%}")
        
        # Create figure
        fig = plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Only show Grad-CAM if not predicting "no tumor"
        if pred_class != 0:  # 0 is the class index for "no tumor"
            # Generate Grad-CAM
            gradcam = GradCAM(self.model, self.target_layer)
            cam_map = gradcam(input_tensor, target_class=pred_class)
            
            # Plot Grad-CAM
            plt.subplot(1, 3, 2)
            plt.imshow(original_image)
            plt.imshow(cam_map, alpha=0.5, cmap='jet')
            plt.title('Grad-CAM Attention')
            plt.axis('off')
            
            # Plot prediction
            plt.subplot(1, 3, 3)
        else:
            # For no tumor case, make prediction subplot wider
            plt.subplot(1, 2, 2)
        
        # Plot prediction
        plt.imshow(original_image)
        plt.title(f'Prediction: {self.class_names[pred_class]}\nConfidence: {confidence:.2%}')
        plt.axis('off')
        
        # Save or show
        if save_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Add file extension if not present
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_path = save_path + '.png'
                
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize brain tumor classification with GradCAM')
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                        help='Path to the classifier checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='results/visualization',
                        help='Directory to save visualizations')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Starting visualization script...")
    print(f"Checkpoint: {args.classifier_checkpoint}")
    print(f"Image path: {args.image_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model
        print("Loading model...")
        model = create_model()
        checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
        
        # Create data loader
        print("Creating data loader...")
        test_dataset = BrainMRIDataset(TEST_DIR, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        print(f"Created data loader with {len(test_dataset)} samples")
        
        # Collect predictions
        print("Running inference...")
        all_outputs = []
        all_labels = []
        all_images = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc="Running inference"):
                images = images.to(device)
                outputs = model(images)
                
                all_outputs.append(outputs)
                all_labels.append(labels)
                all_images.append(images)
        
        # Concatenate results
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_images = torch.cat(all_images)
        
        # Convert outputs to predictions
        predictions = torch.argmax(all_outputs, dim=1).cpu().numpy()
        labels = all_labels.cpu().numpy()
        
        # Compute metrics
        print("Computing metrics...")
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='macro'),
            'recall': recall_score(labels, predictions, average='macro'),
            'f1': f1_score(labels, predictions, average='macro')
        }
        
        # Plot confusion matrix
        print("Plotting confusion matrix...")
        cm_fig = plot_confusion_matrix(labels, predictions, CLASS_NAMES)
        cm_fig.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
        plt.close(cm_fig)
        
        # Plot Grad-CAM visualizations
        print("Plotting Grad-CAM visualizations...")
        gradcam_fig = visualize_gradcam(model, test_loader, device, CLASS_NAMES, 8)
        gradcam_fig.savefig(os.path.join(args.output_dir, 'gradcam_visualization.png'))
        plt.close(gradcam_fig)
        
        # Print metrics
        print("\nModel Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Save metrics to file
        print("Saving metrics...")
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")

        # Create visualizer and generate visualization for the specific image
        print("Generating visualization for specific image...")
        visualizer = BrainTumorVisualizer(args.classifier_checkpoint, device)
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        save_path = os.path.join(args.output_dir, f'prediction_{image_name}.png')
        visualizer.visualize_prediction(args.image_path, save_path)
        print("Visualization complete")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 