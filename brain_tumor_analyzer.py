import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import *
from model.network import create_model
from model.grad_cam import GradCAM
from utils.metrics import compute_metrics_from_outputs
from utils.visualization import plot_training_history, plot_confusion_matrix, visualize_model_predictions, visualize_gradcam

class BrainTumorAnalyzer:
    """
    Comprehensive brain tumor analysis tool that handles both evaluation and prediction.
    Combines functionality from evaluate.py, inference.py, and enhanced analysis.
    """
    
    def __init__(self, model_path=None):
        """Initialize with optional model path"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if model_path:
            self.load_model(model_path)
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model(self, model_path):
        """Load the model from path"""
        self.model = create_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def predict_single_image(self, image_path, visualize=True, save_path=None):
        """
        Predict tumor class for a single image
        
        Args:
            image_path: Path to input image
            visualize: Whether to create visualization
            save_path: Optional path to save visualization
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = get_test_transforms()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        # Create visualization if requested
        if visualize:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Grad-CAM visualization
            grad_cam = GradCAM(self.model)
            cam_map = grad_cam(image_tensor, predicted_class)
            
            ax2.imshow(image)
            ax2.imshow(cam_map, alpha=0.5, cmap='jet')
            ax2.set_title(f'Prediction: {CLASS_NAMES[predicted_class]}\nConfidence: {confidence:.2%}')
            ax2.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        return predicted_class, confidence, probabilities[0].cpu().numpy()
        
    def evaluate_model(self, test_loader, save_dir=None):
        """
        Evaluate model performance on test dataset
        
        Args:
            test_loader: DataLoader containing test data
            save_dir: Directory to save evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        all_preds = []
        all_labels = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = compute_metrics_from_outputs(all_preds, all_labels, all_probs)
        
        # Create visualizations
        if save_dir:
            # Confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=CLASS_NAMES,
                       yticklabels=CLASS_NAMES)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
            
            # Save metrics report
            with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w') as f:
                f.write("Model Evaluation Report\n")
                f.write("=====================\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
                f.write("\nDetailed Metrics:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                    
        return metrics
        
    def process_batch(self, image_paths, output_dir=None):
        """
        Process a batch of images
        
        Args:
            image_paths: List of paths to images
            output_dir: Directory to save results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        results = []
        for i, image_path in enumerate(image_paths):
            save_path = os.path.join(output_dir, f'result_{i}.png') if output_dir else None
            pred_class, confidence, probs = self.predict_single_image(
                image_path, visualize=True, save_path=save_path
            )
            
            results.append({
                'image_path': image_path,
                'predicted_class': CLASS_NAMES[pred_class],
                'confidence': confidence,
                'probabilities': {name: prob for name, prob in zip(CLASS_NAMES, probs)}
            })
            
        return results

def main():
    """Main function for command line usage"""
    import argparse
    parser = argparse.ArgumentParser(description='Brain Tumor Analysis Tool')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', choices=['predict', 'evaluate', 'batch'], required=True,
                      help='Operation mode')
    parser.add_argument('--input', type=str, required=True,
                      help='Input image path or test data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    analyzer = BrainTumorAnalyzer(args.model_path)
    
    if args.mode == 'predict':
        analyzer.predict_single_image(args.input, save_path=os.path.join(args.output_dir, 'prediction.png'))
    elif args.mode == 'evaluate':
        from data.dataset import get_data_loaders
        _, test_loader = get_data_loaders(args.input)
        analyzer.evaluate_model(test_loader, args.output_dir)
    elif args.mode == 'batch':
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        analyzer.process_batch(image_paths, args.output_dir)

if __name__ == '__main__':
    main() 