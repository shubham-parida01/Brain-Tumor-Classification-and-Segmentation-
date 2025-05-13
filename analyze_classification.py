import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from model.network import BrainTumorClassifier
from model.grad_cam import GradCAM
from config import test_transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(checkpoint_path, device):
    """Load classification model."""
    try:
        logging.info(f"Loading model from {checkpoint_path}")
        classifier = BrainTumorClassifier()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier = classifier.to(device)
        classifier.eval()
        logging.info("Model loaded successfully")
        return classifier
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def process_image(image_path, transforms):
    """Load and preprocess an image."""
    try:
        image = Image.open(image_path).convert('RGB')
        tensor_image = transforms(image).unsqueeze(0)
        return image, tensor_image
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def create_visualization(image, gradcam_map, class_name, confidence, save_path=None):
    """Create visualization with Grad-CAM overlay."""
    try:
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Grad-CAM heatmap
        plt.subplot(132)
        plt.imshow(gradcam_map, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        # Combined visualization
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(gradcam_map, alpha=0.5, cmap='jet')
        plt.title(f'Combined View\n{class_name} ({confidence:.2%})')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
    except Exception as e:
        logging.error(f"Error creating visualization: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def analyze_classification(checkpoint_path, test_dir, output_dir, num_samples=10):
    """Analyze tumor classification performance on test images."""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load model
        classifier = load_model(checkpoint_path, device)
        
        # Initialize Grad-CAM
        gradcam = GradCAM(classifier, model_type='classification')
        logging.info("Grad-CAM initialized")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created: {output_dir}")
        
        # Get test images
        test_images = []
        class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']
        for tumor_type in class_names:
            tumor_dir = os.path.join(test_dir, tumor_type)
            if os.path.exists(tumor_dir):
                images = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if f.endswith('.jpg')]
                test_images.extend([(img, tumor_type) for img in images[:num_samples//4]])
        
        logging.info(f"Found {len(test_images)} test images")
        
        # Initialize lists for metrics
        true_labels = []
        pred_labels = []
        confidences = []
        attention_scores = []
        
        # Process each image
        for i, (image_path, true_label) in enumerate(test_images):
            logging.info(f"Processing image {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
            
            try:
                # Process image
                image, tensor_image = process_image(image_path, test_transforms)
                tensor_image = tensor_image.to(device)
                
                # Get predictions
                classifier.eval()  # Ensure model is in eval mode
                with torch.set_grad_enabled(True):  # Enable gradients for GradCAM
                    try:
                        # Classification
                        logits = classifier(tensor_image)
                        probs = torch.softmax(logits, dim=1)
                        confidence, pred_class = torch.max(probs, dim=1)
                        
                        # Grad-CAM - compute for predicted class
                        gradcam_map = gradcam(tensor_image, pred_class.item())
                        
                        if gradcam_map is None:
                            logging.warning(f"GradCAM failed to generate heatmap for {image_path}")
                            continue
                            
                        # Normalize gradcam map to [0, 1]
                        gradcam_map = np.clip(gradcam_map, 0, 1)  # Ensure values are in [0,1]
                        
                    except Exception as e:
                        logging.error(f"Error during GradCAM computation: {str(e)}")
                        logging.error(traceback.format_exc())
                        continue
                
                # Store metrics
                true_labels.append(true_label)
                pred_labels.append(class_names[pred_class.item()])
                confidences.append(confidence.item())
                attention_scores.append(float(np.mean(gradcam_map)))
                
                # Create visualization
                save_path = os.path.join(output_dir, f"viz_{os.path.basename(image_path)}")
                create_visualization(
                    image=np.array(image),
                    gradcam_map=gradcam_map,
                    class_name=pred_labels[-1],
                    confidence=confidences[-1],
                    save_path=save_path
                )
            except Exception as e:
                logging.error(f"Error processing image {image_path}: {str(e)}")
                logging.error(traceback.format_exc())
                continue
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        logging.info("Confusion matrix generated")
        
        # Generate classification report
        report = classification_report(true_labels, pred_labels, target_names=class_names)
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        logging.info("Classification report generated")
        
        # Generate confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7)
        plt.title('Distribution of Classification Confidence')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        plt.close()
        logging.info("Confidence distribution plot generated")
        
        # Generate attention score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(attention_scores, bins=20, alpha=0.7)
        plt.title('Distribution of Grad-CAM Attention Scores')
        plt.xlabel('Average Attention Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'attention_distribution.png'))
        plt.close()
        logging.info("Attention score distribution plot generated")
        
        # Generate scatter plot of confidence vs attention
        plt.figure(figsize=(10, 6))
        plt.scatter(confidences, attention_scores, alpha=0.5)
        plt.title('Classification Confidence vs Attention Score')
        plt.xlabel('Confidence')
        plt.ylabel('Average Attention Score')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'confidence_vs_attention.png'))
        plt.close()
        logging.info("Confidence vs attention plot generated")
        
        # Print and save summary
        summary = "\nAnalysis Summary:\n"
        summary += f"Total images analyzed: {len(test_images)}\n"
        summary += f"Average confidence: {np.mean(confidences):.4f}\n"
        summary += f"Average attention score: {np.mean(attention_scores):.4f}\n"
        summary += "\nClassification Report:\n"
        summary += report
        
        print(summary)
        
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(summary)
        logging.info("Analysis summary saved")
        
    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze brain tumor classification performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to classification model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/testing',
                      help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                      help='Directory to save analysis results')
    parser.add_argument('--num_samples', type=int, default=40,
                      help='Number of samples to analyze (10 per class)')
    
    args = parser.parse_args()
    analyze_classification(
        args.checkpoint,
        args.test_dir,
        args.output_dir,
        args.num_samples
    ) 