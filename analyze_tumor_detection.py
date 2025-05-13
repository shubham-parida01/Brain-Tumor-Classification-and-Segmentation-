import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from model.network import BrainTumorClassifier
from model.segmentation import BrainTumorSegmentation
from model.grad_cam import GradCAM
from utils.visualization_utils import AdvancedVisualizer
from config import test_transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_models(classifier_path, segmentation_path, device):
    """Load classification and segmentation models."""
    # Load classifier
    classifier = BrainTumorClassifier()
    classifier.load_state_dict(torch.load(classifier_path, map_location=device)['model_state_dict'])
    classifier = classifier.to(device)
    classifier.eval()
    
    # Load segmentation model
    segmentation = BrainTumorSegmentation()
    segmentation.load_state_dict(torch.load(segmentation_path, map_location=device)['model_state_dict'])
    segmentation = segmentation.to(device)
    segmentation.eval()
    
    return classifier, segmentation

def process_image(image_path, transforms):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    tensor_image = transforms(image).unsqueeze(0)
    return image, tensor_image

def analyze_tumor_detection(classifier_path, segmentation_path, test_dir, output_dir, num_samples=10):
    """Analyze tumor detection performance on test images."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    classifier, segmentation = load_models(classifier_path, segmentation_path, device)
    
    # Initialize visualizer
    visualizer = AdvancedVisualizer(device=device)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(classifier, model_type='classification')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images = []
    for tumor_type in ['glioma', 'meningioma', 'pituitary', 'notumor']:
        tumor_dir = os.path.join(test_dir, tumor_type)
        if os.path.exists(tumor_dir):
            images = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if f.endswith('.jpg')]
            test_images.extend([(img, tumor_type) for img in images[:num_samples//4]])
    
    # Initialize lists for metrics
    true_labels = []
    pred_labels = []
    confidences = []
    tumor_areas = []
    
    # Process each image
    for i, (image_path, true_label) in enumerate(test_images):
        print(f"Processing image {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
        
        # Process image
        image, tensor_image = process_image(image_path, test_transforms)
        tensor_image = tensor_image.to(device)
        
        # Get predictions
        with torch.no_grad():
            # Classification
            logits = classifier(tensor_image)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
            # Segmentation
            seg_mask = segmentation.predict_mask(tensor_image)
            
            # Grad-CAM
            gradcam_map = gradcam(tensor_image)
        
        # Store metrics
        true_labels.append(true_label)
        pred_labels.append(['glioma', 'meningioma', 'pituitary', 'notumor'][pred_class.item()])
        confidences.append(confidence.item())
        tumor_areas.append(np.sum(seg_mask[0, 0].cpu().numpy()) / (seg_mask.shape[2] * seg_mask.shape[3]))
        
        # Create visualization
        save_path = os.path.join(output_dir, f"viz_{os.path.basename(image_path)}")
        visualizer.create_visualization(
            image=image,
            seg_mask=seg_mask[0, 0].cpu().numpy(),
            gradcam_heatmap=gradcam_map,
            class_name=pred_labels[-1],
            confidence=confidences[-1],
            save_path=save_path
        )
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=['glioma', 'meningioma', 'pituitary', 'notumor'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['glioma', 'meningioma', 'pituitary', 'notumor'],
                yticklabels=['glioma', 'meningioma', 'pituitary', 'notumor'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(true_labels, pred_labels, target_names=['glioma', 'meningioma', 'pituitary', 'notumor'])
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7)
    plt.title('Distribution of Classification Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()
    
    # Generate tumor area distribution
    plt.figure(figsize=(10, 6))
    plt.hist(tumor_areas, bins=20, alpha=0.7)
    plt.title('Distribution of Tumor Areas')
    plt.xlabel('Tumor Area (fraction of image)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tumor_area_distribution.png'))
    plt.close()
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total images analyzed: {len(test_images)}")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Average tumor area: {np.mean(tumor_areas):.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze brain tumor detection performance')
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                      help='Path to classification model checkpoint')
    parser.add_argument('--segmentation_checkpoint', type=str, required=True,
                      help='Path to segmentation model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/testing',
                      help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                      help='Directory to save analysis results')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to analyze per class')
    
    args = parser.parse_args()
    analyze_tumor_detection(
        args.classifier_checkpoint,
        args.segmentation_checkpoint,
        args.test_dir,
        args.output_dir,
        args.num_samples
    ) 