import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from model.network import BrainTumorClassifier
from model.segmentation import BrainTumorSegmentation
from model.grad_cam import GradCAM
from utils.visualization_utils import AdvancedVisualizer
from config import test_transforms

def load_models(classifier_path: str, segmentation_path: str, device: str):
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

def process_image(image_path: str, transforms):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    tensor_image = transforms(image).unsqueeze(0)
    return image, tensor_image

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    classifier, segmentation = load_models(args.classifier_checkpoint, args.segmentation_checkpoint, device)
    
    # Initialize visualizer
    visualizer = AdvancedVisualizer(device=device)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(classifier, model_type='classification')
    
    # Process image
    image, tensor_image = process_image(args.image_path, test_transforms)
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualization
    save_path = os.path.join(args.output_dir, f"viz_{os.path.basename(args.image_path)}")
    visualizer.create_visualization(
        image=image,
        seg_mask=seg_mask[0, 0].cpu().numpy(),
        gradcam_heatmap=gradcam_map,
        class_name=args.class_names[pred_class.item()],
        confidence=confidence.item(),
        save_path=save_path
    )
    
    print(f"Visualization saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize brain tumor detection results')
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                      help='Path to classification model checkpoint')
    parser.add_argument('--segmentation_checkpoint', type=str, required=True,
                      help='Path to segmentation model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='results/visualization',
                      help='Directory to save visualizations')
    parser.add_argument('--class_names', type=list, default=['No Tumor', 'Tumor'],
                      help='List of class names')
    
    args = parser.parse_args()
    main(args) 