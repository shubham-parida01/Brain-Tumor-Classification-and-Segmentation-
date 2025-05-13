import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model.network import BrainTumorClassifier
from utils.visualization import visualize_prediction

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on brain tumor images')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the image or directory of images')
    parser.add_argument('--output_dir', type=str, default='results/inference',
                      help='Directory to save inference results')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference')
    return parser.parse_args()

def load_model(checkpoint_path):
    # Initialize model
    model = BrainTumorClassifier()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image_path):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def run_inference(model, image_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Get class names
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    
    # Get probabilities for each class
    probs = probabilities[0].numpy()
    
    # Print results
    print(f"\nInference Results for {os.path.basename(image_path)}:")
    print("-" * 50)
    for class_name, prob in zip(class_names, probs):
        print(f"{class_name}: {prob*100:.2f}%")
    print(f"\nPredicted Class: {class_names[predicted_class]}")
    
    # Save visualization
    output_path = os.path.join(output_dir, f"prediction_{os.path.basename(image_path)}")
    visualize_prediction(original_image, probs, class_names, output_path)
    print(f"\nVisualization saved to: {output_path}.png")

def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    
    # Check if input path is a directory or single file
    if os.path.isdir(args.image_path):
        # Process all images in directory
        image_files = [f for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(args.image_path, image_file)
            run_inference(model, image_path, args.output_dir)
    else:
        # Process single image
        run_inference(model, args.image_path, args.output_dir)

if __name__ == '__main__':
    main() 