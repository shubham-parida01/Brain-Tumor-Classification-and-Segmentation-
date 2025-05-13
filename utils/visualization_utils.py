import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from typing import Tuple, List, Optional, Union
import torch.nn.functional as F

class AdvancedVisualizer:
    """Advanced visualization utilities for brain tumor detection."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def compute_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute the centroid of a binary mask."""
        if mask.sum() == 0:
            return (0, 0)
        y_coords, x_coords = np.where(mask > 0)
        return (np.mean(x_coords), np.mean(y_coords))
    
    def find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours in a binary mask."""
        return measure.find_contours(mask, 0.5)
    
    def get_bounding_box(self, mask: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
        """Get bounding box coordinates with padding."""
        if mask.sum() == 0:
            return (0, 0, 0, 0)
        y_coords, x_coords = np.where(mask > 0)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        return (
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(mask.shape[1], x_max + padding),
            min(mask.shape[0], y_max + padding)
        )
    
    def crop_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop a region from an image using bounding box coordinates."""
        x_min, y_min, x_max, y_max = bbox
        return image[y_min:y_max, x_min:x_max]
    
    def overlay_mask(self, 
                    image: np.ndarray, 
                    mask: np.ndarray, 
                    color: Tuple[float, float, float] = (1, 0, 0),
                    alpha: float = 0.5) -> np.ndarray:
        """Overlay a mask on an image with specified color and transparency."""
        overlay = image.copy()
        mask_3d = np.stack([mask] * 3, axis=-1)
        overlay[mask_3d > 0] = color
        return cv2.addWeighted(image, 1, overlay, alpha, 0)
    
    def create_visualization(self,
                           image: np.ndarray,
                           seg_mask: np.ndarray,
                           gradcam_heatmap: np.ndarray,
                           class_name: str,
                           confidence: float,
                           save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive visualization combining all features.
        
        Args:
            image: Original image (H, W, C)
            seg_mask: Segmentation mask (H, W)
            gradcam_heatmap: Grad-CAM heatmap (H, W)
            class_name: Predicted class name
            confidence: Classification confidence
            save_path: Optional path to save the visualization
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Original image with Grad-CAM overlay
        plt.subplot(231)
        plt.imshow(image)
        plt.imshow(gradcam_heatmap, alpha=0.5, cmap='jet')
        plt.title('Grad-CAM Attention')
        plt.axis('off')
        
        # 2. Segmentation mask overlay
        plt.subplot(232)
        mask_overlay = self.overlay_mask(image, seg_mask)
        plt.imshow(mask_overlay)
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        # 3. Combined visualization
        plt.subplot(233)
        combined_mask = np.zeros_like(seg_mask)
        combined_mask[gradcam_heatmap > 0.5] = 1
        combined_mask[seg_mask > 0.5] = 1
        combined_overlay = self.overlay_mask(image, combined_mask)
        plt.imshow(combined_overlay)
        plt.title('Combined Visualization')
        plt.axis('off')
        
        # 4. Bounding box and centroid
        plt.subplot(234)
        bbox = self.get_bounding_box(seg_mask)
        cropped = self.crop_region(image, bbox)
        plt.imshow(cropped)
        centroid = self.compute_centroid(seg_mask)
        plt.plot(centroid[0] - bbox[0], centroid[1] - bbox[1], 'rx', markersize=10)
        plt.title('Tumor Region with Centroid')
        plt.axis('off')
        
        # 5. Contours
        plt.subplot(235)
        contours = self.find_contours(seg_mask)
        plt.imshow(image)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
        plt.title('Tumor Contours')
        plt.axis('off')
        
        # 6. Metrics
        plt.subplot(236)
        plt.axis('off')
        plt.text(0.1, 0.8, f'Class: {class_name}', fontsize=12)
        plt.text(0.1, 0.6, f'Confidence: {confidence:.2%}', fontsize=12)
        tumor_area = np.sum(seg_mask) / (seg_mask.shape[0] * seg_mask.shape[1])
        plt.text(0.1, 0.4, f'Tumor Area: {tumor_area:.2%}', fontsize=12)
        plt.text(0.1, 0.2, f'Centroid: ({centroid[0]:.0f}, {centroid[1]:.0f})', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def visualize_batch(self,
                       images: torch.Tensor,
                       seg_masks: torch.Tensor,
                       gradcam_maps: torch.Tensor,
                       class_names: List[str],
                       confidences: torch.Tensor,
                       save_dir: str,
                       num_samples: int = 4) -> None:
        """
        Visualize a batch of predictions.
        
        Args:
            images: Batch of images (B, C, H, W)
            seg_masks: Batch of segmentation masks (B, 1, H, W)
            gradcam_maps: Batch of Grad-CAM heatmaps (B, H, W)
            class_names: List of class names
            confidences: Classification confidences (B,)
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
        """
        num_samples = min(num_samples, images.size(0))
        
        for i in range(num_samples):
            # Convert tensors to numpy arrays
            image = images[i].permute(1, 2, 0).cpu().numpy()
            seg_mask = seg_masks[i, 0].cpu().numpy()
            gradcam_map = gradcam_maps[i].cpu().numpy()
            
            # Create visualization
            save_path = f"{save_dir}/sample_{i}.png"
            self.create_visualization(
                image=image,
                seg_mask=seg_mask,
                gradcam_heatmap=gradcam_map,
                class_name=class_names[i],
                confidence=confidences[i].item(),
                save_path=save_path
            ) 