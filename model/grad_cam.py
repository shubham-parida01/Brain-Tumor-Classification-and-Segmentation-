"""
Grad-CAM implementation for generating attention maps.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import traceback

class GradCAM:
    def __init__(self, model, model_type='classification', target_layer=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The model to use for Grad-CAM
            model_type: Type of model ('classification' or 'segmentation')
            target_layer: The target layer to generate attention maps from.
                         If None, will use the last convolutional layer.
        """
        self.model = model
        self.model_type = model_type
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Find the target layer if not specified
        if self.target_layer is None:
            self.target_layer = self._find_target_layer()
            
        if self.target_layer is None:
            raise ValueError("Could not find a suitable target layer.")
    
    def _find_target_layer(self):
        """Find the appropriate target layer based on model architecture."""
        if self.model_type == 'classification':
            # For ResNet models wrapped in our classifier
            if hasattr(self.model, 'model'):
                model = self.model.model
            else:
                model = self.model
                
            # Get the last conv layer of the last block
            if hasattr(model, 'layer4'):
                return model.layer4[-1].conv3
            
            # Fallback to last conv layer
            last_conv = None
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            return last_conv
            
        else:  # segmentation
            # For U-Net models
            if hasattr(self.model, 'bottleneck'):
                return self.model.bottleneck.double_conv[-2]  # Last conv before ReLU
            
            # Fallback to last conv layer
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            return last_conv
    
    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM attention map for the input tensor and target class
        """
        # Register hooks
        self.activations = None
        self.gradients = None
        
        # Register forward hook to capture activations
        handle_forward = self.target_layer.register_forward_hook(self._save_activation)
        
        # Register backward hook to capture gradients
        handle_backward = self.target_layer.register_full_backward_hook(self._save_gradient)
        
        try:
            # Forward pass
            model_output = self.model(input_tensor)
            
            if target_class is None:
                target_class = model_output.argmax(dim=1)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            target = torch.zeros_like(model_output)
            if isinstance(target_class, torch.Tensor):
                target[range(len(target_class)), target_class] = 1
            else:
                target[0][target_class] = 1
            model_output.backward(gradient=target, retain_graph=True)
            
            # Check if gradients and activations were captured
            if self.gradients is None or self.activations is None:
                logging.error("Failed to capture gradients or activations")
                return None
            
            # Generate attention map
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            for i in range(self.activations.shape[1]):
                self.activations[:, i, :, :] *= pooled_gradients[i]
            
            heatmap = torch.mean(self.activations, dim=1).squeeze()
            heatmap = torch.relu(heatmap)  # ReLU to keep only positive contributions
            
            # Normalize heatmap
            if torch.sum(heatmap) > 0:  # Check if heatmap is not all zeros
                heatmap = heatmap / torch.max(heatmap)
            
            return heatmap.detach().cpu().numpy()
            
        except Exception as e:
            logging.error(f"Error in GradCAM computation: {str(e)}")
            logging.error(traceback.format_exc())
            return None
            
        finally:
            # Clean up hooks
            handle_forward.remove()
            handle_backward.remove()