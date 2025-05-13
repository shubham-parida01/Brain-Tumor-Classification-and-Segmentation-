"""
Neural network architecture for brain tumor classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            
    def forward(self, inputs):
        laterals = [conv(input) for input, conv in zip(inputs, self.lateral_convs)]
        
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
            
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs

class BrainTumorClassifier(nn.Module):
    """
    Enhanced brain tumor classification model with attention and FPN.
    """
    def __init__(self, num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=PRETRAINED, 
                 feature_extract=FEATURE_EXTRACT, dropout_rate=DROPOUT_RATE):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
            backbone (str): Backbone architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained (bool): Whether to use pretrained weights
            feature_extract (bool): If True, only update the reshaped layer params
            dropout_rate (float): Dropout rate for the classifier
        """
        super(BrainTumorClassifier, self).__init__()
        
        # Select backbone architecture
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            feature_size = 512
            fpn_channels = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            feature_size = 512
            fpn_channels = [64, 128, 256, 512]
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            feature_size = 2048
            fpn_channels = [256, 512, 1024, 2048]
        else:  # default to resnet50
            self.model = models.resnet50(pretrained=pretrained)
            feature_size = 2048
            fpn_channels = [256, 512, 1024, 2048]
        
        # Freeze parameters if feature extracting
        if feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(fpn_channels, 256)
        
        # Attention modules
        self.channel_attention = ChannelAttention(256)
        self.spatial_attention = SpatialAttention()
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Store the backbone name for Grad-CAM visualization
        self.backbone = backbone
        
        # Store intermediate features
        self.intermediate_features = None
        self.fpn_features = None
        
    def get_intermediate_features(self, x):
        """Get intermediate features from the backbone"""
        features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        features.append(x)
        x = self.model.layer2(x)
        features.append(x)
        x = self.model.layer3(x)
        features.append(x)
        x = self.model.layer4(x)
        features.append(x)
        
        return features
    
    def get_fpn_features(self, x):
        """Get FPN features"""
        if self.fpn_features is None:
            features = self.get_intermediate_features(x)
            self.fpn_features = self.fpn(features)
        return self.fpn_features
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Get backbone features
        features = self.get_intermediate_features(x)
        
        # Apply FPN
        fpn_features = self.fpn(features)
        self.fpn_features = fpn_features[-1]  # Use the last FPN feature map
        
        # Apply attention mechanisms
        channel_weights = self.channel_attention(self.fpn_features)
        spatial_weights = self.spatial_attention(self.fpn_features)
        
        # Apply attention
        attended_features = self.fpn_features * channel_weights * spatial_weights
        
        # Classification
        x = self.classifier(attended_features)
        
        return x
    
    def get_activation_layer(self):
        """
        Get the activation layer for Grad-CAM visualization.
        
        Returns:
            str: Name of the activation layer
        """
        if self.backbone == 'resnet50':
            return 'model.layer4.2.conv3'
        elif self.backbone == 'resnet101':
            return 'model.layer4.2.conv3'
        else:
            return 'model.layer4.1.conv2'


def create_model(num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=PRETRAINED, 
                feature_extract=FEATURE_EXTRACT, dropout_rate=DROPOUT_RATE):
    """
    Create and initialize the enhanced brain tumor classification model.
    
    Args:
        num_classes (int): Number of output classes
        backbone (str): Backbone architecture
        pretrained (bool): Whether to use pretrained weights
        feature_extract (bool): If True, only update the reshaped layer params
        dropout_rate (float): Dropout rate for the classifier
        
    Returns:
        BrainTumorClassifier: Initialized model
    """
    model = BrainTumorClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        feature_extract=feature_extract,
        dropout_rate=dropout_rate
    )
    
    return model
    
def get_parameter_groups(model, feature_extract=FEATURE_EXTRACT):
    """
    Get parameter groups for optimizer with different learning rates.
    
    Args:
        model (nn.Module): Model
        feature_extract (bool): If True, only update the reshaped layer params
        
    Returns:
        list: Parameter groups for optimizer
    """
    if not feature_extract:
        # If we're fine-tuning the whole model
        # For transfer learning, we may want different learning rates for backbone and new layers
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower learning rate for backbone
            {'params': classifier_params}  # Default learning rate for classifier
        ]
        
        return param_groups
    else:
        # If we're only training the classifier, just return those parameters
        return [param for param in model.parameters() if param.requires_grad]