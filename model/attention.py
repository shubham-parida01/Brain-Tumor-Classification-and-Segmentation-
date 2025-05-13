"""
Enhanced attention mechanisms with multi-scale fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleChannelAttention(nn.Module):
    """Multi-scale channel attention with pyramid pooling."""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        
        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Multi-scale pooling
        y1 = self.pool1(x).view(b, c)
        y2 = self.pool2(x).view(b, c, 4).mean(dim=2)
        y3 = self.pool3(x).view(b, c, 16).mean(dim=2)
        
        # Channel attention
        y1 = self.mlp(y1).view(b, c, 1, 1)
        y2 = self.mlp(y2).view(b, c, 1, 1)
        y3 = self.mlp(y3).view(b, c, 1, 1)
        
        # Feature fusion
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.fusion(y)
        
        return torch.sigmoid(y)

class MultiScaleSpatialAttention(nn.Module):
    """Multi-scale spatial attention with dilated convolutions."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        
        # Multi-scale convolutions
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, dilation=2)
        self.conv3 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, dilation=3)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Multi-scale spatial attention
        x1 = torch.cat([avg_out, max_out], dim=1)
        y1 = self.conv1(x1)
        
        x2 = torch.cat([avg_out, max_out], dim=1)
        y2 = self.conv2(x2)
        
        x3 = torch.cat([avg_out, max_out], dim=1)
        y3 = self.conv3(x3)
        
        # Feature fusion
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.fusion(y)
        
        return torch.sigmoid(y)

class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        # Query and key transformations
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention computation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Transform inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Compute attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi

class MultiScaleAttention(nn.Module):
    """Combined multi-scale attention module."""
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.channel_attention = MultiScaleChannelAttention(in_channels)
        self.spatial_attention = MultiScaleSpatialAttention()
        
    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        
        # Spatial attention
        x = x * self.spatial_attention(x)
        
        return x 