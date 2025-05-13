import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        size = x.size()[2:]
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.project(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BrainTumorSegmentation(nn.Module):
    """Enhanced DeepLabV3+ based brain tumor segmentation model."""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Initial convolution
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks
        for i in range(len(features)-1):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(features[i], features[i+1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(features[i+1], features[i+1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features[i+1]),
                nn.ReLU(inplace=True)
            ))
        
        # ASPP
        self.aspp = ASPP(features[-1], features[-1])
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(features)-1, 0, -1):
            self.decoder.append(DecoderBlock(features[i], features[i-1]))
        
        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Store intermediate features
        self.encoder_features = None
        self.attention_gates = None
        
    def get_encoder_features(self, x):
        """Get features from encoder"""
        if self.encoder_features is None:
            self.encoder_features = self.encoder(x)
        return self.encoder_features
    
    def get_attention_gates(self, x):
        """Get attention gate outputs"""
        if self.attention_gates is None:
            encoder_features = self.get_encoder_features(x)
            aspp_out = self.aspp(encoder_features[-1])
            self.attention_gates = self.decoder.get_attention_outputs(aspp_out, encoder_features)
        return self.attention_gates
    
    def forward(self, x):
        # Get encoder features
        encoder_features = self.get_encoder_features(x)
        
        # ASPP
        aspp_out = self.aspp(encoder_features[-1])
        
        # Decoder with attention
        output = self.decoder[0](aspp_out, encoder_features[0])
        for i in range(1, len(encoder_features)):
            output = self.decoder[i](output, encoder_features[i])
        
        return self.final_conv(output)
    
    def predict_mask(self, x, threshold=0.5):
        """Predict binary mask from input image."""
        with torch.no_grad():
            x = self.forward(x)
            return (torch.sigmoid(x) > threshold).float()

def create_segmentation_model(in_channels=3, out_channels=1):
    """Create and initialize the enhanced segmentation model."""
    model = BrainTumorSegmentation(in_channels=in_channels, out_channels=out_channels)
    return model 