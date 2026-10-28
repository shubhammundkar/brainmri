"""
BrainVision 3D - 3D U-Net Architecture for BraTS 2020 Brain Tumor Segmentation
Target Hardware: NVIDIA RTX 3050 (6 GB VRAM)

Design Principles:
- 4-channel input: T1, T1ce, T2, FLAIR
- 4-class output: Background (0), Necrotic Core (1), Edema (2), Enhancing Tumor (3, mapped from BraTS label 4)
- Instance Normalization (InstanceNorm3d): Optimal for small batch sizes (batch_size=1 or 2) in 3D medical imaging
- Residual shortcut connections for gradient stability
- Lightweight memory footprint (~2.5M params with base_filters=16, ~10M with base_filters=32)
- Trilinear interpolation upsampling to save VRAM and avoid deconvolution artifacts
- Mixed precision (torch.cuda.amp / torch.amp) fully supported
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Dual 3D Convolution block with Instance Normalization and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ResidualBlock3D(nn.Module):
    """Residual 3D block with identity/projection shortcut."""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels, dropout_p=dropout_p)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_block(x)
        return self.act(out + residual)


class EncoderStage3D(nn.Module):
    """Encoder stage: Downsampling followed by convolution block."""
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        if use_residual:
            self.block = ResidualBlock3D(in_channels, out_channels, dropout_p=dropout_p)
        else:
            self.block = ConvBlock3D(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor):
        pooled = self.pool(x)
        features = self.block(pooled)
        return features


class DecoderStage3D(nn.Module):
    """Decoder stage: Trilinear upsampling + concatenation + convolution block."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_residual: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        if use_residual:
            self.block = ResidualBlock3D(out_channels + skip_channels, out_channels, dropout_p=dropout_p)
        else:
            self.block = ConvBlock3D(out_channels + skip_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample using trilinear interpolation
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = self.reduce(x)
        # Skip connection concatenation
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net optimized for BraTS 2020 Multi-Modal Tumor Segmentation.

    Args:
        in_channels: Number of input MRI modalities (default: 4 for T1, T1ce, T2, FLAIR).
        out_channels: Number of target segmentation classes (default: 4).
        base_filters: Channel width multiplier (default: 16 for RTX 3050 6GB; 32 for higher VRAM).
        use_residual: Enable residual connections across encoder/decoder blocks.
        dropout_p: Dropout probability in bottleneck and deep decoder.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_filters: int = 16,
        use_residual: bool = True,
        dropout_p: float = 0.15
    ):
        super().__init__()
        f = [base_filters * (2 ** i) for i in range(4)]  # e.g. [16, 32, 64, 128]

        # Initial stem block (full resolution)
        if use_residual:
            self.stem = ResidualBlock3D(in_channels, f[0], dropout_p=0.0)
        else:
            self.stem = ConvBlock3D(in_channels, f[0], dropout_p=0.0)

        # Encoder stages
        self.enc1 = EncoderStage3D(f[0], f[1], use_residual=use_residual, dropout_p=dropout_p * 0.5)
        self.enc2 = EncoderStage3D(f[1], f[2], use_residual=use_residual, dropout_p=dropout_p)
        self.bottleneck = EncoderStage3D(f[2], f[3], use_residual=use_residual, dropout_p=dropout_p)

        # Decoder stages
        self.dec2 = DecoderStage3D(f[3], f[2], f[2], use_residual=use_residual, dropout_p=dropout_p)
        self.dec1 = DecoderStage3D(f[2], f[1], f[1], use_residual=use_residual, dropout_p=dropout_p * 0.5)
        self.dec0 = DecoderStage3D(f[1], f[0], f[0], use_residual=use_residual, dropout_p=0.0)

        # Final 1x1x1 segmentation projection
        self.out_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        s0 = self.stem(x)         # [B, f0, D, H, W]
        s1 = self.enc1(s0)        # [B, f1, D/2, H/2, W/2]
        s2 = self.enc2(s1)        # [B, f2, D/4, H/4, W/4]
        b = self.bottleneck(s2)   # [B, f3, D/8, H/8, W/8]

        # Decoder path with skip connections
        d2 = self.dec2(b, s2)     # [B, f2, D/4, H/4, W/4]
        d1 = self.dec1(d2, s1)    # [B, f1, D/2, H/2, W/2]
        d0 = self.dec0(d1, s0)    # [B, f0, D, H, W]

        # Output logits
        logits = self.out_conv(d0)
        return logits


class ResidualUNet3D(UNet3D):
    """Convenience alias for UNet3D with residual connections enabled."""
    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_filters: int = 16, dropout_p: float = 0.15):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            use_residual=True,
            dropout_p=dropout_p
        )


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ResidualUNet3D(in_channels=4, out_channels=4, base_filters=16)
    print(f"ResidualUNet3D (base_filters=16) initialized.")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass with dummy patch (B=1, C=4, D=64, H=64, W=64)
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output logits shape: {logits.shape}")
