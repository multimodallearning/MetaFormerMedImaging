# adapted from https://github.com/qubvel-org/segmentation_models.pytorch/blob/5ba632dafe852414d0e6a11abf954b68d795ff44/segmentation_models_pytorch/decoders/segformer/decoder.py
import torch
from torch import nn
from architectures.poolformer import GroupNorm
from torch.nn import functional as F

class SegformerDecoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: int = 256):
        """
        Segformer Decoder
        :param in_channels: list of input channels from the encoder, e.g. [64, 128, 320, 512]
        :param out_channels: number of output channels for the decoder, default is 256
        """
        super(SegformerDecoder, self).__init__()

        self.mlp_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels
        ])
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=len(in_channels) * out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            GroupNorm(out_channels, affine=True),
            nn.GELU()
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        forward pass of the Segformer decoder
        :param features: list of feature maps from the encoder with their C aligned to the list in_channels
        :return: fused feature map of shape (B, out_channels, H, W)
        """
        assert len(features) == len(self.mlp_layers), 'Number of feature maps must match the number of MLP layers'
        target_size = features[0].shape[-2:]
        x = []
        for i, feat in enumerate(features):
            feat = self.mlp_layers[i](feat)
            if i != 0: # resize
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            x.append(feat)
        x = torch.cat(x, dim=1) # (B, out_channels * len(features), H, W)
        y_hat = self.fuse_layer(x)  # (B, out_channels, H, W)
        return y_hat

if __name__ == '__main__':
    sd = SegformerDecoder([64, 128, 320, 512])
    z = [
        torch.randn(128, 64, 96, 56),
        torch.randn(128, 128, 48, 28),
        torch.randn(128, 320, 24, 14),
        torch.randn(128, 512, 12, 7)
    ]

    y_hat = sd(z)
    print(y_hat.shape)