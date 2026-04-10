import torch
from kornia.augmentation import PadTo
from rad_dino import RadDino, utils
from torch import nn
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature

from architectures.dino_head import DINOHead
from architectures.segformer_decoder import SegformerDecoder
from models.classifier_base import ClassifierBase
from models.segmentator_base import SegmentatorBase
from clearml import Task


class RadDinoClassifier(ClassifierBase):
    def __init__(self, ds_name: str, mlp_layers: int = 2, mlp_hidden_dim: int = 2048, mlp_bottleneck_dim: int = 256):
        super().__init__(ds_name)
        assert self.is_2d, 'Rad DINO is only available in 2D'
        self.encoder = RadDino()
        self.encoder.requires_grad_(False)
        self.classifier = DINOHead(768, self.n_classes, nlayers=mlp_layers, hidden_dim=mlp_hidden_dim,
                                   bottleneck_dim=mlp_bottleneck_dim)
        self.upsample = nn.UpsamplingBilinear2d(518)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        x = self.upsample(x)
        if self.n_channels == 1:
            x = x.expand(-1, 3, -1, -1)
        inputs = BatchFeature(dict(pixel_values=x))
        return self.classifier(self.encoder.encode(inputs)[0])

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(
                f'RadDINO_{self.hparams.ds_name}')


class RadDinoSegmentator(SegmentatorBase):
    def __init__(self, ds_name: str, decoder_latent_dim: int = 256, patch_size: int | tuple[int, int] = None):
        super().__init__(ds_name)
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14"
        )
        # download from https://huggingface.co/microsoft/rad-dino/resolve/main/backbone_compatible.safetensors?download=true
        backbone_state_dict = utils.safetensors_to_state_dict("data/backbone_compatible.safetensors")
        self.encoder.load_state_dict(backbone_state_dict, strict=True)
        self.encoder.eval().requires_grad_(False)
        self.decoder = SegformerDecoder([768] * 4, decoder_latent_dim)
        self.seg_head = nn.Conv2d(decoder_latent_dim, self.n_classes, kernel_size=1, bias=True)

        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.pad_to_square = PadTo((max(self.patch_size),) * 2)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        x = self.pad_to_square(x)
        x = F.interpolate(x, (518,) * 2, mode='bilinear')
        if self.n_channels == 1:
            x = x.expand(-1, 3, -1, -1)

        feats = self.encoder.get_intermediate_layers(x, [8, 9, 10, 11], True, False, True)
        feats = self.decoder(feats)
        y_hat = self.seg_head(feats)

        y_hat = F.interpolate(y_hat, self.patch_size, mode='bilinear')
        y_hat = self.pad_to_square.inverse(y_hat)
        return y_hat

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(
                f'RadDINO_{self.hparams.ds_name}')


if __name__ == '__main__':
    import torch

    hw = (256, 256)
    m = RadDinoSegmentator('jsrt', patch_size=256)
    y = m(torch.randn(4, 1, *hw))
    print(y.shape)
