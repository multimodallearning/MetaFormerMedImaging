import torch
from clearml import Task

from models.segmentator_base import SegmentatorBase
from architectures.flex_token_mixer import FlexFormer
from architectures import poolformer as pf
from architectures.segformer_decoder import SegformerDecoder
from torch import nn
from torch.nn import functional as F

class FlexFormerSegmentator(SegmentatorBase):
    def __init__(self, ds_name: str, model_name: str = 'poolformer_s12', pretrained: bool = True, kernel_size: int = 5,
                 head_dim: int = 16, lr: float = 1e-4, drop_path: float = 0.1, use_slopes: bool = False,
                 learn_pe: bool = False, rpl_patch_emb: bool = False, decoder_latent_dim:int=256):
        super().__init__(ds_name, lr=lr)
        try:
            patch_size = {'wristbone': [384, 224]}[ds_name.lower()]
        except KeyError:
            raise NotImplementedError(f'Dataset {ds_name} has not been added yet.')
        self.encoder = FlexFormer(1, self.n_channels, patch_size, kernel_size, head_dim,
                                model_name, pretrained, False, learn_pe, use_slopes, rpl_patch_emb, drop_path,
                                None)
        # put encoder into dense prediction mode
        del self.encoder.model.head
        del self.encoder.model.norm
        self.encoder.model.fork_feat = True
        self.encoder.model.out_indices = [0, 2, 4, 6]
        embed_dims = [64, 128, 320, 512]
        for embed_dim, layer_i in zip(embed_dims, self.encoder.model.out_indices):
            self.encoder.model.add_module(f'norm{layer_i}', pf.GroupNorm(embed_dim))
        self.decoder = SegformerDecoder(embed_dims, decoder_latent_dim)
        self.seg_head = nn.Conv2d(decoder_latent_dim, self.n_classes, kernel_size=1, bias=True)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        y_hat = self.seg_head(x)
        y_hat = F.interpolate(y_hat, scale_factor=4, mode='bilinear', align_corners=False)
        return y_hat

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            model_name = self.hparams.model_name
            model_name = model_name.replace('pool', 'flex')
            Task.current_task().set_name(f'{model_name}_{self.hparams.ds_name}')