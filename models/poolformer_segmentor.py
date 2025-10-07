import torch
from clearml import Task
from timm.models import adapt_input_conv

from models.segmentator_base import SegmentatorBase
from architectures.flex_token_mixer import FlexFormer, FlexTokenMixer
from architectures import poolformer as pf
from architectures.segformer_decoder import SegformerDecoder
from architectures.metaformer import Attention, AddPositionEmb
from torch import nn
from torch.nn import functional as F

class MetaFormerSegmentator(SegmentatorBase):
    def __init__(self, ds_name: str, token_mixer:str, model_name: str = 'poolformer_s12', pretrained: bool = False, kernel_size: int = 5,
                 head_dim: int = 16, lr: float = 1e-3, drop_path: float = 0.1, use_slopes: bool = False,
                 learn_pe: bool = False, decoder_latent_dim:int=256, patch_size:int=None):
        """
        Segmentation model with SegFormer decoder and MetaFormer encoder with definable token mixer.
        :param ds_name: dataset name. has to be jsrt, wristbone or tiger
        :param token_mixer: token mixer to deploy at each stage of MetaFormer encoder
        :param model_name: MetaFormer model to build the encoder on
        :param pretrained: whether to use pretrained weights or not
        :param kernel_size: used kernel size for token mixer in encoder if applicable
        :param head_dim: number of channels per head for attention based token mixers
        :param lr: learning rate to use
        :param drop_path: stochastic depth rate
        :param use_slopes: whether to modify attention scores based on relative positions in directed local self attention.
        :param learn_pe: whether to learn positional embeddings at the beginning of each stage when using attention-based token mixers
        :param decoder_latent_dim: hidden dimension in SegFormer decoder
        :param patch_size: spatial input size (H, W). If None, use default sizes for each dataset.
        """
        super().__init__(ds_name, lr=lr)
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.encoder = getattr(pf, model_name)(pretrained=pretrained)
        if self.n_channels != 3:
            print('Reusing first conv weights and adapt to new number of input channel')
            self.encoder.patch_embed.proj.weight = nn.Parameter(
                adapt_input_conv(self.n_channels, self.encoder.patch_embed.proj.weight))
            self.encoder.patch_embed.proj.in_channels = self.n_channels
        # prepare as feature encoder
        del self.encoder.head
        del self.encoder.norm
        self.encoder.fork_feat = True
        self.encoder.out_indices = [0, 2, 4, 6]
        embed_dims = [64, 128, 320, 512] if model_name.split('_')[1][0] == 's' else [96, 192, 384, 768]
        for embed_dim, layer_i in zip(embed_dims, self.encoder.out_indices):
            self.encoder.add_module(f'norm{layer_i}', pf.GroupNorm(embed_dim))
        self.decoder = SegformerDecoder(embed_dims, decoder_latent_dim)
        self.seg_head = nn.Conv2d(decoder_latent_dim, self.n_classes, kernel_size=1, bias=True)

        if patch_size is None:
            try:
                patch_size = {'wristbone': [384, 224], 'jsrt': [256, 256], 'tiger': [256, 256]}[ds_name.lower()]
            except KeyError:
                raise NotImplementedError(f'Dataset {ds_name} has not been added yet.')
        else:
            assert isinstance(patch_size, int)
            patch_size = [patch_size, patch_size]

        # replace token mixer
        patch_size = torch.tensor(patch_size)
        for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.encoder.network)):
            if drop_path > 0.:
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

            stage_patch_size = patch_size / (4 * 2 ** i)

            if token_mixer == 'loc_attn':
                assert head_dim >= 16, f"head_dim should be at least 16 for local attention, but is {head_dim}"
                num_heads = embed_dims[i] // head_dim
                if stage_patch_size.prod() > 64:  # apply local self attention only when it is worth it
                    block_mask = FlexFormer.generate_block_mask(num_heads, kernel_size, stage_patch_size, 'cuda')
                else:
                    block_mask = None  # apply global self attention
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask, use_slopes=use_slopes,
                                                           learn_pos_emb=(l == 0) and learn_pe, init_as_pooling=False)
                    if not pretrained: # else token mixer initialization mimic avg pooling
                        blocks[l].token_mixer.random_init()
            elif token_mixer == 'full_attn':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    if learn_pe and (l == 0):
                        blocks[l].token_mixer = nn.Sequential(
                            AddPositionEmb(num_channel, stage_patch_size.int().tolist()),
                            Attention(num_channel, head_dim)
                        )
                    else:
                        blocks[l].token_mixer = Attention(num_channel, head_dim)
            elif token_mixer == 'pooling':
                pass  # pooling is already the default token mixer
            elif token_mixer == 'conv':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    blocks[l].token_mixer = nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size,
                                                      stride=1, padding=kernel_size // 2, groups=1)
            elif token_mixer == 'sep_conv':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    blocks[l].token_mixer = nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size,
                                                      stride=1, padding=kernel_size // 2, groups=num_channel)
            elif token_mixer == 'identity':
                for l in range(len(blocks)):
                    blocks[l].token_mixer = nn.Identity()
            else:
                raise ValueError(f'Unknown tokenmixer {token_mixer}')

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
            model_name = model_name.replace('pool', 'meta')
            Task.current_task().set_name(f'{model_name}_{self.hparams.token_mixer}_{self.hparams.ds_name} {self.hparams.kernel_size}²')