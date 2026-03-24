import torch
from clearml import Task
from torch import nn
from torch.nn import functional as F

from architectures import metaformer as mf
from architectures import poolformer as pf
from architectures.flex_token_mixer import FlexFormer, FlexTokenMixer
from architectures.random_token_mixer import RandomMixer
from architectures.segformer_decoder import SegformerDecoder
from models.metaformer_classifier_3d import get_parent_module, convert_conv2d_to_conv3d, AdaptiveMetaformerClassifier3D, \
    Pooling3D
from models.segmentator_base import SegmentatorBase


class MetaFormerSegmentator3D(SegmentatorBase):
    def __init__(self, ds_name: str, token_mixer: str, model_name: str = 'poolformer_s12', kernel_size: int = 3,
                 head_dim: int = 16, lr: float = 1e-3, drop_path: float = 0.1, use_slopes: bool = False,
                 learn_pe: bool = False, decoder_latent_dim: int = 256, patch_size: int = 128):
        """
        Segmentation model with SegFormer decoder and MetaFormer encoder with definable token mixer.
        :param ds_name: dataset name. has to be jsrt, wristbone or tiger
        :param token_mixer: token mixer to deploy at each stage of MetaFormer encoder
        :param model_name: MetaFormer model to build the encoder on
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
        self.encoder = getattr(pf, model_name)(pretrained=False)
        # replace all convs with 3d ones
        conv_moduls = [(name, module) for name, module in self.encoder.named_modules() if isinstance(module, nn.Conv2d)]
        conv_moduls[0][1].in_channels = self.n_channels  # adapt channels to channels of images
        for name, conv in conv_moduls:
            parent, child_name = get_parent_module(self.encoder, name)
            setattr(parent, child_name, convert_conv2d_to_conv3d(conv))
        # reinit MLP convs like linear
        for module in filter(lambda m: isinstance(m, pf.Mlp), self.encoder.modules()):
            module.apply(AdaptiveMetaformerClassifier3D.init_weights)
        # prepare as feature encoder
        del self.encoder.head
        del self.encoder.norm
        self.encoder.fork_feat = True
        self.encoder.out_indices = [0, 2, 4, 6]
        embed_dims = [64, 128, 320, 512] if model_name.split('_')[1][0] == 's' else [96, 192, 384, 768]
        for embed_dim, layer_i in zip(embed_dims, self.encoder.out_indices):
            self.encoder.add_module(f'norm{layer_i}', pf.GroupNorm(embed_dim))
        self.decoder = SegformerDecoder(embed_dims, decoder_latent_dim, spatial_dim=3)
        self.seg_head = nn.Conv3d(decoder_latent_dim, self.n_classes, kernel_size=1, bias=True)

        if isinstance(patch_size, int):
            patch_size = [patch_size] * 3

        # replace token mixer
        patch_size = torch.tensor(patch_size)
        for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.encoder.network)):
            if drop_path > 0.:
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

            stage_patch_size = patch_size / (4 * 2 ** i)
            match token_mixer:
                case 'loc_attn':
                    assert head_dim >= 16, f"head_dim should be at least 16 for local attention, but is {head_dim}"
                    num_heads = embed_dims[0] // head_dim
                    if stage_patch_size.prod() >= 64:  # apply local self attention only when it is worth it
                        block_mask = FlexFormer.generate_block_mask3D(num_heads, kernel_size, stage_patch_size, 'cuda')
                    else:
                        block_mask = None  # apply global self attention
                    for l in range(len(blocks)):
                        num_channel = blocks[l].norm1.num_channels
                        blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask, use_slopes=False,
                                                               learn_pos_emb=False)
                case 'full_attn':
                    for l in range(len(blocks)):
                        num_channel = blocks[l].norm1.num_channels
                        if l == 0:
                            blocks[l].token_mixer = nn.Sequential(
                                mf.AddPositionEmb(num_channel, stage_patch_size.int().tolist()),
                                mf.Attention(num_channel, head_dim)
                            )
                        else:
                            blocks[l].token_mixer = mf.Attention(num_channel, head_dim)
                case 'pooling':
                    kernel_larger_patch_size = torch.any(kernel_size > stage_patch_size).item()
                    if kernel_larger_patch_size:
                        print(
                            f'Kernel ({kernel_size}) is larger then patch size ({stage_patch_size.tolist()}), replacing AvgPool with identity.')
                    for l in range(len(blocks)):
                        if not kernel_larger_patch_size:
                            blocks[l].token_mixer = Pooling3D(pool_size=kernel_size)
                        else:
                            blocks[l].token_mixer = nn.Identity()
                case 'conv':
                    for l in range(len(blocks)):
                        num_channel = blocks[l].norm1.num_channels
                        blocks[l].token_mixer = nn.Conv3d(num_channel, num_channel, kernel_size=kernel_size,
                                                          stride=1, padding=kernel_size // 2, groups=1)
                case 'sep_conv':
                    for l in range(len(blocks)):
                        num_channel = blocks[l].norm1.num_channels
                        blocks[l].token_mixer = nn.Conv3d(num_channel, num_channel, kernel_size=kernel_size,
                                                          stride=1, padding=kernel_size // 2, groups=num_channel)
                case 'identity':
                    for l in range(len(blocks)):
                        blocks[l].token_mixer = nn.Identity()
                case 'random':
                    for l in range(len(blocks)):
                        blocks[l].token_mixer = RandomMixer(stage_patch_size.int().tolist())
                case _:
                    raise ValueError(f'Unknown tokenmixer {token_mixer}')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        y_hat = self.seg_head(x)
        y_hat = F.interpolate(y_hat, scale_factor=4, mode='trilinear', align_corners=False)
        return y_hat

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            model_name = self.hparams.model_name
            model_name = model_name.replace('pool', 'meta')
            Task.current_task().set_name(
                f'{model_name}_{self.hparams.token_mixer}_{self.hparams.ds_name} {self.hparams.kernel_size}³')


if __name__ == '__main__':
    for mixer_name in ['loc_attn', 'pooling', 'conv', 'sep_conv', 'identity', 'random']:
        m = MetaFormerSegmentator3D('abdomenatlas', mixer_name, kernel_size=3, patch_size=128).cuda()
        print('\n', mixer_name)
        # print(m)
        x = torch.randn(2, 1, 128, 128, 128).cuda()
        y = m(x)
        print(f'Output shape: {y.shape}\n')
