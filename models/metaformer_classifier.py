import torch
from clearml import Task
from timm.models import adapt_input_conv
from torch import nn

from architectures import poolformer as pf
from architectures import metaformer as mf
from architectures.flex_token_mixer import FlexFormer, FlexTokenMixer
from models.classifier_base import ClassifierBase


class AdaptiveMetaformerClassifier(ClassifierBase):
    def __init__(self, ds_name, tokenmixer: str, patch_size: int = 224, kernel_size: int = 5,
                 head_dim: int = 16, drop_path: float = 0.1, device: str = "cuda"):
        super().__init__(ds_name)
        self.save_hyperparameters()
        self.model = getattr(pf, "poolformer_s12")(pretrained=False)

        # adjust model to current dataset
        if self.model.head.out_features != self.n_classes:
            print('Replacing classifier head for new numbers of classes.')
            self.model.head = nn.Linear(self.model.head.in_features, self.n_classes)
        if self.n_channels != 3:
            print('Reusing first conv weights and adapt to new number of input channel')
            self.model.patch_embed.proj.weight = nn.Parameter(
                adapt_input_conv(self.n_channels, self.model.patch_embed.proj.weight))
            self.model.patch_embed.proj.in_channels = self.n_channels

        patch_size = torch.tensor([patch_size] * 2)
        embed_dim = self.model.patch_embed.proj.out_channels
        for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
            if drop_path > 0.:
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

            stage_patch_size = patch_size / (4 * 2 ** i)
            stage_embed_dim = embed_dim * 2 ** i if embed_dim * 2 ** i != 256 else 320  # handle outlier of stage 3

            if tokenmixer == 'loc_attn':
                assert head_dim >= 16, f"head_dim should be at least 16 for local attention, but is {head_dim}"
                num_heads = stage_embed_dim // head_dim
                if stage_patch_size.prod() > 64:  # apply local self attention only when it is worth it
                    block_mask = FlexFormer.generate_block_mask(num_heads, kernel_size, stage_patch_size, device)
                else:
                    block_mask = None  # apply global self attention
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask, use_slopes=False,
                                                           learn_pos_emb=False)
            elif tokenmixer == 'full_attn':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    if l == 0:
                        blocks[l].token_mixer = nn.Sequential(
                            mf.AddPositionEmb(num_channel, stage_patch_size.int().tolist()),
                            mf.Attention(num_channel, head_dim)
                        )
                    else:
                        blocks[l].token_mixer = mf.Attention(num_channel, head_dim)
            elif tokenmixer == 'pooling':
                for l in range(len(blocks)):
                    blocks[l].token_mixer = mf.Pooling(pool_size=kernel_size)
            elif tokenmixer == 'conv':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    blocks[l].token_mixer = nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size,
                                                      stride=1, padding=kernel_size // 2, groups=1)
            elif tokenmixer == 'sep_conv':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    blocks[l].token_mixer = nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size,
                                                      stride=1, padding=kernel_size // 2, groups=num_channel)
            else:
                raise ValueError(f'Unknown tokenmixer {tokenmixer}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(
                f'metaformer_{self.hparams.tokenmixer}_{self.hparams.ds_name} {self.hparams.kernel_size}²')


if __name__ == '__main__':
    for mixer_name in ['loc_attn', 'full_attn', 'pooling', 'conv', 'sep_conv']:
        m = AdaptiveMetaformerClassifier('imagewoof', mixer_name).cuda()
        print('\n', mixer_name)
        print(m)
        x = torch.randn(2, 3, 224, 224).cuda()
        y = m(x)
        print(f'Output shape: {y.shape}\n')
