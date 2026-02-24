import torch
from clearml import Task
from timm.models.layers import trunc_normal_
from torch import nn

from architectures import poolformer as pf
from architectures import metaformer as mf
from architectures.flex_token_mixer import FlexFormer, FlexTokenMixer
from models.classifier_base import ClassifierBase


def convert_conv2d_to_conv3d(conv: nn.Conv2d, **kwargs) -> nn.Conv3d:
    def collapse_tuple_to_int(t: tuple[int, ...]) -> int:
        assert all([e == t[0] for e in t]), 'tuple contains different entries'
        return t[0]

    attr = dict(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=collapse_tuple_to_int(conv.kernel_size),
        stride=collapse_tuple_to_int(conv.stride),
        padding=collapse_tuple_to_int(conv.padding),
        dilation=collapse_tuple_to_int(conv.dilation),
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype
    )
    attr.update(kwargs)
    conv3d = nn.Conv3d(**attr)
    return conv3d


def get_parent_module(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class AdaptiveMetaformerClassifier3D(ClassifierBase):
    def __init__(self, ds_name, tokenmixer: str, patch_size: int = 64, kernel_size: int = 5, lr: float = 0.001,
                 head_dim: int = 16, drop_path: float = 0.1, device: str = "cuda"):
        """
        MetaFormerS12 classifier with definable token mixer. Model always trained from scratch.
        Architecture signature: [T, T, T, T] where T is the token mixer.
        :param ds_name: dataset name. Has to be one of the MedMNIST datasets or 'imagewoof'
        :param tokenmixer: token mixer to use at every stage
        :param patch_size: spatial input size (H, W, D)
        :param kernel_size: kernel size for token mixer if applicable
        :param lr: learning rate to use
        :param head_dim: embedding dimension of each head for attention based token mixers
        :param drop_path: stochastic depth rate
        :param device: specify device for creating block masks if local attention is used (needed for block mask creation in flex attention
        """
        super().__init__(ds_name, lr)
        self.save_hyperparameters()
        self.model = getattr(pf, "poolformer_s12")(pretrained=False)
        assert self.is_3d, 'You try to apply a 3D model to a 2D dataset.'

        # adjust model to current dataset
        if self.model.head.out_features != self.n_classes:
            print('Replacing classifier head for new numbers of classes.')
            self.model.head = nn.Linear(self.model.head.in_features, self.n_classes)
        # replace all convs with 3d ones
        conv_moduls = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)]
        conv_moduls[0][1].in_channels = self.n_channels  # adapt channels to channels of images
        for name, conv in conv_moduls:
            parent, child_name = get_parent_module(self.model, name)
            setattr(parent, child_name, convert_conv2d_to_conv3d(conv))
        # reinit MLP convs like linear
        for module in filter(lambda m: isinstance(m, pf.Mlp), self.model.modules()):
            module.apply(self._init_weights)

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
            elif tokenmixer == 'identity':
                for l in range(len(blocks)):
                    blocks[l].token_mixer = nn.Identity()
            else:
                raise ValueError(f'Unknown tokenmixer {tokenmixer}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(
                f'metaformer_{self.hparams.tokenmixer}_{self.hparams.ds_name} {self.hparams.kernel_size}³')

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # import os
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for mixer_name in ['full_attn', 'pooling', 'conv', 'sep_conv', 'identity']:  # 'loc_attn',
        m = AdaptiveMetaformerClassifier3D('NoduleMNIST3D', 'identity')  # .cuda()
        print('\n', mixer_name)
        # print(m)
        x = torch.randn(2, 1, 64, 64, 64)  # .cuda()
        y = m(x)
        print(f'Output shape: {y.shape}\n')
        break
