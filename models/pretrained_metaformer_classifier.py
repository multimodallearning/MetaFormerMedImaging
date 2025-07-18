import torch
from clearml import Task
from timm.models import adapt_input_conv
from torch import nn

from architectures import poolformer as pf
from architectures.flex_token_mixer import FlexFormer, FlexTokenMixer
from architectures.metaformer import Attention, AddPositionEmb
from architectures.poolformer import model_urls
from models.classifier_base import ClassifierBase


class PretrainedMetaformer(ClassifierBase):
    def __init__(self, ds_name, tokenmixer: str, patch_size: int=224, kernel_size: int = 5,
                 head_dim: int = 16, model_name: str = "poolformer_s12", last_stage_pretrained: int = 2,
                 learn_pe: bool = False, use_slopes: bool = False, drop_path: float = 0.1, device: str = "cuda"):
        super().__init__(ds_name)
        self.save_hyperparameters()
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=False)

        # load pretrained weights for first stages
        url = model_urls['poolformer_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        truncated_checkpoint = checkpoint.copy()
        assert 0 <= last_stage_pretrained <= 4, f"last_stage_pretrained must be between 0 and 4, but is {last_stage_pretrained}."
        if last_stage_pretrained > 0:
            highest_allowed_stage = (last_stage_pretrained - 1) * 2  # start counting at 1
            for key in checkpoint.keys():
                if key.startswith('norm.') or key.startswith('head.'):
                    del truncated_checkpoint[key]  # remove norm and head weights

                elif key.startswith('network.'):  # remove weights of stages that should not be pretrained
                    stage_id = int(key.split('.')[1])
                    if stage_id > highest_allowed_stage:
                        del truncated_checkpoint[key]
            self.model.load_state_dict(truncated_checkpoint, strict=False)
            # freeze pretrained weights
            # for name, param in self.model.named_parameters():
            #     if name.startswith('patch_embed.') or (
            #             name.startswith('network.') and int(name.split('.')[1]) <= highest_allowed_stage):
            #         param.requires_grad = False
            #         print(f'Freezing {name}.')
        else:
            print('Not loading any pretrained weights, training from scratch.')

        # adjust model to current dataset
        if self.model.head.out_features != self.n_classes:
            print('Replacing classifier head for new numbers of classes.')
            self.model.head = nn.Linear(self.model.head.in_features, self.n_classes)
        if self.n_channels != 3:
            print('Reusing first conv weights and adapt to new number of input channel')
            self.model.patch_embed.proj.weight = nn.Parameter(
                adapt_input_conv(self.n_channels, self.model.patch_embed.proj.weight))
            self.model.patch_embed.proj.in_channels = self.n_channels

        patch_size = torch.tensor([patch_size]*2)
        embed_dim = self.model.patch_embed.proj.out_channels
        for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
            if drop_path > 0.:
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

            if i < last_stage_pretrained:
                print(f'Leaving stage {i + 1} as is.')
                continue

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
                    blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask, use_slopes=use_slopes,
                                                           learn_pos_emb=(l == 0) and learn_pe)
                    blocks[l].token_mixer.random_init()
            elif tokenmixer == 'full_attn':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    if learn_pe and (l == 0):
                        blocks[l].token_mixer = nn.Sequential(
                            AddPositionEmb(num_channel, stage_patch_size.int().tolist()),
                            Attention(num_channel, head_dim)
                        )
                    else:
                        blocks[l].token_mixer = Attention(num_channel, head_dim, )
            elif tokenmixer == 'pooling':
                pass # pooling is already the default token mixer
            elif tokenmixer == 'conv':
                for l in range(len(blocks)):
                    num_channel = blocks[l].norm1.num_channels
                    blocks[l].token_mixer = nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size,
                                                      stride=1, padding=kernel_size // 2, groups=1)#num_channel)
            else:
                raise ValueError(f'Unknown tokenmixer {tokenmixer}')
        #self.model = torch.compile(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'hybridformer_{self.hparams.tokenmixer}_stage{self.hparams.last_stage_pretrained}_{self.hparams.ds_name}')


if __name__ == '__main__':
    for mixer_name in ['loc_attn']:#, 'full_attn', 'pooling', 'conv']:
        for stages_pretrained in range(0, 5):
            print(f'Testing {mixer_name} with {stages_pretrained} stages pretrained.')
            m = PretrainedMetaformer('imagewoof', mixer_name, 224, 5, 16,
                                     last_stage_pretrained=stages_pretrained).cuda()
            #print(m)
            x = torch.randn(2, 3, 224, 224).cuda()
            y = m(x)
            print(f'Output shape: {y.shape}\n')
