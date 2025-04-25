import torch
from clearml import Task
from torch import nn

from architectures import poolformer as pf
from models.med_mnist_base import MedMNISTBase
from architectures.flex_token_mixer import FlexFormer
from torch.nn import functional as F


class PoolFormerClassifier(MedMNISTBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained: bool = True,
                 train_poolformer: bool = False, lr_poolformer: float = 0.0001, weight_decay: float = 0.05):
        super().__init__(dataset_name)
        assert self.is_2d, "PoolFormer is only implemented for 2D datasets"
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, self.n_classes)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        self.model.requires_grad_(False)
        param_dicts = [{"params": self.model.head.parameters()}, {"params": self.model.norm.parameters()}]
        self.model.head.requires_grad_(True)
        self.model.norm.requires_grad_(True)
        if self.hparams.train_poolformer:
            param_dicts.append({"params": self.model.network.parameters(), "lr": self.hparams.lr_poolformer})
            self.model.network.requires_grad_(True)
            param_dicts.append({"params": self.model.patch_embed.parameters(), "lr": self.hparams.lr_poolformer})
            self.model.patch_embed.requires_grad_(True)

        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'{self.hparams.model_name}_{self.hparams.dataset_name}')

class FlexFormerClassifier(MedMNISTBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained:bool=True,
                 num_heads:int=4, weight_decay: float = 0.05, patch_size: int = 224):
        super().__init__(dataset_name)
        assert self.is_2d, "PoolFormer is only implemented for 2D datasets"
        self.model = FlexFormer(self.n_classes, [patch_size, patch_size], num_heads, model_name, pretrained)

        self.save_hyperparameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
        y_hat = self.model(x)
        return y_hat


    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
    #     return optimizer


    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            model_name = self.hparams.model_name
            model_name = model_name.replace('pool', 'flex')
            Task.current_task().set_name(f'{model_name}_{self.hparams.dataset_name}')


class FlexFormerClassifierV2(MedMNISTBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained:bool=True,
                 num_heads:int=4, weight_decay: float = 0.05, patch_size: int = 224):
        super().__init__(dataset_name)

        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        from architectures.poolformer import poolformer_s12
        poolformer = poolformer_s12(pretrained=True).cuda()
        kernel_options = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32,
                          "BLOCK_N2": 16, }
        class FlexPool(nn.Module):
            def __init__(self, channels=64, patch=16, heads=4):
                super().__init__()
                self.flex_attention5 = torch.compile(flex_attention, dynamic=False)
                p_D = 1  # pool_size = 3
                k_D = 2 * p_D + 1

                def local_patch(b, h, q_idx, kv_idx):
                    x = q_idx // (patch * patch);
                    idx2 = q_idx - x * patch * patch
                    y = idx2 // patch;
                    # z = q_idx % patch
                    x1 = kv_idx // (patch * patch);
                    idx2 = kv_idx - x1 * patch * patch
                    y1 = idx2 // patch;
                    # z1 = idx2 % patch
                    mask = ((x - x1).abs() <= p_D) & ((y - y1).abs() <= p_D)  # & ((z - z1).abs() <= p_D)
                    return mask

                assert patch < 128, 'patch size should be less than 128'
                assert patch ** 2 > 64, 'patch size should be at least 8'

                self.block_mask = create_block_mask(local_patch, B=None, H=None, Q_LEN=patch ** 2, KV_LEN=patch ** 2,
                                                    _compile=True)  ###.to('cuda')
                torch.cuda.empty_cache()

                self.channels = channels
                self.patch = patch
                self.heads = heads

                self.linear_in = nn.Linear(channels, channels * 3, bias=True)
                self.linear_in.bias.data[:] = 0
                self.linear_in.weight.data[-channels:] = torch.eye(channels).cuda()
                self.linear_in.weight.data[:-channels] = torch.randn(channels * 2, channels).mul(0.02).cuda()
                self.linear_out = nn.Linear(channels, channels, bias=True)
                self.linear_out.weight.data = torch.eye(channels).cuda()
                self.linear_out.bias.data[:] = 0

            def forward(self, x):
                channels = self.channels
                patch = self.patch
                heads = self.heads

                x__ = x.flatten(-2).permute(0, 2, 1)
                q__, k__, v__ = self.linear_in(x__).chunk(3, dim=-1)
                v_ = v__.unflatten(-1, (heads, channels // heads)).permute(0, 2, 1, 3).contiguous()
                q_ = q__.unflatten(-1, (heads, channels // heads)).permute(0, 2, 1, 3).contiguous()
                k_ = k__.unflatten(-1, (heads, channels // heads)).permute(0, 2, 1, 3).contiguous()
                y_ = self.flex_attention5(q_, k_, v_, kernel_options=kernel_options, block_mask=self.block_mask)
                # optional linear_out layer so far missing
                y_ = self.linear_out(y_.permute(0, 2, 1, 3).flatten(-2).contiguous())
                y_flex = y_.permute(0, 2, 1).contiguous().view_as(x)
                # y_flex = y_.permute(0, 1, 3, 2).contiguous().view_as(x)

                # x_ = x.unflatten(1,(heads,channels//heads)).flatten(-2).permute(0, 1, 3, 2).contiguous()  # (B, C, H, W) -> (B, H*W, C)
                # q_ = k_ = torch.zeros_like(x_)
                # y_ = self.flex_attention5(q_, k_, x_, kernel_options=kernel_options, block_mask=self.block_mask, score_mod=noscore)
                # y_flex = y_.permute(0, 1, 3, 2).contiguous().view_as(x)

                return y_flex - x

        channels_ = (64, 128, 320, 512)
        patch_ = (64, 32, 16, 8)
        head_ = (4, 8, 10, 16)

        for i in range(0, len(poolformer.network) - 2, 2):
            print('replacing stage', i // 2)
            for j in range(len(poolformer.network[i])):
                channels = channels_[i // 2]
                patch = patch_[i // 2]
                heads = head_[i // 2]
                poolformer.network[i][j].token_mixer = FlexPool(channels=channels, patch=patch, heads=heads).cuda()

        self.model = poolformer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        y_hat = self.model(x)[:, :self.n_classes]
        return y_hat

if __name__ == '__main__':
    from datasets.med_mnist_dataset import MedMNISTDataModule
    from tqdm import trange
    from torch.nn import functional as F

    poolformer = FlexFormerClassifier('OrganAMNIST', 'poolformer_s12', patch_size=224).cuda()

    dm = MedMNISTDataModule('OrganAMNIST', batch_size=128, use_data_aug=False, spatial_size=224)
    dm.setup('fit')
    train_loader = iter(dm.train_dataloader())

    optim = torch.optim.Adam(poolformer.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    run_acc = torch.zeros(3000)
    run_loss = torch.zeros(3000)
    for i in trange(3000):
        try:
            imgs, labels = next(train_loader)
        except StopIteration:
            print('Epoch finished, restarting')
            train_loader = iter(dm.train_dataloader())
            imgs, labels = next(train_loader)
        affine = F.affine_grid(torch.eye(2, 3).cuda().unsqueeze(0) + torch.randn(128, 2, 3).mul(0.06).cuda(),
                               (128, 1, 224, 224), align_corners=False)
        x = F.grid_sample(imgs.cuda(), affine, align_corners=False).expand(-1, 3, -1, -1)
        label = labels.squeeze(-1).cuda()
        optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = poolformer(x)[:, :11]
            loss = loss_fn(output, label)
        loss.backward()
        optim.step()
        run_acc[i] = (output.argmax(1) == label).float().mean()
        run_loss[i] = loss.item()
        if (i % 50 == 40):
            print(i, run_loss[i - 30:i - 1].mean().item(), run_acc[i - 30:i - 1].mean().item())