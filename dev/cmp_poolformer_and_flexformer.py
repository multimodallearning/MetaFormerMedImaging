import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from architectures.poolformer import poolformer_s12
from architectures.flex_token_mixer import FlexFormer
import torch

poolformer = poolformer_s12(pretrained=True).cuda()
flexformer = FlexFormer(1000, 3, [224, 224], 3, 16, rw_percentage=None, drop_path=0.).cuda()
poolformer.eval()
flexformer.eval()

with torch.inference_mode():
    x = torch.randn(8, 3, 224, 224).cuda()
    y_poolformer = poolformer(x)
    y_flexformer = flexformer(x)

print(y_poolformer.shape, y_poolformer.norm())
print(y_flexformer.shape, y_flexformer.norm())
print('Difference:', (y_poolformer - y_flexformer).norm())
