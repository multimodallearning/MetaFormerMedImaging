import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch._inductor.config.realize_opcount_threshold = 500


device = 'cuda'

from architectures.poolformer import poolformer_s12

x = torch.randn(16, 3, 256, 256).cuda()
poolformer = poolformer_s12(pretrained=True).cuda()
output_orig = poolformer(x)
print(output_orig[0][:10])

def noscore(score, b, h, q_idx, kv_idx):
    return score*0+1
kernel_options = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 16, }

class FlexPool(nn.Module):
    def __init__(self, channels=64, patch=16, heads=4):
        super().__init__()
        self.flex_attention5 = torch.compile(flex_attention,dynamic=False)
        p_D = 1 # pool_size = 3
        k_D = 2*p_D+1
        def local_patch(b, h, q_idx, kv_idx):
                x = q_idx//(patch*patch); idx2 = q_idx-x*patch*patch
                y = idx2//patch; z = q_idx%patch
                x1 = kv_idx//(patch*patch); idx2 = kv_idx-x1*patch*patch
                y1 = idx2//patch; z1 = idx2%patch
                mask = ((x-x1).abs() <= p_D) & ((y-y1).abs() <= p_D) & ((z-z1).abs() <= p_D)
                return mask

        assert patch<128, 'patch size should be less than 128'  
        assert patch**2>64, 'patch size should be at least 8'  

        self.block_mask = create_block_mask(local_patch, B=None, H=None, Q_LEN=patch**2, KV_LEN=patch**2,_compile=True)###.to('cuda')
        torch.cuda.empty_cache()

        self.channels = channels
        self.patch = patch
        self.heads = heads

        self.linear_in = nn.Linear(channels, channels*3, bias=False)
        self.linear_in.weight.data[-channels:] = torch.eye(channels).cuda()
        self.linear_in.weight.data[:-channels] = torch.randn(channels*2, channels).mul(0.02).cuda()

    def forward(self, x):
        channels = self.channels
        patch = self.patch
        heads = self.heads
        
        x__ = x.flatten(-2).permute(0, 2, 1)
        q__, k__, v__ = self.linear_in(x__).chunk(3, dim=-1)
        v_ = v__.unflatten(-1,(heads,channels//heads)).permute(0, 2, 1, 3).contiguous()
        q_ = q__.unflatten(-1,(heads,channels//heads)).permute(0, 2, 1, 3).contiguous()
        k_ = k__.unflatten(-1,(heads,channels//heads)).permute(0, 2, 1, 3).contiguous()
        
        y_ = self.flex_attention5(q_, k_, v_, kernel_options=kernel_options, block_mask=self.block_mask)
        #optional linear_out layer so far missing
        y_flex = y_.permute(0, 1, 3, 2).contiguous().view_as(x)


        return y_flex - x


channels_ = (64,128,320,512)
patch_ = (64,32,16,8)
head_ = (4,8,10,16)

for i in range(0,len(poolformer.network),2):
    print('replacing stage', i//2)
    for j in range(len(poolformer.network[i])):
        channels = channels_[i//2]
        patch = patch_[i//2]
        heads = head_[i//2]
        poolformer.network[i][j].token_mixer = FlexPool(channels=channels, patch=patch, heads=heads).cuda()
output = poolformer(x)
print(output[0][:10],(output_orig-output).norm())

label = torch.randint(0, 1000, (16,)).cuda()
optim = torch.optim.Adam(poolformer.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
for i in range(100):
    optim.zero_grad()
    output = poolformer(x)
    loss = loss_fn(output, label)
    loss.backward()
    optim.step()
    if(i%10==0):
        print(i, loss.item())