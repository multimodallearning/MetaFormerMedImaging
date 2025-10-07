import torch
from skimage.draw import polygon2mask
from tqdm import trange

from datasets.jsrt_dataset import JSRTDataset


def get_mask(lm, a_struct):
    (start_idx, end_idx) = JSRTDataset.get_anatomical_structure_index()[a_struct]

    # all images are 256x256
    mask = polygon2mask((W, H), lm[start_idx:end_idx].flip(-1).numpy())  # xy -> yx
    mask = torch.from_numpy(mask)
    return mask


H = W = 256

# load data
data = torch.load('data/JSRT_img0_lms.pth', map_location='cpu')
lms = data['JSRT_lms'].float()
del data

masks = []
for i in trange(len(lms), desc='generate nnUNet data', unit='img'):
    lm = lms[i]
    # load landmarks
    mask_rl = get_mask(lm, 'right_lung')
    mask_ll = get_mask(lm, 'left_lung')
    mask_h = get_mask(lm, 'heart')
    mask_rcla = get_mask(lm, 'right_clavicle')
    mask_lcla = get_mask(lm, 'left_clavicle')

    # stack label maps with same order as JSRTDataset
    lbl = torch.stack([  # channel
        mask_rl,  # 0
        mask_ll,  # 1
        mask_h,  # 2
        mask_rcla,  # 3
        mask_lcla  # 4
    ])
    masks.append(lbl)
masks = torch.stack(masks, dim=0)  # (N, C, H, W)
# save to disk
torch.save(masks, 'data/jsrt_seg_masks.pth')
