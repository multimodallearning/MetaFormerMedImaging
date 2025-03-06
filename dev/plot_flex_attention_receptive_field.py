from matplotlib import pyplot as plt
import seaborn as sns
import torch
from functools import reduce


def compute_mask(patch_size, b, h, q_idx, kv_idx):
    Sx, Sy, Sz = patch_size
    x = q_idx // (Sz * Sy);
    idx2 = q_idx - x * Sz * Sy
    y = idx2 // Sy;
    z = q_idx % Sy
    x1 = kv_idx // (Sz * Sy);
    idx2 = kv_idx - x1 * Sz * Sy
    y1 = idx2 // Sy;
    z1 = idx2 % Sy
    mask = ((x - x1).abs() <= 1) & ((y - y1).abs() <= 1) & ((z - z1).abs() <= 1)
    # mask = ((x-x1).abs() <= (h*2+1)) & ((y-y1).abs() <= (h*2+1)) & ((z-z1).abs() <= (h*2+1))
    return mask


def compute_unravel(patch_size, kernel, b, h, q_idx, kv_idx):
    dilation = h + 1  # each head gets a different dilation
    q_coord = torch.stack(torch.unravel_index(q_idx, patch_size), -1)
    kv_coord = torch.stack(torch.unravel_index(kv_idx, patch_size), -1)
    dist_q_kv = (q_coord - kv_coord).abs()

    # effective receptive field
    dist_rf = kernel + (kernel - 1) * (dilation - 1)
    dist_rf //= 2

    mask_rf = (dist_q_kv <= dist_rf).all(dim=-1)
    mask_dil = (dist_q_kv % dilation == 0).all(dim=-1)
    return mask_rf & mask_dil


def compute_mask_down(kv_patch_size, q_patch_size, q_idx, kv_idx):
    kv_coord = torch.stack(torch.unravel_index(kv_idx, kv_patch_size), -1).float()
    q_coord = torch.stack(torch.unravel_index(q_idx, q_patch_size), -1).float()
    down_sample_factor = torch.tensor(kv_patch_size) // torch.tensor(q_patch_size)
    mask = kv_coord / down_sample_factor - (q_coord - 0.5)
    mask = (mask.abs() <= 1.).all(dim=-1)

    return mask

def compute_mask_up(kv_patch_size, q_patch_size, q_idx, kv_idx):
    kv_coord = torch.stack(torch.unravel_index(kv_idx, kv_patch_size), -1).float()
    q_coord = torch.stack(torch.unravel_index(q_idx, q_patch_size), -1).float()
    sample_factor = torch.tensor(q_patch_size) / torch.tensor(kv_patch_size)
    mask = kv_coord - q_coord / sample_factor
    mask = (mask.abs() <= sample_factor).all(dim=-1)

    return mask


patch_size = [16, 16, 16]
kv_idx = torch.arange(reduce(lambda x, y: x * y, patch_size))
# choose center as query
q_idx = torch.tensor(
    [patch_size[0] // 2 * patch_size[1] * patch_size[2] + patch_size[1] // 2 * patch_size[2] + patch_size[2] // 2])
mask = compute_mask(patch_size, 1, 1, q_idx, kv_idx).view(patch_size)
mask_rvl = compute_unravel(patch_size, 3, 1, 0, q_idx, kv_idx).view(patch_size)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(mask[8], cmap='viridis', linewidths=0.5, ax=axs[0], cbar=False)
sns.heatmap(mask_rvl[8], cmap='viridis', linewidths=0.5, ax=axs[1], cbar=False)

fig, axs = plt.subplots(1, 4, figsize=(12, 6))
for h_idx in range(4):
    mask_rvl = compute_unravel(patch_size, 3, 1, h_idx, q_idx, kv_idx).view(patch_size)
    sns.heatmap(mask_rvl[8], cmap='viridis', linewidths=0.5, ax=axs[h_idx], cbar=False)
    axs[h_idx].set_title(f'Head {h_idx}')

kv_patch = [16, 16, 16]
q_patch = [8, 8, 8]
kv_idx = torch.arange(reduce(lambda x, y: x * y, kv_patch))
q_idx = torch.tensor([q_patch[0] // 2 * q_patch[1] * q_patch[2] + q_patch[1] // 2 * q_patch[2] + q_patch[2] // 2])
mask_down = compute_mask_down(kv_patch, q_patch, q_idx, kv_idx).view(kv_patch)
plt.figure()
sns.heatmap(mask_down[8], cmap='viridis', linewidths=0.5, cbar=False)

kv_patch = [8, 8, 8]
q_patch = [16, 16, 16]
kv_idx = torch.arange(reduce(lambda x, y: x * y, kv_patch))
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
for ax_i, i in enumerate([-1, 0, 1]):
    q_idx = torch.tensor([q_patch[0] // 2 * q_patch[1] * q_patch[2] + q_patch[1] // 2 * q_patch[2] + q_patch[2] // 2]) + i
    mask_up = compute_mask_up(kv_patch, q_patch, q_idx, kv_idx).view(kv_patch)
    sns.heatmap(mask_up[4], cmap='viridis', linewidths=0.5, cbar=False, ax=axs[ax_i])
    axs[ax_i].set_title(f'Query idx {q_idx.item()}')


plt.show()
