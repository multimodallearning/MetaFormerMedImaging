import os
from pathlib import Path
from shutil import copyfile

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.io import decode_image

os.chdir('/home/ron/Documents/FlexConv')

base = Path('data/tiger')
bcss = base / 'tissue-bcss'
cells = base / 'tissue-cells'

all_images_paths = [i for i in (list((bcss / 'images').glob('*.png')) + list((cells / 'images').glob('*.png')))]

img_stem = []
height = []
width = []
path = []
for p in all_images_paths:
    img_stem.append(p.stem)
    path.append(p)
    img = Image.open(p)
    height.append(img.height)
    width.append(img.width)
df = pd.DataFrame({'img_stem': img_stem, 'height': height, 'width': width, 'path': path})

df_big_resolution = df[(df['height'] >= 1000) | (df['width'] >= 1000)].copy()
print('number of images with sufficient resolution:', len(df_big_resolution))
df_big_resolution['split'] = 'test'
train_idx = df_big_resolution.sample(frac=.9, random_state=42)
df_big_resolution.loc[train_idx.index, 'split'] = 'train'
#df_big_resolution.to_csv(base / 'tiger_split.csv', index=False)

sum = torch.zeros(3)
n_pxls = 0
lbl_cnts = torch.zeros(8, dtype=torch.long)
for idx, row in tqdm(df_big_resolution.iterrows(), total=len(df_big_resolution)):
    dst_img_dir = base / ('imagesTr' if row['split'] == 'train' else 'imagesTs')
    dst_lbl_dir = base / ('labelsTr' if row['split'] == 'train' else 'labelsTs')

    img_path = row['path']
    lbl_path = row['path'].parents[1] / 'masks' / (row['img_stem'] + '.png')
    # copyfile(img_path, dst_img_dir / img_path.name)
    # copyfile(lbl_path, dst_lbl_dir / lbl_path.name)

    img = decode_image(img_path).flatten(start_dim=1).float().div(255)
    sum += img.sum(dim=1)
    n_pxls += img.shape[1]

    lbl = decode_image(lbl_path).int()
    lbl_cnts += torch.bincount(lbl.view(-1), minlength=len(lbl_cnts))
print('label counts', lbl_cnts)

mean = sum / n_pxls
print('mean:', mean.tolist())

std = torch.zeros(3)
for idx, row in tqdm(df_big_resolution.iterrows(), total=len(df_big_resolution)):
    img_path = row['path']
    img = decode_image(img_path).flatten(start_dim=1).float().div(255)
    std += ((img - mean.view(3, 1)) ** 2).sum(dim=1)
std = torch.sqrt(std / n_pxls)
print('std:', std.tolist())