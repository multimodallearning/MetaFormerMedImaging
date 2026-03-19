import csv
from pathlib import Path
from random import shuffle, seed
from typing import List
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('path')
path2dataset = parser.parse_args().path

def save_to_csv(l: List[str], split: str):
    with open(f'datasets/AbdomenAtlas{split.capitalize()}Split.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[item] for item in l])


data_dirs = Path(path2dataset)
available_files = [p.stem for p in data_dirs.iterdir() if p.is_dir()]
seed(42)
shuffle(available_files)
split_idx = int(len(available_files) * 0.9)
train_split = available_files[:split_idx]
test_split = available_files[split_idx:]
save_to_csv(train_split, 'train')
save_to_csv(test_split, 'test')
