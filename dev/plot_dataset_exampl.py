from matplotlib import pyplot as plt
from pathlib import Path

base = Path('./data/tiger')
test_data = (base / 'imagesTs').glob('*.png')
test_data = [{'image': str(img_path), 'label': str(base / 'labelsTs' / img_path.name)} for img_path in test_data]

for sample in test_data:
    img = plt.imread(sample['image'])
    lbl = plt.imread(sample['label'])
    plt.figure(sample['image'].split('/')[-1])
    plt.imshow(img)
    plt.imshow(lbl)
plt.show()
