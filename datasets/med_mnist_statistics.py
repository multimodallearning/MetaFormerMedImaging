# calculated over training set
from dataclasses import dataclass
from typing import List


@dataclass
class ImgStats:
    mean: List[float]
    std: List[float]


IMG_MEAN_STD: dict[str, ImgStats] = {
    'pathmnist': ImgStats(mean=[0.74052, 0.53295, 0.70579], std=[0.16514, 0.21737, 0.1574]),
    'chestmnist': ImgStats(mean=[0.49795], std=[0.24798]),
    'dermamnist': ImgStats(mean=[0.76324, 0.53806, 0.56154], std=[0.13685, 0.15866, 0.1769]),
    'octmnist': ImgStats(mean=[0.18955], std=[0.21475]),
    'pneumoniamnist': ImgStats(mean=[0.57166], std=[0.17704]),
    'breastmnist': ImgStats(mean=[0.32769], std=[0.21943]),
    'bloodmnist': ImgStats(mean=[0.79607, 0.65964, 0.69639], std=[0.22646, 0.25977, 0.09622]),
    'tissuemnist': ImgStats(mean=[0.10206], std=[0.09893]),
    'organamnist': ImgStats(mean=[0.46803], std=[0.28758]),
    'organcmnist': ImgStats(mean=[0.49419], std=[0.27756]),
    'organsmnist': ImgStats(mean=[0.49522], std=[0.2769])
}

LOSS_WEIGHTS = {
    'pathmnist': [3.09981, 3.07641, 2.94735, 2.94154, 3.35277, 2.71802, 3.37818, 3.09403, 2.64283],
    'chestmnist': [6.06603, 27.97436, 5.10085, 3.06066, 13.1675, 11.91429, 56.77096, 14.24966, 16.31535, 32.43195,
                   30.40634, 47.79102, 23.79157, 391.36111],
    'dermamnist': [5.54369, 4.41793, 3.01858, 9.35882, 2.99914, 1.22191, 8.41295],
    'octmnist': [1.70621, 3.0894, 3.54559, 1.45529],
    'pneumoniamnist': [1.96929, 1.1608],
    'breastmnist': [1.92725, 1.1698],
    'bloodmnist': [3.74652, 2.34164, 3.31996, 2.42956, 3.75313, 3.47035, 2.26553, 2.69792],
    'tissuemnist': [1.76567, 4.60169, 5.31109, 3.27725, 3.74642, 4.63413, 2.05445, 2.59308],
    'organamnist': [4.20348, 4.98638, 5.04665, 4.84222, 2.95312, 3.00907, 2.36789, 2.96965, 2.96587, 3.37676, 3.11535],
    'organcmnist': [3.36188, 4.57834, 4.66977, 4.65027, 3.45334, 3.33013, 2.08453, 3.59849, 3.5631, 3.32587, 2.87295],
    'organsmnist': [3.48366, 4.70258, 4.76346, 4.39581, 3.50819, 3.52851, 2.00548, 4.33608, 4.16533, 2.63668, 2.99228]

}

ALLOW_FLIPPING = {
    'pathmnist': True,
    'chestmnist': False,
    'dermamnist': True,
    'octmnist': False,
    'pneumoniamnist': False,
    'breastmnist': True,
    'bloodmnist': True,
    'tissuemnist': True,
    'organamnist': False,
    'organcmnist': False,
    'organsmnist': False
}

assert IMG_MEAN_STD.keys() == LOSS_WEIGHTS.keys() == ALLOW_FLIPPING.keys(), \
    "IMG_MEAN_STD, LOSS_WEIGHTS and ALLOW_FLIPPING keys must be the same"