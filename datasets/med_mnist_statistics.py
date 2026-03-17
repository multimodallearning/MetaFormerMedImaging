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
    'organsmnist': ImgStats(mean=[0.49522], std=[0.2769]),
    'nodulemnist3d': ImgStats(mean=[0.26745], std=[0.27127]),
    'synapsemnist3d': ImgStats(mean=[0.50973], std=[0.23681]),
    'fracturemnist3d': ImgStats(mean=[0.02485], std=[0.11808]),
    'organmnist3d': ImgStats(mean=[0.50595], std=[0.27877])
}

LOSS_WEIGHTS = {
    'pathmnist': [1.17291, 1.16406, 1.11523, 1.11303, 1.26863, 1.02845, 1.27824, 1.17073, 1.0],
    'chestmnist': [1.98194, 9.13998, 1.66659, 1.0, 4.30218, 3.89272, 18.5486, 4.65575, 5.33066, 10.59639, 9.93457,
                   15.61461, 7.77335, 127.86822],
    'dermamnist': [4.53691, 3.61559, 2.47038, 7.65917, 2.45447, 1.0, 6.88508],
    'octmnist': [1.17242, 2.12288, 2.43635, 1.0],
    'pneumoniamnist': [1.69649, 1.0],
    'breastmnist': [1.6475, 1.0],
    'bloodmnist': [1.65371, 1.03359, 1.46542, 1.0724, 1.65662, 1.5318, 1.0, 1.19086],
    'tissuemnist': [1.0, 2.6062, 3.00797, 1.85609, 2.12181, 2.62457, 1.16355, 1.46861],
    'organamnist': [1.7752, 2.10583, 2.13129, 2.04495, 1.24715, 1.27078, 1.0, 1.25413, 1.25254, 1.42606, 1.31567],
    'organcmnist': [1.61278, 2.19634, 2.2402, 2.23085, 1.65665, 1.59754, 1.0, 1.72628, 1.70931, 1.5955, 1.37822],
    'organsmnist': [1.73707, 2.34487, 2.37522, 2.1919, 1.7493, 1.75943, 1.0, 2.16212, 2.07697, 1.31474, 1.49205],
    'nodulemnist3d': [1.0, 1.71039],
    'synapsemnist3d': [1.64803, 1.0],
    'fracturemnist3d': [1.0, 1.1113, 1.66315],
    'organmnist3d': [1.0, 1.0, 1.0, 1.12416, 1.11201, 1.10608, 1.69558, 1.71718, 1.71718, 1.0, 1.0]
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
    'organsmnist': False,
    'nodulemnist3d': True,
    'synapsemnist3d': True,
    'fracturemnist3d': True,
    'organmnist3d': True
}

assert IMG_MEAN_STD.keys() == LOSS_WEIGHTS.keys() == ALLOW_FLIPPING.keys(), \
    "IMG_MEAN_STD, LOSS_WEIGHTS and ALLOW_FLIPPING keys must be the same"
