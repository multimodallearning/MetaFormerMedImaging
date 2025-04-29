# calculated over training set
from dataclasses import dataclass


@dataclass
class ImgStats:
    mean: float
    std: float


IMG_MEAN_STD: dict[str, ImgStats] = {
    'organamnist': ImgStats(mean=0.46802595, std=0.28758164573365136)
}

LOSS_WEIGHTS = {
    'organamnist': [4.203477382659912, 4.986384391784668, 5.046650409698486, 4.84221887588501, 2.953120231628418,
                    3.009068489074707, 2.367891788482666, 2.969651699066162, 2.9658701419830322, 3.3767600059509277,
                    3.115352153778076],
}

ALLOW_FLIPPING = {
    'organamnist': False,
}