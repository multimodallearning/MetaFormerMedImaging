from typing import List


def unravel_index(idx: int, shape: List[int]) -> List[int]:
    unraveled = []
    for dim in reversed(shape):
        unraveled.append(idx % dim)
        idx //= dim
        print(f'dim: {dim}, idx: {idx}, unraveled: {unraveled}')
    return unraveled[::-1]

def unravel_index_2d(idx: int, shape: List[int]) -> tuple[int, int]:
    w = idx % shape[1]
    h = idx // shape[1]

    return h, w

def unravel_index_3d(idx: int, shape: List[int]) -> tuple[int, int, int]:
    d = idx % shape[2]
    w = (idx // shape[2]) % shape[1]
    h = idx // (shape[2] * shape[1])

    return h, w, d
