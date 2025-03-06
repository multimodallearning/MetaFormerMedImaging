from typing import List


def unravel_index(idx: int, shape: List[int]) -> List[int]:
    unraveled = []
    for dim in reversed(shape):
        unraveled.append(idx % dim)
        idx //= dim
        print(f'dim: {dim}, idx: {idx}, unraveled: {unraveled}')
    return unraveled[::-1]

def unravel_index_2d(idx: int, shape: List[int]) -> tuple[int, int]:
    x = idx % shape[1]
    y = idx // shape[1]

    return x, y
