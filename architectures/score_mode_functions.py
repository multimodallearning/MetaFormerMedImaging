# functions to modify attention scores based on relative positions in directed local self attention.
# The input resolutions have to be hardcoded hence the wrapper function

def slopes_4_neighbours(score, b, h, q_idx, kv_idx, W):
    # unravel index
    q_x = q_idx % W
    q_y = q_idx // W
    kv_x = kv_idx % W
    kv_y = kv_idx // W

    dx = kv_x - q_x
    dy = kv_y - q_y

    invert_direction = 1 - (h % 2) * 2  # 1 or -1
    look_on_dx = h // 2  # 0 or 1
    look_on_dy = 1 - (h // 2)  # 0 or 1 (inverse to dx)
    bias = invert_direction * look_on_dx * dx + invert_direction * look_on_dy * dy

    return score + bias * 0.1


def wrapper_s4_56(score, b, h, q_idx, kv_idx):
    return slopes_4_neighbours(score, b, h, q_idx, kv_idx, 56)


def wrapper_s4_28(score, b, h, q_idx, kv_idx):
    return slopes_4_neighbours(score, b, h, q_idx, kv_idx, 28)


def wrapper_s4_14(score, b, h, q_idx, kv_idx):
    return slopes_4_neighbours(score, b, h, q_idx, kv_idx, 14)


def wrapper_s4_7(score, b, h, q_idx, kv_idx):
    return slopes_4_neighbours(score, b, h, q_idx, kv_idx, 7)
