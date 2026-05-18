import numpy as np


def add_velocity(xy_seq: np.ndarray) -> np.ndarray:
    """
    xy_seq: [T,2] -> features [T,4] with dx,dy appended.
    """
    vel = np.zeros_like(xy_seq, dtype=np.float32)
    vel[1:] = xy_seq[1:] - xy_seq[:-1]
    return np.concatenate([xy_seq.astype(np.float32), vel], axis=-1)
