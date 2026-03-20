import numpy as np


def ade(pred: np.ndarray, target: np.ndarray) -> float:
    """
    pred,target: [N,T,2]
    """
    err = np.linalg.norm(pred - target, axis=-1)
    return float(np.mean(err))


def fde(pred: np.ndarray, target: np.ndarray) -> float:
    err = np.linalg.norm(pred[:, -1, :] - target[:, -1, :], axis=-1)
    return float(np.mean(err))
