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


def min_ade_at_k(pred_modes: np.ndarray, target: np.ndarray) -> float:
    """
    pred_modes: [N, K, T, 2], target: [N, T, 2]
    Per sample: min over k of mean_t ||pred - tgt||; then mean over N.
    """
    err = np.linalg.norm(pred_modes - target[:, np.newaxis, :, :], axis=-1)  # N,K,T
    ade_k = err.mean(axis=-1)  # N,K
    return float(ade_k.min(axis=1).mean())


def min_fde_at_k(pred_modes: np.ndarray, target: np.ndarray) -> float:
    err = np.linalg.norm(pred_modes[:, :, -1, :] - target[:, np.newaxis, -1, :], axis=-1)  # N,K
    return float(err.min(axis=1).mean())
