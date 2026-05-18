import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_infos(path: str) -> List[Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _group_by_sequence(infos: List[Dict]) -> Dict[str, Dict[int, Dict]]:
    grouped = defaultdict(dict)
    for info in infos:
        pc = info.get("point_cloud", {})
        seq = pc.get("lidar_sequence", "")
        idx = int(pc.get("sample_idx", -1))
        if seq and idx >= 0:
            grouped[seq][idx] = info
    return grouped


def _xy_by_obj(info: Dict) -> Dict[str, np.ndarray]:
    ann = info.get("annos", {})
    obj_ids = ann.get("obj_ids", [])
    loc = np.asarray(ann.get("location", np.zeros((0, 3), dtype=np.float32)))
    out = {}
    for i, obj_id in enumerate(obj_ids):
        if i < loc.shape[0]:
            out[str(obj_id)] = loc[i, :2].astype(np.float32)
    return out


def build_xy_windows(
    infos_path: str,
    past_len: int = 10,
    future_len: int = 20,
    max_windows: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    infos = load_infos(infos_path)
    grouped = _group_by_sequence(infos)

    xs, ys = [], []
    for _, frames in grouped.items():
        ids = sorted(frames.keys())
        id_set = set(ids)
        obj_xy = {fid: _xy_by_obj(frames[fid]) for fid in ids}

        for anchor in ids:
            past_ids = [anchor - k for k in range(past_len - 1, -1, -1)]
            fut_ids = [anchor + k for k in range(1, future_len + 1)]
            if not all(fid in id_set for fid in past_ids + fut_ids):
                continue

            common = None
            for fid in past_ids + fut_ids:
                ids_here = set(obj_xy[fid].keys())
                common = ids_here if common is None else (common & ids_here)
                if not common:
                    break
            if not common:
                continue

            for obj_id in common:
                x_abs = np.stack([obj_xy[fid][obj_id] for fid in past_ids], axis=0)
                y_abs = np.stack([obj_xy[fid][obj_id] for fid in fut_ids], axis=0)
                origin = x_abs[-1].copy()
                xs.append((x_abs - origin).astype(np.float32))
                ys.append((y_abs - origin).astype(np.float32))
                if len(xs) >= max_windows:
                    return np.stack(xs), np.stack(ys)

    if not xs:
        raise RuntimeError("No windows built; try smaller past/future lengths.")
    return np.stack(xs), np.stack(ys)


class TrajectoryDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
