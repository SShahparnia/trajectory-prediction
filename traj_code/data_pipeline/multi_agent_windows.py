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


def build_multi_agent_windows(
    infos_path: str,
    past_len: int = 10,
    future_len: int = 20,
    max_neighbors: int = 8,
    max_windows: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    infos = load_infos(infos_path)
    grouped = _group_by_sequence(infos)

    ego_x, neigh_x, neigh_mask, y = [], [], [], []
    for _, frames in grouped.items():
        ids = sorted(frames.keys())
        id_set = set(ids)
        obj_xy = {fid: _xy_by_obj(frames[fid]) for fid in ids}

        for anchor in ids:
            past_ids = [anchor - k for k in range(past_len - 1, -1, -1)]
            fut_ids = [anchor + k for k in range(1, future_len + 1)]
            if not all(fid in id_set for fid in past_ids + fut_ids):
                continue

            common_all = None
            for fid in past_ids + fut_ids:
                ids_here = set(obj_xy[fid].keys())
                common_all = ids_here if common_all is None else (common_all & ids_here)
                if not common_all:
                    break
            if not common_all:
                continue

            for target_obj in common_all:
                x_abs = np.stack([obj_xy[fid][target_obj] for fid in past_ids], axis=0)
                y_abs = np.stack([obj_xy[fid][target_obj] for fid in fut_ids], axis=0)
                origin = x_abs[-1].copy()
                ego_local = (x_abs - origin).astype(np.float32)
                y_local = (y_abs - origin).astype(np.float32)

                candidates = [oid for oid in common_all if oid != target_obj]
                if candidates:
                    anchor_xy = obj_xy[anchor][target_obj]
                    candidates = sorted(
                        candidates,
                        key=lambda oid: float(
                            np.linalg.norm(obj_xy[anchor][oid] - anchor_xy)
                        ),
                    )
                selected = candidates[:max_neighbors]

                neigh_arr = np.zeros((max_neighbors, past_len, 2), dtype=np.float32)
                mask_arr = np.zeros((max_neighbors,), dtype=np.float32)
                for ni, oid in enumerate(selected):
                    tr = np.stack([obj_xy[fid][oid] for fid in past_ids], axis=0)
                    neigh_arr[ni] = (tr - origin).astype(np.float32)
                    mask_arr[ni] = 1.0

                ego_x.append(ego_local)
                neigh_x.append(neigh_arr)
                neigh_mask.append(mask_arr)
                y.append(y_local)

                if len(ego_x) >= max_windows:
                    return (
                        np.stack(ego_x),
                        np.stack(neigh_x),
                        np.stack(neigh_mask),
                        np.stack(y),
                    )

    if not ego_x:
        raise RuntimeError("No multi-agent windows built; reduce lengths or inspect data.")
    return np.stack(ego_x), np.stack(neigh_x), np.stack(neigh_mask), np.stack(y)


class MultiAgentTrajectoryDataset(Dataset):
    def __init__(
        self,
        ego_x: np.ndarray,
        neigh_x: np.ndarray,
        neigh_mask: np.ndarray,
        y: np.ndarray,
    ):
        self.ego_x = torch.from_numpy(ego_x)
        self.neigh_x = torch.from_numpy(neigh_x)
        self.neigh_mask = torch.from_numpy(neigh_mask)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.ego_x.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.ego_x[idx],
            self.neigh_x[idx],
            self.neigh_mask[idx],
            self.y[idx],
        )
