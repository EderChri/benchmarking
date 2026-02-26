import torch
import numpy as np
import pandas as pd


def _to_numpy_array(data) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    if hasattr(data, "to_pd"):
        data = data.to_pd()

    if isinstance(data, pd.DataFrame):
        return data.values

    if isinstance(data, pd.Series):
        return data.values

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return np.array([])
        parts = [_to_numpy_array(d) for d in data]
        try:
            return np.concatenate(parts, axis=0)
        except Exception:
            return np.array(parts, dtype=object)

    return np.asarray(data)

def _to_flat_array(data, dtype=np.int64) -> np.ndarray:
    arr = _to_numpy_array(data)

    if arr.dtype == object:
        parts = [np.asarray(x).reshape(-1) for x in arr]
        arr = np.concatenate(parts, axis=0) if len(parts) > 0 else np.array([])

    if arr.ndim == 0:          # scalar — wrap it
        return np.array([arr.item()], dtype=dtype)

    arr = np.squeeze(arr)

    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = np.argmax(arr, axis=1)
    elif arr.ndim >= 2:
        arr = arr.squeeze()

    return arr.astype(dtype).flatten()


def _to_2d_array(data, dtype=np.float64) -> np.ndarray:
    arr = _to_numpy_array(data)

    if arr.dtype == object:
        rows = [np.asarray(x, dtype=dtype).reshape(-1) for x in arr]
        if len(rows) == 0:
            return np.empty((0, 0), dtype=dtype)
        arr = np.vstack(rows)

    if arr.ndim == 0:
        arr = np.array([[arr.item()]], dtype=dtype)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(dtype)


