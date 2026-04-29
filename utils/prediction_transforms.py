import numpy as np


def rescale_to_range(values, target_range):
    """Min-max rescale a prediction vector into [low, high]."""
    if target_range is None:
        return values

    if len(target_range) != 2:
        raise ValueError("prediction_return_rescale_range must have two values")

    low, high = float(target_range[0]), float(target_range[1])
    if not low < high:
        raise ValueError("prediction_return_rescale_range must satisfy low < high")

    arr = np.asarray(values, dtype=float)
    src_min = float(np.min(arr))
    src_max = float(np.max(arr))
    if np.isclose(src_min, src_max):
        return np.full_like(arr, (low + high) / 2.0, dtype=float)

    return low + (arr - src_min) * (high - low) / (src_max - src_min)
