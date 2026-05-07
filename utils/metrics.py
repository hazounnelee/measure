from __future__ import annotations
import json
import math
import typing as tp
import numpy as np


def normalize_image_to_uint8(arr_img: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 [0, 255]."""
    arr_f = arr_img.astype(np.float32)
    float_min = float(arr_f.min())
    float_max = float(arr_f.max())
    if float_max - float_min < 1e-6:
        return np.zeros_like(arr_img, dtype=np.uint8)
    arr_norm = (arr_f - float_min) / (float_max - float_min) * 255.0
    return arr_norm.astype(np.uint8)


def convert_pixels_to_micrometers(
    float_pixels: float,
    float_scalePixels: float,
    float_scaleMicrometers: float,
) -> float:
    """Convert pixel length to micrometers using scale bar calibration."""
    if float_scalePixels <= 0:
        return 0.0
    return float_pixels * (float_scaleMicrometers / float_scalePixels)


def calculate_mean_from_optional_values(
    list_values: tp.Iterable[tp.Optional[float]],
) -> tp.Optional[float]:
    """Return mean of non-None, non-NaN values, or None if no valid values."""
    valid = [v for v in list_values if v is not None and not math.isnan(float(v))]
    return float(np.mean(valid)) if valid else None


def calculate_percentage(
    int_part: int,
    int_total: int,
) -> tp.Optional[float]:
    """Return part/total as percentage (0-100), or None if total is 0."""
    if int_total == 0:
        return None
    return round(100.0 * int_part / int_total, 2)


def pooled_stats(
    list_vals: tp.List[float],
) -> tp.Dict[str, tp.Optional[float]]:
    """Return mean/median/std dict for a list of floats, filtering NaN. Returns None fields if empty."""
    if not list_vals:
        return {"mean": None, "median": None, "std": None}
    arr = np.array(list_vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def _safe_float(v: float) -> tp.Optional[float]:
    """Return None for NaN/Inf, otherwise return the float unchanged."""
    return None if math.isnan(v) or math.isinf(v) else v


class _SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf floats to null and handles numpy types."""

    def default(self, obj: tp.Any) -> tp.Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return _safe_float(float(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def iterencode(self, obj: tp.Any, _one_shot: bool = False) -> tp.Iterator[str]:
        # Intercept float NaN/Inf before the C encoder can emit them as bare NaN/Infinity
        return super().iterencode(self._sanitize(obj), _one_shot)

    @classmethod
    def _sanitize(cls, obj: tp.Any) -> tp.Any:
        if isinstance(obj, float):
            return _safe_float(obj)
        if isinstance(obj, dict):
            return {k: cls._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._sanitize(v) for v in obj]
        if isinstance(obj, np.floating):
            return _safe_float(float(obj))
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return [cls._sanitize(v) for v in obj.tolist()]
        return obj



def json_dump_safe(obj: tp.Any, fp: tp.Any, **kwargs: tp.Any) -> None:
    """Write JSON to file, replacing NaN/Inf with null for valid output."""
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    json.dump(obj, fp, cls=_SafeJSONEncoder, **kwargs)
