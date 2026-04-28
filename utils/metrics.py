from __future__ import annotations
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
    list_values: tp.List[tp.Optional[float]],
) -> tp.Optional[float]:
    """Return mean of non-None values, or None if list is empty."""
    valid = [v for v in list_values if v is not None]
    return float(np.mean(valid)) if valid else None


def calculate_percentage(
    int_part: int,
    int_total: int,
) -> tp.Optional[float]:
    """Return part/total as percentage (0-100), or None if total is 0."""
    if int_total == 0:
        return None
    return round(100.0 * int_part / int_total, 2)


def json_default(obj: tp.Any) -> tp.Any:
    """Custom JSON default: convert numpy scalar/array to Python native type."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
