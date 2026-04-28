from __future__ import annotations
import typing as tp

try:
    from ultralytics import SAM as SAM
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    SAM = None  # type: ignore[assignment,misc]
    _ULTRALYTICS_AVAILABLE = False


def load_sam2_model(str_weights_path: str) -> tp.Any:
    """Load and return a SAM2 model instance.

    Raises ImportError if ultralytics is not installed.
    """
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "ultralytics 패키지가 없습니다. SAM2 모드에는 ultralytics가 필요합니다."
        )
    return SAM(str_weights_path)
