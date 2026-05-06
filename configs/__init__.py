from __future__ import annotations
import typing as tp
from pathlib import Path
import yaml

_PRESETS_PATH = Path(__file__).parent / "presets.yaml"
_PRESETS: tp.Optional[tp.Dict[str, tp.Any]] = None

_PATHS_KEYS = {"input", "output_dir", "model", "model_cfg", "device"}

# 배율-스케일 기준 (1024px 폭 기준)
CONST_SCALE_REFERENCE_MAG: float = 20000.0
CONST_SCALE_REFERENCE_PIXELS: float = 74.0  # 20000x @ 1024px


def mag_to_scale_pixels(float_mag: float) -> float:
    """배율에서 scale_pixels (px/µm, 1024px 기준) 자동 계산."""
    return CONST_SCALE_REFERENCE_PIXELS * (float_mag / CONST_SCALE_REFERENCE_MAG)


def parse_magnification(str_or_num: tp.Optional[tp.Any]) -> tp.Optional[float]:
    """'20k', '1.5k', 20000, 1500 등 다양한 형식을 float으로 변환."""
    if str_or_num is None:
        return None
    s = str(str_or_num).strip().lower()
    if s.endswith("k"):
        return float(s[:-1]) * 1000.0
    return float(s)


def mag_to_preset_key(float_mag: float) -> str:
    """20000 → '20k',  50000 → '50k' 등 프리셋 키 문자열로 변환."""
    float_k = float_mag / 1000.0
    if float_k == int(float_k):
        return f"{int(float_k)}k"
    return f"{float_k:g}k"


def _load() -> tp.Dict[str, tp.Any]:
    global _PRESETS
    if _PRESETS is None:
        with _PRESETS_PATH.open(encoding="utf-8") as f:
            _PRESETS = yaml.safe_load(f)
    return _PRESETS


def get_analysis_preset(
    str_particleType: str,
    str_magnification: str,
) -> tp.Dict[str, tp.Any]:
    """Return preset dict for particle_type x magnification, or {} if not found."""
    data = _load()
    return dict(data.get(str_particleType, {}).get(str_magnification, {}))


def load_paths_config(str_config_path: str) -> tp.Dict[str, tp.Any]:
    """Load a paths config YAML and return only non-empty recognised keys.

    Unknown keys and empty-string values are silently ignored so they never
    shadow argparse defaults.  Returns {} if the file does not exist.
    """
    try:
        with Path(str_config_path).open(encoding="utf-8") as obj_f:
            obj_raw = yaml.safe_load(obj_f) or {}
    except FileNotFoundError:
        return {}
    return {
        str_k: str_v
        for str_k, str_v in obj_raw.items()
        if str_k in _PATHS_KEYS and isinstance(str_v, str) and str_v.strip()
    }
