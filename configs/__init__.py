from __future__ import annotations
import typing as tp
from pathlib import Path
import yaml

_PRESETS_PATH = Path(__file__).parent / "presets.yaml"
_PRESETS: tp.Optional[tp.Dict[str, tp.Any]] = None


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
