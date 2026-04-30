from __future__ import annotations
import typing as tp
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sam2AspectRatioConfig:
    path_input: Path = Path(".")
    path_outputDir: Path = Path("out")
    path_modelConfig: Path = Path("model/sam2.1_hiera_t.yaml")
    path_modelWeights: Path = Path("model/sam2.1_hiera_base_plus.pt")
    int_roiXMin: int = 0
    int_roiYMin: int = 0
    int_roiXMax: int = 1024
    int_roiYMax: int = 768
    int_bboxEdgeMargin: int = 8
    int_tileEdgeMargin: int = 8
    float_particleAreaThreshold: float = 1500.0
    float_maskBinarizeThreshold: float = 0.0
    int_minValidMaskArea: int = 1
    int_maskMorphKernelSize: int = 0
    int_maskMorphOpenIterations: int = 0
    int_maskMorphCloseIterations: int = 0
    int_imgSize: int = 1536
    int_tileSize: int = 512
    int_stride: int = 256
    int_pointsPerTile: int = 80
    int_pointMinDistance: int = 14
    float_pointQualityLevel: float = 0.03
    int_pointBatchSize: int = 32
    float_dedupIou: float = 0.60
    float_bboxDedupIou: float = 0.85
    bool_usePointPrompts: bool = True
    bool_smallParticle: bool = False
    float_scalePixels: float = 147.0
    float_scaleMicrometers: float = 1.0
    str_device: tp.Optional[str] = None
    bool_retinaMasks: bool = True
    bool_saveIndividualMasks: bool = True
    bool_useEqDiameter: bool = True


@dataclass
class ObjectMeasurement:
    int_index: int
    str_category: str
    int_maskArea: int
    float_confidence: tp.Optional[float]
    int_bboxX: int
    int_bboxY: int
    int_bboxWidth: int
    int_bboxHeight: int
    float_bboxWidthUm: float
    float_bboxHeightUm: float
    float_centroidX: float
    float_centroidY: float
    int_longestHorizontal: int
    int_longestVertical: int
    float_longestHorizontalUm: float
    float_longestVerticalUm: float
    float_eqDiameterUm: float
    float_sphericity: tp.Optional[float]


@dataclass
class Sam2AspectRatioResult:
    list_objects: tp.List[ObjectMeasurement]
    dict_summary: tp.Dict[str, tp.Any]


@dataclass
class PrimaryParticleConfig(Sam2AspectRatioConfig):
    float_acicularThreshold: float = 0.40
    bool_autoCenterCrop: bool = True
    float_centerCropRatio: float = 0.60
    int_targetParticleCount: int = 10
    str_particleMode: str = "auto"
    bool_autoDetectSphere: bool = False
    float_sphereCapFraction: float = 0.45
    str_particleType: str = "unknown"
    str_magnification: str = "unknown"
    str_measureMode: str = "sam2"
    bool_lsdAdaptiveThresh: bool = False

    bool_arScreen: bool = False
    int_lsdMinLengthPx: int = 20
    bool_fuseContours: bool = False


@dataclass
class PrimaryParticleMeasurement:
    int_index: int
    str_category: str
    int_maskArea: int
    float_confidence: tp.Optional[float]
    int_bboxX: int
    int_bboxY: int
    int_bboxWidth: int
    int_bboxHeight: int
    float_centroidX: float
    float_centroidY: float
    float_thicknessPx: float
    float_longAxisPx: float
    float_minRectAngle: float
    float_thicknessUm: float
    float_longAxisUm: float
    float_aspectRatio: float
    int_longestHorizontal: int
    int_longestVertical: int
    float_longestHorizontalUm: float
    float_longestVerticalUm: float


@dataclass
class PrimaryParticleResult:
    list_objects: tp.List[PrimaryParticleMeasurement]
    dict_summary: tp.Dict[str, tp.Any]
