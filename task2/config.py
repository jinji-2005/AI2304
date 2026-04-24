from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameConfig:
    sample_rate: int = 16000
    frame_size: float = 0.032
    frame_shift: float = 0.008


@dataclass
class FeatureConfig:
    feature_type: str = "fbank"  # e.g. mfcc / fbank / plp
    feature_dim: int = 40


@dataclass
class ModelConfig:
    model_type: str = "gmm"  # e.g. gmm / dnn


@dataclass
class PathConfig:
    data_root: Path
    wav_root: Path
    train_label_path: Path
    dev_label_path: Path


@dataclass
class Task2Config:
    frame: FrameConfig
    feature: FeatureConfig
    model: ModelConfig
    paths: PathConfig
