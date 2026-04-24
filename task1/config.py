from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameConfig:
    sample_rate: int = 16000
    frame_size: float = 0.032
    frame_shift: float = 0.008


@dataclass
class PathConfig:
    data_root: Path
    wav_root: Path
    train_label_path: Path
    dev_label_path: Path


@dataclass
class ThresholdConfig:
    threshold: float = 0.5
    high_threshold: float = 0.6
    low_threshold: float = 0.4


@dataclass
class Task1Config:
    frame: FrameConfig
    paths: PathConfig
    threshold: ThresholdConfig
