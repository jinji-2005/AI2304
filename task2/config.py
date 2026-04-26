from dataclasses import dataclass, field
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
    model_type: str = "dnn"  # e.g. gmm / dnn


@dataclass
class PathConfig:
    data_root: Path
    wav_root: Path = field(init=False)
    train_label_path: Path = field(init=False)
    dev_label_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.wav_root = self.data_root / "wavs"
        self.train_label_path = self.data_root / "data" / "train_label.txt"
        self.dev_label_path = self.data_root / "data" / "dev_label.txt"


@dataclass
class Task2Config:
    frame: FrameConfig
    feature: FeatureConfig
    model: ModelConfig
    paths: PathConfig
