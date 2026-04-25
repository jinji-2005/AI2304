from dataclasses import dataclass,field
from pathlib import Path


@dataclass
class FrameConfig:
    sample_rate: int = 16000
    frame_size: float = 0.032
    frame_shift: float = 0.008


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
class ThresholdConfig:
    threshold: float = 0.5
    high_threshold: float = 0.6
    low_threshold: float = 0.4


@dataclass
class Task1Config:
    frame: FrameConfig
    paths: PathConfig
    threshold: ThresholdConfig

# Task1Config 可以用这个加载 config 么
# split_wav_and_label 是不是没用
# auc 和 eer 都是什么
# 