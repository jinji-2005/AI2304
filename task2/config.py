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
    use_cmvn: bool = True
    context_size= 1
    use_delta=True
    use_delta_delta= False


@dataclass
class ModelConfig:
    model_type: str = "dnn"  # e.g. gmm / dnn
    train_epoch: int = 10
    batch_size: int = 1024*5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim_1: int = 64
    hidden_dim_2: int = 128
    dropout: float = 0.2
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    smooth_kernel_size: int = 3
    threshold_min: float = 0.10
    threshold_max: float = 0.90
    threshold_step: float = 0.01

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
