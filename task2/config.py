from dataclasses import dataclass, field
from pathlib import Path
import os


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class FrameConfig:
    sample_rate: int = field(default_factory=lambda: _env_int("TASK2_SAMPLE_RATE", 16000))
    frame_size: float = field(default_factory=lambda: _env_float("TASK2_FRAME_SIZE", 0.032))
    frame_shift: float = field(default_factory=lambda: _env_float("TASK2_FRAME_SHIFT", 0.008))


@dataclass
class FeatureConfig:
    feature_type: str = field(default_factory=lambda: _env_str("TASK2_FEATURE_TYPE", "fbank"))  # mfcc/fbank/plp
    feature_dim: int = field(default_factory=lambda: _env_int("TASK2_FEATURE_DIM", 40))
    use_cmvn: bool = field(default_factory=lambda: _env_bool("TASK2_USE_CMVN", True))
    context_size: int = field(default_factory=lambda: _env_int("TASK2_CONTEXT_SIZE", 6))
    use_delta: bool = field(default_factory=lambda: _env_bool("TASK2_USE_DELTA", True))
    use_delta_delta: bool = field(default_factory=lambda: _env_bool("TASK2_USE_DELTA_DELTA", True))


@dataclass
class ModelConfig:
    model_type: str = field(default_factory=lambda: _env_str("TASK2_MODEL_TYPE", "dnn"))  # gmm/dnn
    dnn_variant: str = field(default_factory=lambda: _env_str("TASK2_DNN_VARIANT", "mlp"))  # mlp/bn_dropout
    train_epoch: int = field(default_factory=lambda: _env_int("TASK2_TRAIN_EPOCH", 20))
    batch_size: int = field(default_factory=lambda: _env_int("TASK2_BATCH_SIZE", 5210))
    learning_rate: float = field(default_factory=lambda: _env_float("TASK2_LEARNING_RATE", 1e-3))
    weight_decay: float = field(default_factory=lambda: _env_float("TASK2_WEIGHT_DECAY", 1e-5))
    hidden_dim_1: int = field(default_factory=lambda: _env_int("TASK2_HIDDEN_DIM_1", 256))
    hidden_dim_2: int = field(default_factory=lambda: _env_int("TASK2_HIDDEN_DIM_2", 512))
    dropout: float = field(default_factory=lambda: _env_float("TASK2_DROPOUT", 0.2))
    use_focal_loss: bool = field(default_factory=lambda: _env_bool("TASK2_USE_FOCAL_LOSS", True))
    focal_alpha: float = field(default_factory=lambda: _env_float("TASK2_FOCAL_ALPHA", 0.25))
    focal_gamma: float = field(default_factory=lambda: _env_float("TASK2_FOCAL_GAMMA", 2.0))
    smooth_kernel_size: int = field(default_factory=lambda: _env_int("TASK2_SMOOTH_KERNEL_SIZE", 15))
    threshold_min: float = field(default_factory=lambda: _env_float("TASK2_THRESHOLD_MIN", 0.10))
    threshold_max: float = field(default_factory=lambda: _env_float("TASK2_THRESHOLD_MAX", 0.90))
    threshold_step: float = field(default_factory=lambda: _env_float("TASK2_THRESHOLD_STEP", 0.01))


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
