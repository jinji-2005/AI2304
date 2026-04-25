from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from tqdm.auto import tqdm

from config import FeatureConfig, FrameConfig, ModelConfig, PathConfig
from dataset import (
    align_frame_labels_to_num_frames,
    list_wav_files,
    load_waveform,
    read_label_from_file,
)
from features import extract_spectral_features
from model import StatisticalModelParams, StatisticalVAD


class DevResult(Dict[str, float]):
    """Container for development metrics."""


def _evaluate_py_path(project_root: Path) -> Path:
    return (
        project_root
        / "voice-activity-detection-sjtu-spring-2024"
        / "vad"
        / "evaluate.py"
    )


@lru_cache(maxsize=None)
def _load_official_get_metrics(evaluate_py_abs: str) -> Callable:
    """Load official `get_metrics` from provided evaluate.py path."""
    evaluate_path = Path(evaluate_py_abs)
    if not evaluate_path.exists():
        raise FileNotFoundError(f"Official evaluate.py not found: {evaluate_path}")

    module_name = f"_official_vad_evaluate_{abs(hash(evaluate_py_abs))}"
    spec = spec_from_file_location(module_name, str(evaluate_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {evaluate_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    get_metrics = getattr(module, "get_metrics", None)
    if not callable(get_metrics):
        raise AttributeError(f"`get_metrics` is missing in: {evaluate_path}")
    return get_metrics


def compute_auc_eer(
    pred_scores: np.ndarray, label_binary: np.ndarray, project_root: Path
) -> Tuple[float, float]:
    """Compute AUC/EER by reusing official `evaluate.py` implementation."""
    pred_arr = np.asarray(pred_scores).reshape(-1)
    label_arr = np.asarray(label_binary).reshape(-1)
    if pred_arr.shape != label_arr.shape:
        raise ValueError(
            f"pred_scores and labels shape mismatch: {pred_arr.shape} vs {label_arr.shape}"
        )

    evaluate_py = _evaluate_py_path(project_root).resolve()
    get_metrics = _load_official_get_metrics(str(evaluate_py))
    auc, eer = get_metrics(pred_arr.tolist(), label_arr.tolist())
    return float(auc), float(eer)


def build_runtime_components(project_root: Path) -> Tuple[PathConfig, FrameConfig, FeatureConfig, StatisticalVAD]:
    """Build common configs/model used by Task2 train/dev/test pipeline."""
    data_root = (
        Path(project_root)
        / "voice-activity-detection-sjtu-spring-2024"
        / "vad"
    )
    path_cfg = PathConfig(data_root=data_root)
    frame_cfg = FrameConfig()
    feature_cfg = FeatureConfig()
    model_cfg = ModelConfig()
    model = StatisticalVAD(StatisticalModelParams(model_type=model_cfg.model_type))
    return path_cfg, frame_cfg, feature_cfg, model


def build_xy(
    split: str,
    label_path: Path,
    path_cfg: PathConfig,
    frame_cfg: FrameConfig,
    feature_cfg: FeatureConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build concatenated frame-level features/labels for one supervised split."""
    wav_files = list_wav_files(path_cfg.wav_root / split)
    label_dict = read_label_from_file(
        label_path,
        frame_size=frame_cfg.frame_size,
        frame_shift=frame_cfg.frame_shift,
    )

    x_list = []
    y_list = []
    iterator = tqdm(
        wav_files,
        total=len(wav_files),
        desc=f"[{split}] feature extraction",
        unit="utt",
    )
    for wav_path in iterator:
        utt_id = wav_path.stem
        if utt_id not in label_dict:
            continue

        waveform = load_waveform(wav_path, frame_cfg.sample_rate)
        x = extract_spectral_features(
            waveform=waveform,
            sample_rate=frame_cfg.sample_rate,
            frame_size=frame_cfg.frame_size,
            frame_shift=frame_cfg.frame_shift,
            feature_type=feature_cfg.feature_type,
            feature_dim=feature_cfg.feature_dim,
        )
        y = align_frame_labels_to_num_frames(label_dict[utt_id], x.shape[0])

        x_list.append(x)
        y_list.append(y)
        iterator.set_postfix({"kept": len(x_list)})

    if not x_list:
        raise RuntimeError(f"No matched labeled samples in split={split}")

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return x_all, y_all


def run_dev_pipeline(project_root: Path) -> DevResult:
    """Run Task2 development experiment and return metric dict.

    Suggested metric keys: acc, auc, eer
    """
    # Data usage plan:
    # 1) Train side (recommended):
    #    wavs/train + data/train_label.txt
    # 2) Dev side:
    #    wavs/dev + data/dev_label.txt
    # 3) Build frame-level train features/labels -> fit model
    # 4) On dev, gather utterance scores + aligned labels
    # 5) Concatenate all dev frames and compute acc + official auc/eer:
    #    auc, eer = compute_auc_eer(pred_scores, labels, project_root)
    # TODO: implement
    raise NotImplementedError


def run_test_pipeline(project_root: Path, output_path: Path) -> None:
    """Run Task2 inference on test set and write test_label.txt."""
    # Data usage plan:
    # 1) Read test wavs from wavs/test
    # 2) Apply trained model to obtain frame predictions
    # 3) Convert each utterance to timestamp segments
    # 4) Write `utt_id <space> start,end ...` per line to output_path
    # TODO: implement
    raise NotImplementedError


def compute_acc(pred_binary: np.ndarray, label_binary: np.ndarray) -> float:
    """Compute frame-level accuracy.

    Data usage:
    - Input: concatenated frame-level predictions and ground truth on dev
    - Output: scalar accuracy for report table
    """
    pred_arr = np.asarray(pred_binary).reshape(-1)
    label_arr = np.asarray(label_binary).reshape(-1)
    if pred_arr.shape != label_arr.shape:
        raise ValueError(
            f"pred_binary and label_binary shape mismatch: {pred_arr.shape} vs {label_arr.shape}"
        )

    pred_arr = pred_arr.astype(np.int64)
    label_arr = label_arr.astype(np.int64)
    return float(np.mean(pred_arr == label_arr))
