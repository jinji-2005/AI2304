from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm.auto import tqdm

from config import FeatureConfig, FrameConfig, ModelConfig, PathConfig
from dataset import (
    align_frame_labels_to_num_frames,
    list_wav_files,
    load_waveform,
    read_label_from_file,
)
from features import extract_spectral_features
from model import DNNClassifier
from postprocess import frame_prediction_to_label_line,smooth_predictions

class DevResult(Dict[str, float]):
    """Container for development metrics."""


def save_roc_curve(
    pred_scores: np.ndarray,
    label_binary: np.ndarray,
    mode_name: str,
    out_dir: Path,
) -> Path:
    """Save one ROC curve image with AUC annotation."""
    scores = np.asarray(pred_scores).reshape(-1)
    labels = np.asarray(label_binary).reshape(-1).astype(np.int64)
    if scores.shape != labels.shape:
        raise ValueError(f"score/label shape mismatch: {scores.shape} vs {labels.shape}")

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2.0, label=f"AUC = {roc_auc:.6f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", label="random")
    ax.set_title(f"ROC Curve ({mode_name})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path = out_dir / "roc_curve.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def evaluate_metrics(
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    label_binary: np.ndarray,
    project_root: Path,
) -> Dict[str, float]:
    pred_arr = np.asarray(pred_scores).reshape(-1)
    label_arr = np.asarray(label_binary).reshape(-1)
    if pred_arr.shape != label_arr.shape:
        raise ValueError("pred_scores and labels shape mismatch in evaluate_metrics")

    auc, eer = compute_auc_eer(pred_arr, label_arr, project_root)

    pred_bin = np.asarray(pred_labels).reshape(-1).astype(np.int64)
    label_bin = label_arr.astype(np.int64)
    if pred_bin.shape != label_bin.shape:
        raise ValueError("pred_labels and labels shape mismatch in evaluate_metrics")

    tp = float(np.sum((pred_bin == 1) & (label_bin == 1)))
    fp = float(np.sum((pred_bin == 1) & (label_bin == 0)))
    fn = float(np.sum((pred_bin == 0) & (label_bin == 1)))
    tn = float(np.sum((pred_bin == 0) & (label_bin == 0)))

    eps = 1e-8
    recall = float(tp / (tp + fn + eps))  # 召回率
    precision = float(tp / (tp + fp + eps))  # 精确度
    F1 = float(2 * precision * recall / (precision + recall + eps))  # 平衡召回率与精确度
    FAR = float(fp / (fp + tn + eps))  # 误报率
    FRR = float(fn / (fn + tp + eps))  # 漏检率

    return {
        "auc": auc,
        "eer": eer,
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "FAR": FAR,
        "FRR": FRR,
    }

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


def sweep_best_threshold_by_acc(
    scores_by_utt: List[np.ndarray],
    labels_by_utt: List[np.ndarray],
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_step: float = 0.01,
    smooth_kernel_size: int = 3,
) -> Tuple[float, float]:
    """Sweep thresholds on dev scores and return (best_threshold, best_acc).

    Smoothing is performed utterance by utterance to avoid cross-utterance leakage.
    """
    if len(scores_by_utt) != len(labels_by_utt):
        raise ValueError(
            f"scores_by_utt and labels_by_utt length mismatch: {len(scores_by_utt)} vs {len(labels_by_utt)}"
        )

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    total_frames = 0
    for idx, (scores, labels) in enumerate(zip(scores_by_utt, labels_by_utt)):
        score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if score_arr.shape != label_arr.shape:
            raise ValueError(
                f"utt[{idx}] score/label shape mismatch: {score_arr.shape} vs {label_arr.shape}"
            )
        if score_arr.size == 0:
            continue
        pairs.append((score_arr, label_arr))
        total_frames += int(score_arr.size)
    if not pairs:
        raise RuntimeError("No valid dev utterance found for threshold sweeping.")

    thresholds = np.arange(threshold_min, threshold_max + 1e-8, threshold_step, dtype=np.float32)
    best_threshold = float(thresholds[0])
    best_acc = -1.0

    iterator = tqdm(
        thresholds,
        total=thresholds.size,
        desc="[dev] threshold sweep",
        unit="thr",
    )
    for threshold in iterator:
        correct = 0
        for score_arr, label_arr in pairs:
            pred = (score_arr >= float(threshold)).astype(np.int64)
            # pred = smooth_predictions(pred, kernel_size=smooth_kernel_size)
            correct += int(np.sum(pred == label_arr))
        acc = correct / total_frames
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
        iterator.set_postfix(
            {
                "thr": f"{float(threshold):.2f}",
                "best_thr": f"{best_threshold:.2f}",
                "best_acc": f"{best_acc:.6f}",
            }
        )

    return best_threshold, float(best_acc)


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
            use_cmvn=feature_cfg.use_cmvn,
            use_delta=feature_cfg.use_delta,
            use_delta_delta=feature_cfg.use_delta_delta,
            context_size=feature_cfg.context_size,
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


def build_split_utterances(
    split: str,
    label_path: Path,
    path_cfg: PathConfig,
    frame_cfg: FrameConfig,
    feature_cfg: FeatureConfig,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Build per-utterance features and aligned labels for one split."""
    wav_files = list_wav_files(path_cfg.wav_root / split)
    label_dict = read_label_from_file(
        label_path,
        frame_size=frame_cfg.frame_size,
        frame_shift=frame_cfg.frame_shift,
    )

    utt_ids: List[str] = []
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
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
            use_cmvn=feature_cfg.use_cmvn,
            use_delta=feature_cfg.use_delta,
            use_delta_delta=feature_cfg.use_delta_delta,
            context_size=feature_cfg.context_size,
        )
        y = align_frame_labels_to_num_frames(label_dict[utt_id], x.shape[0])

        utt_ids.append(utt_id)
        x_list.append(x)
        y_list.append(y)
        iterator.set_postfix({"kept": len(x_list)})

    if not x_list:
        raise RuntimeError(f"No matched labeled samples in split={split}")
    return utt_ids, x_list, y_list


def tune_threshold_on_dev(
    model: DNNClassifier,
    path_cfg: PathConfig,
    frame_cfg: FrameConfig,
    feature_cfg: FeatureConfig,
    model_cfg: ModelConfig,
) -> Tuple[float, float, List[np.ndarray], List[np.ndarray]]:
    """Score dev utterances and return the dev-selected decoding threshold."""
    _, x_dev_list, y_dev_list = build_split_utterances(
        "dev",
        path_cfg.dev_label_path,
        path_cfg,
        frame_cfg,
        feature_cfg,
    )
    score_iterator = tqdm(
        x_dev_list,
        total=len(x_dev_list),
        desc="[dev] scoring",
        unit="utt",
    )
    scores_dev_list = [model.score_frames(x_dev) for x_dev in score_iterator]

    best_threshold, best_acc = sweep_best_threshold_by_acc(
        scores_by_utt=scores_dev_list,
        labels_by_utt=y_dev_list,
        threshold_min=model_cfg.threshold_min,
        threshold_max=model_cfg.threshold_max,
        threshold_step=model_cfg.threshold_step,
        smooth_kernel_size=model_cfg.smooth_kernel_size,
    )
    return float(best_threshold), float(best_acc), scores_dev_list, y_dev_list


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
    data_root = (
        Path(project_root)
        / "voice-activity-detection-sjtu-spring-2024"
        / "vad"
    )
    path_cfg = PathConfig(data_root=data_root)
    model = DNNClassifier()
    model_cfg = ModelConfig()
    frame_cfg = FrameConfig()
    fea_cfg = FeatureConfig()
    x_all, y_all = build_xy("train", path_cfg.train_label_path, path_cfg, frame_cfg, fea_cfg)
    model.fit(x_all, y_all)

    best_threshold, best_acc, scores_dev_list, y_dev_list = tune_threshold_on_dev(
        model,
        path_cfg,
        frame_cfg,
        fea_cfg,
        model_cfg,
    )
    scores_dev = np.concatenate(scores_dev_list, axis=0)
    y_dev = np.concatenate(y_dev_list, axis=0)

    res = {}
    res["acc"] = best_acc
    preds_dev = (scores_dev >= float(best_threshold)).astype(np.int64)
    res.update(evaluate_metrics(scores_dev, preds_dev, y_dev, project_root))
    mode_name = model.model_type
    if model.model_type == "dnn":
        mode_name = f"dnn-{model.model_cfg.dnn_variant}"
    roc_dir = model.output_dir / "images" / model.model_type
    save_roc_curve(scores_dev, y_dev, mode_name, roc_dir)

    res["best_threshold"] = float(best_threshold)
    return res


def run_test_pipeline(project_root: Path, output_path: Path) -> Dict[str, object]:
    """Train on train split, tune threshold on dev, and write test_label.txt."""
    # Data usage plan:
    # 1) Train with wavs/train + data/train_label.txt
    # 2) Tune the decoding threshold with wavs/dev + data/dev_label.txt
    # 3) Apply the trained model and dev-selected threshold to wavs/test
    # 4) Write `utt_id <space> start,end ...` per line to output_path
    data_root = (
        Path(project_root)
        / "voice-activity-detection-sjtu-spring-2024"
        / "vad"
    )
    frame_cfg = FrameConfig()
    model_cfg = ModelConfig()
    path_cfg = PathConfig(data_root=data_root)
    model = DNNClassifier()
    fea_cfg = FeatureConfig()
    x_all, y_all = build_xy("train", path_cfg.train_label_path, path_cfg, frame_cfg, fea_cfg)
    model.fit(x_all, y_all)

    test_threshold, dev_best_acc, _, _ = tune_threshold_on_dev(
        model,
        path_cfg,
        frame_cfg,
        fea_cfg,
        model_cfg,
    )

    wave_files = list_wav_files(path_cfg.wav_root / "test")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        iterator = tqdm(
            wave_files,
            total=len(wave_files),
            desc="[test] inference",
            unit="utt",
        )
        for wav_path in iterator:
            utt_id = wav_path.stem

            waveform = load_waveform(wav_path, frame_cfg.sample_rate)
            x_test = extract_spectral_features(
                waveform=waveform,
                sample_rate=frame_cfg.sample_rate,
                frame_size=frame_cfg.frame_size,
                frame_shift=frame_cfg.frame_shift,
                feature_type=fea_cfg.feature_type,
                feature_dim=fea_cfg.feature_dim,
                use_cmvn=fea_cfg.use_cmvn,
                use_delta=fea_cfg.use_delta,
                use_delta_delta=fea_cfg.use_delta_delta,
                context_size=fea_cfg.context_size,
            )

            score = model.score_frames(x_test)
            pred = (score >= test_threshold).astype(np.int64)
            pred = smooth_predictions(pred, kernel_size=model_cfg.smooth_kernel_size)
            seg_str = frame_prediction_to_label_line(
                pred,
                frame_cfg.frame_size,
                frame_cfg.frame_shift,
            )

            line = f"{utt_id} {seg_str}".rstrip()  # seg_str为空时只保留utt_id
            f.write(line + "\n")

    return {
        "output_path": str(output_path),
        "threshold": test_threshold,
        "dev_best_acc": dev_best_acc,
        "num_test_utts": len(wave_files),
    }



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
