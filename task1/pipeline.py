from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, Dict, Tuple
import librosa
import numpy as np
from tqdm.auto import tqdm

from config import FrameConfig, PathConfig
from dataset import (
    list_wav_files,
    read_label_from_file,
    load_waveform,
    align_frame_labels_to_num_frames,
)
from features import framing, apply_window, extract_short_time_features, stack_features
from model import ThresholdParams, VADclassifier
from postprocess import frame_prediction_to_label_line,smooth_predictions


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

def build_xy(split: str, label_path: Path , project_root: Path): # split = 'train' 'dev' 'test'
    data_root = (
        Path(project_root)
        / "voice-activity-detection-sjtu-spring-2024"
        / "vad"
    )
    path_cfg = PathConfig(data_root=data_root)
    frame_cfg = FrameConfig()
    wav_files = list_wav_files(path_cfg.wav_root / split)
    label_dict = read_label_from_file(
        label_path, 
        frame_size=frame_cfg.frame_size,
        frame_shift=frame_cfg.frame_shift,
    )

    x_list, y_list = [], []
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
        frames = framing(
            waveform,
            sample_rate=frame_cfg.sample_rate,
            frame_size=frame_cfg.frame_size,
            frame_shift=frame_cfg.frame_shift,
        )
        frames = apply_window(frames, window="hamming")

        feat_dict = extract_short_time_features(frames)
        x = stack_features(feat_dict)  # (T, D)

        y = align_frame_labels_to_num_frames(label_dict[utt_id], x.shape[0])  # (T,)

        x_list.append(x)
        y_list.append(y)
        iterator.set_postfix({"kept": len(x_list)})
    if not x_list:
        raise RuntimeError(f"No matched labeled samples in split={split}")

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return x_all, y_all


def run_dev_pipeline(project_root: Path) -> Dict:
    """Run Task1 development experiment and return metric dict.

    Suggested metric keys: acc, auc, eer
    """
    # Data usage plan:
    # 1) Read dev wavs from:
    #    project_root/voice-activity-detection-sjtu-spring-2024/vad/wavs/dev
    # 2) Read dev labels from:
    #    project_root/voice-activity-detection-sjtu-spring-2024/vad/data/dev_label.txt
    # 3) For each utterance:
    #    wav -> features -> scores/pred -> aligned frame labels
    # 4) Concatenate all frame scores and labels over the ENTIRE dev split
    # 5) Compute acc + official auc/eer and return as dict:
    #    auc, eer = compute_auc_eer(pred_scores, labels, project_root)
    # TODO: implement
    data_root = (
        Path(project_root)
        / "voice-activity-detection-sjtu-spring-2024"
        / "vad"
    )
    path_cfg = PathConfig(data_root=data_root)
    model = VADclassifier(ThresholdParams())

    x_all,y_all = build_xy('train',path_cfg.train_label_path,project_root)
    #x_all,y_all = build_xy('dev',path_cfg.dev_label_path,project_root)

    # print(np.mean(x_all))
    # print(np.min(x_all))
    # print(np.max(x_all))
    model.fit(x_all,y_all)
    
    #print(model.params.threshold)
    ThresholdParams.high_threshold = model.params.high_threshold
    ThresholdParams.low_threshold = model.params.low_threshold
    ThresholdParams.threshold = (model.params.high_threshold +model.params.low_threshold)/2
    print(model.params.high_threshold)
    print(model.params.low_threshold)
    
    # 开发集评估

    x_dev , y_dev = build_xy('dev',path_cfg.dev_label_path,project_root)
    scores_dev = model.score_frames(x_dev)
    preds_dev = model.predict_frames(x_dev)
    preds_dev = smooth_predictions(preds_dev,kernel_size=3)
    acc = compute_acc(preds_dev,y_dev)
    auc , eer = compute_auc_eer(scores_dev,y_dev,Path(project_root))
    
    out = {}
    out['acc'] = acc
    out['auc'] = auc
    out['eer'] = eer
    return out
    
    
    


def run_test_pipeline(project_root: Path, output_path: Path) -> None:
    """Run Task1 inference on test set and write test_label.txt."""
    # Data usage plan:
    # 1) Read test wavs from:
    #    project_root/voice-activity-detection-sjtu-spring-2024/vad/wavs/test
    # 2) No labels available on test
    # 3) For each utterance:
    #    wav -> features -> score/pred -> timestamp label string
    # 4) Write one line per utterance:
    #    utt_id <space> start,end start,end ...
    #    (if empty prediction, usually keep only utt_id)
    # TODO: implement
    data_root = (
    Path(project_root)
    / "voice-activity-detection-sjtu-spring-2024"
    / "vad")
    path_cfg = PathConfig(data_root=data_root)
    model = VADclassifier(ThresholdParams())
    frame_cfg = FrameConfig()
    # 开发集求得阈值
    x_dev,y_dev = build_xy('dev',path_cfg.dev_label_path,project_root=project_root)
    model.fit(x_dev,y_dev)
    
    # 测试测试集
    wave_files = list_wav_files(path_cfg.wav_root / 'test')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for wav_path in wave_files:
            utt_id = wav_path.stem

            waveform = load_waveform(wav_path, frame_cfg.sample_rate)
            frames = framing(waveform, frame_cfg.sample_rate, frame_cfg.frame_size, frame_cfg.frame_shift)
            frames = apply_window(frames, window="hamming")
            feat_dict = extract_short_time_features(frames)
            x_test = stack_features(feat_dict)

            pred = model.predict_frames(x_test)  # shape (T,), 0/1 from dual-threshold decode
            seg_str = frame_prediction_to_label_line(
                pred,
                frame_cfg.frame_size,
                frame_cfg.frame_shift,
            )

            line = f"{utt_id} {seg_str}".rstrip()  # seg_str为空时只保留utt_id
            f.write(line + "\n")
        
    

        
    
    


def compute_acc(pred_binary: np.ndarray, label_binary: np.ndarray) -> float:
    """Compute frame-level accuracy.

    Data usage:
    - Input: concatenated frame-level binary predictions + labels on dev
    - Output: scalar frame accuracy
    - Used by: run_dev_pipeline metric report
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
