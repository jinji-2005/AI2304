from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import librosa
import numpy as np

"""
Task1 dataset conventions (recommended):
- dataset root: <project_root>/voice-activity-detection-sjtu-spring-2024/vad
- wav dirs: wavs/train, wavs/dev, wavs/test
- label files:
  - data/train_label.txt
  - data/dev_label.txt
- one label line format: utt_id start1,end1 start2,end2 ...

To keep imports simple, label parsing helpers are placed in this file.
"""


def parse_vad_label(
    line: str,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
) -> List[int]:
    """Parse timestamp string to frame-wise labels.

    Example input line:
    "0.14,1.79 1.82,2.88 3.30,3.97"
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    frames: List[int] = []
    frame_n = 0
    for time_pairs in line.split():
        start, end = map(float, time_pairs.split(","))
        if end <= start:
            raise ValueError(f"Invalid time segment: {start},{end}")
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


def read_label_from_file(
    path: Path,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
) -> Dict[str, List[int]]:
    """Read one label file and convert each utterance to frame labels."""
    data: Dict[str, List[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            if len(sps) != 2:
                raise ValueError(f'Invalid label format at {path}:{linenum}: "{line.strip()}"')
            utt_id, timestamp_line = sps
            if utt_id in data:
                raise RuntimeError(f"{utt_id} is duplicated ({path}:{linenum})")
            data[utt_id] = parse_vad_label(
                timestamp_line,
                frame_size=frame_size,
                frame_shift=frame_shift,
            )
    return data


def list_wav_files(split_wav_dir: Path | str) -> List[Path]:
    """Return sorted wav file list for one split.

    Data usage:
    - Input: one split directory, e.g. wavs/dev
    - Output: ordered list of wav file paths, used by pipeline for iteration
    - Note: keep deterministic ordering (e.g. lexicographic) for reproducibility
    """
    split_wav_dir = Path(split_wav_dir)
    if not split_wav_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_wav_dir}")
    if not split_wav_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {split_wav_dir}")
    wav_files = [
        p for p in split_wav_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav" # p.suffix 取出后缀
    ]
    return sorted(wav_files, key=lambda p: p.name)


def load_waveform(wav_path: Path, sample_rate: int) -> np.ndarray:
    """Load one wav as mono float waveform.

    Data usage:
    - Input: one wav path from split list
    - Output: waveform array with shape [num_samples]
    - Used by: framing + short-time feature extraction
    """
    # TODO: implement
    # sr=sample_rate: 自动重采样到目标采样率
    # mono=True: 自动转单通道
    y ,_ = librosa.load(wav_path,sr= sample_rate,mono=True)
    y = np.array(y,dtype=np.float32).reshape(-1)
    return y


def align_frame_labels_to_num_frames(labels: List[int], num_frames: int) -> np.ndarray:
    """Pad/truncate labels to match feature frame count.

    Data usage:
    - Input: frame labels from timestamps + feature frame count
    - Output: fixed-length label vector with shape [num_frames]
    - Used by: metric computation and model fitting
    """
    # TODO: implement
    labels_arr = np.array(labels).reshape(-1)
    len_arr = len(labels_arr)
    if(len_arr == num_frames):
        return labels_arr
    elif(len_arr > num_frames):
        labels_arr = labels_arr[:num_frames]
    else:
        pad_len = num_frames - len_arr
        labels_arr = np.pad(labels_arr,(0,pad_len),'constant',constant_values=0)
    return labels_arr
        
    


def split_wav_and_label(
    wav_files: List[Path],
    label_dict: Dict[str, List[int]],
) -> List[Tuple[Path, List[int]]]:
    """Return matched pairs (wav_path, frame_labels) for supervised splits.

    Data usage:
    - Input: wav file list of one split + label dict (frame labels)
    - Output: list[(wav_path, frame_labels)] for only matched utt_id
    - Used by: train/dev loops
    """
    # TODO: implement
    pairs = []
    for path in wav_files:
        utt_id = path.stem # stem 去掉后缀
        if utt_id in label_dict:
            label = label_dict[utt_id]
            pairs.append((path,label))
    return pairs
        

        
