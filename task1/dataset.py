from __future__ import annotations

from pathlib import Path
from typing import Dict, List
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

def prediction_to_vad_label(
    prediction,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
    threshold: float = 0.5,
):
    """Convert model prediction to VAD labels.

    Args:
        prediction (List[float]): predicted speech activity of each **frame** in one sample
            e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
        threshold (float): prediction values that are higher than `threshold` are set to 1,
                            and those lower than or equal to `threshold` are set to 0
    Returns:
        vad_label (str): converted VAD label
            e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"

    NOTE: Each frame is converted to the timestamp according to its center time point.
    Thus the converted labels may not exactly coincide with the original VAD label, depending
    on the specified `frame_size` and `frame_shift`.
    See the following exmaple for more detailed explanation.

    Examples:
        >>> label = parse_vad_label("0.31,0.52 0.75,0.92")
        >>> prediction_to_vad_label(label)
        '0.31,0.53 0.75,0.92'
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_frames = []
    prev_state = False
    start, end = 0, 0
    end_prediction = len(prediction) - 1
    for i, pred in enumerate(prediction):
        state = pred > threshold
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        prev_state = state
    return " ".join(speech_frames)


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

print(Path(__file__).resolve().parent)
print(Path(__file__).parent)