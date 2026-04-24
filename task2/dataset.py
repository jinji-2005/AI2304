from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

"""
Task2 dataset conventions are the same as Task1:
- wavs: train/dev/test
- labels: train_label.txt and dev_label.txt
Task2 mainly differs in feature/model, not in I/O format.

To keep imports simple, label parsing helpers are placed in this file.
"""


def parse_vad_label(
    line: str,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
) -> List[int]:
    """Parse timestamp string to frame-wise labels."""
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
    - Input: split wav dir (train/dev/test)
    - Output: ordered wav path list for deterministic loops
    """
    split_wav_dir = Path(split_wav_dir)
    if not split_wav_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_wav_dir}")
    if not split_wav_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {split_wav_dir}")

    wav_files = [
        p for p in split_wav_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"
    ]
    return sorted(wav_files, key=lambda p: p.name)


def load_waveform(wav_path: Path, sample_rate: int) -> np.ndarray:
    """Load one wav as mono float waveform.

    Data usage:
    - Input: single wav file
    - Output: waveform used by spectral feature extractor
    """
    # TODO: implement
    raise NotImplementedError


def align_frame_labels_to_num_frames(labels: List[int], num_frames: int) -> np.ndarray:
    """Pad/truncate labels to match feature frame count.

    Data usage:
    - Input: frame labels + actual feature frame count
    - Output: fixed-length frame labels aligned to feature matrix
    """
    # TODO: implement
    raise NotImplementedError


def split_wav_and_label(
    wav_files: List[Path],
    label_dict: Dict[str, List[int]],
) -> List[Tuple[Path, List[int]]]:
    """Return matched pairs (wav_path, frame_labels) for supervised splits.

    Data usage:
    - Input: wav list + label dict (frame labels)
    - Output: matched training/evaluation pairs by utt_id
    """
    # TODO: implement
    raise NotImplementedError
