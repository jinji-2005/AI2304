from typing import Dict
import math
import numpy as np
from dataset import load_waveform
from config import FrameConfig,PathConfig
"""
Task1 feature flow:
waveform -> framing -> windowing -> per-frame features -> stacked features.
"""

def framing(
    waveform: np.ndarray,
    sample_rate: int,
    frame_size: float,
    frame_shift: float,
) -> np.ndarray:
    """Convert waveform to shape [num_frames, frame_length].

    Data usage:
    - Input: one waveform from dataset.load_waveform
    - Output: 2D frame matrix, each row is one short-time frame
    - Used by: all Task1 feature extraction
    """
    # Keep frame parameters consistent with label conversion.
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    wave_len = waveform.shape[0] #原始音频的点数
    frame_length = int(round(sample_rate * frame_size)) #每一帧的采样点数
    hop_length = int(round(sample_rate * frame_shift)) #每两帧之间间隔的点数
    if wave_len<frame_length:
        num_frames = 1
    else:
        num_frames = 1+ math.ceil((wave_len - frame_length)/hop_length)
    pad_len = (num_frames-1)*hop_length + frame_length - wave_len # 填补最后一段不足 一帧长度的部分
    waveform = np.pad(waveform,(0,pad_len),'constant',constant_values=0.0)
    starts = np.arange(num_frames) * hop_length #用矩阵形式来提取 [num_frames, frame_length].
    offset = np.arange(frame_length).reshape(-1)
    indices = starts[:,None] + offset
    data = waveform[indices]
    return data

# d = np.array([1,3,4,5,6])
# a = np.array([0,1,2])
# b = np.array([0,1,2])
# b = b.reshape(-1)
# print(b,b.shape)
# b = b[:,None]
# print(b,b.shape)
# c = a+b
# print(c)
# print(d[c])
# # b = b[None,:]
# # print(b,b.shape)

def apply_window(frames: np.ndarray, window: str = "hamming") -> np.ndarray:
    """Apply analysis window to each frame.

    Data usage:
    - Input: framed waveform [num_frames, frame_length]
    - Output: windowed frames with the same shape
    - Used by: energy / zcr / spectral statistics
    """
    # Default recommendation: Hamming window for robust short-time analysis.
    # TODO: implement
    raise NotImplementedError


def extract_short_time_features(frames: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract short-time features needed in Task1.

    Suggested outputs:
    - "energy": short-time energy or log-energy
    - "zcr": zero-crossing rate
    - add your own features if needed
    """
    # Data usage:
    # - Input: windowed frames from `apply_window`
    # - Output: dict of 1D arrays, each with length `num_frames`
    # - Used by: threshold model scoring in task1/model.py
    #
    # Keep each feature aligned by frame index.
    # TODO: implement
    raise NotImplementedError


def stack_features(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack multiple feature streams into [num_frames, feat_dim].

    Data usage:
    - Input: feature dict from `extract_short_time_features`
    - Output: 2D matrix consumed by ThresholdVAD.fit/score_frames
    - Used by: training, dev evaluation, and test inference
    """
    # Recommendation:
    # - choose a fixed feature order
    # - ensure all streams have identical num_frames
    # TODO: implement
    raise NotImplementedError
