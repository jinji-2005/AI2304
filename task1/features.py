from typing import Dict

import numpy as np

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
    # TODO: implement
    raise NotImplementedError


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
