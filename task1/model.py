from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdParams:
    threshold: float = 0.5
    high_threshold: float = 0.6
    low_threshold: float = 0.4


class ThresholdVAD:
    """Threshold-based frame classifier scaffold."""

    def __init__(self, params: ThresholdParams):
        self.params = params

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Optional calibration stage on development data.

        Data usage:
        - Input:
          - features: [N, D] stacked frame features from train/dev
          - labels: [N] frame labels (0/1), aligned with features
        - Output: update internal thresholds/weights in `self.params`
        - Used by: run_dev_pipeline (optional), then reused in test
        """
        # If you use pure manual thresholding, this can be a no-op.
        # TODO: implement (or keep as no-op if you use manual thresholds)
        raise NotImplementedError

    def score_frames(self, features: np.ndarray) -> np.ndarray:
        """Return frame-level speech scores/probabilities in [0, 1].

        Data usage:
        - Input: per-utterance or concatenated features [T, D] / [N, D]
        - Output: continuous score array [T] / [N]
        - Used by:
          - AUC/EER calculation on dev (prefer continuous scores)
          - threshold decoding in predict_frames/postprocess
        """
        # Score can come from simple linear combination of handcrafted features.
        # TODO: implement
        raise NotImplementedError

    def predict_frames(self, features: np.ndarray) -> np.ndarray:
        """Return binary frame prediction (0/1).

        Data usage:
        - Input: features [T, D] of one utterance
        - Output: binary frame decision [T]
        - Used by:
          - frame-level accuracy
          - frame-to-segment conversion for test_label.txt
        """
        # Usually: scores = score_frames(features) -> apply threshold rule.
        # TODO: implement
        raise NotImplementedError
