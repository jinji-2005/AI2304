from dataclasses import dataclass

import numpy as np


@dataclass
class StatisticalModelParams:
    model_type: str = "gmm"


class StatisticalVAD:
    """Statistical classifier scaffold for Task2."""

    def __init__(self, params: StatisticalModelParams):
        self.params = params

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train model with frame-level supervision.

        Data usage:
        - Input:
          - features: [N, D] concatenated frame features
          - labels: [N] frame labels (0/1), aligned to features
        - Output: trained model state in this object
        - Used by: run_dev_pipeline before evaluation and test inference
        """
        # `features/labels` usually come from train split (or train+dev strategy).
        # TODO: implement
        raise NotImplementedError

    def score_frames(self, features: np.ndarray) -> np.ndarray:
        """Return frame-level speech scores/probabilities.

        Data usage:
        - Input: one utterance feature matrix [T, D]
        - Output: frame scores [T] in continuous range (preferably [0,1])
        - Used by: AUC/EER and threshold decoding
        """
        # TODO: implement
        raise NotImplementedError

    def predict_frames(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary frame prediction (0/1).

        Data usage:
        - Input: feature matrix [T, D] + decoding threshold
        - Output: frame decisions [T]
        - Used by: accuracy and timestamp label generation
        """
        # Usually call score_frames first, then apply threshold.
        # TODO: implement
        raise NotImplementedError
