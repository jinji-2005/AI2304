from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm


@dataclass
class ThresholdParams:
    threshold: float = 0.5
    high_threshold: float = 0.6
    low_threshold: float = 0.4


class VADclassifier:
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
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"features must be 2D, got shape={x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("特征和标签维度不对齐")
        if x.shape[1] < 2:
            raise ValueError("特征维度不足，至少需要 energy 和 zcr 两维")

        scores = self.score_frames(x)
        # Search (high, low) directly with the same hysteresis rule as predict_frames.
        # This avoids mismatch between fit-time threshold and inference-time decoding.
        candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 49)))

        best_acc = -1.0
        best_high = float(candidates[-1])
        best_low = float(candidates[0])
        total_pairs = len(candidates) * (len(candidates) + 1) // 2
        with tqdm(total=total_pairs, desc="threshold search", unit="pair", leave=False) as pbar:
            for low_idx, low in enumerate(candidates):
                for high in candidates[low_idx:]:
                    pbar.update(1)
                    pred = self._decode_hysteresis(scores, float(high), float(low))
                    acc = float(np.mean(pred == y))
                    if acc > best_acc:
                        best_acc = acc
                        best_high = float(high)
                        best_low = float(low)
                        pbar.set_postfix(
                            {
                                "best_acc": f"{best_acc:.4f}",
                                "high": f"{best_high:.4f}",
                                "low": f"{best_low:.4f}",
                            }
                        )

        self.params.high_threshold = best_high
        self.params.low_threshold = best_low
        self.params.threshold = (best_high + best_low) / 2.0

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
        score = features[:,0] *1 + (1-features[:,1]) * 0
        score = np.asarray(score)
        return score.astype(np.float32)
                
            
    @staticmethod
    def _decode_hysteresis(scores: np.ndarray, high: float, low: float) -> np.ndarray:
        """Decode frame-wise labels using dual-threshold hysteresis."""
        state = 0
        out = []
        for sco in scores:
            if sco > high and state == 0:
                state = 1
            elif sco < low and state == 1:
                state = 0
            out.append(state)
        return np.asarray(out, dtype=np.int64)

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
        score = self.score_frames(features=features)
        return self._decode_hysteresis(
            score,
            high=self.params.high_threshold,
            low=self.params.low_threshold,
        )


# a = np.array([0.3,0.3,0.4])
# t = 0.24
# print(np.array(a>t).astype(np.int64))

# a =[1,23]
# print(type(np.array(a)))

# print(np.arange(-1,-0.5,0.005))

# a = np.array([[1,1,1],[2,2,2]])
# print(a[0])
# print(a[1])
