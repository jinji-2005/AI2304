from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm
from config import FeatureConfig

@dataclass
class ThresholdParams:
    threshold: float = 0.5
    high_threshold: float = 0.6
    low_threshold: float = 0.4


class VADclassifier:
    """Threshold-based frame classifier scaffold."""

    def __init__(self, params: ThresholdParams):
        self.params = params
        self.threshold_candidates: np.ndarray | None = None
        self.threshold_acc_matrix: np.ndarray | None = None

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
        candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 25)))

        best_acc = -1.0
        best_high = float(candidates[-1])
        best_low = float(candidates[0])
        acc_matrix = np.full((len(candidates), len(candidates)), np.nan, dtype=np.float32)
        total_pairs = len(candidates) * (len(candidates) + 1) // 2
        with tqdm(total=total_pairs, desc="threshold search", unit="pair", leave=False) as pbar:
            for low_idx, low in enumerate(candidates):
                for high_idx in range(low_idx, len(candidates)):
                    high = float(candidates[high_idx])
                    pbar.update(1)
                    pred = self._decode_hysteresis(scores, high, float(low))
                    acc = float(np.mean(pred == y))
                    acc_matrix[low_idx, high_idx] = acc
                    if acc > best_acc:
                        best_acc = acc
                        best_high = high
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
        self.threshold_candidates = candidates.astype(np.float32)
        self.threshold_acc_matrix = acc_matrix

    def score_frames(self, features: np.ndarray) -> np.ndarray:
        """Return frame-level speech scores/probabilities in [0, 1].

        Data usage:
        - Input: per-utterance or concatenated features [T, D] / [N, D]
        - Output: continuous score array [T] / [N]
        - Used by:
          - AUC/EER calculation on dev (prefer continuous scores)
          - threshold decoding in predict_frames/postprocess
        """
        # Score from multiple handcrafted features:
        # - energy: higher tends to speech
        # - zcr: too high often noisy/unvoiced, give a mild negative weight
        # - short-time spectrum statistic: higher tends to speech-rich frames
        # - pitch(F0): periodic voiced cue
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        t = x.shape[0]

        energy = x[:, 0] if x.shape[1] > 0 else np.zeros(t, dtype=np.float32)
        zcr = x[:, 1] if x.shape[1] > 1 else np.zeros(t, dtype=np.float32)
        st_spectrum = x[:, 2] if x.shape[1] > 2 else np.zeros(t, dtype=np.float32)
        pitch = x[:, 3] if x.shape[1] > 3 else np.zeros(t, dtype=np.float32)

        fea_cfg = FeatureConfig()

        score = (
            fea_cfg.energy * energy
            - fea_cfg.zcr * zcr
            + fea_cfg.st_spectrum * st_spectrum
            + fea_cfg.pitch * pitch
        )
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
