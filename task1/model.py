from dataclasses import dataclass

import numpy as np


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
        if(features.shape[0]!=labels.shape[0]):
            raise ValueError("特征和标签维度不对齐")
        if(features.shape[1]!=2):
            raise ValueError("特征不是2维")
        
        score = features[:,0] *1 + (1-features[:,1]) * 0
        score = np.asarray(score)
        thres = [t for t in np.arange(-0.6,-0.4,0.005)]
        best_thre = 0.05
        best_acc = 0
        for thre in thres:
            pred = np.array(score > thre).astype(dtype=int)
            acc = np.mean(pred == labels)
            if acc > best_acc:
                best_acc = acc
                best_thre = thre
        
        margin = 0.05 * (np.ptp(score)) + 1e-6
        self.params.threshold = best_thre
        self.params.high_threshold = best_thre + 0.1* margin
        self.params.low_threshold = best_thre - 0.1* margin

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
        score = features[:,0] *0.8 + (1-features[:,1]) * 0.2
        score = np.asarray(score)
        return score.astype(np.float32)
                
            

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
        score = np.asarray(score)
        labels = []
        state = 0
        for sco in score:
            if sco > self.params.high_threshold and state == 0:
                state = 1
            elif sco < self.params.low_threshold and state == 1:
                state  = 0
            labels.append(state)
        return np.array(labels)


# a = np.array([0.3,0.3,0.4])
# t = 0.24
# print(np.array(a>t).astype(np.int64))

# a =[1,23]
# print(type(np.array(a)))

# print(np.arange(-1,-0.5,0.005))

# a = np.array([[1,1,1],[2,2,2]])
# print(a[0])
# print(a[1])