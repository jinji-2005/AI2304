from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # sigmoid

import numpy as np

from config import ModelConfig


class DNNClassifier:
    """Statistical classifier scaffold for Task2."""

    def __init__(self, model_type: str | None = None):
        if model_type is None:
            model_type = ModelConfig().model_type
        if model_type not in ("gmm", "dnn"):
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type

        # 特征标准化
        self.scaler = StandardScaler()

        # GMM 两类模型（语音 / 非语音）
        self.gmm_speech: GaussianMixture | None = None
        self.gmm_noise: GaussianMixture | None = None

        # DNN 分类器
        self.dnn: MLPClassifier | None = None

        # 先验与状态
        self.prior_speech: float = 0.5
        self.is_fitted: bool = False


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
        model_type = self.model_type
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        
        if x.shape[0] != y.shape[0]:
            raise ValueError("维度不对齐")
        
        self.scaler = StandardScaler().fit(x)
        x_std = self.scaler.transform(x)
        if model_type == "gmm":
            self.prior_speech = float(y.mean())  # 先验概率
            x_speech = x_std[y == 1]
            x_non = x_std[y == 0]
            if x_speech.size == 0 or x_non.size == 0:
                raise ValueError("GMM 训练需要同时包含语音和非语音样本")
            
            self.gmm_speech = GaussianMixture(n_components=4, covariance_type="diag").fit(x_speech)
            self.gmm_noise = GaussianMixture().fit(x_non)
            self.is_fitted = True
        
        elif model_type == "dnn":
            self.dnn = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=100,
                early_stopping=True,
                random_state=42,
            )
            self.dnn.fit(x_std, y)
            self.is_fitted = True



    def score_frames(self, features: np.ndarray) -> np.ndarray:
        """Return frame-level speech scores/probabilities.

        Data usage:
        - Input: one utterance feature matrix [T, D]
        - Output: frame scores [T] in continuous range (preferably [0,1])
        - Used by: AUC/EER and threshold decoding
        """
        # TODO: implement
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before score_frames().")

        model_type = self.model_type
        x = np.asarray(features, dtype=np.float32)
        x_std = self.scaler.transform(x)
        
        if model_type == "gmm":
            ll_s = self.gmm_speech.score_samples(x_std)
            ll_n = self.gmm_noise.score_samples(x_std)
            llr = ll_s - ll_n
            score = expit(llr)
        elif model_type == "dnn":
            score = self.dnn.predict_proba(x_std)[:, 1]
        
        return np.asarray(score, dtype=np.float32)

    def predict_frames(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary frame prediction (0/1).

        Data usage:
        - Input: feature matrix [T, D] + decoding threshold
        - Output: frame decisions [T]
        - Used by: accuracy and timestamp label generation
        """
        # Usually call score_frames first, then apply threshold.
        # TODO: implement
        score = self.score_frames(features)
        labels = (score > threshold).astype(np.int64)
        return labels


    
# x = np.array([1,2,3]).reshape(3,1)
# scaler = StandardScaler().fit(x)
# print(scaler.transform(x))
