from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # sigmoid

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import sigmoid_focal_loss
from tqdm.auto import tqdm

from config import ModelConfig


class DNNClassifier:
    """Statistical classifier scaffold for Task2."""

    def __init__(self, model_type: str | None = None):
        if model_type is None:
            model_type = ModelConfig().model_type
        if model_type not in ("gmm", "dnn"):
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        self.epoch = ModelConfig().train_epoch
        self.batch_size = ModelConfig().batch_size
        # 特征标准化
        self.scaler = StandardScaler()

        # GMM 两类模型（语音 / 非语音）
        self.gmm_speech: GaussianMixture | None = None
        self.gmm_noise: GaussianMixture | None = None

        # DNN 分类器（PyTorch）
        self.dnn: nn.Module | None = None
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

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
            input_dim = x_std.shape[1]
            self.dnn = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.dnn.to(self.device) # 网络参数需要 to device

            x_tensor = torch.from_numpy(x_std).float() # 转换成 torch 向量
            y_tensor = torch.from_numpy(y.astype(np.float32)).float().unsqueeze(1) # unsqueeze 把单维度向量增加一个维度
            dataset = TensorDataset(x_tensor, y_tensor)

            epochs = self.epoch
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

            # 使用 BCE criterion
            pos_count = float(max(1, np.sum(y == 1)))
            neg_count = float(max(1, np.sum(y == 0)))
            pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # Adam 优化器
            optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)

            epoch_bar = tqdm(range(1, epochs + 1), desc=f"[fit:{self.model_type}] epochs", unit="epoch")
            for epoch in epoch_bar:
                self.dnn.train()
                running_loss = 0.0
                seen = 0

                batch_bar = tqdm(
                    train_loader,
                    desc=f"[fit:{self.model_type}] epoch {epoch}/{epochs}",
                    unit="batch",
                    leave=False,
                )
                for xb, yb in batch_bar:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    logits = self.dnn(xb)
                    loss = sigmoid_focal_loss(
                        logits,     # [B, 1]
                        yb,         # [B, 1], float
                        alpha=0.25, # 对正类(语音)的权重
                        gamma=2.0,
                        reduction="mean",
                    )
                    
                    loss.backward()
                    optimizer.step()

                    bs = xb.shape[0]
                    running_loss += float(loss.item()) * bs
                    seen += bs
                    batch_bar.set_postfix({"loss": f"{loss.item():.5f}"})

                epoch_loss = running_loss / max(seen, 1)
                epoch_bar.set_postfix({"loss": f"{epoch_loss:.5f}", "device": str(self.device)})

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
            if self.gmm_speech is None or self.gmm_noise is None:
                raise RuntimeError("GMM models are not initialized. Please call fit() first.")
            ll_s = self.gmm_speech.score_samples(x_std)
            ll_n = self.gmm_noise.score_samples(x_std)
            llr = ll_s - ll_n
            score = expit(llr)
        elif model_type == "dnn":

            self.dnn.eval()
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_std).float().to(self.device)
                logits = self.dnn(x_tensor).squeeze(1)
                score = torch.sigmoid(logits).cpu().numpy()
        
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
