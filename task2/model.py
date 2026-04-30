from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import expit  # sigmoid

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import sigmoid_focal_loss
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from config import ModelConfig


class DNNClassifier:
    """Statistical classifier scaffold for Task2."""

    def __init__(self, model_type: str | None = None):
        self.model_cfg = ModelConfig()
        if model_type is None:
            model_type = self.model_cfg.model_type
        if model_type not in ("gmm", "dnn"):
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        self.epoch = self.model_cfg.train_epoch
        self.batch_size = self.model_cfg.batch_size
        self.learning_rate = self.model_cfg.learning_rate
        self.weight_decay = self.model_cfg.weight_decay
        # 特征标准化
        self.scaler = StandardScaler()

        # GMM 两类模型（语音 / 非语音）
        self.gmm_speech: GaussianMixture | None = None
        self.gmm_noise: GaussianMixture | None = None

        # DNN 分类器（PyTorch）
        self.dnn: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 先验与状态
        self.prior_speech: float = 0.5
        self.is_fitted: bool = False
        default_output = "/home/lihaochen/Work_Dir/AI2304/experiment_logs"
        self.output_dir = Path(os.getenv("TASK2_OUTPUT_DIR", default_output)).resolve()
        self._gmm_plot_dir = self.output_dir / "images" / "gmm"
        self._dnn_plot_dir = self.output_dir / "images" / "dnn"

    def _build_dnn(self, input_dim: int) -> nn.Module:
        variant = self.model_cfg.dnn_variant.strip().lower()
        if variant == "bn_dropout":
            return nn.Sequential(
                nn.Linear(input_dim, self.model_cfg.hidden_dim_1),
                nn.BatchNorm1d(self.model_cfg.hidden_dim_1),
                nn.ReLU(),
                nn.Dropout(self.model_cfg.dropout),
                nn.Linear(self.model_cfg.hidden_dim_1, self.model_cfg.hidden_dim_2),
                nn.BatchNorm1d(self.model_cfg.hidden_dim_2),
                nn.ReLU(),
                nn.Dropout(self.model_cfg.dropout),
                nn.Linear(self.model_cfg.hidden_dim_2, 1),
            )
        if variant == "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, self.model_cfg.hidden_dim_1),
                nn.ReLU(),
                nn.Linear(self.model_cfg.hidden_dim_1, self.model_cfg.hidden_dim_2),
                nn.ReLU(),
                nn.Linear(self.model_cfg.hidden_dim_2, 1),
            )
        raise ValueError(f"Unsupported dnn_variant: {self.model_cfg.dnn_variant}")

    def _save_gmm_speech_plot(self, x_std: np.ndarray, y: np.ndarray) -> None:
        """Save one visualization figure for both-class GMM components and sample points."""
        if self.gmm_speech is None or self.gmm_noise is None:
            return
        self._gmm_plot_dir.mkdir(parents=True, exist_ok=True)

        # Project high-dimensional features to 2D for visualization.
        pca = PCA(n_components=2, random_state=0)
        x2d = pca.fit_transform(x_std)
        y = np.asarray(y).reshape(-1).astype(np.int64)
        sample_cap = min(5000, x2d.shape[0])
        if x2d.shape[0] > sample_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(x2d.shape[0], size=sample_cap, replace=False)
            x2d = x2d[idx]
            y = y[idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        # Color palette:
        # - Non-speech points: muted blue
        # - Speech points: muted coral
        # - Non-speech ellipses: deep teal
        # - Speech ellipses: warm orange
        point_noise_color = "#6FA8DC"
        point_speech_color = "#F4A582"
        ellipse_noise_color = "#1B7F79"
        ellipse_speech_color = "#C96A00"
        speech_mask = y == 1
        noise_mask = y == 0
        ax.scatter(
            x2d[noise_mask, 0],
            x2d[noise_mask, 1],
            s=5,
            alpha=0.22,
            c=point_noise_color,
            label="non-speech samples",
        )
        ax.scatter(
            x2d[speech_mask, 0],
            x2d[speech_mask, 1],
            s=5,
            alpha=0.22,
            c=point_speech_color,
            label="speech samples",
        )

        pca_comp = pca.components_

        def _draw_gmm_components(
            gmm: GaussianMixture,
            edge_color: str,
            center_label: str,
            ellipse_label_once: str,
        ) -> None:
            means = gmm.means_
            covs = gmm.covariances_
            for k in range(means.shape[0]):
                mean2d = pca.transform(means[k].reshape(1, -1))[0]
                cov_full = np.diag(covs[k])
                cov2d = pca_comp @ cov_full @ pca_comp.T
                eigvals, eigvecs = np.linalg.eigh(cov2d)
                eigvals = np.clip(eigvals, 1e-8, None)
                order = eigvals.argsort()[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                width, height = 2.0 * 2.0 * np.sqrt(eigvals)

                ellipse = Ellipse(
                    xy=mean2d,
                    width=width,
                    height=height,
                    angle=angle,
                    fill=False,
                    linewidth=2.2,
                    edgecolor=edge_color,
                    alpha=0.95,
                    label=ellipse_label_once if k == 0 else None,
                )
                ax.add_patch(ellipse)
                ax.scatter(
                    mean2d[0],
                    mean2d[1],
                    s=36,
                    c=edge_color,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=3,
                    label=center_label if k == 0 else None,
                )

        _draw_gmm_components(
            self.gmm_noise,
            edge_color=ellipse_noise_color,
            center_label="non-speech centers",
            ellipse_label_once="non-speech Gaussian kernels",
        )
        _draw_gmm_components(
            self.gmm_speech,
            edge_color=ellipse_speech_color,
            center_label="speech centers",
            ellipse_label_once="speech Gaussian kernels",
        )

        ax.set_title("Speech/Non-Speech Samples with Class-wise GMM Kernels")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(self._gmm_plot_dir / "gmm_speech_components.png", dpi=200)
        plt.close(fig)

    def _save_dnn_loss_plot(self, epoch_losses: list[float]) -> None:
        """Save one training loss curve for DNN."""
        if len(epoch_losses) == 0:
            return
        self._dnn_plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = np.arange(1, len(epoch_losses) + 1)
        ax.plot(epochs, epoch_losses, marker="o", linewidth=1.8, markersize=3)
        ax.set_title("DNN Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._dnn_plot_dir / "dnn_train_loss.png", dpi=200)
        plt.close(fig)


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
        print("Training device:", self.device)
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

            stage_bar = tqdm(total=2, desc="[fit:gmm] stages", unit="stage")
            # Limit BLAS/OpenMP thread pools to avoid OpenBLAS thread-region overflow.
            with threadpool_limits(limits=1):
                stage_bar.set_postfix({"current": "speech_gmm"})
                self.gmm_speech = GaussianMixture(
                    n_components=4,
                    covariance_type="diag",
                    reg_covar=1e-4,
                    n_init=3,
                    max_iter=200,
                    random_state=0,
                ).fit(x_speech)
                stage_bar.update(1)

                stage_bar.set_postfix({"current": "noise_gmm"})
                self.gmm_noise = GaussianMixture(
                    n_components=4,
                    covariance_type="diag",
                    reg_covar=1e-4,
                    n_init=3,
                    max_iter=200,
                    random_state=0,
                ).fit(x_non)
                stage_bar.update(1)
            stage_bar.close()
            self._save_gmm_speech_plot(x_std, y)
            self.is_fitted = True
        
        elif model_type == "dnn":
            input_dim = x_std.shape[1]
            self.dnn = self._build_dnn(input_dim)
            self.dnn.to(self.device) # 网络参数需要 to device

            x_tensor = torch.from_numpy(x_std).float() # 转换成 torch 向量
            y_tensor = torch.from_numpy(y.astype(np.float32)).float().unsqueeze(1) # unsqueeze 把单维度向量增加一个维度
            dataset = TensorDataset(x_tensor, y_tensor)

            epochs = self.epoch
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

            pos_count = float(max(1, np.sum(y == 1)))
            neg_count = float(max(1, np.sum(y == 0)))
            pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=self.device)

            optimizer = torch.optim.Adam(
                self.dnn.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            epoch_losses: list[float] = []

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
                    if self.model_cfg.use_focal_loss:
                        loss = sigmoid_focal_loss(
                            logits,
                            yb,
                            alpha=self.model_cfg.focal_alpha,
                            gamma=self.model_cfg.focal_gamma,
                            reduction="mean",
                        )
                    else:
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits,
                            yb,
                            pos_weight=pos_weight,
                        )
                    
                    loss.backward()
                    optimizer.step()

                    bs = xb.shape[0]
                    running_loss += float(loss.item()) * bs
                    seen += bs
                    batch_bar.set_postfix({"loss": f"{loss.item():.5f}"})

                epoch_loss = running_loss / max(seen, 1)
                epoch_losses.append(float(epoch_loss))
                current_lr = optimizer.param_groups[0]["lr"]
                epoch_bar.set_postfix(
                    {
                        "loss": f"{epoch_loss:.5f}",
                        "lr": f"{current_lr:.2e}",
                        "device": str(self.device),
                    }
                )
                scheduler.step()

            self._save_dnn_loss_plot(epoch_losses)
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
            # Keep inference consistent with training thread limits.
            with threadpool_limits(limits=1):
                ll_s = self.gmm_speech.score_samples(x_std)
                ll_n = self.gmm_noise.score_samples(x_std)
            prior = float(np.clip(self.prior_speech, 1e-6, 1 - 1e-6))
            llr = ll_s - ll_n + np.log(prior / (1.0 - prior))
            score = expit(llr)
        elif model_type == "dnn":
            if self.dnn is None:
                raise RuntimeError("DNN model is not initialized. Please call fit() first.")

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
