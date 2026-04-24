from pathlib import Path
from typing import Dict

import numpy as np


class DevResult(Dict[str, float]):
    """Container for development metrics."""


def run_dev_pipeline(project_root: Path) -> DevResult:
    """Run Task2 development experiment and return metric dict.

    Suggested metric keys: acc, auc, eer
    """
    # Data usage plan:
    # 1) Train side (recommended):
    #    wavs/train + data/train_label.txt
    # 2) Dev side:
    #    wavs/dev + data/dev_label.txt
    # 3) Build frame-level train features/labels -> fit model
    # 4) On dev, gather utterance scores + aligned labels
    # 5) Concatenate all dev frames and compute acc/auc/eer
    # TODO: implement
    raise NotImplementedError


def run_test_pipeline(project_root: Path, output_path: Path) -> None:
    """Run Task2 inference on test set and write test_label.txt."""
    # Data usage plan:
    # 1) Read test wavs from wavs/test
    # 2) Apply trained model to obtain frame predictions
    # 3) Convert each utterance to timestamp segments
    # 4) Write `utt_id <space> start,end ...` per line to output_path
    # TODO: implement
    raise NotImplementedError


def compute_acc(pred_binary: np.ndarray, label_binary: np.ndarray) -> float:
    """Compute frame-level accuracy.

    Data usage:
    - Input: concatenated frame-level predictions and ground truth on dev
    - Output: scalar accuracy for report table
    """
    # TODO: implement
    raise NotImplementedError
