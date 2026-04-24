from pathlib import Path
from typing import Dict

import numpy as np


class DevResult(Dict[str, float]):
    """Container for development metrics."""


def run_dev_pipeline(project_root: Path) -> DevResult:
    """Run Task1 development experiment and return metric dict.

    Suggested metric keys: acc, auc, eer
    """
    # Data usage plan:
    # 1) Read dev wavs from:
    #    project_root/voice-activity-detection-sjtu-spring-2024/vad/wavs/dev
    # 2) Read dev labels from:
    #    project_root/voice-activity-detection-sjtu-spring-2024/vad/data/dev_label.txt
    # 3) For each utterance:
    #    wav -> features -> scores/pred -> aligned frame labels
    # 4) Concatenate all frame scores and labels over the ENTIRE dev split
    # 5) Compute acc/auc/eer and return as dict
    # TODO: implement
    raise NotImplementedError


def run_test_pipeline(project_root: Path, output_path: Path) -> None:
    """Run Task1 inference on test set and write test_label.txt."""
    # Data usage plan:
    # 1) Read test wavs from:
    #    project_root/voice-activity-detection-sjtu-spring-2024/vad/wavs/test
    # 2) No labels available on test
    # 3) For each utterance:
    #    wav -> features -> score/pred -> timestamp label string
    # 4) Write one line per utterance:
    #    utt_id <space> start,end start,end ...
    #    (if empty prediction, usually keep only utt_id)
    # TODO: implement
    raise NotImplementedError


def compute_acc(pred_binary: np.ndarray, label_binary: np.ndarray) -> float:
    """Compute frame-level accuracy.

    Data usage:
    - Input: concatenated frame-level binary predictions + labels on dev
    - Output: scalar frame accuracy
    - Used by: run_dev_pipeline metric report
    """
    # Ensure same shape and dtype before computing.
    # TODO: implement
    raise NotImplementedError
