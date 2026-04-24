import numpy as np


def smooth_predictions(binary_pred: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth frame-level binary predictions.

    Data usage:
    - Input: raw binary predictions [T]
    - Output: smoothed binary predictions [T]
    - Used by: cleaner endpoint boundaries before label export
    """
    # TODO: implement
    raise NotImplementedError


def frame_prediction_to_label_line(
    prediction: np.ndarray,
    frame_size: float,
    frame_shift: float,
    threshold: float = 0.5,
) -> str:
    """Convert frame prediction to 'start,end start,end ...' format.

    Data usage:
    - Input: one utterance frame prediction [T]
    - Output: timestamp string for one utterance line
    - Used by: test_label.txt generation in run_test_pipeline
    """
    # Keep formatting consistent with provided label files.
    # TODO: implement
    raise NotImplementedError
