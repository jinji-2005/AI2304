import numpy as np


def smooth_predictions(binary_pred: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth frame-level binary predictions.

    Data usage:
    - Input: raw frame decisions [T] from threshold classifier
    - Output: denoised frame decisions [T]
    - Used by: both dev reporting and final test label generation
    """
    # Typical operations: median filter / remove too-short speech islands.
    pred = np.asarray(binary_pred).reshape(-1).astype(np.int64)
    if pred.size == 0 or kernel_size <= 1:
        return pred
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    padded = np.pad(pred, (half, half), mode="edge")
    out = np.empty_like(pred)
    for i in range(pred.size):
        out[i] = int(np.median(padded[i : i + kernel_size]))
    return out



def frame_prediction_to_label_line(
    prediction: np.ndarray,
    frame_size: float,
    frame_shift: float,
    threshold: float = 0.5,
) -> str:
    """Convert frame prediction to 'start,end start,end ...' format.

    Data usage:
    - Input: frame-level prediction for ONE utterance
    - Output: timestamp string for one output line in test_label.txt
    - Used by: run_test_pipeline when writing submission file
    """
    # Output format example:
    # "0.14,1.79 1.82,2.88"
    # If no speech is detected, return empty string.
    pred = np.asarray(prediction).reshape(-1)
    if pred.size == 0:
        return ""

    # IMPORTANT:
    # - If prediction is already binary (e.g. model.predict_frames from dual-threshold),
    #   do NOT apply threshold again.
    # - Only threshold when input is continuous score.
    unique_vals = np.unique(pred)
    if np.all(np.isin(unique_vals, [0, 1])):
        binary = pred.astype(np.int64)
    else:
        binary = (pred > threshold).astype(np.int64)

    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_segments = []
    start = 0
    prev_state = 0
    end_prediction = len(binary) - 1
    for i, state in enumerate(binary):
        if prev_state == 0 and state == 1:
            start = i
        elif prev_state == 1 and state == 0:
            end = i
            speech_segments.append(
                f"{frame2time(start):.2f},{frame2time(end):.2f}"
            )
        elif i == end_prediction and state == 1:
            end = i
            speech_segments.append(
                f"{frame2time(start):.2f},{frame2time(end):.2f}"
            )
        prev_state = int(state)

    return " ".join(speech_segments)

# a = np.array([-1,0,10,1])
# mean = np.mean(a)
# std = np.std(a)
# a = (a-mean)/std
# print(a)

import torch
print(torch.cuda.is_available())

