import numpy as np


def extract_spectral_features(
    waveform: np.ndarray,
    sample_rate: int,
    frame_size: float,
    frame_shift: float,
    feature_type: str,
    feature_dim: int,
) -> np.ndarray:
    """Return spectral features with shape [num_frames, feature_dim].

    feature_type can be: mfcc, fbank, plp.
    """
    # Data usage:
    # - Input: one utterance waveform
    # - Output: feature matrix [num_frames, feature_dim]
    # - Used by:
    #   - model.fit on train/dev
    #   - model.score_frames on dev/test
    #
    # Important:
    # - Keep frame_size/frame_shift identical to label conversion.
    # - Feature matrix frame count is the reference length for label alignment.
    # TODO: implement
    raise NotImplementedError
