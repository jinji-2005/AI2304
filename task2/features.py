import numpy as np
import librosa


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
    y = np.asarray(waveform,dtype=np.float32).reshape(-1)
    win_length = int(round(frame_size * sample_rate))
    hop_length = int(round(frame_shift * sample_rate))
    
    
    if(feature_type == 'fbank'):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_fft=win_length,
            hop_length=hop_length,
            win_length=win_length,
            window="hamming",
            center=False,
            n_mels=feature_dim,
            power=2.0,
        )
        feat = np.log(np.maximum(mel, 1e-10))
    
    elif(feature_type == 'mfcc'):
        feat = librosa.feature.mfcc(
            y=y,
            sr=sample_rate,
            n_mfcc=feature_dim,
            n_fft=win_length,
            hop_length=hop_length,
            win_length=win_length,
            window="hamming",
            center=False,
        )

    elif(feature_type == 'plp'):
        raise NotImplementedError("PLP is not implemented yet in this baseline")
    else:
        raise ValueError(f"{feature_type} not in prepared tyeps")

    # librosa returns [D, T], while pipeline expects [T, D].
    return np.asarray(feat.T, dtype=np.float32)
