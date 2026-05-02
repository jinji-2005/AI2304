import numpy as np
import librosa


def _apply_cmvn(feat_dt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-utterance cepstral mean-variance normalization on [D, T]."""
    mean = np.mean(feat_dt, axis=1, keepdims=True)
    std = np.std(feat_dt, axis=1, keepdims=True)
    return (feat_dt - mean) / (std + eps)


def _stack_context(feat_td: np.ndarray, context_size: int) -> np.ndarray:
    """Concatenate left/right contextual frames on [T, D]."""
    if context_size <= 0:
        return feat_td
    t, _ = feat_td.shape
    padded = np.pad(feat_td, ((context_size, context_size), (0, 0)), mode="edge")
    chunks = [padded[i : i + t] for i in range(2 * context_size + 1)]
    return np.concatenate(chunks, axis=1)


def extract_spectral_features(
    waveform: np.ndarray,
    sample_rate: int,
    frame_size: float,
    frame_shift: float,
    feature_type: str,
    feature_dim: int,
    use_cmvn: bool = True,
    use_delta: bool = True,
    use_delta_delta: bool = True,
    context_size: int = 1,
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

    # Add temporal derivatives in [D, T] domain.
    feat_parts = [feat]
    if use_delta:
        feat_parts.append(librosa.feature.delta(feat, order=1))
    if use_delta_delta:
        feat_parts.append(librosa.feature.delta(feat, order=2))
    feat = np.concatenate(feat_parts, axis=0)

    if use_cmvn:
        feat = _apply_cmvn(feat)

    # librosa returns [D, T], while pipeline expects [T, D].
    feat_td = np.asarray(feat.T, dtype=np.float32)
    feat_td = _stack_context(feat_td, context_size=context_size)
    return np.asarray(feat_td, dtype=np.float32)
