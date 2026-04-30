from typing import Dict
import math
import numpy as np
"""
Task1 feature flow:
waveform -> framing -> windowing -> per-frame features -> stacked features.
"""

def framing(
    waveform: np.ndarray,
    sample_rate: int,
    frame_size: float,
    frame_shift: float,
) -> np.ndarray:
    """Convert waveform to shape [num_frames, frame_length].

    Data usage:
    - Input: one waveform from dataset.load_waveform
    - Output: 2D frame matrix, each row is one short-time frame
    - Used by: all Task1 feature extraction
    """
    # Keep frame parameters consistent with label conversion.
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    wave_len = waveform.shape[0] #原始音频的点数
    frame_length = int(round(sample_rate * frame_size)) #每一帧的采样点数
    hop_length = int(round(sample_rate * frame_shift)) #每两帧之间间隔的点数
    if wave_len<frame_length:
        num_frames = 1
    else:
        num_frames = 1+ math.ceil((wave_len - frame_length)/hop_length)
    pad_len = (num_frames-1)*hop_length + frame_length - wave_len # 填补最后一段不足 一帧长度的部分
    waveform = np.pad(waveform,(0,pad_len),'constant',constant_values=0.0)
    starts = np.arange(num_frames) * hop_length #用矩阵形式来提取 [num_frames, frame_length].
    offset = np.arange(frame_length).reshape(-1)
    indices = starts[:,None] + offset
    data = waveform[indices]
    return data

# d = np.array([1,3,4,5,6])
# a = np.array([0,1,2])
# b = np.array([0,1,2])
# b = b.reshape(-1)
# print(b,b.shape)
# b = b[:,None]
# print(b,b.shape)
# c = a+b
# print(c)
# print(d[c])
# # b = b[None,:]
# # print(b,b.shape)

def apply_window(frames: np.ndarray, window: str = "hamming") -> np.ndarray:
    """Apply analysis window to each frame.

    Data usage:
    - Input: framed waveform [num_frames, frame_length]
    - Output: windowed frames with the same shape
    - Used by: energy / zcr / spectral statistics
    """
    # Default recommendation: Hamming window for robust short-time analysis.
    # TODO: implement
    str = window.lower()
    if str=="hamming":
        frame_len = frames.shape[1]
        window = np.hamming(frame_len) # 生成一段权重函数
        frames = frames * window
    elif str =='hanning':
        frame_len = frames.shape[1]
        window = np.hanning(frame_len)
        frames = frames * window
    else:
        return frames
    
    return frames

# a = np.array([[1,1],[1,1]])
# window = np.hamming(2)
# a = a* window
# print(a)

def extract_short_time_features(frames: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract short-time features needed in Task1.

    Suggested outputs:
    - "energy": short-time energy or log-energy
    - "zcr": zero-crossing rate
    - add your own features if needed
    """
    # Data usage:
    # - Input: windowed frames from `apply_window`
    # - Output: dict of 1D arrays, each with length `num_frames`
    # - Used by: threshold model scoring in task1/model.py
    #
    # Keep each feature aligned by frame index.
    feature: Dict[str, np.ndarray] = {}
    eps = 1e-8

    def _zscore(x: np.ndarray) -> np.ndarray:
        mean = float(np.mean(x))
        std = float(np.std(x))
        return (x - mean) / (std + eps)

    # 1) 短时能量
    energy = np.sum(frames**2, axis=1)
    energy = np.log(energy + eps)
    feature["energy"] = _zscore(energy).astype(np.float32)

    # 2) 过零率
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = 0.5 * np.mean(np.abs(np.diff(signs, axis=1)), axis=1)
    feature["zcr"] = _zscore(zcr).astype(np.float32)


    # 3) 短时频谱特征（每帧 log-magnitude 谱均值）
    mag = np.abs(np.fft.rfft(frames, axis=1))
    st_spectrum = np.mean(np.log1p(mag), axis=1)
    feature["st_spectrum"] = _zscore(st_spectrum).astype(np.float32)

    # 4) 基频(F0)近似（每帧自相关峰值对应频率）
    #    这里将 "进频" 按 "基频" 实现，搜索范围 [60, 400] Hz。
    sample_rate = 16000.0
    f0_min, f0_max = 60.0, 400.0
    lag_min = max(1, int(sample_rate // f0_max))
    lag_max = max(lag_min + 1, int(sample_rate // f0_min))
    f0 = np.zeros(frames.shape[0], dtype=np.float32)
    for i, frame in enumerate(frames):
        x = frame - np.mean(frame)
        ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
        if ac.size <= lag_min:
            continue
        ac0 = float(ac[0] + eps)
        stop = min(lag_max, ac.size)
        if stop <= lag_min:
            continue
        seg = ac[lag_min:stop]
        peak_idx = int(np.argmax(seg))
        peak_val = float(seg[peak_idx])
        # 无声/弱周期帧不强行估计基频
        if peak_val / ac0 < 0.3:
            continue
        lag = lag_min + peak_idx
        f0[i] = float(sample_rate / (lag + eps))
    feature["pitch"] = _zscore(f0).astype(np.float32)
    return feature
    

def stack_features(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack multiple feature streams into [num_frames, feat_dim].

    Data usage:
    - Input: feature dict from `extract_short_time_features`
    - Output: 2D matrix consumed by ThresholdVAD.fit/score_frames
    - Used by: training, dev evaluation, and test inference
    """
    # Recommendation:
    # - choose a fixed feature order
    # - ensure all streams have identical num_frames
    # feature_dict = {
    # "energy": np.array([1.0, 2.0, 3.0]),
    # "zcr":    np.array([0.2, 0.1, 0.4]),}
    # X = np.array([
    # [1.0, 0.2],
    # [2.0, 0.1],
    # [3.0, 0.4],
    # ], dtype=np.float32)   # shape=(3,2)

    order = ["energy", "zcr", "st_spectrum", "pitch"]
    feature = [np.asarray(feature_dict[o],dtype=np.float32) for o in order]
    feature = np.stack(feature,axis=1) # [[1,2,3],[2,3,4]]
    return feature

# a= np.array([1,2,3])
# b= np.array([2,3,4])
# c = [a,b]
# c = np.array([[1,1],[2,2]])
# c = c.reshape(-1)
# print(c)
# c = np.stack(c,axis=0)
# print(c)
# c = np.stack(c,axis=1)
# print(c)

# c = np.array([[1,1],[2,2]])
# print(np.mean(c,axis=0))
# print(np.mean(c,axis=1))
