from pathlib import Path

from config import FrameConfig, PathConfig
from dataset import list_wav_files, load_waveform
from features import framing


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "voice-activity-detection-sjtu-spring-2024" / "vad"

    path_cfg = PathConfig(data_root=data_root)
    frame_cfg = FrameConfig()

    dev_wav_files = list_wav_files(path_cfg.wav_root / "dev")
    if not dev_wav_files:
        raise RuntimeError(f"No wav files found in: {path_cfg.wav_root / 'dev'}")

    wav_path = dev_wav_files[0]
    waveform = load_waveform(wav_path, frame_cfg.sample_rate)
    frames = framing(
        waveform=waveform,
        sample_rate=frame_cfg.sample_rate,
        frame_size=frame_cfg.frame_size,
        frame_shift=frame_cfg.frame_shift,
    )

    frame_length = int(round(frame_cfg.sample_rate * frame_cfg.frame_size))
    hop_length = int(round(frame_cfg.sample_rate * frame_cfg.frame_shift))

    assert frames.ndim == 2, f"frames should be 2D, got shape={frames.shape}"
    assert frames.shape[1] == frame_length, (
        f"frame_length mismatch: expected={frame_length}, got={frames.shape[1]}"
    )

    print("framing test passed")
    print(f"wav: {wav_path.name}")
    print(f"waveform shape: {waveform.shape}")
    print(f"frames shape: {frames.shape}")
    print(f"frame_length: {frame_length}, hop_length: {hop_length}")
    print(f"first frame (first 8 samples): {frames[0, :8]}")


if __name__ == "__main__":
    main()

