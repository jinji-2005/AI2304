# Task 1: short-time features and threshold VAD

This implementation performs frame-level voice activity detection using short-time energy, zero-crossing rate, spectrum and pitch features. It searches a pair of hysteresis thresholds on the development split, smooths frame decisions and converts them back to speech time segments.

## Run

From the repository root:

```bash
python task1/run_dev.py
python task1/run_test.py --output task1/test_label.txt
```

The dataset must first be placed under `voice-activity-detection-sjtu-spring-2024/vad/` as described in the project-level README.

## Files

- `config.py`: frame, feature and threshold settings.
- `dataset.py`: waveform and frame-label loading.
- `features.py`: short-time feature extraction.
- `model.py`: weighted scoring and hysteresis-threshold search.
- `postprocess.py`: smoothing and frame-to-segment conversion.
- `pipeline.py`: development evaluation and test inference.
