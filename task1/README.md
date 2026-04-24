# Task1 Scaffold (Short-Time Features + Threshold Classifier)

This folder is a code scaffold only. Fill TODO sections by yourself.

## Suggested workflow

1. Implement waveform loading and frame-level label alignment in `dataset.py`.
2. Implement short-time **feature extraction** in `features.py`.
3. Implement **threshold-based classifier** in `model.py`.
4. Implement smoothing / segment conversion in `postprocess.py`.
5. Connect the full pipeline in `pipeline.py`.
6. Use `run_dev.py` for development evaluation.
7. Use `run_test.py` to generate `test_label.txt`.

## Files

- `config.py`: task configuration dataclasses.
- `dataset.py`: dataset and label I/O.
- `features.py`: short-time feature extraction.
- `model.py`: threshold classifier skeleton.
- `postprocess.py`: frame-to-segment conversion.
- `pipeline.py`: train/eval/inference orchestration.
- `run_dev.py`: CLI entry for dev set evaluation.
- `run_test.py`: CLI entry for test set inference.
