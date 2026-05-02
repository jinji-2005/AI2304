# Task2 Scaffold (Spectral Features + Statistical Model)

This folder is a code scaffold only. Fill TODO sections by yourself.

## Suggested workflow

1. Implement data loading and frame-level supervision in `dataset.py`.
2. Implement spectral feature extraction (MFCC/FBank/PLP) in `features.py`.
3. Implement statistical model training/inference in `model.py`.
4. Add post-processing and segment conversion in `postprocess.py`.
5. Connect everything in `pipeline.py`.
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
