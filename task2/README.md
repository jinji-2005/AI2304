# Task 2: spectral features with GMM/DNN VAD

This implementation extracts FBank or MFCC features, optionally adds first- and second-order deltas, applies CMVN and context splicing, then trains either class-conditional GMMs or a PyTorch MLP. The decoding threshold is selected on the development split.

## Run

From the repository root:

```bash
python task2/run_dev.py
python task2/run_test.py --output task2/test_label.txt
```

Task 2 can be configured through `TASK2_*` environment variables. For example:

```bash
TASK2_MODEL_TYPE=gmm TASK2_FEATURE_TYPE=mfcc python task2/run_dev.py
```

## Files

- `config.py`: frame, feature and model settings.
- `dataset.py`: waveform and frame-label loading.
- `features.py`: FBank/MFCC, CMVN, deltas and context splicing.
- `model.py`: GMM and DNN classifiers.
- `postprocess.py`: smoothing and frame-to-segment conversion.
- `pipeline.py`: training, threshold tuning, evaluation and inference.
