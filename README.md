# ECG Arrhythmia Classification — CNN + BiLSTM with Transfer Learning

A deep learning pipeline for ECG beat classification that pretrains a residual CNN + BiLSTM
network on the **PhysioNet/CinC 2017** dataset, then transfers and fine-tunes it on the
**MIT-BIH Arrhythmia Database (mitdb)** for 5-class beat-level arrhythmia classification.

## Overview

The notebook implements a two-stage transfer learning workflow:

1. **Pretraining** on PhysioNet/CinC 2017 (4-class rhythm classification: Normal, AFib,
   Other, Noisy).
2. **Fine-tuning** the pretrained model on MIT-BIH (5-class beat classification following
   the AAMI beat groupings: N, S, V, F, Q).

The model architecture combines a residual 1D-CNN feature extractor (with Squeeze-and-Excitation
attention blocks) followed by a 2-layer Bidirectional LSTM, designed to capture both local
morphological features and longer-range temporal dependencies in the ECG waveform.

## Pipeline

### 1. Signal Preprocessing
- All signals resampled to a common target frequency (`TARGET_FS = 125 Hz`).
- Min-max normalization to `[0, 1]`.
- R-peak detection via `neurokit2.ecg_peaks`.
- Beat segmentation using **RR-interval-adaptive windows** (`-0.3 × RR_median` to
  `+0.6 × RR_median` around each R-peak), zero-padded/trimmed to a fixed length
  (`BEAT_LEN = 375` samples, i.e. 3 seconds at 125 Hz).

### 2. Stage 1 — Pretraining on PhysioNet/CinC 2017
- Loads `.mat` signals + `REFERENCE-original.csv` labels.
- Filters to valid classes: `N` (normal), `A` (AFib), `O` (other), `~` (noisy).
- Trains the CNN+BiLSTM model from scratch for 10 epochs.
- Saves the pretrained backbone as `pretrained_physio.h5`.

### 3. Stage 2 — Fine-tuning on MIT-BIH
- Downloads the 48-record `mitdb` database via `wfdb`.
- Remaps the 15 raw MIT-BIH annotation symbols to the 5 standard AAMI superclasses:
  - **N** (Normal): `N, L, R, e, j`
  - **S** (Supraventricular ectopic): `A, a, J, S`
  - **V** (Ventricular ectopic): `V, E`
  - **F** (Fusion): `F`
  - **Q** (Unknown/paced): `/, f, Q`
- Loads the pretrained model, swaps the final classification head for 5 classes, and
  fine-tunes all layers (full network unfrozen).
- Uses **class-balanced weighting** to handle the natural class imbalance in MIT-BIH
  (the `N` class vastly outnumbers `S`, `V`, `F`, `Q`).
- Trains with `EarlyStopping` (patience=10, restores best weights on `val_loss`).

### 4. Evaluation
- Classification report (precision/recall/F1 per class).
- Confusion matrix visualization.
- Training/validation accuracy and loss curves.

## Model Architecture

```
Input (375, 1)
  → Conv1D(32) → BatchNorm
  → Conv1D(32) → BatchNorm
  → MaxPooling1D
  → Residual Block(64)   ─┐
  → Residual Block(96)    │  each block: 3× Conv1D + BatchNorm
  → Residual Block(128)   │  + Squeeze-and-Excitation gating
  → Residual Block(160)  ─┘  + skip connection
  → BiLSTM(64, return_sequences=True) → Dropout(0.2)
  → BiLSTM(64) → Dropout(0.5)
  → Dense(64, relu) → Dropout(0.5)
  → Dense(num_classes, softmax)
```

Each **residual block** includes a Squeeze-and-Excitation (SE) sub-block that learns
per-channel attention weights via global average pooling + a small bottleneck MLP,
re-scaling feature maps before the residual addition.

## Datasets

| Dataset | Used for | Classes |
|---|---|---|
| [PhysioNet/CinC 2017](https://physionet.org/content/challenge-2017/1.0.0/) | Pretraining | N, A, O, ~ |
| [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) | Fine-tuning / evaluation | N, S, V, F, Q (AAMI) |

## Requirements

```
numpy
pandas
scipy
scikit-learn
tensorflow
wfdb
neurokit2
matplotlib
```

Install with:
```bash
pip install numpy pandas scipy scikit-learn tensorflow wfdb neurokit2 matplotlib
```

## Usage

1. Place the PhysioNet/CinC 2017 `training2017` directory (with `REFERENCE-original.csv`
   and `.mat` files) at the path referenced by `BASE_PATH`.
2. Run the preprocessing + pretraining cells to produce `pretrained_physio.h5`.
3. Run the MIT-BIH download/preprocessing cells (`wfdb.dl_database('mitdb', ...)`
   handles this automatically).
4. Run the transfer learning / fine-tuning cells to adapt the model to 5-class
   beat classification.
5. Run the evaluation cells to view the classification report, confusion matrix,
   and training curves.

> **Note:** File paths (`BASE_PATH`, `MIT_PATH`) are currently set for a Kaggle
> environment (`/kaggle/input/...`, `/kaggle/working/...`) and should be updated
> if running locally or on another platform.

## Notes / Possible Improvements

- The PhysioNet pretraining stage currently trains for a fixed 10 epochs without
  early stopping or validation-based checkpointing — consider adding callbacks
  for consistency with the fine-tuning stage.
- `BEAT_LEN` is redefined locally inside the MIT-BIH extraction loop (375), which
  happens to match the global value but is duplicated — worth consolidating into
  a single constant.
- Layer-slicing when transferring the pretrained backbone (`base_model.layers[-2].output`)
  assumes a fixed architecture; if the backbone architecture changes, this index
  will need to be re-verified.
