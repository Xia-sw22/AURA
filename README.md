This repository provides the code for [Multimodal transformer-based model for predicting prognosis after radiotherapy plus systemic therapy in hepatocellular carcinoma]. Based on the code, you can easily train your own TRIM by configuring your own dataset and modifying the training details (such as optimizer, learning rate, etc).

## Overview

TRIM is a transformer-based multimodal medical prediction model for survival analysis tasks. It processes data from three modalities (imaging, text, and structured metrics) and integrates them using parallel cross-attention mechanisms with cohort-specific processing channels for fusion.

This repository contains a training script for the TRIM model:
- `train_survival.py`: Survival analysis task for time-to-event prediction

## Setup the Environment

This software was implemented in a system running Windows 10, with Python 3.9, PyTorch 2.5.1, and CUDA 12.1.

You can adjust the batch size to adapt to your own hardware environment. Personally, we recommend the use of four NVIDIA GPUs.

## Code Description

The main architecture of TRIM lies in the models/ folder. The file modeling_trim_surv.py serves as the main backbone for survival tasks, while the rest necessary modules are distributed into different files based on their own functions, i.e., attention.py, block.py, configs.py, embed.py, encoder_parallel.py, and mlp.py.

### Model Architecture
The TRIM model features a novel parallel cross-attention architecture:
- **Embedding Layer**: Processes images, text features, and clinical data into unified representations
- **Parallel Cross-Attention Channels**: Two independent channels (ETS and ES) for cohort-specific processing
  - ETS Channel: Processes ETS cohort samples (cohort=0)
  - ES Channel: Processes ES cohort samples (cohort=1)
- **Shared Self-Attention Layers**: 10 layers of shared self-attention after cross-attention processing
- **Classification Head**: Final prediction layer for survival risk scores

Please refer to each file to acquire more implementation details.

## Data Preparation

The script requires three data files (pickle format):
- Training data (`train_data.pkl`)
- Validation data (`val_data.pkl`)
- Test data (`test_data.pkl`)

### Survival Task Data Format
Each data file should contain:
- `images`: Image data
- `text_features`: Text features
- `structured_data`: Structured data
- `age`: Age information
- `sex`: Gender information
- `event`: Event indicator (0/1)
- `time`: Time to event
- `cohort`: Cohort identifier (0: ETS, 1: ES) - optional, defaults to 0 if not provided
- `patient_ids`: Patient identifiers

## Running Training

### Basic Usage

For Survival Task:
```bash
python train_survival.py \
    --train_data path/to/train_data.pkl \
    --val_data path/to/val_data.pkl \
    --test_data path/to/test_data.pkl
```

### Common Parameters

The script uses these parameters:
- `--train_data`: Path to training data file (required)
- `--val_data`: Path to validation data file (required)
- `--test_data`: Path to test data file (required)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--num_epochs`: Number of training epochs
- `--save_path`: Path to save models
- `--resume_from_checkpoint`: Path to resume training from checkpoint 
- `--seed`: Random seed for reproducibility
- `--device`: Training device, either 'cuda' or 'cpu' 
- `--num_workers`: Number of data loading workers
- `--save_every_n_epochs`: Save model every N epochs
- `--early_stopping`: Early stopping patience

### Example Commands

Survival task with custom parameters:
```bash
python train_survival.py \
    --train_data ./data/train.pkl \
    --val_data ./data/val.pkl \
    --test_data ./data/test.pkl \
    --batch_size 32 \
    --lr 5e-5 \
    --num_epochs 50 \
```

## Output Description

### Survival Task Outputs
- Model checkpoints in `{save_path}/`:
  - `best_model.pth`: Best model
  - `model_epoch_X.pth`: Periodic checkpoints
- Training logs in `{save_path}/logs/`
- Evaluation metrics in `{save_path}/surv_results/`:
  - `train_metrics.csv`: Training metrics (loss, C-index)
  - `val_metrics.csv`: Validation metrics
  - `test_metrics.csv`: Test metrics
  - `train_preds.csv`: Training predictions with patient IDs
  - `val_preds.csv`: Validation predictions with patient IDs
  - `test_preds.csv`: Test predictions with patient IDs

## Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir ./checkpoints/task_surv/logs
```

## Important Notes

1. Ensure sufficient GPU memory (if using GPU training)
2. Data files must be in the correct format for the survival task
3. Recommended to test the script with a small dataset first
4. For survival task:
   - Monitor C-index as the primary metric
   - Ensure event and time data are properly normalized
5. The task supports early stopping and model checkpointing
6. Training logs are saved to `trim_log.log` for the survival task
7. Cohort-specific processing:
   - Provide `cohort` field in data for optimal performance (0: ETS, 1: ES)
   - If `cohort` field is missing, all samples default to ETS channel (cohort=0)
   - Mixed batches with different cohorts are supported

## Metrics Tracked

### Survival Task
- Cox Partial Likelihood Loss
- C-index (Concordance Index)

