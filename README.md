This repository provides the code for [Multimodal transformer-based model for predicting prognosis after radiotherapy plus systemic therapy in hepatocellular carcinoma]. Based on the code, you can easily train your own TRIM by configuring your own dataset and modifying the training details (such as optimizer, learning rate, etc).

## Overview

TRIM is a transformer-based multimodal medical prediction model that can perform both classification and survival prediction tasks. It processes data from three modalities *[imaging, text, and structured metrics]** and integrates them using cross-attention mechanisms for fusion.*

This repository contains two training scripts for the TRIM model:
- `train_response.py`: Binary classification task for treatment response prediction
- `train_survival.py`: Survival analysis task for time-to-event prediction

## Setup the Environment

This software was implemented in a system running Windows 10, with Python 3.9, PyTorch 2.5.1, and CUDA 12.1.

You can adjust the batch size to adapt to your own hardware environment. Personally, we recommend the use of four NVIDIA GPUs.

## Code Description

The main architecture of TRIM lies in the models/ folder. The files modeling_aura.py and modeling_aura_surv serve as the main backbone for classification tasks and survival tasks, while the rest necessary modules are distributed into different files based on their own functions, i.e., attention.py, block.py, configs.py, embed.py, encoder.py, and mlp.py. Please refer to each file to acquire more implementation details.

## Data Preparation

Both scripts require three data files (pickle format):
- Training data (`train_data.pkl`)
- Validation data (`val_data.pkl`)
- Test data (`test_data.pkl`)

### Classification Task Data Format
Each data file should contain:
- `images`: Image data
- `text_features`: Text features
- `structured_data`: Structured data
- `age`: Age information
- `sex`: Gender information
- `labels`: Binary labels (0/1)

### Survival Task Data Format
Each data file should contain:
- `images`: Image data
- `text_features`: Text features
- `structured_data`: Structured data
- `age`: Age information
- `sex`: Gender information
- `event`: Event indicator (0/1)
- `time`: Time to event

## Running Training

### Basic Usage

1. For Classification Task:
```bash
python train_response.py \
    --train_data path/to/train_data.pkl \
    --val_data path/to/val_data.pkl \
    --test_data path/to/test_data.pkl
```

2. For Survival Task:
```bash
python train_survival.py \
    --train_data path/to/train_data.pkl \
    --val_data path/to/val_data.pkl \
    --test_data path/to/test_data.pkl
```

### Common Parameters

Both scripts share these common parameters:
- `--train_data`: Path to training data file (required)
- `--val_data`: Path to validation data file (required)
- `--test_data`: Path to test data file (required)
- `--batch_size`: Training batch size (default: 16)
- `--lr`: Learning rate (default: 1e-5 for classification, 3e-5 for survival)
- `--num_epochs`: Number of training epochs (default: 30)
- `--save_path`: Path to save models (default: ./checkpoints/task_response or ./checkpoints/task_surv)
- `--resume_from_checkpoint`: Path to resume training from checkpoint (default: None)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Training device, either 'cuda' or 'cpu' (default: cuda)
- `--num_workers`: Number of data loading workers (default: 0)
- `--save_every_n_epochs`: Save model every N epochs (default: 10 for classification, 30 for survival)
- `--early_stopping`: Early stopping patience (default: 5)

### Example Commands

1. Classification task with custom parameters:
```bash
python train_response.py \
    --train_data ./data/train.pkl \
    --val_data ./data/val.pkl \
    --test_data ./data/test.pkl \
    --batch_size 32 \
    --lr 2e-5 \
    --num_epochs 50 \
    --contrastive_weight 0.2 \
    --temperature 0.2
```

2. Survival task with custom parameters:
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

### Classification Task Outputs
- Model checkpoints in `{save_path}/`:
  - `best_model.pth`: Best model
  - `model_epoch_X.pth`: Periodic checkpoints
- Training logs in `{save_path}/logs/`
- Evaluation metrics in `{save_path}/response_results/`:
  - `train_metrics.csv`: Training metrics (accuracy, precision, recall, F1, AUC, etc.)
  - `val_metrics.csv`: Validation metrics
  - `test_metrics.csv`: Test metrics
  - `train_preds.csv`: Training predictions
  - `val_preds.csv`: Validation predictions
  - `test_preds.csv`: Test predictions

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
tensorboard --logdir ./checkpoints/task_response/logs  # for classification
tensorboard --logdir ./checkpoints/task_surv/logs      # for survival
```

## Important Notes

1. Ensure sufficient GPU memory (if using GPU training)
2. Data files must be in the correct format for each task
3. Recommended to test the scripts with a small dataset first
4. For classification task:
   - Adjust `contrastive_weight` and `temperature` to optimize model performance
   - If training is unstable, try adjusting learning rate and weight decay
5. For survival task:
   - Monitor C-index as the primary metric
   - Ensure event and time data are properly normalized
6. Both tasks support early stopping and model checkpointing
7. Training logs are saved to `aura_log.log` for the survival task

## Metrics Tracked

### Classification Task
- Loss
- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Specificity
- False Positive Rate (FPR)
- False Negative Rate (FNR)

### Survival Task
- Cox Partial Likelihood Loss
- C-index (Concordance Index)

