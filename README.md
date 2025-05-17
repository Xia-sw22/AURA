This repository provides the code for [Multimodal transformer-based model for predicting prognosis after radiotherapy plus systemic therapy in hepatocellular carcinoma]. Based on the code, you can easily train your own AURA by configuring your own dataset and modifying the training details (such as optimizer, learning rate, etc).

## Overview

AURA is a transformer-based multimodal medical prediction model that can perform both classification and survival prediction tasks. It processes data from three modalities *[imaging, text, and structured metrics]** and integrates them using cross-attention mechanisms for fusion.*

## Setup the Environment

This software was implemented in a system running Windows 10, with Python 3.9, PyTorch 2.5.1, and CUDA 12.1.

You can adjust the batch size to adapt to your own hardware environment. Personally, we recommend the use of four NVIDIA GPUs.

## Code Description

The main architecture of AURA lies in the models/ folder. The files modeling_aura.py and modeling_aura_surv serve as the main backbone for classification tasks and survival tasks, while the rest necessary modules are distributed into different files based on their own functions, i.e., attention.py, block.py, configs.py, embed.py, encoder.py, and mlp.py. Please refer to each file to acquire more implementation details.

Parameter description:

The training script requires three data files (pickle format):
- Training data (`train_data.pkl`)
- Validation data (`val_data.pkl`) 
- Test data (`test_data.pkl`)

Each data file should contain the following fields:
- `images`: Image data
- `text_features`: Text features
- `structured_data`: Structured data
- `age`: Age information
- `sex`: Gender information
- `labels`: Label data

## Running Training

### Basic Usage

```bash
python train_response.py \
    --train_data path/to/train_data.pkl \
    --val_data path/to/val_data.pkl \
    --test_data path/to/test_data.pkl
```

### Complete Parameter Description

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_data` | Path to training data file | Required |
| `--val_data` | Path to validation data file | Required |
| `--test_data` | Path to test data file | Required |
| `--batch_size` | Training batch size | 16 |
| `--lr` | Learning rate | 1e-5 |
| `--num_epochs` | Number of training epochs | 30 |
| `--num_classes` | Number of classes | 2 |
| `--save_path` | Path to save models | ./checkpoints/task_response |
| `--resume_from_checkpoint` | Resume training from checkpoint | None |
| `--contrastive_weight` | Weight for contrastive loss | 0.1 |
| `--temperature` | Temperature parameter for contrastive loss | 0.1 |
| `--weight_decay` | Weight decay for optimizer | 0.01 |
| `--seed` | Random seed | 42 |
| `--device` | Training device (cuda/cpu) | cuda |
| `--num_workers` | Number of data loading workers | 0 |
| `--save_every_n_epochs` | Save model every N epochs | 10 |
| `--early_stopping` | Early stopping patience | 5 |

### Example Commands

1. Basic training:
```bash
python train_response.py \
    --train_data ./data/train.pkl \
    --val_data ./data/val.pkl \
    --test_data ./data/test.pkl
```

2. Training with custom parameters:
```bash
python train_response.py \
    --train_data ./data/train.pkl \
    --val_data ./data/val.pkl \
    --test_data ./data/test.pkl \
    --batch_size 32 \
    --lr 2e-5 \
    --num_epochs 50 \
    --device cuda \
    --save_path ./my_checkpoints \
    --early_stopping 10
```

3. Resume training from checkpoint:
```bash
python train_response.py \
    --train_data ./data/train.pkl \
    --val_data ./data/val.pkl \
    --test_data ./data/test.pkl \
    --resume_from_checkpoint ./checkpoints/task_response/best_model.pth
```

## Output Description

The training process generates the following files:

1. Model checkpoints:
   - `best_model.pth`: Best model
   - `model_epoch_X.pth`: Periodically saved model checkpoints

2. Training logs:
   - Saved in `{save_path}/logs` directory
   - Viewable using TensorBoard

3. Evaluation metrics:
   - Saved in `{save_path}/response_results` directory
   - `train_metrics.csv`: Training set metrics
   - `val_metrics.csv`: Validation set metrics
   - `test_metrics.csv`: Test set metrics
   - `train_preds.csv`: Training set predictions
   - `val_preds.csv`: Validation set predictions
   - `test_preds.csv`: Test set predictions

## Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir ./checkpoints/task_response/logs
```

## Important Notes

1. Ensure sufficient GPU memory (if using GPU training)
2. Data files must be in the correct format
3. Recommended to test the script with a small dataset first
4. Model performance can be optimized by adjusting `contrastive_weight` and `temperature` parameters
5. If training is unstable, try adjusting learning rate and weight decay parameters

## Metrics Tracked

The training script tracks the following metrics:
- Loss
- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Specificity
- False Positive Rate (FPR)
- False Negative Rate (FNR)

All metrics are logged to TensorBoard and saved in CSV files for detailed analysis.
```

