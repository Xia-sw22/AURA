import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import logging
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import DataLoader
from models.modeling_aura_surv import AURA
from models.configs import get_AURA_config
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
from lifelines.utils import concordance_index
import pandas as pd

logger = logging.getLogger(__name__)
log_filename = "aura_log.log"  

logging.basicConfig(level=logging.INFO)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def get_config():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--train_data', type=str, required=True, help='Path to the training pickle file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to the validation pickle file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test pickle file')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='checkpoints/task_surv', help='Path to save the model checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to resume the model from a checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--save_every_n_epochs', type=int, default=30, help='Save model every n epochs')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience (epochs without improvement)')

    args = parser.parse_args()
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def cox_ph_loss(logits, events, times, eps=1e-7):
    logits = logits.view(-1)
    events = events.view(-1)
    times = times.view(-1)
    
    sort_idx = torch.argsort(times, descending=True)
    logits_sorted = logits[sort_idx]
    events_sorted = events[sort_idx]
    times_sorted = times[sort_idx]
    
    max_logit = logits_sorted.max()
    logits_stable = logits_sorted - max_logit
    exp_terms = torch.exp(logits_stable)
    
    cumsum = torch.cumsum(exp_terms, dim=0)
    log_cumsum = torch.log(cumsum + eps) + max_logit
    
    event_mask = events_sorted.bool()
    if not event_mask.any():
        return torch.tensor(0.0, device=logits.device)
    
    loss_terms = logits_sorted[event_mask] - log_cumsum[event_mask]
    loss = -torch.mean(loss_terms)
    
    return loss

class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.patient_ids = data['patient_ids']

    def __getitem__(self, idx):
        sample = {
            'patient_id': self.patient_ids[idx],
            'images': self.data['images'][idx],
            'text_features': self.data['text_features'][idx],
            'structured_data': self.data['structured_data'][idx],
            'age': self.data['age'][idx],
            'sex': self.data['sex'][idx],
            'event': self.data['event'][idx],
            'time': self.data['time'][idx],
        }
        return sample
    
    def __len__(self):
        return len(self.patient_ids)

def load_data(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def initialize_model(config):
    config_model = get_AURA_config()
    model = AURA(config_model)
    model.to(config.device)

    epoch_start = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    if config.resume_from_checkpoint:
        checkpoint = torch.load(config.resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        early_stopping_counter = checkpoint['early_stopping_counter']
        logger.info(f"Resumed from checkpoint: {config.resume_from_checkpoint}")

    return model, epoch_start, best_val_loss, early_stopping_counter

def save_checkpoint(model, optimizer, epoch, best_val_loss, early_stopping_counter, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'early_stopping_counter': early_stopping_counter
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved at epoch {epoch}")

def train(model, train_dataloader, optimizer, criterion, epoch, config, writer):
    model.train()
    running_loss = 0.0
    all_risk = []
    all_events = []
    all_times = []
    all_patient_ids = []

    for data in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
        inputs = data['images'].to(config.device)
        cc = data['text_features'].to(config.device)
        lab = data['structured_data'].to(config.device)
        sex = data['sex'].to(config.device)
        age = data['age'].to(config.device)
        event = data['event'].to(config.device)
        time = data['time'].to(config.device)
        patient_ids = data['patient_id']

        optimizer.zero_grad()
        risk, _ = model(x=inputs, cc=cc, lab=lab, sex=sex, age=age)
        loss = criterion(risk, event, time)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_risk.extend(risk.detach().cpu().numpy().flatten())
        all_events.extend(event.cpu().numpy().flatten())
        all_times.extend(time.cpu().numpy().flatten())
        all_patient_ids.extend(patient_ids)

    epoch_loss = running_loss / len(train_dataloader)
    c_index = concordance_index(all_times, -np.array(all_risk), all_events)

    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('C-Index/Train', c_index, epoch)
    logger.info(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}, C-Index: {c_index:.4f}")

    return epoch_loss, c_index, np.array(all_risk), np.array(all_events), np.array(all_times), all_patient_ids

def evaluate(model, dataloader, criterion, config, writer, epoch, stage='val'):
    model.eval()
    running_loss = 0.0
    all_risk = []
    all_events = []
    all_times = []
    all_patient_ids = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"{stage.capitalize()} Evaluation"):
            inputs = data['images'].to(config.device)
            cc = data['text_features'].to(config.device)
            lab = data['structured_data'].to(config.device)
            sex = data['sex'].to(config.device)
            age = data['age'].to(config.device)
            event = data['event'].to(config.device)
            time = data['time'].to(config.device)
            patient_ids = data['patient_id']

            risk, _ = model(x=inputs, cc=cc, lab=lab, sex=sex, age=age)
            loss = criterion(risk, event, time)

            running_loss += loss.item()
            all_risk.extend(risk.cpu().numpy().flatten())
            all_events.extend(event.cpu().numpy().flatten())
            all_times.extend(time.cpu().numpy().flatten())
            all_patient_ids.extend(patient_ids)

    epoch_loss = running_loss / len(dataloader)
    c_index = concordance_index(all_times, -np.array(all_risk), all_events)

    writer.add_scalar(f'Loss/{stage.capitalize()}', epoch_loss, epoch)
    writer.add_scalar(f'C-Index/{stage.capitalize()}', c_index, epoch)
    logger.info(f"{stage.capitalize()} Loss: {epoch_loss:.4f}, C-Index: {c_index:.4f}")

    return epoch_loss, c_index, np.array(all_risk), np.array(all_events), np.array(all_times), all_patient_ids

def main():
    config = get_config()
    set_random_seed(config.seed)
    
    logger.info("Training configuration:")
    for arg in vars(config):
        logger.info(f"{arg:>20}: {getattr(config, arg)}")
    
    writer = SummaryWriter(log_dir=os.path.join(config.save_path, "logs"))

    train_data = load_data(config.train_data)
    val_data = load_data(config.val_data)
    test_data = load_data(config.test_data)

    train_dataset = MultiModalDataset(train_data)
    val_dataset = MultiModalDataset(val_data)
    test_dataset = MultiModalDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                num_workers=config.num_workers, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                               num_workers=config.num_workers, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=False)

    model, epoch_start, best_val_loss, early_stopping_counter = initialize_model(config)

    criterion = cox_ph_loss
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config.lr,
                            weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config.num_epochs, 
                                  eta_min=1e-6)

    results_path = os.path.join(config.save_path, 'surv_results')
    os.makedirs(results_path, exist_ok=True)

    metrics_columns = ['epoch', 'loss', 'c_index']
    train_metrics_df = pd.DataFrame(columns=metrics_columns)
    val_metrics_df = pd.DataFrame(columns=metrics_columns)
    test_metrics_df = pd.DataFrame(columns=metrics_columns)

    train_preds_df = pd.DataFrame(index=train_dataset.patient_ids)
    train_preds_df['event'] = train_data['event'].squeeze()
    train_preds_df['time'] = train_data['time'].squeeze()

    val_preds_df = pd.DataFrame(index=val_dataset.patient_ids)
    val_preds_df['event'] = val_data['event'].squeeze()
    val_preds_df['time'] = val_data['time'].squeeze()

    test_preds_df = pd.DataFrame(index=test_dataset.patient_ids)
    test_preds_df['event'] = test_data['event'].squeeze()
    test_preds_df['time'] = test_data['time'].squeeze()

    for epoch in range(epoch_start, config.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{config.num_epochs}")

        train_loss, train_c_index, train_risk, train_events, train_times, train_pids = train(
            model, train_dataloader, optimizer, criterion, epoch, config, writer
        )
        
        val_loss, val_c_index, val_risk, val_events, val_times, val_pids = evaluate(
            model, val_dataloader, criterion, config, writer, epoch, 'val'
        )
        
        test_loss, test_c_index, test_risk, test_events, test_times, test_pids = evaluate(
            model, test_dataloader, criterion, config, writer, epoch, 'test'
        )

        train_metrics = {'epoch': epoch+1, 'loss': train_loss, 'c_index': train_c_index}
        val_metrics = {'epoch': epoch+1, 'loss': val_loss, 'c_index': val_c_index}
        test_metrics = {'epoch': epoch+1, 'loss': test_loss, 'c_index': test_c_index}

        train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([train_metrics])], ignore_index=True)
        val_metrics_df = pd.concat([val_metrics_df, pd.DataFrame([val_metrics])], ignore_index=True)
        test_metrics_df = pd.concat([test_metrics_df, pd.DataFrame([test_metrics])], ignore_index=True)

        train_current_risk = pd.Series(train_risk, index=train_pids, name=f'risk_epoch_{epoch+1}')
        train_preds_df = train_preds_df.join(train_current_risk)

        val_current_risk = pd.Series(val_risk, index=val_pids, name=f'risk_epoch_{epoch+1}')
        val_preds_df = val_preds_df.join(val_current_risk)

        test_current_risk = pd.Series(test_risk, index=test_pids, name=f'risk_epoch_{epoch+1}')
        test_preds_df = test_preds_df.join(test_current_risk)

        train_metrics_df.to_csv(os.path.join(results_path, 'train_metrics.csv'), index=False)
        val_metrics_df.to_csv(os.path.join(results_path, 'val_metrics.csv'), index=False)
        test_metrics_df.to_csv(os.path.join(results_path, 'test_metrics.csv'), index=False)
        
        train_preds_df.reset_index().rename(columns={'index': 'patient_id'}).to_csv(
            os.path.join(results_path, 'train_preds.csv'), index=False)
        val_preds_df.reset_index().rename(columns={'index': 'patient_id'}).to_csv(
            os.path.join(results_path, 'val_preds.csv'), index=False)
        test_preds_df.reset_index().rename(columns={'index': 'patient_id'}).to_csv(
            os.path.join(results_path, 'test_preds.csv'), index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, early_stopping_counter,
                           os.path.join(config.save_path, 'best_model.pth'))
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= config.early_stopping:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(config.save_path, f'model_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, best_val_loss, early_stopping_counter, checkpoint_path)

        scheduler.step()

    logger.info("Training complete!")
    writer.close()

if __name__ == "__main__":
    main()