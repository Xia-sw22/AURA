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
from models.modeling_aura import AURA
from models.configs import get_AURA_config
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_config():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--train_data', type=str, required=True, help='Path to the training pickle file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to the validation pickle file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test pickle file')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--save_path', type=str, default='./checkpoints/task_response', help='Path to save the model checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to resume the model from a checkpoint')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for contrastive loss')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--save_every_n_epochs', type=int, default=10, help='Save model every n epochs')
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

def calculate_metrics(labels, preds, probs=None):
    metrics = {}
    metrics['auc'] = roc_auc_score(labels, probs[:, 1]) if probs is not None else None
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics['specificity'] = tn / (tn + fp)
    metrics['fpr'] = fp / (fp + tn)
    metrics['fnr'] = fn / (fn + tp)
    return metrics

def info_nce_loss(image_embeds, text_embeds, clinical_embeds, temperature=0.1):

    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)
    clinical_embeds = F.normalize(clinical_embeds, dim=1)
    
    batch_size = image_embeds.size(0)
    labels = torch.arange(batch_size, device=image_embeds.device)
    
    i2t_logits = torch.matmul(image_embeds, text_embeds.T) / temperature
    loss_i2t = F.cross_entropy(i2t_logits, labels)
    
    i2c_logits = torch.matmul(image_embeds, clinical_embeds.T) / temperature
    loss_i2c = F.cross_entropy(i2c_logits, labels)
    
    t2c_logits = torch.matmul(text_embeds, clinical_embeds.T) / temperature
    loss_t2c = F.cross_entropy(t2c_logits, labels)
    
    total_loss = (loss_i2t + loss_i2c + loss_t2c) / 3
    return total_loss

class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        sample = {
            'images': self.data['images'][idx],
            'text_features': self.data['text_features'][idx],
            'structured_data': self.data['structured_data'][idx],
            'age': self.data['age'][idx],
            'sex': self.data['sex'][idx],
            'label': F.one_hot(self.data['labels'], num_classes=2).squeeze(1)[idx],
        }
        return sample
    
    def __len__(self):
        return len(self.data['images'])

def load_data(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def initialize_model(config):
    config_model = get_AURA_config()
    model = AURA(config_model, num_classes=config.num_classes, vis=True)
    model.to(config.device)

    epoch_start = 0
    early_stopping_counter = 0
    best_val_loss = float('inf')
    if config.resume_from_checkpoint:
        checkpoint = torch.load(config.resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch'] + 1  
        best_val_acc = checkpoint['best_val_acc']  
        early_stopping_counter = checkpoint['early_stopping_counter']  
        logger.info(f"Resumed from checkpoint: {config.resume_from_checkpoint}")

    return model, epoch_start, best_val_loss, early_stopping_counter

def save_checkpoint(model, optimizer, epoch, val_acc, early_stopping_counter, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': val_acc,
        'early_stopping_counter': early_stopping_counter
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved at epoch {epoch}")

def train(model, train_dataloader, optimizer, criterion, epoch, config, writer):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for _, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
        inputs, labels = data['images'].to(config.device), data['label'].to(config.device)
        cc = data['text_features'].to(config.device)
        lab = data['structured_data'].to(config.device)
        sex = data['sex'].to(config.device)
        age = data['age'].to(config.device)

        optimizer.zero_grad()

        outputs, _, image_embeds, text_embeds, clinical_embeds = model(x=inputs, cc=cc, lab=lab, sex=sex, age=age)        

        cls_loss = criterion(outputs, labels.float()) 

        contrastive_loss = info_nce_loss(image_embeds, text_embeds, clinical_embeds, temperature=config.temperature)
        contrastive_weight = config.contrastive_weight
        
        total_loss = cls_loss + contrastive_weight * contrastive_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct_preds += torch.sum(preds == labels.argmax(dim=1)).item()
        total_preds += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.argmax(dim=1).cpu().numpy())
        all_probs.extend(F.softmax(outputs, dim=1).cpu().detach().numpy())

    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = correct_preds / total_preds * 100

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics = calculate_metrics(all_labels, all_preds, np.array(all_probs))

    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
    writer.add_scalar('Precision/Train', precision, epoch)
    writer.add_scalar('Recall/Train', recall, epoch)
    writer.add_scalar('F1/Train', f1, epoch)
    writer.add_scalar('AUC/Train', metrics['auc'], epoch)
    writer.add_scalar('Specificity/Train', metrics['specificity'], epoch)
    writer.add_scalar('FPR/Train', metrics['fpr'], epoch)
    writer.add_scalar('FNR/Train', metrics['fnr'], epoch)

    logger.info(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {metrics['auc']:.4f}, Specificity: {metrics['specificity']:.4f}, FPR: {metrics['fpr']:.4f}, FNR: {metrics['fnr']:.4f}")

    return epoch_loss, epoch_acc, precision, recall, f1, metrics['auc'], metrics['specificity'], metrics['fpr'], metrics['fnr'], all_labels, all_preds


def evaluate(model, dataloader, criterion, config, writer, epoch, stage='val'):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"{stage.capitalize()} Evaluation"):
            inputs, labels = data['images'].to(config.device), data['label'].to(config.device)
            cc = data['text_features'].to(config.device)
            lab = data['structured_data'].to(config.device)
            sex = data['sex'].to(config.device)
            age = data['age'].to(config.device)

            outputs, all_attentions, image_embeds, text_embeds, clinical_embeds = model(x=inputs, cc=cc, lab=lab, sex=sex, age=age)

            cls_loss = criterion(outputs, labels.float())
            contrastive_loss = info_nce_loss(image_embeds, text_embeds, clinical_embeds, temperature=config.temperature)
            total_loss = cls_loss + config.contrastive_weight * contrastive_loss

            running_loss += total_loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_preds += torch.sum(preds == labels.argmax(dim=1)).item()
            total_preds += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_preds / total_preds * 100

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics = calculate_metrics(all_labels, all_preds, np.array(all_probs))

    writer.add_scalar(f'Loss/{stage.capitalize()}', epoch_loss, epoch)
    writer.add_scalar(f'Accuracy/{stage.capitalize()}', epoch_acc, epoch)
    writer.add_scalar(f'Precision/{stage.capitalize()}', precision, epoch)
    writer.add_scalar(f'Recall/{stage.capitalize()}', recall, epoch)
    writer.add_scalar(f'F1/{stage.capitalize()}', f1, epoch)
    writer.add_scalar(f'AUC/{stage.capitalize()}', metrics['auc'], epoch)
    writer.add_scalar(f'Specificity/{stage.capitalize()}', metrics['specificity'], epoch)
    writer.add_scalar(f'FPR/{stage.capitalize()}', metrics['fpr'], epoch)
    writer.add_scalar(f'FNR/{stage.capitalize()}', metrics['fnr'], epoch)

    logger.info(f"{stage.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                f"AUC: {metrics['auc']:.4f}, Specificity: {metrics['specificity']:.4f}, "
                f"FPR: {metrics['fpr']:.4f}, FNR: {metrics['fnr']:.4f}")

    return (epoch_loss, epoch_acc, precision, recall, f1, metrics['auc'], 
            metrics['specificity'], metrics['fpr'], metrics['fnr'], 
            all_labels, all_preds)

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

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False)

    model, epoch_start, best_val_loss, early_stopping_counter = initialize_model(config)

    optimizer = optim.AdamW(model.parameters(), 
                            lr=config.lr,
                            weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config.num_epochs, 
                                  eta_min=1e-6)

    results_path = os.path.join(config.save_path, 'response_results')
    os.makedirs(results_path, exist_ok=True)

    train_metrics_df = pd.DataFrame()
    val_metrics_df = pd.DataFrame()
    test_metrics_df = pd.DataFrame()
    train_preds_df = pd.DataFrame()
    val_preds_df = pd.DataFrame()
    test_preds_df = pd.DataFrame()

    for epoch in range(epoch_start, config.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{config.num_epochs}")

        train_loss, train_acc, train_precision, train_recall, train_f1, train_auc, train_specificity, train_fpr, train_fnr, train_labels, train_preds = train(model, train_dataloader, optimizer, criterion, epoch, config, writer)

        val_metrics = evaluate(model, val_dataloader, criterion, config, writer, epoch, 'val')
        (val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, 
         val_specificity, val_fpr, val_fnr, val_labels, val_preds) = val_metrics

        test_metrics = evaluate(model, test_dataloader, criterion, config, writer, epoch, 'test')
        (test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, 
         test_specificity, test_fpr, test_fnr, test_labels, test_preds) = test_metrics

        train_metrics = {'epoch': epoch+1, 'auc': train_auc, 'loss': train_loss, 
                         'accuracy': train_acc, 'precision': train_precision, 
                         'recall': train_recall, 'f1': train_f1, 'specificity': train_specificity, 
                         'fpr': train_fpr, 'fnr': train_fnr}
        val_metrics_dict = {'epoch': epoch+1, 'auc': val_auc, 'loss': val_loss, 
                            'accuracy': val_acc, 'precision': val_precision, 
                            'recall': val_recall, 'f1': val_f1, 'specificity': val_specificity, 
                           'fpr': val_fpr, 'fnr': val_fnr}
        test_metrics_dict = {'epoch': epoch+1, 'auc': test_auc, 'loss': test_loss, 
                             'accuracy': test_acc, 'precision': test_precision, 
                             'recall': test_recall, 'f1': test_f1, 'specificity': test_specificity, 
                            'fpr': test_fpr, 'fnr': test_fnr}

        train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([train_metrics])], ignore_index=True)
        val_metrics_df = pd.concat([val_metrics_df, pd.DataFrame([val_metrics_dict])], ignore_index=True)
        test_metrics_df = pd.concat([test_metrics_df, pd.DataFrame([test_metrics_dict])], ignore_index=True)

        if epoch == epoch_start:
            train_preds_df['labels'] = train_labels
            val_preds_df['labels'] = val_labels
            test_preds_df['labels'] = test_labels
        train_preds_df[f'pred_epoch_{epoch+1}'] = train_preds
        val_preds_df[f'pred_epoch_{epoch+1}'] = val_preds
        test_preds_df[f'pred_epoch_{epoch+1}'] = test_preds

        train_metrics_df.to_csv(os.path.join(results_path, 'train_metrics.csv'), index=False)
        val_metrics_df.to_csv(os.path.join(results_path, 'val_metrics.csv'), index=False)
        test_metrics_df.to_csv(os.path.join(results_path, 'test_metrics.csv'), index=False)
        train_preds_df.to_csv(os.path.join(results_path, 'train_preds.csv'), index=False)
        val_preds_df.to_csv(os.path.join(results_path, 'val_preds.csv'), index=False)
        test_preds_df.to_csv(os.path.join(results_path, 'test_preds.csv'), index=False)

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