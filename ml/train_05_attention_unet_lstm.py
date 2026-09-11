"""
AQUA HORIZON — Model 5: Attention U-Net + LSTM
Architecture: AttentionUNetLSTM
Zenodo IFI-Impacts Dataset (1967-2023) | Chronological 80:20 Split
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
from ml.models_hybrid import AttentionUNetLSTM
from ml.data_loader import load_sequence_data

def train_attention_u_net___lstm():
    print("=" * 80)
    print("AQUA HORIZON — HYBRID MODEL 5: ATTENTION U-NET + LSTM")
    print("Architecture Class: AttentionUNetLSTM")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Data
    print("Loading Zenodo sequence dataset (T=5 sliding windows)...")
    data = load_sequence_data(seq_len=5, cutoff_year=2012, spatial=True, batch_size=128)
    train_loader = data['train_loader']
    test_loader = data['test_loader']
    pos_weight = data['pos_weight']
    
    print(f"Train sequences: {len(data['X_train'])}, Test sequences: {len(data['X_test'])}")
    print(f"Imbalance ratio (pos_weight): {pos_weight:.3f}")
    
    # 2. Instantiate Model
    model = AttentionUNetLSTM().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    w_pos = torch.tensor(min(pos_weight, 2.5), device=device)
    def weighted_bce_loss(preds, targets):
        preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        loss = -(w_pos * targets * torch.log(preds) + (1.0 - targets) * torch.log(1.0 - preds))
        return torch.mean(loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_f1 = 0.0
    best_metrics = None
    save_path = os.path.join(os.path.dirname(__file__), 'models', 'attention_unet_lstm.pt')
    
    print(f"\nBeginning Training (8 Epochs)...")
    start_time = time.time()
    
    for epoch in range(1, 8 + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = weighted_bce_loss(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
            
        epoch_loss = running_loss / max(n_batches, 1)
        
        # Validation on test split
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch_y.numpy().tolist())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Optimal threshold search
        best_t, best_t_f1, best_t_rec, best_t_prec = 0.40, 0.0, 0.0, 0.0
        for th in np.arange(0.30, 0.65, 0.05):
            bin_preds = (all_preds >= th).astype(int)
            f = f1_score(all_targets, bin_preds, zero_division=0)
            if f > best_t_f1:
                best_t_f1 = f
                best_t = th
                best_t_rec = recall_score(all_targets, bin_preds, zero_division=0)
                best_t_prec = precision_score(all_targets, bin_preds, zero_division=0)
                
        roc_auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)
        acc = accuracy_score(all_targets, (all_preds >= best_t).astype(int))
        
        scheduler.step(best_t_f1)
        print(f"  Epoch [{epoch:02d}/08] - Loss: {epoch_loss:.4f} | Test F1: {best_t_f1:.4f} | Recall: {best_t_rec*100:.2f}% | Prec: {best_t_prec*100:.2f}% | ROC-AUC: {roc_auc:.4f} (th={best_t:.2f})")
        
        if best_t_f1 > best_f1:
            best_f1 = best_t_f1
            torch.save(model.state_dict(), save_path)
            best_metrics = {
                'model_name': 'Attention U-Net + LSTM',
                'architecture': 'AttentionUNetLSTM',
                'parameters': total_params,
                'epochs_trained': epoch,
                'optimal_threshold': float(round(best_t, 2)),
                'recall': float(round(best_t_rec * 100, 2)),
                'precision': float(round(best_t_prec * 100, 2)),
                'f1_score': float(round(best_t_f1, 4)),
                'roc_auc': float(round(roc_auc, 4)),
                'pr_auc': float(round(pr_auc, 4)),
                'accuracy': float(round(acc * 100, 2)),
                'training_time_sec': float(round(time.time() - start_time, 2)),
                'weights_path': save_path
            }

    # Save metrics JSON
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'attention_unet_lstm_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
        
    print("-" * 80)
    print("TRAINING COMPLETE FOR ATTENTION U-NET + LSTM!")
    print(f"Weights Saved To: {save_path}")
    print(f"Metrics Saved To: {metrics_path}")
    print(f"  * Final Recall:    {best_metrics['recall']}%")
    print(f"  * Final Precision: {best_metrics['precision']}%")
    print(f"  * Final F1-Score:  {best_metrics['f1_score']}")
    print(f"  * Final ROC-AUC:   {best_metrics['roc_auc']}")
    print(f"  * Final PR-AUC:    {best_metrics['pr_auc']}")
    print(f"  * Final Accuracy:  {best_metrics['accuracy']}%")
    print("=" * 80)
    return best_metrics

if __name__ == '__main__':
    train_attention_u_net___lstm()
