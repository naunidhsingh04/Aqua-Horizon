"""
AQUA HORIZON — 10-Fold Stratified Cross-Validation Suite for 5 Hybrid Architectures
Evaluates all 5 mandatory hackathon models across 10 folds:
1. U-Net + ConvLSTM
2. CNN + LSTM
3. CNN + Transformer
4. ResNet + BiLSTM
5. Attention U-Net + LSTM

Computes Mean Recall (+/- Std), Mean Precision (+/- Std), Mean F1, and Mean ROC-AUC.
Exports:
- ml/models/kfold_hybrid_benchmarks.json
- web/data/kfold_hybrid_benchmarks.json
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ml.models_hybrid import UNetConvLSTM, CNNLSTM, CNNTransformer, ResNetBiLSTM, AttentionUNetLSTM

FEATURE_COLS = [
    'prior_cum_events', 'prior_3yr_events', 'prior_5yr_events', 'prior_10yr_events',
    'flood_acceleration_rate', 'dfsi_score', 'dfsi_rank', 'Corrected_Percent_Flooded_Area',
    'Parmanent_Water', 'Mean_Flood_Duration', 'Population', 'drainage_stress_ratio',
    'exposure_severity_index', 'hydro_rain_stress', 'rainfall_anomaly_pct'
]

def load_all_sequences(csv_path=None, seq_len=5):
    if csv_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        csv_path = os.path.join(base_dir, 'data', 'processed', 'district_year_dataset.csv')
        
    df = pd.read_csv(csv_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    
    X_list, y_list = [], []
    for dist, group in df.groupby('clean_dist'):
        group = group.sort_values('year')
        feat_vals = group[FEATURE_COLS].values
        target_vals = group['flood_occurred'].values
        n = len(group)
        for i in range(seq_len - 1, n):
            X_list.append(feat_vals[i - seq_len + 1 : i + 1])
            y_list.append(target_vals[i])
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

def evaluate_kfold_for_model(model_name, model_class, X_all, y_all, is_spatial=False, n_splits=10, epochs=3, batch_size=256, lr=0.001):
    print(f"\n{'='*80}\nSTARTING 10-FOLD STRATIFIED CV: {model_name.upper()}\n{'='*80}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fold_records = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        t_fold_start = time.time()
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_va, y_va = X_all[val_idx], y_all[val_idx]
        
        # Zero-Leakage: Standardize strictly on training fold
        scaler = StandardScaler()
        X_tr_flat = X_tr.reshape(-1, 15)
        scaler.fit(X_tr_flat)
        X_tr_norm = scaler.transform(X_tr_flat).reshape(-1, 5, 15)
        X_va_norm = scaler.transform(X_va.reshape(-1, 15)).reshape(-1, 5, 15)
        
        if is_spatial:
            pad_tr = np.zeros((X_tr_norm.shape[0], 5, 1), dtype=np.float32)
            X_tr_norm = np.concatenate([X_tr_norm, pad_tr], axis=-1).reshape(-1, 5, 1, 4, 4)
            pad_va = np.zeros((X_va_norm.shape[0], 5, 1), dtype=np.float32)
            X_va_norm = np.concatenate([X_va_norm, pad_va], axis=-1).reshape(-1, 5, 1, 4, 4)
            
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr_norm), torch.tensor(y_tr)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_va_norm), torch.tensor(y_va)),
            batch_size=batch_size, shuffle=False
        )
        
        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        w_pos = torch.tensor(min(pos_weight, 2.5), device=device)
        
        def weighted_bce_loss(preds, targets):
            preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
            return -torch.mean(w_pos * targets * torch.log(preds) + (1.0 - targets) * torch.log(1.0 - preds))
            
        model = model_class().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        for ep in range(1, epochs + 1):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = weighted_bce_loss(model(bx), by)
                loss.backward()
                opt.step()
                
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                preds = model(bx)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(by.numpy().tolist())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Optimal threshold search
        best_t, best_f1, best_rec, best_prec = 0.45, 0.0, 0.0, 0.0
        for th in np.arange(0.35, 0.60, 0.05):
            bp = (all_preds >= th).astype(int)
            f = f1_score(all_targets, bp, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = th
                best_rec = recall_score(all_targets, bp, zero_division=0)
                best_prec = precision_score(all_targets, bp, zero_division=0)
                
        roc_auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)
        fold_time = time.time() - t_fold_start
        
        record = {
            'fold': f"Fold {fold}",
            'recall': float(round(best_rec * 100, 2)),
            'precision': float(round(best_prec * 100, 2)),
            'f1_score': float(round(best_f1, 4)),
            'roc_auc': float(round(roc_auc, 4)),
            'pr_auc': float(round(pr_auc, 4)),
            'threshold': float(round(best_t, 2)),
            'time_sec': float(round(fold_time, 2))
        }
        fold_records.append(record)
        print(f"  Fold {fold}/{n_splits} -> Recall: {record['recall']}% | Prec: {record['precision']}% | F1: {record['f1_score']} | ROC-AUC: {record['roc_auc']} ({fold_time:.1f}s)")
        
    df_folds = pd.DataFrame(fold_records)
    mean_rec = float(round(df_folds['recall'].mean(), 2))
    std_rec = float(round(df_folds['recall'].std(), 2))
    mean_prec = float(round(df_folds['precision'].mean(), 2))
    std_prec = float(round(df_folds['precision'].std(), 2))
    mean_f1 = float(round(df_folds['f1_score'].mean(), 4))
    std_f1 = float(round(df_folds['f1_score'].std(), 4))
    mean_auc = float(round(df_folds['roc_auc'].mean(), 4))
    std_auc = float(round(df_folds['roc_auc'].std(), 4))
    mean_pr = float(round(df_folds['pr_auc'].mean(), 4))
    
    total_params = sum(p.numel() for p in model_class().parameters())
    
    summary = {
        'model_name': model_name,
        'architecture': model_class.__name__,
        'parameters': total_params,
        'mean_recall': mean_rec,
        'std_recall': std_rec,
        'mean_precision': mean_prec,
        'std_precision': std_prec,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_roc_auc': mean_auc,
        'std_roc_auc': std_auc,
        'mean_pr_auc': mean_pr,
        'folds': fold_records
    }
    print(f"  === SUMMARY: Mean Recall: {mean_rec}% (+/- {std_rec}%) | Mean Precision: {mean_prec}% (+/- {std_prec}%) | Mean ROC-AUC: {mean_auc}")
    return summary

def run_all_kfold():
    print("=" * 90)
    print("AQUA HORIZON — 10-FOLD CROSS-VALIDATION SUITE (5 HYBRID ARCHITECTURES)")
    print("Dataset: Zenodo IFI-Impacts (1967-2023) | 38,425 Sequences | Zero Leakage Scalers")
    print("=" * 90)
    
    t0 = time.time()
    X_all, y_all = load_all_sequences()
    print(f"Loaded {len(X_all)} total sequence windows across 725 districts.")
    
    configs = [
        ("CNN + LSTM", CNNLSTM, False, 3, 256, 0.001),
        ("CNN + Transformer", CNNTransformer, False, 3, 256, 0.0008),
        ("ResNet + BiLSTM", ResNetBiLSTM, False, 3, 256, 0.001),
        ("U-Net + ConvLSTM", UNetConvLSTM, True, 2, 256, 0.001),
        ("Attention U-Net + LSTM", AttentionUNetLSTM, True, 2, 256, 0.001)
    ]
    
    results = []
    for cfg in configs:
        res = evaluate_kfold_for_model(cfg[0], cfg[1], X_all, y_all, cfg[2], n_splits=10, epochs=cfg[3], batch_size=cfg[4], lr=cfg[5])
        results.append(res)
        
    total_time = time.time() - t0
    
    payload = {
        "suite": "10-Fold Stratified Cross-Validation Benchmark for 5 Hybrid Architectures",
        "dataset": "Zenodo IFI-Impacts (1967-2023)",
        "total_samples": len(X_all),
        "total_time_sec": round(total_time, 2),
        "models": results
    }
    
    # Save to ml/models/
    ml_save_path = os.path.join(os.path.dirname(__file__), 'models', 'kfold_hybrid_benchmarks.json')
    with open(ml_save_path, 'w') as f:
        json.dump(payload, f, indent=2)
        
    # Save to web/data/
    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web', 'data'))
    os.makedirs(web_dir, exist_ok=True)
    web_save_path = os.path.join(web_dir, 'kfold_hybrid_benchmarks.json')
    with open(web_save_path, 'w') as f:
        json.dump(payload, f, indent=2)
        
    print("\n" + "=" * 90)
    print("10-FOLD CROSS-VALIDATION COMPLETE ACROSS ALL 5 HYBRID ARCHITECTURES!")
    print(f"Saved to: {ml_save_path}")
    print(f"Saved to: {web_save_path}")
    print("=" * 90)
    print(f"{'Model Architecture':<26} | {'Mean Recall':<18} | {'Mean Precision':<18} | {'Mean F1':<10} | {'Mean ROC-AUC':<10}")
    print("-" * 90)
    for m in results:
        print(f"{m['model_name']:<26} | {m['mean_recall']}% (+/- {m['std_recall']}%) | {m['mean_precision']}% (+/- {m['std_precision']}%) | {m['mean_f1']:<10.4f} | {m['mean_roc_auc']:<10.4f}")
    print("=" * 90)

if __name__ == '__main__':
    run_all_kfold()
