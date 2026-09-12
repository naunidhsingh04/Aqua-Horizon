"""
AQUA HORIZON — Hybrid Sequence Data Loader
Constructs multi-year sliding window sequences (T=5) for 725 Indian districts (1967-2023)
from Zenodo IFI-Impacts dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

FEATURE_COLS = [
    'prior_cum_events',
    'prior_3yr_events',
    'prior_5yr_events',
    'prior_10yr_events',
    'flood_acceleration_rate',
    'dfsi_score',
    'dfsi_rank',
    'Corrected_Percent_Flooded_Area',
    'Parmanent_Water',
    'Mean_Flood_Duration',
    'Population',
    'drainage_stress_ratio',
    'exposure_severity_index',
    'hydro_rain_stress',
    'rainfall_anomaly_pct'
]

TARGET_COL = 'flood_occurred'

def load_sequence_data(csv_path=None, seq_len=5, cutoff_year=2012, spatial=False, batch_size=128, balance_smote=False):
    if csv_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        csv_path = os.path.join(base_dir, 'data', 'processed', 'district_year_dataset.csv')
        
    df = pd.read_csv(csv_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Chronological Scaler (Zero Information Leakage from test set)
    train_mask = df['year'] <= cutoff_year
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATURE_COLS].values)
    
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS].values)
    
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for dist, group in df_scaled.groupby('clean_dist'):
        group = group.sort_values('year')
        feat_vals = group[FEATURE_COLS].values
        target_vals = group[TARGET_COL].values
        years = group['year'].values
        
        n = len(group)
        for i in range(seq_len - 1, n):
            seq_x = feat_vals[i - seq_len + 1 : i + 1]  # shape (seq_len, 15)
            seq_y = target_vals[i]
            yr = years[i]
            
            if yr <= cutoff_year:
                X_train_list.append(seq_x)
                y_train_list.append(seq_y)
            else:
                X_test_list.append(seq_x)
                y_test_list.append(seq_y)
                
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)
    
    if balance_smote:
        from imblearn.over_sampling import SMOTE
        n_samples, s_len, n_feats = X_train.shape
        X_flat = X_train.reshape(n_samples, s_len * n_feats)
        smote = SMOTE(random_state=42)
        X_flat_res, y_train = smote.fit_resample(X_flat, y_train)
        X_train = X_flat_res.reshape(-1, s_len, n_feats).astype(np.float32)
        y_train = y_train.astype(np.float32)
    
    if spatial:
        pad_train = np.zeros((X_train.shape[0], seq_len, 1), dtype=np.float32)
        X_train_padded = np.concatenate([X_train, pad_train], axis=-1)
        X_train = X_train_padded.reshape(-1, seq_len, 1, 4, 4)
        
        pad_test = np.zeros((X_test.shape[0], seq_len, 1), dtype=np.float32)
        X_test_padded = np.concatenate([X_test, pad_test], axis=-1)
        X_test = X_test_padded.reshape(-1, seq_len, 1, 4, 4)

    t_X_train = torch.tensor(X_train)
    t_y_train = torch.tensor(y_train)
    t_X_test = torch.tensor(X_test)
    t_y_test = torch.tensor(y_test)
    
    train_dataset = TensorDataset(t_X_train, t_y_train)
    test_dataset = TensorDataset(t_X_test, t_y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = float(num_neg / max(num_pos, 1))
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': FEATURE_COLS,
        'pos_weight': pos_weight
    }
