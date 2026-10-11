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

# the 15 columns of district_year_dataset.csv that the deep-learning models read for each year
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

TARGET_COL = 'flood_occurred'   # what the models predict: 1 = a flood happened that year

def load_sequence_data(csv_path=None, seq_len=5, cutoff_year=2012, spatial=False, batch_size=128, balance_smote=False):
    # Returns train/test loaders. Each sample = 5 consecutive years of 15 features for one district,
    # labelled with whether it flooded in the 5th year. Windows ending <= cutoff_year -> train,
    # later ones -> test (chronological split, no random shuffling across time).
    if csv_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        csv_path = os.path.join(base_dir, 'data', 'processed', 'district_year_dataset.csv')
        
    df = pd.read_csv(csv_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Chronological Scaler (Zero Information Leakage from test set)
    # Standardise each feature as z = (x - mean) / std, using mean/std from training years only
    train_mask = df['year'] <= cutoff_year
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATURE_COLS].values)
    
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS].values)
    
    # build the windows district by district, so a window never mixes years of two different districts
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
    
    # Optional: add synthetic flood windows (SMOTE) to the TRAINING set only
    if balance_smote:
        from imblearn.over_sampling import SMOTE
        n_samples, s_len, n_feats = X_train.shape
        X_flat = X_train.reshape(n_samples, s_len * n_feats)
        smote = SMOTE(random_state=42)
        X_flat_res, y_train = smote.fit_resample(X_flat, y_train)
        X_train = X_flat_res.reshape(-1, s_len, n_feats).astype(np.float32)
        y_train = y_train.astype(np.float32)
    
    # For U-Net models: pad 15 features with one 0 -> 16 values -> reshape as a 4x4 single-channel "image"
    if spatial:
        pad_train = np.zeros((X_train.shape[0], seq_len, 1), dtype=np.float32)
        X_train_padded = np.concatenate([X_train, pad_train], axis=-1)
        X_train = X_train_padded.reshape(-1, seq_len, 1, 4, 4)
        
        pad_test = np.zeros((X_test.shape[0], seq_len, 1), dtype=np.float32)
        X_test_padded = np.concatenate([X_test, pad_test], axis=-1)
        X_test = X_test_padded.reshape(-1, seq_len, 1, 4, 4)

    # wrap the arrays as PyTorch tensors / loaders; training batches are shuffled, test batches keep their order
    t_X_train = torch.tensor(X_train)
    t_y_train = torch.tensor(y_train)
    t_X_test = torch.tensor(X_test)
    t_y_test = torch.tensor(y_test)
    
    train_dataset = TensorDataset(t_X_train, t_y_train)
    test_dataset = TensorDataset(t_X_test, t_y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # pos_weight = (#non-flood / #flood) in training data; trainers use it to weight missed floods
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = float(num_neg / max(num_pos, 1))
    
    # everything a training script needs, in one dictionary
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


if __name__ == '__main__':
    # Self-test: build the 5-year windows and show what the deep-learning models will be trained on.
    print("=" * 70)
    print("AQUA HORIZON - SEQUENCE DATA LOADER (self-test)")
    print("=" * 70)
    d = load_sequence_data(seq_len=5, cutoff_year=2012, spatial=False, batch_size=128)
    n_tr, n_te = len(d['X_train']), len(d['X_test'])
    print(f"Window shape (years x features): {tuple(d['X_train'].shape[1:])}")
    print(f"Train windows (ending <= 2012): {n_tr:,}  | flood share: {d['y_train'].mean()*100:.1f}%")
    print(f"Test windows  (ending 2013-2023): {n_te:,} | flood share: {d['y_test'].mean()*100:.1f}%")
    print(f"Train : test ratio = {n_tr/(n_tr+n_te)*100:.0f} : {n_te/(n_tr+n_te)*100:.0f}")
    print(f"Class weight for missed floods (non-flood / flood): {d['pos_weight']:.2f}")
    print(f"Features used ({len(d['feature_cols'])}): {', '.join(d['feature_cols'])}")
    xb, yb = next(iter(d['train_loader']))
    print(f"One training batch: inputs {tuple(xb.shape)}, labels {tuple(yb.shape)}")
    ds = load_sequence_data(seq_len=5, cutoff_year=2012, spatial=True, batch_size=128)
    print(f"Spatial version for the U-Net models: {tuple(ds['X_train'].shape[1:])} (years, channels, 4, 4)")
