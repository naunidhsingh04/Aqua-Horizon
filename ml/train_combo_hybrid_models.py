"""
AQUA HORIZON - Combined (ensemble) hybrid models built from the TOP 3 deep-learning models.

Top 3 (ranked by 10-fold F1, ROC-AUC and accuracy in ml/models/kfold_hybrid_benchmarks.json):
    1st ResNet + BiLSTM   2nd CNN + LSTM   3rd CNN + Transformer

Combinations (soft voting = average of the members' flood probabilities):
    first 2   : ResNet+BiLSTM + CNN+LSTM
    last 2    : CNN+LSTM + CNN+Transformer
    first+last: ResNet+BiLSTM + CNN+Transformer
    all 3     : ResNet+BiLSTM + CNN+LSTM + CNN+Transformer
The 3 single models are re-scored under the SAME protocol, so singles and combos are compared fairly.

Protocol (no information from test data is ever used for training or tuning):
  1. 80:20 chronological split (windows ending <= 2012 train, later years test)  AND
     10-fold stratified cross-validation.
  2. Inside every training part: scale features (mean/std from training data only), then balance the WHOLE
     training part to 50:50 with SMOTE (synthetic flood rows). The test part is never balanced.
  3. Because the networks are trained on 50:50 data, a window is called flood when P(flood) >= 0.5 (fixed,
     nothing is tuned on any held-out data).
  4. Accuracy, recall, precision, F1, ROC-AUC, PR-AUC are measured on the untouched, real test data.

Outputs: ml/models/combo_hybrid_benchmarks.json, web/data/combo_hybrid_benchmarks.json,
         ml/models/combo_district_predictions.csv (per-district probabilities used by the website)
Run:  python ml/train_combo_hybrid_models.py        (add --quick for a fast smoke test)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, average_precision_score)

from ml.models_hybrid import CNNLSTM, CNNTransformer, ResNetBiLSTM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEQ_LEN, N_FEATS, SEED = 5, 15, 42
THRESHOLD = 0.5   # models are trained on 50:50 data, so 0.5 is the natural flood cut-off

FEATURE_COLS = [
    'prior_cum_events', 'prior_3yr_events', 'prior_5yr_events', 'prior_10yr_events',
    'flood_acceleration_rate', 'dfsi_score', 'dfsi_rank', 'Corrected_Percent_Flooded_Area',
    'Parmanent_Water', 'Mean_Flood_Duration', 'Population', 'drainage_stress_ratio',
    'exposure_severity_index', 'hydro_rain_stress', 'rainfall_anomaly_pct'
]

# The top 3 single models, in rank order: key -> (display name, class, learning rate, epochs)
TOP3 = {
    'resnet_bilstm':   ('ResNet + BiLSTM',   ResNetBiLSTM,   0.001,  3),
    'cnn_lstm':        ('CNN + LSTM',        CNNLSTM,        0.001,  3),
    'cnn_transformer': ('CNN + Transformer', CNNTransformer, 0.0008, 3),
}

# Every model that gets scored: key -> (display name, type, member keys)
CANDIDATES = {k: (v[0], 'single', [k]) for k, v in TOP3.items()}
CANDIDATES['combo_resnet_cnnlstm'] = ('ResNet+BiLSTM + CNN+LSTM', 'pair', ['resnet_bilstm', 'cnn_lstm'])
CANDIDATES['combo_cnnlstm_cnntransformer'] = ('CNN+LSTM + CNN+Transformer', 'pair', ['cnn_lstm', 'cnn_transformer'])
CANDIDATES['combo_resnet_cnntransformer'] = ('ResNet+BiLSTM + CNN+Transformer', 'pair', ['resnet_bilstm', 'cnn_transformer'])
CANDIDATES['combo_top3'] = ('ResNet+BiLSTM + CNN+LSTM + CNN+Transformer', 'triple', list(TOP3))


def load_windows(df):
    """Sliding 5-year windows per district: X (N,5,15), label = flood in the last year, plus that year."""
    X_list, y_list, yr_list = [], [], []
    for _, g in df.groupby('clean_dist'):
        g = g.sort_values('year')
        feats, tgt, yrs = g[FEATURE_COLS].values, g['flood_occurred'].values, g['year'].values
        for i in range(SEQ_LEN - 1, len(g)):
            X_list.append(feats[i - SEQ_LEN + 1: i + 1])
            y_list.append(tgt[i])
            yr_list.append(yrs[i])
    return (np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32),
            np.array(yr_list))


def latest_district_windows(df):
    """Each district's most recent 5 years (2019-2023) -> used for the website's annual probabilities."""
    names, wins, labels = [], [], []
    for cd, g in df.groupby('clean_dist'):
        g = g.sort_values('year').tail(SEQ_LEN)
        if len(g) == SEQ_LEN:
            names.append((cd, g['Dist_Name'].iloc[-1]))
            wins.append(g[FEATURE_COLS].values)
    return names, np.array(wins, dtype=np.float32)


def scale(scaler, X):
    """z = (x - mean) / std applied to every feature of every year of every window."""
    return scaler.transform(X.reshape(-1, N_FEATS)).reshape(-1, SEQ_LEN, N_FEATS).astype(np.float32)


def train_one(model_cls, lr, epochs, X, y, seed, batch_size=256):
    """Train one network on (already balanced) data with plain binary cross-entropy."""
    torch.manual_seed(seed)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            p = torch.clamp(model(Xt[idx]), 1e-7, 1 - 1e-7)
            # loss = -mean( y*log(p) + (1-y)*log(1-p) )  (data is balanced, so no class weight needed)
            F.binary_cross_entropy(p, yt[idx]).backward()
            opt.step()
    return model


def predict(model, X, batch_size=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.tensor(X[i:i + batch_size])).numpy())
    return np.concatenate(out)


def score(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        'accuracy': accuracy_score(y, pred) * 100,   # (TP+TN)/all
        'recall': recall_score(y, pred, zero_division=0) * 100,       # TP/(TP+FN)
        'precision': precision_score(y, pred, zero_division=0) * 100,  # TP/(TP+FP)
        'f1_score': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, p),
        'pr_auc': average_precision_score(y, p),
        'threshold': thr,
    }


def run_split(X_tr, y_tr, X_te, y_te, seed, district_X=None):
    """
    Train the 3 base models on one train/test split and score all 7 candidates.
    Returns {candidate key: metrics} (+ per-district probabilities of each base model if district_X given).
    """
    # scale with training-data statistics only
    scaler = StandardScaler().fit(X_tr.reshape(-1, N_FEATS))
    X_tr, X_te = scale(scaler, X_tr), scale(scaler, X_te)

    # SMOTE: add synthetic flood windows to the WHOLE training part until flood : non-flood is 50 : 50
    n = len(X_tr)
    X_bal, y_bal = SMOTE(k_neighbors=5, random_state=seed).fit_resample(X_tr.reshape(n, -1), y_tr)
    X_bal = X_bal.reshape(-1, SEQ_LEN, N_FEATS).astype(np.float32)
    y_bal = y_bal.astype(np.float32)

    p_te, p_dist = {}, {}
    for key, (_, cls, lr, epochs) in TOP3.items():
        model = train_one(cls, lr, epochs, X_bal, y_bal, seed)
        p_te[key] = predict(model, X_te)
        if district_X is not None:
            p_dist[key] = predict(model, scale(scaler, district_X))

    results = {}
    for ckey, (_, _, members) in CANDIDATES.items():
        pt = np.mean([p_te[m] for m in members], axis=0)    # soft voting on the real test data
        results[ckey] = score(y_te, pt, THRESHOLD)
    return results, p_dist


def summarise(fold_rows):
    df = pd.DataFrame(fold_rows)
    out = {}
    for m in ['accuracy', 'recall', 'precision', 'f1_score', 'roc_auc', 'pr_auc']:
        out[f'mean_{m}'] = float(round(df[m].mean(), 4))
        out[f'std_{m}'] = float(round(df[m].std(), 4))
    return out


def main(quick=False):
    t0 = time.time()
    print("=" * 90)
    print("AQUA HORIZON - TOP-3 HYBRID MODEL COMBINATIONS (80:20 HOLDOUT + 10-FOLD CV, SMOTE IN TRAINING ONLY)")
    print("=" * 90)

    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'district_year_dataset.csv'))
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    X, y, years = load_windows(df)
    names, dist_X = latest_district_windows(df)
    n_splits = 10
    if quick:   # tiny run just to check the code works
        rng = np.random.RandomState(SEED)
        keep = rng.choice(len(X), 4000, replace=False)
        X, y, years = X[keep], y[keep], years[keep]
        n_splits = 2
        for k in TOP3:
            TOP3[k] = TOP3[k][:3] + (1,)
    print(f"{len(X):,} five-year windows | {int(y.sum()):,} flood ({y.mean()*100:.1f}%) | {len(names)} districts")

    # ---- A. 80:20 chronological holdout (train: windows ending <= 2012, test: 2013-2023) ----
    print("\n[A] 80:20 chronological split ...")
    tr, te = years <= 2012, years > 2012
    print(f"    train {tr.sum():,} | test {te.sum():,}")
    holdout, _ = run_split(X[tr], y[tr], X[te], y[te], SEED)

    # ---- B. 10-fold stratified cross-validation ----
    print("\n[B] 10-fold stratified cross-validation ...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_rows = {k: [] for k in CANDIDATES}
    dist_sum = {k: np.zeros(len(names)) for k in TOP3}
    for fold, (a, b) in enumerate(skf.split(X, y), 1):
        tf = time.time()
        res, p_dist = run_split(X[a], y[a], X[b], y[b], SEED + fold, district_X=dist_X)
        for ckey, r in res.items():
            fold_rows[ckey].append({'fold': f'Fold {fold}', **{k: float(round(v, 4)) for k, v in r.items()}})
        for k in TOP3:
            dist_sum[k] += p_dist[k]
        print(f"    fold {fold}/{n_splits}  acc: ResNet {res['resnet_bilstm']['accuracy']:.1f}% | "
              f"first-2 combo {res['combo_resnet_cnnlstm']['accuracy']:.1f}% | "
              f"all-3 combo {res['combo_top3']['accuracy']:.1f}%  ({time.time()-tf:.0f}s)")

    # ---- C. assemble results ----
    models = []
    for ckey, (name, ctype, members) in CANDIDATES.items():
        h = {k: float(round(v, 4)) for k, v in holdout[ckey].items()}
        models.append({
            'key': ckey, 'model_name': name, 'type': ctype, 'members': members,
            'holdout': h,
            'kfold': {**summarise(fold_rows[ckey]), 'folds': fold_rows[ckey]},
        })

    payload = {
        'suite': 'Top-3 hybrid models and their combinations',
        'top3_ranked': [TOP3[k][0] for k in TOP3],
        'protocol': {
            'holdout': '80:20 chronological split (train: windows ending <=2012, test: 2013-2023)',
            'cross_validation': f'{n_splits}-fold stratified',
            'balancing': 'SMOTE (k=5) applied to the whole training part (80% in the holdout, 90% in each CV fold); test data is real and never balanced',
            'threshold': 'fixed 0.5 (training data is balanced 50:50); nothing is tuned on held-out data',
            'combination': 'soft voting (average of member probabilities)',
        },
        'total_windows': int(len(X)),
        'runtime_sec': round(time.time() - t0, 1),
        'models': models,
    }
    save_dirs = [] if quick else [os.path.join(BASE_DIR, 'ml', 'models'), os.path.join(BASE_DIR, 'web', 'data')]
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'combo_hybrid_benchmarks.json'), 'w') as f:
            json.dump(payload, f, indent=2)

    # ---- D. per-district probabilities (average of the fold models) for the website ----
    cols = {k: dist_sum[k] / n_splits for k in TOP3}
    out = pd.DataFrame({'clean_dist': [n[0] for n in names], 'Dist_Name': [n[1] for n in names]})
    for ckey, (_, ctype, members) in CANDIDATES.items():
        if ctype != 'single':
            out[ckey] = np.round(np.mean([cols[m] for m in members], axis=0), 4)
    if not quick:   # the smoke test never overwrites real result files
        out.to_csv(os.path.join(BASE_DIR, 'ml', 'models', 'combo_district_predictions.csv'), index=False)

    # ---- E. summary ----
    print("\n" + "=" * 100)
    print(f"{'Model':<54}| {'Holdout Acc':<12}| {'10-Fold Acc (+/- std)':<24}| Recall | Prec  | F1     | ROC-AUC")
    print("-" * 100)
    for m in models:
        k = m['kfold']
        print(f"{m['model_name']:<54}| {m['holdout']['accuracy']:>9.2f}%  | "
              f"{k['mean_accuracy']:>8.2f}% (+/- {k['std_accuracy']:.2f}%)  | {k['mean_recall']:>5.1f}% | "
              f"{k['mean_precision']:>4.1f}% | {k['mean_f1_score']:.4f} | {k['mean_roc_auc']:.4f}")
    print("=" * 100)
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='tiny smoke test (2 folds, 1 epoch, 4000 windows)')
    main(ap.parse_args().quick)
