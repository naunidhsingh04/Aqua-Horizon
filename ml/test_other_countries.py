"""
Tests the India-trained Sentinel-1 flood model on other countries (never seen in training).

  Train : Sen1Floods11 India chips only (CLAHE + DCGAN augmentation, same recipe as image_pipeline_india.py)
  Test  : USA, Pakistan, Sri-Lanka chips from the same Sen1Floods11 hand-labelled set
  Task  : for each 64x64 dual-pol (VV, VH) chip, predict flood / no flood.
          Ground truth = the chip has >= 10% water pixels in the hand-labelled mask.

Nothing from the other countries is used for training or for choosing the best epoch
(the epoch is chosen on a held-out 20% of India only), so these scores are a fair
"does it work in a new country?" test. Expect them to be lower than the India result.

Outputs (the website is NOT touched):
  ml/models/other_countries_metrics.json      scores per country
  ml/models/other_countries_predictions.csv   one row per test chip: input stats, truth, probability, prediction
  ml/models/other_countries_samples/<country>.png   sample inputs (VV, VH, true water mask) with the model output

Run:  python ml/download_sen1floods11_india.py India Pakistan Sri-Lanka USA   (once)
      python ml/test_other_countries.py
"""
import os
import sys
import json
import time
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, jaccard_score, confusion_matrix)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'ml'))
import image_pipeline_india as ip   # reuses the loader, CLAHE, DCGAN and the classifier network

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = ip.device
TEST_COUNTRIES = ['USA', 'Pakistan', 'Sri-Lanka']
OUT_DIR = os.path.join(ROOT, 'ml', 'models')
SAMPLE_DIR = os.path.join(OUT_DIR, 'other_countries_samples')


# Trains SARFloodNet on India chips; the best epoch is picked on the India validation part only.
def train_on_india(X_tr, y_tr, X_val, y_val, epochs=6, lr=0.001, wd=1e-4):
    tr_loader = DataLoader(ip.SARImageDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    model = ip.SARFloodNet(in_channels=2, num_classes=2, dropout_p=0.3).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        probs = predict_proba(model, X_val)
        acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
        print(f"    epoch {ep + 1}/{epochs}  India-validation accuracy {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


# P(flood) for every chip
def predict_proba(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in DataLoader(ip.SARImageDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=64):
            out.append(torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def score(y, probs):
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y.tolist())) == 2
    return {
        'chips': int(len(y)),
        'flood_chips': int(y.sum()),
        'flood_share_pct': round(float(y.mean() * 100), 1),
        'accuracy': round(accuracy_score(y, pred) * 100, 2),
        'recall': round(recall_score(y, pred, zero_division=0) * 100, 2),
        'precision': round(precision_score(y, pred, zero_division=0) * 100, 2),
        'f1_score': round(f1_score(y, pred, zero_division=0), 4),
        'iou': round(jaccard_score(y, pred, zero_division=0) * 100, 2),
        'roc_auc': round(roc_auc_score(y, probs), 4) if both else None,
        'confusion': {'true_flood_found': int(tp), 'missed_flood': int(fn),
                      'false_alarm': int(fp), 'true_no_flood': int(tn)},
        # a model that always answers "no flood" would score this accuracy:
        'always_no_flood_accuracy': round(float(1 - y.mean()) * 100, 2),
    }


# Picture of sample test chips: VV | VH | true water mask, with truth and model output underneath
def save_samples(country, raw, masks, y, probs, n_each=4):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    pick = []
    for cls in (1, 0):
        idx = np.where(y == cls)[0]
        if len(idx):
            pick += list(rng.choice(idx, size=min(n_each, len(idx)), replace=False))
    scale, pad = 3, 6
    tiles = []
    for i in pick:
        def gray(ch):
            c = raw[i][:, :, ch]
            c = ((c - c.min()) / max(c.max() - c.min(), 1e-5) * 255).astype(np.uint8)
            return cv2.cvtColor(cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        m = cv2.resize(masks[i].astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        m = np.where(m[:, :, None] == 1, np.array([220, 140, 30], np.uint8), np.array([70, 70, 70], np.uint8)).astype(np.uint8)  # water = blue, land = dark grey (BGR)
        sep = np.full((gray(0).shape[0], pad, 3), 255, np.uint8)
        row = np.hstack([gray(0), sep, gray(1), sep, m])
        pred = int(probs[i] >= 0.5)
        ok = pred == int(y[i])
        banner = np.full((34, row.shape[1], 3), 255, np.uint8)
        text = f"truth: {'FLOOD' if y[i] else 'no flood'} | model: {'FLOOD' if pred else 'no flood'} ({probs[i] * 100:.0f}%)"
        cv2.putText(banner, text, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0) if ok else (0, 0, 220), 1, cv2.LINE_AA)
        tiles.append(np.vstack([row, banner]))
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack([tiles[r * cols + c] if c == 0 else np.hstack([np.full((tiles[0].shape[0], 12, 3), 255, np.uint8), tiles[r * cols + c]]) for c in range(cols)])
            for r in range(len(tiles) // cols)]
    title = np.full((36, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"{country}: columns = VV radar | VH radar | true water (blue) on land (grey). Green text = correct, red = wrong",
                (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(SAMPLE_DIR, f"{country}.png"), np.vstack([title] + rows))


def main():
    t0 = time.time()
    print("=" * 80)
    print("AQUA HORIZON - TRAIN ON INDIA, TEST ON USA / PAKISTAN / SRI-LANKA (SENTINEL-1)")
    print("=" * 80)

    # ---- training data: India only ----
    raw, masks, labels = ip.load_sen1floods11_india_chips(patch_size=64, country='India')
    clahe = ip.apply_clahe_sar(raw)
    tr, va = train_test_split(np.arange(len(labels)), test_size=0.20, random_state=SEED, stratify=labels)
    print(f"[*] India: {len(tr)} train chips, {len(va)} held-out chips")

    print("\n[GAN] DCGAN sees only India training flood chips")
    gen = ip.train_sar_dcgan(clahe[tr][labels[tr] == 1], epochs=10, batch_size=32)
    X_tr, y_tr = ip.augment_with_gan(gen, clahe[tr], labels[tr], n_synthetic=100)

    print("\n[Train] SARFloodNet on India")
    model = train_on_india(X_tr, y_tr, clahe[va], labels[va])
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'sar_flood_net_india.pt'))

    results = {'India (held-out 20%, same country)': score(labels[va], predict_proba(model, clahe[va]))}
    rows = []

    # ---- test on the other countries ----
    for country in TEST_COUNTRIES:
        folder = os.path.join(ROOT, 'data', 'raw', 'sen1floods11_' + country.lower().replace('-', '_'), 'LabelHand')
        if not os.path.isdir(folder) or not os.listdir(folder):
            print(f"[!] {country}: data missing, run: python ml/download_sen1floods11_india.py {country}")
            continue
        c_raw, c_masks, c_y = ip.load_sen1floods11_india_chips(patch_size=64, max_chips=None, country=country)
        c_x = ip.apply_clahe_sar(c_raw)
        probs = predict_proba(model, c_x)
        results[country] = score(c_y, probs)
        save_samples(country, c_raw, c_masks, c_y, probs)
        for i in range(len(c_y)):
            rows.append((country, i, round(float(c_masks[i].mean()) * 100, 2), int(c_y[i]), round(float(probs[i]), 4), int(probs[i] >= 0.5)))

    # ---- report ----
    print("\n" + "=" * 100)
    print(f"{'Test set':<38}{'Chips':>7}{'Flood%':>8}{'Acc':>8}{'Recall':>8}{'Prec':>8}{'F1':>8}{'IoU':>8}{'AUC':>8}{'All-no':>8}")
    print("-" * 100)
    for name, r in results.items():
        auc = f"{r['roc_auc']:.3f}" if r['roc_auc'] is not None else 'n/a'
        print(f"{name:<38}{r['chips']:>7}{r['flood_share_pct']:>8}{r['accuracy']:>7.1f}%{r['recall']:>7.1f}%{r['precision']:>7.1f}%"
              f"{r['f1_score']:>8.3f}{r['iou']:>7.1f}%{auc:>8}{r['always_no_flood_accuracy']:>7.1f}%")
    print("=" * 100)
    print("'All-no' = accuracy of a model that always says 'no flood'. Beating it matters more than raw accuracy when floods are rare.")

    payload = {
        'suite': 'Train on India, test on other countries (Sentinel-1, Sen1Floods11 hand-labelled)',
        'trained_on': 'India chips only (CLAHE + DCGAN on training part); best epoch picked on 20% of India',
        'task': '64x64 dual-pol chip -> flood (>=10% water pixels) / no flood, threshold 0.5',
        'results': results,
        'runtime_sec': round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, 'other_countries_metrics.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(OUT_DIR, 'other_countries_predictions.csv'), 'w') as f:
        f.write('country,chip_index,water_pixels_pct,true_flood,model_probability,model_prediction\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')
    print(f"\n[+] Saved metrics, per-chip predictions and sample pictures in {OUT_DIR}")
    print(f"[DONE] {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
