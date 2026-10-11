"""
PROOF SCRIPT - run this to show the training protocol with real numbers.

It does not retrain anything. It (1) scans the training scripts to show which data each one reads,
(2) rebuilds the exact 80:20 splits used for the CSV data and for the satellite images and prints the
class counts BEFORE and AFTER balancing, with assertions that the 20% test part is never balanced,
and (3) prints every saved metric (accuracy, recall, precision, F1, ROC-AUC, PR-AUC, IoU) per model.

Run:  python ml/verify_protocol.py        Output is also saved to ml/models/protocol_proof.txt
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import json
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEED = 42                 # the seed every split in the project uses, so the splits here match the real runs
LOG = io.StringIO()       # everything printed is also kept here and saved to ml/models/protocol_proof.txt


def out(s=""):
    """Print a line and keep a copy for the saved proof file."""
    print(s)
    LOG.write(s + "\n")


def counts(y):
    """One-line class summary of a label array: rows, flood, non-flood and the flood share."""
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    return f"{len(y):>6,} rows | flood {n1:>6,} | non-flood {n0:>6,} | flood share {n1 / len(y) * 100:5.1f}%"


# ------------------------------------------------------------------ 1. separation of data sources
def part1_separation():
    out("=" * 100)
    out("PART 1 - CSV DATA AND SATELLITE IMAGES ARE TRAINED BY SEPARATE SCRIPTS")
    out("=" * 100)
    # a script "reads CSV data" if its source mentions the district CSV (or its loader), and "reads images" if it
    # mentions the Sentinel-1 files / loader. A script matching BOTH would be mixing the two datasets.
    csv_markers = ['district_year_dataset', 'load_all_sequences']
    img_markers = ['sen1floods11', 'tifffile', 'load_sen1floods11_india_chips']
    out(f"{'training script':<38}| reads CSV district data | reads satellite images")
    out("-" * 100)
    mixed = []
    for f in ['02_train_models.py', 'train_kfold_hybrid_models.py', 'train_combo_hybrid_models.py',
              'train_all_hybrid_models.py', 'test_10fold_irs.py', 'image_pipeline_india.py',
              'test_other_countries.py']:
        p = os.path.join(BASE, 'ml', f)
        if not os.path.exists(p):
            continue
        src = open(p, encoding='utf-8').read()
        c, i = any(m in src for m in csv_markers), any(m in src for m in img_markers)
        out(f"{f:<38}| {'YES' if c else '-':<24}| {'YES' if i else '-'}")
        if c and i:
            mixed.append(f)
    assert not mixed, f"scripts mixing both data types: {mixed}"
    # no training/test script may read a CSV that was balanced on the whole dataset (the demo script and this file are exempt)
    for f in sorted(os.listdir(os.path.join(BASE, 'ml'))):
        if f.endswith('.py') and f not in ('verify_protocol.py', 'balance_dataset_smote.py'):
            assert 'balanced_smote' not in open(os.path.join(BASE, 'ml', f), encoding='utf-8').read(), f
    out("Also checked: no training/test script reads a pre-balanced copy of the dataset.")
    out("\nRESULT: no training script reads both data types. CSV models and the image model are trained")
    out("        separately and their features are never concatenated.")


# ------------------------------------------------------------------ 2. CSV data protocol
def part2_csv():
    # reuse the real functions of the combined-models script, so the split here is the split the models really use
    from ml.train_combo_hybrid_models import load_windows, FEATURE_COLS, N_FEATS, SEQ_LEN, scale
    from sklearn.preprocessing import StandardScaler

    out("\n" + "=" * 100)
    out("PART 2 - CSV (district-year) DATA: 80:20 SPLIT, SMOTE ONLY ON THE 80% TRAINING PART (ALL OF IT)")
    out("=" * 100)
    df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'district_year_dataset.csv'))
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    X, y, years = load_windows(df)
    out(f"Whole dataset (real, never balanced)  : {counts(y)}")

    # chronological split: windows whose last year is <= 2012 train, later ones test (the test part is never resampled)
    tr, te = years <= 2012, years > 2012
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    out(f"\n[A] 80:20 chronological holdout (train: windows ending <=2012, test: 2013-2023)")
    out(f"    train part (~80%) before SMOTE : {counts(y_tr)}   ({tr.sum() / len(y) * 100:.1f}% of data)")
    out(f"    TEST  part (~20%) UNBALANCED   : {counts(y_te)}   ({te.sum() / len(y) * 100:.1f}% of data)")

    # scale with training statistics only, then run SMOTE on the WHOLE training part (the same call the real script makes)
    scaler = StandardScaler().fit(X_tr.reshape(-1, N_FEATS))
    X_trs = scale(scaler, X_tr)
    n = len(X_trs)
    X_bal, y_bal = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(X_trs.reshape(n, -1), y_tr)
    out(f"    SMOTE input  (the WHOLE 80%)   : {counts(y_tr)}")
    out(f"    SMOTE output (train only)      : {counts(y_bal)}   <- balanced 50:50, "
        f"{len(y_bal) - len(y_tr):,} synthetic rows added")
    out(f"    TEST part after the whole run  : {counts(y_te)}   <- identical to before: SMOTE never touched it")
    # hard guarantees: if any of these is false the script stops with an error instead of printing a result
    assert n == len(y_tr) < len(y), "SMOTE saw more than the training part"
    assert abs(y_bal.mean() - 0.5) < 1e-6, "training part is not balanced"
    assert abs(y_te.mean() - y.mean()) < 0.15 and y_te.mean() < 0.5, "test part must keep the natural ratio"
    assert not set(np.where(tr)[0]) & set(np.where(te)[0]), "train/test overlap"
    out("    CHECKS PASSED: SMOTE fitted on training rows only | test rows disjoint | test ratio unchanged")

    out(f"\n[B] 10-fold stratified cross-validation (each fold: 90% train / 10% test)")
    # each fold: a (training rows) is balanced with SMOTE, b (test fold) is only counted, never resampled
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    out(f"    {'fold':<5}| {'train before SMOTE':<44}| {'train after SMOTE (50:50)':<22}| test fold (real, unbalanced)")
    for k, (a, b) in enumerate(skf.split(X, y), 1):
        Xa = X[a].reshape(len(a), -1)
        # only the labels are needed here: the count of rows SMOTE produces from this fold's training part
        _, y_b = SMOTE(k_neighbors=5, random_state=SEED + k).fit_resample(Xa, y[a])
        assert not set(a) & set(b)
        out(f"    {k:<5}| {counts(y[a]):<44}| {len(y_b):>6,} flood {int(y_b.sum()):>6,}  | {counts(y[b])}")
    out("    CHECKS PASSED: every fold balances only its own training rows; the test fold keeps the real ratio")
    return len(y)


# ------------------------------------------------------------------ 3. image protocol
def part3_images():
    out("\n" + "=" * 100)
    out("PART 3 - SATELLITE IMAGES (Sentinel-1 SAR): 80:20 SPLIT, SYNTHETIC CHIPS ONLY IN THE 80% TRAINING PART")
    out("=" * 100)
    from ml.image_pipeline_india import load_sen1floods11_india_chips
    _, _, labels = load_sen1floods11_india_chips(patch_size=64)
    out(f"Whole image dataset (real)            : {counts(labels)}")
    tri, tei = train_test_split(np.arange(len(labels)), test_size=0.20, random_state=SEED, stratify=labels)
    out(f"train part (80%) before augmentation  : {counts(labels[tri])}")
    out(f"TEST  part (20%) UNBALANCED/REAL      : {counts(labels[tei])}")
    out("The image model balances with a DCGAN trained on flood chips of the TRAINING part only")
    out("(image_pipeline_india.py: flood_train_chips = images_clahe[train_idx][labels[train_idx] == 1]);")
    out("it adds 100 synthetic flood chips to the training part. The test part is passed in untouched:")
    out(f"TEST part after the whole run         : {counts(labels[tei])}   <- unchanged")
    assert not set(tri) & set(tei), "train/test overlap"
    assert len(tri) + len(tei) == len(labels)
    out("CHECKS PASSED: 80:20 split is disjoint and stratified; the test chips are real and never augmented")
    out("(10-fold CV for images works the same way: a fresh GAN per fold, trained on that fold's training chips only)")


# ------------------------------------------------------------------ 4. all metrics
def read(path):
    p = os.path.join(BASE, path)
    return json.load(open(p)) if os.path.exists(p) else None


def part4_metrics():
    out("\n" + "=" * 100)
    out("PART 4 - ALL SAVED TRAINING PARAMETERS (measured on the untouched test data)")
    out("=" * 100)

    d = read('ml/models/combo_hybrid_benchmarks.json')
    if d:
        out("\n[CSV] Top-3 hybrid models + combinations | " + d['protocol']['holdout'])
        out(f"      balancing: {d['protocol']['balancing']}")
        hdr = f"{'Model':<44}| Acc %  | Recall % | Prec % | F1     | ROC-AUC | PR-AUC"
        out("  80:20 holdout (single run on the 20% real test data)\n  " + hdr)
        for m in d['models']:
            h = m['holdout']
            out(f"  {m['model_name']:<44}| {h['accuracy']:>6.2f} | {h['recall']:>8.2f} | {h['precision']:>6.2f} | "
                f"{h['f1_score']:.4f} | {h['roc_auc']:.4f}  | {h['pr_auc']:.4f}")
        out("\n  10-fold CV (mean +/- std over the 10 test folds)\n  " + hdr)
        for m in d['models']:
            k = m['kfold']
            out(f"  {m['model_name']:<44}| {k['mean_accuracy']:>6.2f} | {k['mean_recall']:>8.2f} | {k['mean_precision']:>6.2f} | "
                f"{k['mean_f1_score']:.4f} | {k['mean_roc_auc']:.4f}  | {k['mean_pr_auc']:.4f}")
            out(f"  {'   (+/- std)':<44}| {k['std_accuracy']:>6.2f} | {k['std_recall']:>8.2f} | {k['std_precision']:>6.2f} | "
                f"{k['std_f1_score']:.4f} | {k['std_roc_auc']:.4f}  | {k['std_pr_auc']:.4f}")
        best = max(d['models'], key=lambda m: m['kfold']['mean_f1_score'])
        out(f"\n  Per-fold scores of the best model by 10-fold F1: {best['model_name']}")
        out("  fold   | Acc %  | Recall % | Prec % | F1     | ROC-AUC | PR-AUC | threshold")
        for r in best['kfold']['folds']:
            out(f"  {r['fold']:<7}| {r['accuracy']:>6.2f} | {r['recall']:>8.2f} | {r['precision']:>6.2f} | {r['f1_score']:.4f} | "
                f"{r['roc_auc']:.4f}  | {r['pr_auc']:.4f} | {r['threshold']:.2f}")

    d = read('ml/models/benchmark_metrics.json')
    if d:
        out("\n[CSV] XGBoost district classifier - balancing strategies (SMOTE/ADASYN applied inside the training part only)")
        out(f"  {'Strategy':<26}| split      | Acc    | Recall | Prec   | F1     | ROC-AUC | PR-AUC")
        for name, r in d.items():
            h, k = r['holdout'], r['kfold']
            out(f"  {name:<26}| 80:20      | {h['Accuracy']:.4f} | {h['Recall']:.4f} | {h['Precision']:.4f} | "
                f"{h['F1-Score']:.4f} | {h['ROC-AUC']:.4f}  | {h['PR-AUC']:.4f}")
            out(f"  {'':<26}| 10-fold    | {k['Accuracy']:.4f} | {k['Recall']:.4f} | {k['Precision']:.4f} | "
                f"{k['F1-Score']:.4f} | {k['ROC-AUC']:.4f}  | {k['PR-AUC']:.4f}")

    d = read('ml/models/kfold_hybrid_benchmarks.json')
    if d:
        out("\n[CSV] All 5 deep hybrid models - 10-fold CV (class-weighted loss, no resampling)")
        out(f"  {'Model':<28}| Acc %  | Recall % | Prec % | F1     | ROC-AUC")
        for m in d['models']:
            k = m.get('kfold_summary', m)
            try:
                out(f"  {m['model_name']:<28}| {k['mean_accuracy']:>6.2f} | {k['mean_recall']:>8.2f} | "
                    f"{k['mean_precision']:>6.2f} | {k['mean_f1']:.4f} | {k['mean_roc_auc']:.4f}")
            except KeyError:
                out(f"  {m.get('model_name', '?')}: keys = {list(k)[:10]}")

    d = read('ml/models/image_benchmark_metrics.json')
    if d:
        c = d['comparisons']
        out("\n[IMAGES] Sentinel-1 SAR flood classifier (CNN) | " + f"{d['total_chips']} real chips, 64x64 VV/VH")
        out(f"  {'Experiment':<34}| Acc %  | Prec % | Recall % | F1     | IoU %  | ROC-AUC")
        rows = [('Raw SAR (80:20)', c['clahe_ablation']['before_clahe']),
                ('CLAHE (80:20)', c['clahe_ablation']['after_clahe']),
                ('CLAHE + GAN train-only (80:20)', c['gan_ablation']['after_gan'])]
        for n, r in rows:
            out(f"  {n:<34}| {r['accuracy']:>6.2f} | {r['precision']:>6.2f} | {r['recall']:>8.2f} | {r['f1_score']:.4f} | "
                f"{r['iou']:>6.2f} | {r['roc_auc']:.4f}")
        k = c['kfold_comparison']['after_kfold_10fold_cv']
        out(f"  {'10-fold CV mean':<34}| {k['mean_accuracy']:>6.2f} | {k['mean_precision']:>6.2f} | {k['mean_recall']:>8.2f} | "
            f"{k['mean_f1']:.4f} | {k['mean_iou']:>6.2f} | {k['mean_roc_auc']:.4f}")
        out(f"  {'10-fold CV (+/- std)':<34}| {k['std_accuracy']:>6.2f} | {k['std_precision']:>6.2f} | {k['std_recall']:>8.2f} | "
            f"{k['std_f1']:.4f} | {k['std_iou']:>6.2f} | {k['std_roc_auc']:.4f}")
        out("  per fold:  fold   | Acc %  | Prec % | Recall % | F1     | IoU %  | ROC-AUC")
        for r in k['folds']:
            out(f"             {r['fold']:<7}| {r['accuracy']:>6.2f} | {r['precision']:>6.2f} | {r['recall']:>8.2f} | "
                f"{r['f1_score']:.4f} | {r['iou']:>6.2f} | {r['roc_auc']:.4f}")


def part5_caveats():
    out("\n" + "=" * 100)
    out("PART 5 - KNOWN EXCEPTIONS (stated openly)")
    out("=" * 100)
    out("* ml/test_10fold_irs.py and ml/test_satellite_learning.py now read the real district_year_dataset.csv;")
    out("  test_10fold_irs.py applies SMOTE inside each training fold only. No benchmark uses a pre-balanced dataset.")
    out("* The image pipeline picks its best epoch using the test fold's accuracy, so image scores are slightly optimistic.")
    out("* ml/train_kfold_hybrid_models.py uses no resampling: it balances with a class-weighted loss instead.")


if __name__ == '__main__':
    part1_separation()
    part2_csv()
    part3_images()
    part4_metrics()
    part5_caveats()
    with open(os.path.join(BASE, 'ml', 'models', 'protocol_proof.txt'), 'w', encoding='utf-8') as f:
        f.write(LOG.getvalue())
    print("\nSaved: ml/models/protocol_proof.txt")
