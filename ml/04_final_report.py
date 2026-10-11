"""
STEP 4 - Final report.
Reads the result files written by every earlier script and prints one consolidated report:
  1. which script produced which file (and whether the file exists),
  2. dataset summary,
  3. accuracy of all 9 flood models (5 single + 4 combined),
  4. the other benchmarks (XGBoost district model, Inundation Risk Score, satellite images),
  5. the final output of the system: the districts with the highest flood risk today.
Also saves the report as ml/models/final_report.json and ml/models/final_report.md.
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)   # relative paths work from any folder (e.g. Google Colab)

# artifact -> script that produces it (this table is printed first so every file's role is visible)
ARTIFACTS = [
    ("data/processed/district_year_dataset.csv",        "ml/01_preprocess.py"),
    ("data/processed/district_profiles.json",           "ml/01_preprocess.py"),
    ("ml/models/champion_classifier.joblib",            "ml/02_train_models.py"),
    ("ml/models/severity_regressor.joblib",             "ml/02_train_models.py"),
    ("ml/models/benchmark_metrics.json",                "ml/02_train_models.py"),
    ("ml/models/feature_importance.json",               "ml/02_train_models.py"),
    ("ml/models/hybrid_benchmarks.json",                "ml/train_all_hybrid_models.py (runs train_01 to train_05)"),
    ("ml/models/unet_convlstm.pt",                      "ml/train_01_unet_convlstm.py"),
    ("ml/models/cnn_lstm.pt",                           "ml/train_02_cnn_lstm.py"),
    ("ml/models/cnn_transformer.pt",                    "ml/train_03_cnn_transformer.py"),
    ("ml/models/resnet_bilstm.pt",                      "ml/train_04_resnet_bilstm.py"),
    ("ml/models/attention_unet_lstm.pt",                "ml/train_05_attention_unet_lstm.py"),
    ("ml/models/kfold_hybrid_benchmarks.json",          "ml/train_kfold_hybrid_models.py"),
    ("ml/models/kfold_district_predictions.csv",        "ml/train_kfold_hybrid_models.py"),
    ("ml/models/combo_hybrid_benchmarks.json",          "ml/train_combo_hybrid_models.py"),
    ("ml/models/combo_district_predictions.csv",        "ml/train_combo_hybrid_models.py"),
    ("ml/models/irs_10fold_benchmarks.json",            "ml/test_10fold_irs.py"),
    ("ml/models/satellite_learning_metrics.json",       "ml/test_satellite_learning.py"),
    ("data/raw/sen1floods11_india/S1Hand",              "ml/download_sen1floods11_india.py"),
    ("ml/models/image_benchmark_metrics.json",          "ml/image_pipeline_india.py"),
    ("web/data/flood_intelligence_data.json",           "ml/03_export_web_data.py"),
    ("web/data/india_districts.geojson",                "ml/03_export_web_data.py"),
]

SINGLE_KEYS = [  # (key in kfold json architecture, label)
    ("CNNTransformer", "CNN + Transformer"), ("UNetConvLSTM", "U-Net + ConvLSTM"), ("CNNLSTM", "CNN + LSTM"),
    ("ResNetBiLSTM", "ResNet + BiLSTM"), ("AttentionUNetLSTM", "Attention U-Net + LSTM"),
]


def read_json(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full, encoding='utf-8') as f:
        return json.load(f)


def pct(v, nd=1):
    return "N/A" if v is None else f"{v:.{nd}f}%"


def main():
    lines = []   # the report text (printed and saved)
    report = {}

    def out(text=""):
        lines.append(text)
        print(text)

    out("=" * 96)
    out("AQUA HORIZON - FINAL REPORT (all pipeline outputs in one place)")
    out("=" * 96)

    # ---- 1. pipeline status ----
    out("\n1. PIPELINE FILES: which script produced which file")
    out("-" * 96)
    status = []
    for art, script in ARTIFACTS:
        full = os.path.join(ROOT, art)
        exists = os.path.exists(full)
        if exists and os.path.isfile(full):
            size = f"{os.path.getsize(full)/1024:,.0f} KB"
        elif exists:
            size = f"{len(os.listdir(full))} files"
        else:
            size = "-"
        status.append({'file': art, 'script': script, 'exists': exists})
        out(f"  {'FOUND  ' if exists else 'MISSING'}  {art:<58} {size:>9}   <- {script}")
    n_ok = sum(s['exists'] for s in status)
    out(f"  {n_ok}/{len(status)} output files present")
    report['pipeline_files'] = status

    # ---- 2. dataset ----
    out("\n2. DATASET")
    out("-" * 96)
    ds_path = os.path.join(ROOT, 'data', 'processed', 'district_year_dataset.csv')
    if os.path.exists(ds_path):
        df = pd.read_csv(ds_path, usecols=['clean_dist', 'year', 'flood_occurred'])
        out(f"  {len(df):,} district-year rows | {df['clean_dist'].nunique()} districts | years {df['year'].min()}-{df['year'].max()}")
        out(f"  Flood years: {int(df['flood_occurred'].sum()):,} ({df['flood_occurred'].mean()*100:.1f}%) | non-flood: {int((1-df['flood_occurred']).sum()):,}")
        report['dataset'] = {'rows': int(len(df)), 'districts': int(df['clean_dist'].nunique()),
                             'flood_share_pct': float(df['flood_occurred'].mean() * 100)}
    else:
        out("  (not found - run ml/01_preprocess.py)")

    # ---- 3. all 9 flood models ----
    out("\n3. FLOOD MODELS: accuracy of the 5 single models and the 4 combined models")
    out("-" * 96)
    hyb = read_json('ml/models/hybrid_benchmarks.json')
    kf = read_json('ml/models/kfold_hybrid_benchmarks.json')
    combo = read_json('ml/models/combo_hybrid_benchmarks.json')
    hold = {m['architecture']: m for m in (hyb or {}).get('models', [])}
    kfm = {m['architecture']: m for m in (kf or {}).get('models', [])}
    head = f"  {'Model':<46} | {'80:20 acc':>9} | {'10-fold acc':>13} | {'Recall':>7} | {'Prec.':>6} | {'F1':>6} | {'ROC-AUC':>7}"
    out(head)
    out("  " + "-" * (len(head) - 2))
    rows = []
    out("  Single models (older protocol: threshold picked on the test fold)")
    for arch, label in SINGLE_KEYS:
        h, k = hold.get(arch), kfm.get(arch)
        if not (h or k):
            out(f"  {label:<46} | (not found - run train_all_hybrid_models.py / train_kfold_hybrid_models.py)")
            continue
        row = {'model': label, 'group': 'single', 'protocol': 'older', 'acc_8020': h and h.get('accuracy'),
               'acc_10fold': k and k.get('mean_accuracy'), 'std': k and k.get('std_accuracy'),
               'recall': k and k.get('mean_recall'), 'precision': k and k.get('mean_precision'),
               'f1': k and k.get('mean_f1'), 'auc': k and k.get('mean_roc_auc')}
        rows.append(row)
        acc10 = f"{row['acc_10fold']:.1f}% +/-{row['std']:.1f}" if row['acc_10fold'] is not None else "N/A"
        out(f"  {label:<46} | {pct(row['acc_8020']):>9} | {acc10:>13} | {pct(row['recall']):>7} | {pct(row['precision']):>6} | "
            f"{(row['f1'] if row['f1'] is not None else float('nan')):>6.3f} | {(row['auc'] if row['auc'] is not None else float('nan')):>7.3f}")
    if combo:
        out("  Top-3 models re-scored with the stricter protocol, and their combinations")
        for m in combo['models']:
            k = m['kfold']
            tag = '' if m['type'] == 'single' else f" ({m['type']})"
            row = {'model': m['model_name'] + tag, 'group': m['type'], 'protocol': 'strict', 'acc_8020': m['holdout']['accuracy'],
                   'acc_10fold': k['mean_accuracy'], 'std': k['std_accuracy'], 'recall': k['mean_recall'],
                   'precision': k['mean_precision'], 'f1': k['mean_f1_score'], 'auc': k['mean_roc_auc']}
            rows.append(row)
            out(f"  {row['model']:<46} | {pct(row['acc_8020']):>9} | {row['acc_10fold']:.1f}% +/-{row['std']:.1f} | "
                f"{pct(row['recall']):>7} | {pct(row['precision']):>6} | {row['f1']:>6.3f} | {row['auc']:>7.3f}")
        best = max((r for r in rows if r['protocol'] == 'strict'), key=lambda r: r['auc'])
        out(f"  Highest ROC-AUC among the combined-protocol models: {best['model']} ({best['auc']:.3f})")
    else:
        out("  (combined models not found - run ml/train_combo_hybrid_models.py)")
    out("  Note: differences of 1 to 2% are within the fold-to-fold spread (the +/- value).")
    report['flood_models'] = rows

    # ---- 4. other benchmarks ----
    out("\n4. OTHER BENCHMARKS")
    out("-" * 96)
    xgb = read_json('ml/models/benchmark_metrics.json')
    if xgb:
        out("  District-data XGBoost classifier (class-balancing comparison)")
        for name, m in xgb.items():
            out(f"    {name:<26} 80:20 acc {m['holdout']['Accuracy']*100:5.1f}% | 10-fold acc {m['kfold']['Accuracy']*100:5.1f}% "
                f"+/-{m['kfold']['Accuracy_std']*100:.1f} | recall {m['kfold']['Recall']*100:5.1f}% | ROC-AUC {m['kfold']['ROC-AUC']:.3f}")
        report['xgboost'] = xgb
    irs = read_json('ml/models/irs_10fold_benchmarks.json')
    if irs:
        s = irs['summary']
        out("  Inundation Risk Score (IRS = P(flood) x ln(1 + flooded area %)), 10-fold on the real dataset, SMOTE inside each training fold only")
        out(f"    classifier accuracy {s['mean_accuracy']:.1f}% | recall {s['mean_recall']:.1f}% | ROC-AUC {s['mean_roc_auc']:.3f} | "
            f"flooded-area R2 {s['mean_sat_r2']:.1f}% (inflated, see note) | IRS rank correlation {s['mean_irs_correlation']:.3f}")
        out("    (SMOTE touches only each fold's training rows; every test fold is real and unbalanced)")
        out("    (the flooded-area R2 is inflated: dfsi_score and drainage_stress_ratio are computed from the flooded-area value itself)")
        report['irs'] = s
    sat = read_json('ml/models/satellite_learning_metrics.json')
    if sat:
        out("  District features vs satellite flooded-area %, 10-fold")
        out(f"    R2 {sat['cv_r2_mean']:.3f} (inflated, see note) | MAE {sat['cv_mae_mean']:.3f}% | top features: " +
            ", ".join(list(sat['top_features'])[:4]))
        out("    (a leakage check, not predictive skill: the top features are computed from the flooded-area value being predicted)")
        report['satellite_learning'] = sat
    img = read_json('ml/models/image_benchmark_metrics.json')
    if img:
        c = img['comparisons']
        kcv = c['kfold_comparison']['after_kfold_10fold_cv']
        out("  Sentinel-1 satellite images (real Sen1Floods11 India chips)")
        out(f"    chips: {img.get('total_chips')} | before CLAHE {c['clahe_ablation']['before_clahe']['accuracy']:.1f}% -> after CLAHE "
            f"{c['clahe_ablation']['after_clahe']['accuracy']:.1f}% | after GAN {c['gan_ablation']['after_gan']['accuracy']:.1f}%")
        out(f"    10-fold: accuracy {kcv['mean_accuracy']:.1f}% +/-{kcv['std_accuracy']:.1f} | recall {kcv['mean_recall']:.1f}% | "
            f"IoU {kcv['mean_iou']:.1f}% | ROC-AUC {kcv['mean_roc_auc']:.3f}")
        report['satellite_images'] = {'kfold': {k: v for k, v in kcv.items() if k != 'folds'}}

    # ---- 5. final output: highest-risk districts today ----
    out("\n5. FINAL OUTPUT: highest flood risk today (best combined model: all 3 top models together)")
    out("-" * 96)
    geo = read_json('web/data/india_districts.geojson')
    if geo:
        meta = read_json('web/data/flood_intelligence_data.json') or {}
        dates = meta.get('metadata', {}).get('forecast_dates', [])
        recs = []
        for f in geo['features']:
            p = f.get('properties', {})
            probs = (p.get('model_daily_probs') or {}).get('combo_top3')
            if probs and p.get('name'):
                recs.append((probs[0], p['name'], p.get('st_nm', ''), (p.get('daily_rains_mm') or [0])[0],
                             p.get('weather_zone', ''), p.get('dfsi_rank')))
        recs.sort(reverse=True)
        out(f"  Forecast date: {dates[0] if dates else 'n/a'} | districts scored: {len(recs)}")
        out(f"  {'#':>2} {'District':<26} {'State':<22} {'Risk':>6}  {'Rain':>8}  Flood-danger rank")
        top = []
        for i, (pr, name, st, rain, zone, rk) in enumerate(recs[:10], 1):
            out(f"  {i:>2} {name:<26} {st:<22} {pr*100:>5.1f}%  {rain:>6.1f}mm  #{rk}")
            top.append({'district': name, 'state': st, 'risk_pct': round(pr * 100, 1), 'rain_mm': rain})
        levels = {'Severe (>=70%)': 0, 'High (45-70%)': 0, 'Moderate (25-45%)': 0, 'Low (<25%)': 0}
        for pr, *_ in recs:
            levels['Severe (>=70%)' if pr >= .70 else 'High (45-70%)' if pr >= .45 else 'Moderate (25-45%)' if pr >= .25 else 'Low (<25%)'] += 1
        out("  Districts by alert level today: " + " | ".join(f"{k}: {v}" for k, v in levels.items()))
        report['top_risk_districts'] = top
        report['alert_levels_today'] = levels
    else:
        out("  (not found - run ml/03_export_web_data.py)")

    out("\n" + "=" * 96)
    out("End of report. Start the website with:  python run_app.py")
    out("=" * 96)

    os.makedirs(os.path.join(ROOT, 'ml', 'models'), exist_ok=True)
    with open(os.path.join(ROOT, 'ml', 'models', 'final_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=lambda o: None if o is None else float(o))
    with open(os.path.join(ROOT, 'ml', 'models', 'final_report.md'), 'w', encoding='utf-8') as f:
        f.write("```\n" + "\n".join(lines) + "\n```\n")
    print("\nSaved: ml/models/final_report.json and ml/models/final_report.md")


if __name__ == '__main__':
    main()
